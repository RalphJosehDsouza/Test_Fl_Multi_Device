"""
Drives the real ServerApp and the real ClientApp in one process: no SuperLink, no network.

The federation used to stall on the second tree with "no client returned statistics after 3
resync attempts". That is a statement about the *protocol*, not about the machines, so it can
be reproduced and settled here in seconds instead of on three devices with a shared printer.

So this script hands ``server_app.main`` a Grid whose ``send_and_receive`` is answered by
``client_app.train`` in the same process, over the two real shards and the real schema. The
ServerApp loop, the client app, the binning and the boosting maths are all the production
code path; only the transport is replaced. It complements ``simulate.py``, which drives
``server_core`` and therefore never exercises the messages at all.

Three runs, each one an assertion:

    clean    two clients, N trees        the run finishes, every tree is grown from the
                                         statistics of *both* clients (no silent shrinking
                                         to one machine), and each node really did read its
                                         own shard - including the node whose data-path
                                         cannot be resolved
    drift    one client comes back       the server notices, rebuilds it once, and the model
             cold at a chosen round      comes out byte-identical to the clean run, so the
             (restart mid-run)           recovery is exact rather than approximate
    silent   the same cold client,       the server sees NO_REPLY, retries with the
             but it does not answer      authoritative forest, and still finishes

    python fl/protocol_check.py                     # 12 trees, injections at tree 3
    python fl/protocol_check.py --trees 60          # the demo's size
    python fl/protocol_check.py --clean-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from flwr.common import Context, Error, Message, RecordDict
from flwr.serverapp.grid import Grid
from flwr.supercore.task_identity import TaskIdentity

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fl"))

from upi_fl import client_app, server_app  # noqa: E402

FL_DIR = ROOT / "data" / "fl"
W = 78
# One SuperNode per shard, named the way the demo names them: partition-id N wants
# client_N.csv. Node 1's data-path is deliberately a relative path that cannot be resolved
# from anywhere - the config that killed the run at ``hello`` with FileNotFoundError. The
# client app now falls back to the node's own standard shard name, so it must still load
# client_1.csv (150,345 rows for node 0 and 132,444 for node 1; see the shard manifest).
NODES = {
    0: {"partition-id": 0, "num-partitions": 2, "data-path": str(FL_DIR / "client_0.csv")},
    1: {"partition-id": 1, "num-partitions": 2, "data-path": "shards/client_1.csv"},
}


def rule(char: str = "-") -> None:
    print(char * W)


def context_for(nid: int, run_config: dict, node_config: dict | None = None) -> Context:
    """The Context the runtime would hand node *nid*."""
    return Context(run_id=TaskIdentity.run_id, node_id=nid,
                   node_config=dict(NODES[nid] if node_config is None else node_config),
                   state=RecordDict(), run_config=dict(run_config))


def check_shard_guard() -> bool:
    """Two nodes pointed at one shard must be refused, not silently share a score buffer.

    One process can serve several nodes, and one node advancing a shared score would drag the
    other out of step - the same drift the protocol now detects, but caused by configuration
    and invisible in the totals.
    """
    run_config = {"schema-path": str(ROOT / "fl" / "upi_fl" / "schema.json")}
    saved = dict(client_app._STATE)
    client_app._STATE.clear()
    try:
        client_app._state(context_for(0, run_config))
        twin = dict(NODES[0], **{"partition-id": 1})       # node 1 sent node 0's shard
        try:
            client_app._state(context_for(1, run_config, twin))
        except RuntimeError as exc:
            return "own shard" in str(exc)
        return False
    finally:
        client_app._STATE.clear()
        client_app._STATE.update(saved)


class LocalGrid(Grid):
    """A Grid that answers from this very process instead of from a SuperLink.

    One process plays every node - which is exactly what the in-process federation does -
    so the full message protocol runs without any networking. It is also the injection
    point for the two failure modes: a client that comes back cold, and a client that goes
    silent.
    """

    def __init__(self, run_config: dict, *, cold_at=None, cold_node: int = 1,
                 silent: bool = False, cold_process: bool = False) -> None:
        self._run_config = run_config
        self.cold_at = cold_at        # (tree-index, level) at which to cold-start a client
        self.cold_node = cold_node    # which client that is (one restart, not two)
        self.silent = silent          # ... and whether that client also stops answering
        self.cold_process = cold_process   # true = every message in a brand new process
        self._run = None
        self.rounds = 0               # server<->client exchanges (one send_and_receive each)
        self.messages = 0             # individual requests carried inside those exchanges
        self.colds = 0                # times a client really was cold-started
        self.fresh_starts = 0         # times a node began a message with empty memory
        self.expected_gt0 = 0         # requests that ask a client to hold >0 tree(s) (retries too)
        self.rejected = 0             # replies that came back as need-resync
        self.missing = 0              # replies that never came back at all
        self.levels: dict[tuple[int, int, int], int] = {}   # (node, tree, level) -> requests

    # -- Grid plumbing ------------------------------------------------------------------
    def set_run(self, run) -> None:
        self._run = run

    @property
    def run(self):
        return self._run

    def get_node_ids(self) -> list[int]:
        return sorted(NODES)

    def create_message(self, content, message_type: str, dst_node_id: int,
                       group_id: str, ttl: float | None = None) -> Message:
        return Message(content, dst_node_id, message_type, group_id=group_id, ttl=ttl)

    def push_messages(self, messages):
        raise NotImplementedError("this check answers from this process, not a SuperLink")

    def pull_messages(self, message_ids):
        raise NotImplementedError("this check answers from this process, not a SuperLink")

    def send_and_receive(self, messages, *, timeout: float | None = None) -> list[Message]:
        """Answer every request here and now. One process plays every node."""
        msgs = list(messages)
        self.messages += len(msgs)
        if any(m.content["config"]["op"] == "hist" for m in msgs):
            self.rounds += 1          # the hello round is an exchange too, but not a round
        replies: list[Message] = []
        for msg in msgs:
            replies.extend(self._deliver(msg))
        return replies

    # -- the transport itself -----------------------------------------------------------
    def _deliver(self, msg: Message) -> list[Message]:
        """Answer one request by calling the production client app on this node's shard."""
        nid = msg.metadata.dst_node_id
        cfg = msg.content["config"]
        cold = False
        if self.cold_process:
            # The runtime starts this app in a new process for every message, so nothing the
            # previous message left behind is still here. Model that before the message runs.
            self._fresh_process(nid)
        if cfg["op"] == "hist":
            if int(cfg["expected-trees"]) > 0:
                self.expected_gt0 += 1
            key = (nid, int(cfg["tree-index"]), int(cfg["level"]))
            self.levels[key] = self.levels.get(key, 0) + 1
            cold = self._cold_start(nid, int(cfg["tree-index"]), int(cfg["level"]))
        context = context_for(nid, self._run_config)
        try:
            reply = client_app.train(msg, context)
        except Exception as exc:            # reported the way the runtime reports it
            print(f"    node {nid}: {type(exc).__name__}: {exc}", flush=True)
            return [Message(Error(code=1, reason=f"{type(exc).__name__}: {exc}"),
                            reply_to=msg)]
        if cold and self.silent:
            self.missing += 1
            print(f"    node {nid}: [inject] went silent for this round", flush=True)
            return []
        if int(reply.content["metrics"].get("need-resync", 0)):
            self.rejected += 1
        return [reply]

    def _fresh_process(self, nid: int) -> None:
        """Empty one node's memory, the way the runtime does before every message.

        A SuperNode runs the ClientApp in a new process per message, so no module-level
        cache survives: the shard is binned again and the score starts empty. That is not a
        fault to inject - it is how the real federation differs from this harness - so it is
        modelled by rebuilding the state rather than by corrupting it. The parsed shard is
        reused purely to keep the check fast; the state it is put into (empty score, zero
        trees) is exactly a new process's, which is the only part the protocol can see.
        """
        state = client_app._STATE.get(NODES[nid]["partition-id"])
        if state is None:
            return
        client_app._STATE.pop(NODES[nid]["partition-id"])
        self.fresh_starts += 1
        state["shard"].raw[:] = 0.0
        state["shard"].trees_seen = 0

    def _cold_start(self, nid: int, k: int, level: int) -> bool:
        """Forget everything one client knew, as a SuperNode restart would.

        The client is put back on an empty score with zero finished trees while the server
        still believes it is up to date - the exact mismatch that used to abort the run.
        The injection fires once: it models a restart, so it must not wipe the client again
        when the server retries, or the retry could never succeed.
        """
        if self.cold_at != (k, level) or nid != self.cold_node:
            return False
        state = client_app._STATE.get(NODES[nid]["partition-id"])
        if state is None:
            return False
        self.cold_at = None               # a restart happens once, not on every attempt
        shard = state["shard"]
        shard.raw[:] = 0.0
        shard.trees_seen = 0
        self.colds += 1
        print(f"    node {nid}: [inject] cold start at tree {k} level {level} "
              f"(the server expects it to hold {k} tree(s))", flush=True)
        return True


def run_case(name: str, trees: int, out: Path, *, depth: int = 4,
             cold_at=None, silent: bool = False, cold_process: bool = False) -> dict:
    """Run the real ServerApp once, over the real client app, and report on the exchange."""
    run_config = {
        "num-trees": trees, "max-depth": depth, "learning-rate": 0.2, "lambda-l2": 1.0,
        "min-child-samples": 50, "min-split-gain": 1e-6,
        "eval-every-trees": 0, "aggregation": "sum",
        "schema-path": str(ROOT / "fl" / "upi_fl" / "schema.json"),
        "out-dir": str(out),
    }
    grid = LocalGrid(run_config, cold_at=cold_at, silent=silent, cold_process=cold_process)
    print()
    rule("=")
    print(f" {name.upper()}".center(W))
    rule("=")
    started = time.perf_counter()
    server_app.main(grid, Context(run_id=TaskIdentity.run_id, node_id=0, node_config={},
                                  state=RecordDict(), run_config=run_config))
    return {
        "name": name,
        "grid": grid,
        "seconds": time.perf_counter() - started,
        "log": json.loads((out / "reports" / "fl" / "fl_rounds.json").read_text()),
        "model": (out / "models" / "fl" / "global_final.json").read_text(),
    }


def first_difference(a: str, b: str) -> str:
    """Where two documents start to disagree, for a failure message that is worth reading."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"first disagreement at character {i}"
    return "one document is a prefix of the other"


def repeat_reports(grid: LocalGrid) -> str:
    """The (node, tree, level) requests that were sent more than once, or 'none'."""
    repeats = sorted(key for key, n in grid.levels.items() if n > 1)
    return ", ".join(f"node {nid} tree {k} level {lv} x{grid.levels[(nid, k, lv)]}"
                     for nid, k, lv in repeats) or "none"


def check(clean: dict, cases: list[dict], trees: int) -> int:
    """Every claim this script makes, in one place. Returns the number of failures."""
    failures = 0

    def ok(label: str, passed: bool, detail: str = "") -> None:
        nonlocal failures
        print(f"  [{'ok  ' if passed else 'FAIL'}] {label}"
              + ("" if passed else f"  <- {detail}"))
        failures += 0 if passed else 1

    grid = clean["grid"]
    print()
    rule("=")
    print(" RESULT".center(W))
    rule("=")
    ok(f"clean run finished all {trees} trees", len(clean["log"]["trees"]) == trees,
       f"{len(clean['log']['trees'])} trees in the log")
    ok("clean run needed no resend or rebuild",
       (grid.rejected, grid.missing, grid.colds) == (0, 0, 0),
       f"rejected={grid.rejected} missing={grid.missing} cold={grid.colds}")
    ok("the report counts the same exchanges the transport did",
       clean["log"]["federation_rounds"] == grid.rounds,
       f"report {clean['log']['federation_rounds']} vs transport {grid.rounds}")
    manifest = json.loads((FL_DIR / "partition_summary.json").read_text())
    want = {int(c["client"]): int(c["rows"]) for c in manifest["clients"]}
    got = {nid: int(clean["log"]["nodes"][str(nid)]["rows"]) for nid in NODES}
    ok("each node read its own shard, though node 1's data-path is unusable", got == want,
       f"rows {got}, shard manifest {want}")
    ok("clean run asked each client for each level exactly once",
       set(grid.levels.values()) == {1},
       f"repeat requests: {repeat_reports(grid)}")

    # A rejection and a lost message cost the same single retry; they differ only in how
    # the server learns about them (the client said so, or it said nothing at all).
    expected = {"clean": (grid.rounds, 0, 0), "drift": (grid.rounds + 1, 1, 0),
                "silent": (grid.rounds + 1, 0, 1)}
    for case in cases:
        name = case["name"].split(":")[0]
        g = case["grid"]
        if name == "stateless":
            # Not an injected fault: this is how the runtime runs the client app, so the
            # budget is not "one retry" but "one retry per round that carries state".
            ok("stateless: a fresh client really did start every message from scratch",
               g.fresh_starts >= 2 * trees, f"{g.fresh_starts} fresh start(s)")
            # Every rejected round costs exactly one extra exchange (both nodes together),
            # and the report's own recovery count has to agree with what the transport did.
            rebuilt = case["log"]["client_recoveries"]
            ok("stateless: every rebuild cost exactly one extra exchange, no more",
               g.rounds == clean["grid"].rounds + rebuilt and 0 < rebuilt <= g.expected_gt0,
               f"rounds={g.rounds} vs clean {clean['grid'].rounds} + {rebuilt} rebuild(s)")
            ok("stateless: both clients were rebuilt on every rebuilt round",
               g.rejected == 2 * rebuilt,
               f"{g.rejected} rejection(s) for {rebuilt} rebuilt round(s)")
            ok("stateless: no single request was ever sent more than twice",
               set(g.levels.values()) <= {1, 2},
               f"repeat requests: {repeat_reports(g)}")
            ok("stateless: the report counts the same exchanges the transport did",
               case["log"]["federation_rounds"] == g.rounds,
               f"report {case['log']['federation_rounds']} vs transport {g.rounds}")
        else:
            rounds, rejected, missing = expected[name]
            ok(f"{name}: recovery cost exactly one extra exchange per injection",
               (g.rounds, g.rejected, g.missing) == (rounds, rejected, missing),
               f"rounds={g.rounds} (want {rounds}), rejected={g.rejected} (want {rejected}), "
               f"missing={g.missing} (want {missing})")
            repeats = sorted(n for n in g.levels.values() if n > 1)
            ok(f"{name}: the retry happened once, at one level, for one client",
               repeats == ([] if name == "clean" else [2]),
               f"repeat requests: {repeat_reports(g)}")
            if name != "clean":
                ok(f"{name}: the client really was cold and had to be rebuilt",
                   g.colds == 1, f"{g.colds} cold start(s)")
        lonely = [t["tree"] for t in case["log"]["trees"]
                  if len(t["participants"]) < len(NODES)]
        ok(f"{name}: every tree still used both clients' statistics", not lonely,
           f"trees {lonely} averaged over one machine")
        ok(f"{name}: model identical to the clean run", case["model"] == clean["model"],
           first_difference(clean["model"], case["model"]))
    ok("two nodes pointed at one shard are refused, not merged", check_shard_guard(),
       "the second node silently shared the first node's score buffer")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Drive server_app and client_app in-process")
    parser.add_argument("--trees", type=int, default=12, help="trees to grow")
    parser.add_argument("--depth", type=int, default=4, help="max depth per tree")
    parser.add_argument("--at", type=int, default=3,
                        help="tree at whose first level to inject the failure")
    parser.add_argument("--clean-only", action="store_true",
                        help="skip the two injected runs")
    args = parser.parse_args()

    # The runtime sets these before an app sees its first message; messages need a run id.
    TaskIdentity.task_id, TaskIdentity.run_id, TaskIdentity.node_id = 1, 1, 0

    started = time.perf_counter()
    out_root = ROOT / "fl_output" / "protocol_check"
    clean = run_case("clean: two clients, in step", args.trees, out_root / "clean",
                     depth=args.depth)
    cases = [clean]
    if not args.clean_only:
        cases.append(run_case("drift: one client comes back cold", args.trees,
                              out_root / "drift", depth=args.depth,
                              cold_at=(args.at, 0)))
        cases.append(run_case("silent: the cold client also stops answering", args.trees,
                              out_root / "silent", depth=args.depth,
                              cold_at=(args.at, 0), silent=True))
        cases.append(run_case("stateless: every message in a new process", args.trees,
                              out_root / "stateless", depth=args.depth,
                              cold_process=True))

    failures = check(clean, cases, args.trees)
    rule("=")
    print(f"  {len(cases)} run(s), {args.trees} trees each, "
          f"{time.perf_counter() - started:.1f}s total")
    print("  the federation cannot drift out of step and cannot silently shrink to fewer"
          " machines" if not failures else f"  {failures} check(s) FAILED")
    rule("=")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

