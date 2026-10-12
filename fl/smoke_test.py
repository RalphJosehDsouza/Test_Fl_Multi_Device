"""Run a three-tree smoke test using real Flower processes on this machine."""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def command(name):
    entry = next(e for e in importlib.metadata.distribution("flwr").entry_points
                 if e.name == name)
    return [sys.executable, "-c",
            f"from {entry.module} import {entry.attr}; {entry.attr}()"]


def wait_for(predicate, processes, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        if any(p.poll() is not None for p in processes):
            raise RuntimeError("Flower process exited early; inspect the saved logs")
        time.sleep(0.5)
    raise TimeoutError("Flower startup timed out; inspect the saved logs")


def main():
    if importlib.metadata.version("flwr") != "1.39.0":
        raise RuntimeError("This smoke test requires Flower 1.39.0")
    output = ROOT / "reports" / "fl" / "smoke"
    output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="run-", dir=output)).resolve()
    print(f"Smoke test logs and outputs: {work}", flush=True)
    home = work / "flower"
    home.mkdir()
    env = {**os.environ, "FLWR_HOME": str(home), "PYTHONUNBUFFERED": "1",
           "PYTHONIOENCODING": "utf-8", "FLWR_TELEMETRY_ENABLED": "0"}
    # Reserve distinct free ports before launching. Another process can still race
    # after release; startup checks surface that as a failed smoke test.
    reservations = [socket.socket() for _ in range(4)]
    try:
        for sock in reservations:
            sock.bind(("127.0.0.1", 0))
        control, fleet, *client_ports = [s.getsockname()[1] for s in reservations]
    finally:
        for sock in reservations:
            sock.close()
    (home / "config.toml").write_text(
        '[superlink]\ndefault = "smoke"\n[superlink.smoke]\n'
        f'address = "127.0.0.1:{control}"\ninsecure = true\n', encoding="utf-8")
    overrides = work / "run.toml"
    overrides.write_text(
        'num-trees = 3\nmax-depth = 2\nmin-child-samples = 2\n'
        'min-clients = 2\neval-every-trees = 0\nforest-every-level = 1\n'
        f'out-dir = {json.dumps(work.as_posix())}\n', encoding="utf-8")
    schema = json.loads((ROOT / "fl/upi_fl/schema.json").read_text())
    for part in range(2):
        frame = pd.DataFrame(0, index=range(64), columns=schema["features"])
        frame["amount (INR)"] = np.tile([100, 10000], 32)
        frame["fraud_flag"] = np.tile([0, 1], 32)
        frame.to_csv(work / f"client_{part}.csv", index=False)

    processes, handles = [], []

    def start(name, args):
        handle = (work / f"{name}.log").open("w", encoding="utf-8")
        handles.append(handle)
        process = subprocess.Popen(args, cwd=work, env=env, stdout=handle,
                                   stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        processes.append(process)
        return process

    try:
        start("server", command("flower-superlink") + ["--insecure", "--host", "127.0.0.1",
            "--port", str(control), "--fleet-api-address", f"127.0.0.1:{fleet}",
            "--disable-runtime-dependency-installation"])
        for part, port in enumerate(client_ports):
            start(f"client-{part}", command("flower-supernode") + ["--insecure",
                "--superlink", f"127.0.0.1:{fleet}", "--port", str(port),
                "--node-config", f"partition-id={part} num-partitions=2 "
                f"data-path={json.dumps((work / f'client_{part}.csv').as_posix())}"])
        wait_for(lambda: all("SuperNode ID:" in (work / f"client-{p}.log").read_text(
            encoding="utf-8", errors="replace") for p in range(2)), processes)
        print("Two real client processes connected. Submitting three trees...", flush=True)
        run = start("run", command("flwr") + ["run", str(ROOT / "fl"), "smoke",
                    "--stream", "--run-config", str(overrides)])
        try:
            run.wait(timeout=300)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("Training exceeded five minutes; inspect run.log") from exc
        if run.returncode:
            raise RuntimeError(f"Flower run exited with code {run.returncode}; inspect run.log")
        report = json.loads((work / "reports/fl/fl_rounds.json").read_text())
        if len(report["trees"]) != 3 or len(report["nodes"]) != 2:
            raise RuntimeError("Expected three trees and two participating clients")
        expected = {int(n) for n in report["nodes"]}
        if any(set(t["participants"]) != expected for t in report["trees"]):
            raise RuntimeError("A tree is missing a client's contribution")
        model = json.loads((work / "models/fl/global_final.json").read_text())
        if len(model["trees"]) != 3 or not any(len(t["feature"]) > 1 for t in model["trees"]):
            raise RuntimeError("Smoke model must contain three trees and at least one split")
        print(f"PASS: three trees, two clients, {report['federation_rounds']} rounds.", flush=True)
        print("This proves local process communication, not a multi-laptop run.", flush=True)
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
