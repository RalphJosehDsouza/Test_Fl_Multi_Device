"""Path resolution that works both in the repo and inside a Flower app bundle.

A FAB is unpacked and run from a temporary directory, so ``data/fl/server_test.csv`` is not
necessarily reachable by a relative path once the app is running. Every path a user can
point at is therefore resolved here: an explicit absolute path wins, and otherwise we walk
up from the working directory and from this module looking for the file.

When nothing resolves the error says exactly what to pass, rather than failing later with a
missing column.
"""

from __future__ import annotations

from pathlib import Path

REPO_MARKERS = ("requirements.txt", "README.md", "data")


def _candidates(value: str, default: Path | None):
    seen: list[Path] = []
    if value:
        p = Path(value)
        if p.is_absolute():
            seen.append(p)
        else:
            seen.append(Path.cwd() / p)
            seen.extend(parent / p for parent in Path.cwd().parents)
    if default is not None:
        seen.append(default)
    here = Path(__file__).resolve()
    seen.extend(here.parents[i] / value for i in range(1, min(5, len(here.parents))))
    out = []
    for c in seen:
        if c not in out:
            out.append(c)
    return out


def resolve_path(value: str, default: Path | None = None, must_exist: bool = True,
                 what: str = "file") -> Path:
    candidates = _candidates(value, default)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if must_exist:
        tried = "\n".join(f"      {c}" for c in candidates)
        raise FileNotFoundError(
            f"could not find the {what} '{value or default}'.\n"
            f"    this app is running from {Path.cwd()}, so these were tried:\n{tried}\n"
            f"    pass an absolute path, e.g.\n"
            f"      --run-config \"{what}-path=/abs/path/to/{Path(value or '.').name}\"\n"
            f"    or, for a SuperNode, --node-config \"{what}-path=/abs/path/to/"
            f"{Path(value or '.').name}\""
        )
    return Path.cwd() / value


def resolve_out_dir(value: str) -> Path:
    """Where to write models and reports. Prefers an explicit path, then the repo root,
    then the working directory."""
    if value:
        p = Path(value)
        p = p if p.is_absolute() else (Path.cwd() / p)
        p.mkdir(parents=True, exist_ok=True)
        return p.resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if all((parent / marker).exists() for marker in REPO_MARKERS):
            (parent / "models" / "fl").mkdir(parents=True, exist_ok=True)
            (parent / "reports" / "fl").mkdir(parents=True, exist_ok=True)
            return parent
    out = Path.cwd() / "fl_output"
    out.mkdir(parents=True, exist_ok=True)
    return out
