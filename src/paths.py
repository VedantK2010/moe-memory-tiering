"""
Project path resolution
=========================
Every script in this project used to hard-code an absolute path from
whichever machine happened to run it last, which meant nothing ran
anywhere else. This module resolves the three directories the project
cares about relative to the repository itself, so the scripts work from
any checkout and any working directory.

Layout:
    <repo>/src/        this file
    <repo>/data/       input traces (gitignored -- see README)
    <repo>/results/    generated CSVs and figures (committed)
"""

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"


def data(name: str) -> Path:
    """Path to a trace under data/, creating the dir so writes succeed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / name


def result(name: str) -> Path:
    """Path to a generated artefact under results/, creating the dir."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / name


def require_data(name: str) -> Path:
    """
    Path to a required input trace, with an actionable error if it is
    missing -- the trace CSVs are gitignored because of their size, so a
    fresh clone genuinely will not have them until they are generated or
    downloaded.
    """
    p = data(name)
    if not p.exists():
        raise SystemExit(
            f"Missing input trace: {p}\n"
            f"\n"
            f"The trace CSVs are excluded from git (see .gitignore). To create them:\n"
            f"  python run_all.py --stage traces     # synthetic traces\n"
            f"  python src/convert_real_data.py      # real Mixtral traces (needs internet)\n"
        )
    return p
