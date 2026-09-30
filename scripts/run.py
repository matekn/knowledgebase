#!/usr/bin/env python3
"""Launcher for the knowledgebase skill.

Runs kb.py using the skill's isolated venv, bootstrapping it (venv + deps) on
first use. Use the system Python to invoke this file; it re-execs into the venv.

Examples:
    python run.py add ./docs --collection research
    python run.py search "how does auth work" -k 5
    python run.py remove --source ./docs
    python run.py list
    python run.py stats
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import setup as kb_setup  # noqa: E402


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    py = kb_setup.ensure()
    cmd = [str(py), str(HERE / "kb.py"), *sys.argv[1:]]
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
