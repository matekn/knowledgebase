#!/usr/bin/env python3
"""One-time environment setup for the knowledgebase skill.

Creates an isolated venv at $KB_HOME/.venv and installs the RAG dependencies
(sqlite-vec, fastembed, pypdf). Safe to re-run; it is idempotent.

Usage:
    python setup.py              # install deps (no-op if already ready)
    python setup.py --force      # reinstall deps
    python setup.py --predownload  # also download the embedding model now
    python setup.py --status     # report readiness and locations
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

DEPS = [
    "sqlite-vec==0.1.9",
    "fastembed==0.8.1",
    "pypdf>=4",
]

REQ_FILE = Path(__file__).resolve().parent.parent / "requirements.txt"


def load_deps() -> list[str]:
    """Prefer requirements.txt so pins live in one place; fall back to DEPS."""
    if REQ_FILE.exists():
        deps = [
            line.strip()
            for line in REQ_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if deps:
            return deps
    return list(DEPS)


def kb_home() -> Path:
    return Path(os.environ.get("KB_HOME", Path.home() / ".knowledgebase")).expanduser()


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _run(cmd, env=None, quiet=False):
    if quiet:
        proc = subprocess.run(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        if proc.returncode != 0 and proc.stdout:
            sys.stderr.write(proc.stdout)
        return proc.returncode
    return subprocess.call(cmd, env=env)


def _pip_ok(py: Path) -> bool:
    try:
        return (
            subprocess.call(
                [str(py), "-m", "pip", "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            == 0
        )
    except OSError:
        return False


class _Lock:
    """Cross-process lock so parallel invocations don't bootstrap the same venv at once."""

    def __init__(self, path: Path, timeout: float = 600.0, stale: float = 900.0):
        self.path = path
        self.timeout = timeout
        self.stale = stale
        self.fd = None

    def __enter__(self):
        deadline = time.time() + self.timeout
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, str(os.getpid()).encode())
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > self.stale:
                        self.path.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                if time.time() > deadline:
                    raise TimeoutError(f"timed out waiting for lock {self.path}")
                time.sleep(0.5)

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        self.path.unlink(missing_ok=True)


def is_ready(home: Path) -> bool:
    return venv_python(home / ".venv").exists() and (home / ".venv" / ".ready").exists()


def _repair_pip(py: Path, venv: Path, quiet: bool) -> None:
    if not quiet:
        print("[setup] repairing pip")
    os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    _run([str(py), "-m", "ensurepip", "--upgrade"], quiet=quiet)
    if not _pip_ok(py):
        if not quiet:
            print("[setup] recreating venv")
        _run([sys.executable, "-m", "venv", "--clear", str(venv)], quiet=quiet)
        _run([str(py), "-m", "ensurepip", "--upgrade"], quiet=quiet)


def ensure(home: Path | None = None, force: bool = False, quiet: bool = True) -> Path:
    home = home or kb_home()
    venv = home / ".venv"
    py = venv_python(venv)
    sentinel = venv / ".ready"
    if py.exists() and sentinel.exists() and not force:
        return py

    home.mkdir(parents=True, exist_ok=True)
    lock = _Lock(home / ".setup.lock")
    with lock:
        if py.exists() and sentinel.exists() and not force:
            return py

        if not py.exists():
            if not quiet:
                print(f"[setup] creating venv at {venv}")
            rc = _run([sys.executable, "-m", "venv", str(venv)], quiet=quiet)
            if rc != 0 or not py.exists():
                raise SystemExit(
                    "[setup] could not create a virtualenv with "
                    f"'{sys.executable}'. Ensure this Python has the venv module "
                    "(e.g. install python3-venv on Debian/Ubuntu) and that "
                    f"{home} is writable."
                )

        if not _pip_ok(py):
            _repair_pip(py, venv, quiet)
        if not _pip_ok(py):
            raise SystemExit("[setup] could not obtain a working pip in the venv")

        deps = load_deps()
        env = dict(os.environ)
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        env["PIP_NO_INPUT"] = "1"
        pip = [str(py), "-m", "pip", "install"]
        if not quiet:
            print("[setup] upgrading pip")
        _run(pip + ["--upgrade", "pip"], env=env, quiet=quiet)

        if not quiet:
            print(f"[setup] installing {', '.join(deps)}")
        last = 0
        for attempt in range(3):
            last = _run(pip + list(deps), env=env, quiet=quiet)
            if last == 0:
                break
            if not quiet:
                print(f"[setup] install attempt {attempt + 1} failed; retrying")
            time.sleep(2 * (attempt + 1))
        if last != 0:
            raise SystemExit(
                "[setup] dependency installation failed after retries. This step "
                "needs network access to PyPI on first run."
            )

        sentinel.write_text("ready\n", encoding="utf-8")
        return py


def predownload(home: Path, py: Path) -> None:
    cache = home / "models"
    cache.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["FASTEMBED_CACHE_PATH"] = str(cache)
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["HF_HUB_VERBOSITY"] = "error"
    print("[setup] downloading embedding model (one-time)…")
    subprocess.check_call(
        [str(py), "-c",
         "from fastembed import TextEmbedding; "
         "TextEmbedding(model_name='BAAI/bge-small-en-v1.5'); "
         "print('model ready')"],
        env=env,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Set up the knowledgebase skill environment")
    ap.add_argument("--force", action="store_true", help="Reinstall dependencies")
    ap.add_argument("--predownload", action="store_true", help="Download the embedding model now")
    ap.add_argument("--status", action="store_true", help="Report readiness")
    args = ap.parse_args()

    home = kb_home()
    if args.status:
        py = venv_python(home / ".venv")
        print(f"kb_home : {home}")
        print(f"venv py : {py}  ({'present' if py.exists() else 'MISSING'})")
        print(f"ready   : {is_ready(home)}")
        return

    py = ensure(home, force=args.force, quiet=False)
    if args.predownload:
        predownload(home, py)
    print(f"[setup] ready: {py}")


if __name__ == "__main__":
    main()
