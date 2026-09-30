# knowledgebase

A fast, self-contained local knowledge base (RAG) for adding, searching, listing, and removing files. No server, no daemon, no manual dependency setup.

- **Storage:** SQLite + [`sqlite-vec`](https://github.com/asg017/sqlite-vec) — a single `kb.db` file.
- **Embeddings:** [`fastembed`](https://github.com/qdrant/fastembed) (ONNX, in-process) — no model server.
- **Self-bootstrapping:** the first run creates an isolated venv and downloads the model; later runs are fast and offline.
- **Packaged as an agent skill** (`SKILL.md`) so an assistant can use it automatically.

## Requirements

- Python 3.x with the `venv` module on `PATH` (tested on 3.13).
- Network access on first run (installs deps and downloads the ~90 MB embedding model). Offline afterwards.

## Install

Clone the repo straight into your agent's skills directory:

```bash
# macOS / Linux
git clone https://github.com/matekn/knowledgebase.git ~/.agents/skills/knowledgebase

# Windows (PowerShell)
git clone https://github.com/matekn/knowledgebase.git "$env:USERPROFILE\.agents\skills\knowledgebase"
```

Prefer not to use git? Copy the folder instead:

```bash
cp -r knowledgebase ~/.agents/skills/knowledgebase
```

No `pip install` needed — `scripts/run.py` bootstraps everything on first use.

## Usage

Always invoke through the launcher (it manages the venv):

```bash
python <skill-dir>/scripts/run.py <command> [options]
```

| Command | Description |
| --- | --- |
| `add PATH... [--collection NAME] [--force]` | Index files or directories (recursive). Skips unchanged files by content hash. |
| `search QUERY [-k N] [-c NAME ...] [--exclude-collection NAME ...]` | Semantic top-k search with source path and char range. |
| `remove --source PATH [--collection NAME]` | Remove by path (file or dir), `--id CHUNK_ID`, or `--all`. |
| `list [-c NAME ...] [--exclude-collection NAME ...]` | List indexed sources. |
| `collections` | List collection tags with counts. |
| `stats` | Store location, size, model, totals. |

Global flags: `--json` for machine-readable output, `--kb-home PATH` to override the store for one run.

### Examples

```bash
# index a folder, tag it
python scripts/run.py add ./docs --collection research

# search by meaning (spans all collections)
python scripts/run.py search "how do we restore backups" -k 5

# include some collections, exclude others
python scripts/run.py search "deploy process" -c work -c personal --exclude-collection scratch

# remove a source
python scripts/run.py remove --source ./docs/old-runbook.md --collection research

# inspect
python scripts/run.py list
python scripts/run.py collections
python scripts/run.py stats
```

## Collections

Every source is tagged with a `--collection` (default `default`). The store is global, so collections let you keep unrelated material separable.

- `add` targets exactly one collection.
- `search` / `list` accept a repeatable `--collection` (include any) and `--exclude-collection` (exclude). Omit both to span everything.
- `remove` is scoped to a single `--collection` (or `--all`).

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `KB_HOME` | `~/.knowledgebase` | Store location (`kb.db`, `.venv`, `models/`). |
| `KB_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model (384-dim). One model per store. |
| `KB_MAX_FILE_MB` | `25` | Skip files larger than this. |

## Supported files

Plain text, Markdown, and common code/config formats, plus PDF (text layer). Directories are walked recursively, skipping noise like `.git`, `node_modules`, `.venv`. Unsupported/binary files are skipped with a warning.

## How it works

1. **Chunk** — ~1000 chars with ~150 overlap, preferring newline boundaries.
2. **Embed** — in-process ONNX model; vectors stored in a `vec0` virtual table (cosine distance).
3. **Search** — KNN over vectors, then filtered by collection include/exclude.

Re-adding a file is idempotent: it is only re-embedded when its content hash changes (`--force` overrides).

## Repository layout

```
SKILL.md            # agent-facing instructions
requirements.txt    # pinned runtime deps
scripts/
  run.py            # launcher (bootstraps venv, re-execs kb.py)
  setup.py          # venv + dependency bootstrap
  kb.py             # the CLI
evals/              # test prompts + fixtures
```

## License

Not yet specified.
