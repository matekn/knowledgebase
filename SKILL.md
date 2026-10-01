---
name: knowledgebase
description: Manage a fast local knowledge base (RAG) for indexing, semantically searching, listing, and removing files. Use this skill whenever the user wants to add, remember, index, search, recall, or delete documents/notes/code/PDFs from a local retrieval store — e.g. "add these docs to my knowledge base", "index this folder", "search my notes for X", "what did I save about Y", "remove that file from the KB", "set up a local RAG", or any mention of semantic search over local files, embeddings, sqlite-vec, or a personal/global knowledge base. Prefer this skill over ad-hoc grep when the user asks to recall meaning rather than match exact text.
---

# Knowledge Base (local RAG)

A single-file, no-daemon retrieval store: SQLite + `sqlite-vec` for vectors, `fastembed` (ONNX) for in-process embeddings. Everything lives under `$KB_HOME` (default `~/.knowledgebase`).

## When to use

Use this skill when the user wants durable, searchable storage of files rather than reading them once:

- "add / index / save these files (or this folder) to the knowledge base"
- "search / find / what do I know about …" across previously indexed material
- "remove / delete / drop X from the knowledge base"
- "how big is my knowledge base", "what's indexed", "list collections"
- setting up a local RAG or semantic search over their own files

Do **not** use it for one-off reads of a file the user just pointed at — read the file directly. Use it when the content should persist and be retrievable by meaning later.

## Quick start

Always invoke through the launcher, which bootstraps the venv on first run:

```bash
python <skill-dir>/scripts/run.py add <path> [--collection NAME]
python <skill-dir>/scripts/run.py search "<query>" [-k N] [--collection NAME]
python <skill-dir>/scripts/run.py remove --source <path> [--collection NAME]
python <skill-dir>/scripts/run.py list [--collection NAME]
python <skill-dir>/scripts/run.py collections
python <skill-dir>/scripts/run.py stats
```

Replace `<skill-dir>` with the directory this SKILL.md lives in (the skill's own location, wherever it was installed). On Windows/PowerShell use the call operator and quote the path:

```powershell
& python "<skill-dir>\scripts\run.py" add .\docs --collection research
```

## Commands

| Command | What it does |
| --- | --- |
| `add PATH...` | Index one or more files or directories (recursive). Skips unchanged files by content hash, so re-running is cheap and safe. |
| `search QUERY` | Semantic top-k search. Returns score, source path, char range, and a snippet. `--collection` (repeatable) to include, `--exclude-collection` (repeatable) to exclude. |
| `remove` | Delete by `--source PATH` (file or directory), `--id CHUNK_ID`, or `--all` (scope with `--collection`). |
| `list` | Show indexed sources with collection, chunk count, size. |
| `collections` | Show collection tags with source/chunk counts. |
| `stats` | Store location, size, model, dimensions, totals. |

Global flags: `--json` (machine-readable output), `--kb-home PATH` (override store for a run).

## Collections

The store is global, but every source is tagged with a `--collection` (default `default`). Use collections to keep unrelated material separable: e.g. index a repo as `--collection myproject` and personal notes as `--collection notes`.

- `add` puts a source into exactly one collection (`--collection NAME`).
- `search` and `list` can **include several collections** by repeating `--collection` (results from any of them) and **exclude** collections with `--exclude-collection`, also repeatable. Omit both to span everything.

```bash
# only work + personal, never the scratch collection
python <skill-dir>/scripts/run.py search "deploy process" -c work -c personal --exclude-collection scratch
python <skill-dir>/scripts/run.py list --exclude-collection archive
```

- `remove` is scoped to a single `--collection` (or `--all`).

## Languages and models

The default model (`BAAI/bge-small-en-v1.5`) is English-only. For Hungarian or mixed-language content, pick a multilingual model with `--model` (or the `KB_MODEL` env var):

| Preset | Languages | Dim | Speed |
| --- | --- | --- | --- |
| `en` | English | 384 | fastest (default) |
| `multilingual` | ~50 incl. Hungarian | 384 | fast |
| `multilingual-base` | ~50 incl. Hungarian | 768 | slower |
| `multilingual-e5-large` | 100+ | 1024 | slow, best quality |

```bash
python <skill-dir>/scripts/run.py models                        # list options
python <skill-dir>/scripts/run.py add ./docs --model multilingual
python <skill-dir>/scripts/run.py search "..." --model multilingual
```

A store remembers the model it was built with, and later commands reuse it automatically — pass `--model` only when creating a store or overriding. Switching models on a populated store is refused; build a separate store via a different `KB_HOME` instead.

## Supported files

Plain text, Markdown, and common code/config formats, plus PDF (text layer). Directories are walked recursively, skipping noise dirs like `.git`, `node_modules`, `.venv`. Binary or unsupported files are skipped with a warning.

## Behavior notes

- **First run is slow-ish**: builds the venv and downloads the embedding model (~one time, cached in `$KB_HOME/models`). Subsequent runs are fast and offline.
- **Idempotent adds**: a file is only re-embedded when its content hash changes. Use `--force` to re-embed regardless.
- **Chunks** are ~1000 chars with ~150 char overlap, preferring newline boundaries. Search results include the source path and char offsets so you can open the exact location.
- **One model per store**: the store records its embedding model/dimension. Switching `KB_MODEL` on a populated store errors — reindex into a fresh `KB_HOME` instead.

## Reporting results to the user

After a search, summarize the hits in your own words and cite the source file paths (and char ranges when useful). Don't dump raw JSON unless the user asked for it. After an add, report how many files were added/updated/skipped and the collection used.
