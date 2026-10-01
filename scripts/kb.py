#!/usr/bin/env python3
"""Local knowledge base (RAG) CLI.

A single-file vector store (SQLite + sqlite-vec) with in-process embeddings
(fastembed / ONNX). Supports add, remove, search, list, stats, collections.

Store location: $KB_HOME (default ~/.knowledgebase), containing kb.db.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Must be set before fastembed / huggingface_hub import to take effect.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
warnings.filterwarnings("ignore", message=r".*progress bars.*")

try:
    import sqlite_vec
    from fastembed import TextEmbedding
except ImportError as exc:  # pragma: no cover - handled by run.py bootstrap
    sys.stderr.write(
        f"Missing dependency: {exc}\n"
        "Run the skill via scripts/run.py, which bootstraps a venv, or run scripts/setup.py.\n"
    )
    raise SystemExit(3)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# Friendly aliases for common choices. Any fastembed model id also works directly.
MODEL_PRESETS = {
    "en": "BAAI/bge-small-en-v1.5",
    "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "multilingual-base": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "multilingual-e5-large": "intfloat/multilingual-e5-large",
}


def resolve_model(name: str) -> str:
    return MODEL_PRESETS.get(name, name)


MODEL_NAME = resolve_model(os.environ.get("KB_MODEL", DEFAULT_MODEL))
SCHEMA_VERSION = "1"
MAX_FILE_MB = float(os.environ.get("KB_MAX_FILE_MB", "100"))

TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".rst", ".org", ".tex", ".log", ".csv", ".tsv",
}
CODE_EXTS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".java", ".kt", ".kts",
    ".go", ".rs", ".rb", ".php", ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".swift",
    ".scala", ".sh", ".bash", ".zsh", ".ps1", ".psm1", ".sql", ".r", ".lua", ".pl",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".json", ".xml", ".html",
    ".htm", ".css", ".scss", ".vue", ".svelte",
}
PDF_EXTS = {".pdf"}
SUPPORTED_EXTS = TEXT_EXTS | CODE_EXTS | PDF_EXTS

SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build", ".next",
    ".knowledgebase", ".idea", ".vscode",
}


def kb_home() -> Path:
    return Path(os.environ.get("KB_HOME", Path.home() / ".knowledgebase")).expanduser()


def set_model_cache(home: Path) -> None:
    # Keep the model inside the KB home so the store is self-contained.
    cache = home / "models"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("FASTEMBED_CACHE_PATH", str(cache))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def eprint(*a):
    sys.stderr.write(" ".join(str(x) for x in a) + "\n")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in PDF_EXTS:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pypdf is required for PDF files") from exc
        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages):
            try:
                t = page.extract_text() or ""
            except Exception as exc:  # noqa: BLE001
                eprint(f"  ! page {i + 1} failed in {path.name}: {exc}")
                t = ""
            if t.strip():
                pages.append(f"[page {i + 1}]\n{t}")
        return "\n\n".join(pages)
    data = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def chunk_text(text: str, size: int = 1000, overlap: int = 150):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    n = len(text)
    out = []
    i = 0
    while i < n:
        end = min(i + size, n)
        if end < n:
            lo = max(i, end - 200)
            brk = text.rfind("\n", lo, end)
            if brk > i + size // 2:
                end = brk
        piece = text[i:end]
        if piece.strip():
            out.append((i, end, piece))
        if end >= n:
            break
        i = max(end - overlap, i + 1)
    return out


def iter_files(target: Path):
    if target.is_file():
        yield target
        return
    for root, dirs, files in os.walk(target):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for name in files:
            p = Path(root) / name
            if p.suffix.lower() in SUPPORTED_EXTS:
                yield p


class Store:
    def __init__(self, home: Path):
        self.home = home
        home.mkdir(parents=True, exist_ok=True)
        self.db_path = home / "kb.db"
        self.db = sqlite3.connect(str(self.db_path), timeout=30)
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        # WAL + busy_timeout keep concurrent add/search processes from clashing.
        try:
            self.db.execute("pragma journal_mode=WAL")
            self.db.execute("pragma busy_timeout=30000")
            self.db.execute("pragma synchronous=NORMAL")
        except sqlite3.DatabaseError:
            pass
        self._ensure_schema()
        self._model = None
        self._dim = None

    def _ensure_schema(self):
        db = self.db
        db.execute("create table if not exists meta(key text primary key, value text)")
        db.execute(
            "create table if not exists sources("
            "path text not null, collection text not null, hash text, size integer, "
            "mtime real, chunk_count integer, added_at text, updated_at text, "
            "primary key(path, collection))"
        )
        db.execute(
            "create table if not exists chunks("
            "id integer primary key, collection text not null, source_path text not null, "
            "chunk_index integer, content text, char_start integer, char_end integer)"
        )
        db.execute("create index if not exists idx_chunks_source on chunks(source_path, collection)")
        db.execute("create index if not exists idx_chunks_collection on chunks(collection)")
        db.commit()

    def model(self):
        if self._model is None:
            self._model = TextEmbedding(model_name=MODEL_NAME)
            self._dim = len(list(self._model.embed(["dimension probe"]))[0])
            self._init_vec(self._dim)
        return self._model

    def _init_vec(self, dim: int):
        row = self.db.execute("select value from meta where key='dim'").fetchone()
        model_row = self.db.execute("select value from meta where key='model'").fetchone()
        if row is None:
            self.db.execute(
                f"create virtual table if not exists vec_chunks using vec0("
                f"embedding float[{dim}] distance_metric=cosine)"
            )
            self.db.execute("insert or replace into meta values('dim', ?)", (str(dim),))
            self.db.execute("insert or replace into meta values('model', ?)", (MODEL_NAME,))
            self.db.execute("insert or replace into meta values('schema', ?)", (SCHEMA_VERSION,))
            self.db.commit()
            return

        stored_dim = int(row[0])
        stored_model = model_row[0] if model_row else "(unknown)"
        if stored_model == MODEL_NAME and stored_dim == dim:
            return

        # Model or dimension changed. Only safe when nothing is indexed yet.
        n = self.db.execute("select count(*) from chunks").fetchone()[0]
        if n == 0:
            self.db.execute("drop table if exists vec_chunks")
            self.db.execute(
                f"create virtual table vec_chunks using vec0("
                f"embedding float[{dim}] distance_metric=cosine)"
            )
            self.db.execute("insert or replace into meta values('dim', ?)", (str(dim),))
            self.db.execute("insert or replace into meta values('model', ?)", (MODEL_NAME,))
            self.db.commit()
            return

        raise SystemExit(
            f"Store was built with model '{stored_model}' (dim={stored_dim}) but this run "
            f"uses '{MODEL_NAME}' (dim={dim}). A populated store cannot switch embedding "
            "models. Point KB_HOME at a new location to build a separate store for this "
            "model, or clear the current one first."
        )

    def ensure_vec(self):
        self.model()

    def embed(self, texts):
        return list(self.model().embed(texts))

    def query_embed(self, text):
        return list(self.model().query_embed([text]))[0]

    # --- mutations -------------------------------------------------------
    def _delete_source_rows(self, path: str, collection: str):
        ids = [
            r[0]
            for r in self.db.execute(
                "select id from chunks where source_path=? and collection=?", (path, collection)
            )
        ]
        for cid in ids:
            self.db.execute("delete from vec_chunks where rowid=?", (cid,))
        self.db.execute(
            "delete from chunks where source_path=? and collection=?", (path, collection)
        )
        self.db.execute(
            "delete from sources where path=? and collection=?", (path, collection)
        )
        return len(ids)

    def add_file(self, path: Path, collection: str, force: bool = False) -> dict:
        try:
            size = path.stat().st_size
        except OSError as exc:
            return {"path": str(path), "status": "error", "error": str(exc)}
        if size > MAX_FILE_MB * 1024 * 1024:
            return {
                "path": str(path),
                "status": "skipped",
                "reason": f"larger than KB_MAX_FILE_MB ({MAX_FILE_MB:g} MB)",
            }
        try:
            text = extract_text(path)
        except Exception as exc:  # noqa: BLE001
            return {"path": str(path), "status": "error", "error": str(exc)}
        if not text.strip():
            return {"path": str(path), "status": "empty"}

        digest = file_hash(path)
        stat = path.stat()
        key = str(path.resolve())
        row = self.db.execute(
            "select hash from sources where path=? and collection=?", (key, collection)
        ).fetchone()
        if row and row[0] == digest and not force:
            return {"path": key, "status": "skipped"}

        existed = row is not None
        self._delete_source_rows(key, collection)

        pieces = chunk_text(text)
        if not pieces:
            return {"path": key, "status": "empty"}

        vectors = self.embed([p[2] for p in pieces])
        for idx, ((start, end, content), vec) in enumerate(zip(pieces, vectors)):
            cur = self.db.execute(
                "insert into chunks(collection, source_path, chunk_index, content, char_start, char_end)"
                " values(?,?,?,?,?,?)",
                (collection, key, idx, content, start, end),
            )
            cid = cur.lastrowid
            self.db.execute(
                "insert into vec_chunks(rowid, embedding) values(?,?)",
                (cid, sqlite_vec.serialize_float32(vec)),
            )
        ts = now_iso()
        self.db.execute(
            "insert or replace into sources(path, collection, hash, size, mtime, chunk_count, added_at, updated_at)"
            " values(?,?,?,?,?,?,?,?)",
            (key, collection, digest, stat.st_size, stat.st_mtime, len(pieces), ts, ts),
        )
        self.db.commit()
        return {
            "path": key,
            "status": "updated" if existed else "added",
            "chunks": len(pieces),
        }

    def remove(self, source: str | None, collection: str | None, all_in_collection: bool,
               remove_id: int | None) -> dict:
        removed_sources = 0
        removed_chunks = 0
        if remove_id is not None:
            row = self.db.execute(
                "select source_path, collection from chunks where id=?", (remove_id,)
            ).fetchone()
            if row:
                removed_chunks = self._delete_source_rows(row[0], row[1])
                removed_sources = 1
            self.db.commit()
            return {"removed_sources": removed_sources, "removed_chunks": removed_chunks}

        if source is not None:
            target = Path(source).expanduser()
            if target.exists():
                key = str(target.resolve())
                if target.is_dir():
                    allrows = self.db.execute(
                        "select path, collection from sources"
                        + (" where collection=?" if collection else ""),
                        (collection,) if collection else (),
                    ).fetchall()
                    prefix = key.rstrip("\\/") + os.sep
                    rows = [(p, c) for p, c in allrows if p == key or p.startswith(prefix)]
                else:
                    colls = [
                        r[0]
                        for r in self.db.execute(
                            "select collection from sources where path=?", (key,)
                        )
                    ]
                    rows = [(key, c) for c in ([collection] if collection else colls)]
                for p, c in rows:
                    removed_chunks += self._delete_source_rows(p, c)
                    removed_sources += 1
            else:
                rows = self.db.execute(
                    "select path, collection from sources where (path=? or path like ?)"
                    + (" and collection=?" if collection else ""),
                    (source, f"%{source}%", *([collection] if collection else [])),
                ).fetchall()
                for p, c in rows:
                    removed_chunks += self._delete_source_rows(p, c)
                    removed_sources += 1
            self.db.commit()
            return {"removed_sources": removed_sources, "removed_chunks": removed_chunks}

        if all_in_collection:
            q = "select path, collection from sources"
            params = ()
            if collection:
                q += " where collection=?"
                params = (collection,)
            rows = self.db.execute(q, params).fetchall()
            for p, c in rows:
                removed_chunks += self._delete_source_rows(p, c)
                removed_sources += 1
            self.db.commit()
            return {"removed_sources": removed_sources, "removed_chunks": removed_chunks}

        raise SystemExit("Provide --source, --id, or --all (optionally with --collection).")

    # --- queries ---------------------------------------------------------
    @staticmethod
    def _coll_clause(collections=None, exclude=None, base="where 1=1"):
        clause = base
        params: list = []
        if collections:
            clause += " and collection in (%s)" % ",".join("?" * len(collections))
            params += list(collections)
        if exclude:
            clause += " and collection not in (%s)" % ",".join("?" * len(exclude))
            params += list(exclude)
        return clause, params

    def search(self, query: str, k: int, collections=None, exclude=None):
        if not query or not query.strip():
            raise SystemExit("Search query is empty.")
        include = set(collections) if collections else None
        excluded = set(exclude) if exclude else None
        clause, params = self._coll_clause(collections, exclude)
        total = self.db.execute("select count(*) from chunks " + clause, params).fetchone()[0]
        if total == 0:
            return []
        qv = self.query_embed(query)
        fetch = min(max(k * 6, 30), total)
        rows = self.db.execute(
            "select rowid, distance from vec_chunks where embedding match ? and k = ?"
            " order by distance",
            (sqlite_vec.serialize_float32(qv), fetch),
        ).fetchall()
        out = []
        for rid, dist in rows:
            row = self.db.execute(
                "select collection, source_path, chunk_index, content, char_start, char_end"
                " from chunks where id=?",
                (rid,),
            ).fetchone()
            if not row:
                continue
            coll, spath, cidx, content, cs, ce = row
            if include is not None and coll not in include:
                continue
            if excluded and coll in excluded:
                continue
            out.append(
                {
                    "score": round(1.0 - dist, 4),
                    "distance": round(dist, 4),
                    "collection": coll,
                    "source": spath,
                    "chunk_index": cidx,
                    "char_start": cs,
                    "char_end": ce,
                    "content": content,
                }
            )
            if len(out) >= k:
                break
        return out

    def list_sources(self, collections=None, exclude=None):
        clause, params = self._coll_clause(collections, exclude)
        q = (
            "select path, collection, chunk_count, size, updated_at from sources "
            + clause
            + " order by path"
        )
        return [
            {"path": p, "collection": c, "chunks": n, "size": s, "updated_at": u}
            for p, c, n, s, u in self.db.execute(q, params).fetchall()
        ]

    def collections(self):
        return [
            {"collection": c, "sources": s, "chunks": ch}
            for c, s, ch in self.db.execute(
                "select collection, count(*), sum(chunk_count) from sources group by collection order by collection"
            ).fetchall()
        ]

    def stats(self):
        n_chunks = self.db.execute("select count(*) from chunks").fetchone()[0]
        n_sources = self.db.execute("select count(*) from sources").fetchone()[0]
        size = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {
            "kb_home": str(self.home),
            "db_path": str(self.db_path),
            "model": MODEL_NAME,
            "dim": self._dim,
            "sources": n_sources,
            "chunks": n_chunks,
            "db_bytes": size,
        }


def fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n} B"


def snippet(text: str, width: int = 400) -> str:
    t = " ".join(text.split())
    return t if len(t) <= width else t[:width] + " …"


def cmd_add(args, store: Store):
    targets = [Path(p) for p in args.path]
    files = []
    for t in targets:
        if not t.exists():
            eprint(f"warning: {t} does not exist")
            continue
        if t.is_file() and t.suffix.lower() not in SUPPORTED_EXTS:
            eprint(f"warning: unsupported file type, skipping {t}")
            continue
        files.extend(iter_files(t))
    files = sorted(set(files))
    results = []
    t0 = time.time()
    for f in files:
        r = store.add_file(f, args.collection, force=args.force)
        results.append(r)
        eprint(f"  {r['status']:8} {r.get('path', '')}" + (f"  ({r.get('chunks', 0)} chunks)" if r.get("chunks") else ""))
    added = sum(1 for r in results if r["status"] == "added")
    updated = sum(1 for r in results if r["status"] == "updated")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = [r for r in results if r["status"] == "error"]
    summary = {
        "collection": args.collection,
        "files_seen": len(files),
        "added": added,
        "updated": updated,
        "skipped": skipped,
        "errors": len(errors),
        "seconds": round(time.time() - t0, 2),
        "results": results,
    }
    print(json.dumps(summary, indent=2) if args.json else
          f"collection={args.collection} seen={len(files)} added={added} updated={updated} "
          f"skipped={skipped} errors={len(errors)} in {summary['seconds']}s")


def cmd_remove(args, store: Store):
    res = store.remove(args.source, args.collection, args.all, args.id)
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(f"removed {res['removed_sources']} source(s), {res['removed_chunks']} chunk(s)")


def cmd_search(args, store: Store):
    hits = store.search(args.query, args.k, args.collection, args.exclude_collection)
    if args.json:
        print(json.dumps(hits, indent=2))
        return
    if not hits:
        print("No results. The knowledge base may be empty for this collection.")
        return
    for i, h in enumerate(hits, 1):
        name = Path(h["source"]).name
        print(f"\n[{i}] score={h['score']}  {name}  (collection={h['collection']}, chars {h['char_start']}-{h['char_end']})")
        print(f"    source: {h['source']}")
        print(f"    {snippet(h['content'])}")


def cmd_list(args, store: Store):
    rows = store.list_sources(args.collection, args.exclude_collection)
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    if not rows:
        print("No sources indexed.")
        return
    print(f"{'collection':<16} {'chunks':>7} {'size':>10}  path")
    for r in rows:
        print(f"{r['collection']:<16} {r['chunks']:>7} {fmt_size(r['size']):>10}  {r['path']}")


def cmd_collections(args, store: Store):
    rows = store.collections()
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    if not rows:
        print("No collections yet.")
        return
    print(f"{'collection':<16} {'sources':>8} {'chunks':>8}")
    for r in rows:
        print(f"{r['collection']:<16} {r['sources']:>8} {r['chunks'] or 0:>8}")


def cmd_stats(args, store: Store):
    store.ensure_vec()
    s = store.stats()
    if args.json:
        print(json.dumps(s, indent=2))
        return
    print(f"kb_home : {s['kb_home']}")
    print(f"db      : {s['db_path']} ({fmt_size(s['db_bytes'])})")
    print(f"model   : {s['model']} (dim={s['dim']})")
    print(f"sources : {s['sources']}")
    print(f"chunks  : {s['chunks']}")


def cmd_models(args):
    rows = [
        ("en", "BAAI/bge-small-en-v1.5", 384, "English only, fastest (default)"),
        ("multilingual", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
         384, "~50 languages incl. Hungarian; small and fast"),
        ("multilingual-base", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
         768, "Better multilingual quality, slower"),
        ("multilingual-e5-large", "intfloat/multilingual-e5-large",
         1024, "Best multilingual quality; large and slow"),
    ]
    if args.json:
        print(json.dumps(
            [{"preset": p, "model": m, "dim": d, "notes": n} for p, m, d, n in rows], indent=2
        ))
        return
    print(f"current model: {MODEL_NAME}\n")
    print(f"{'preset':<22} {'dim':>5}  model")
    for p, m, d, n in rows:
        print(f"{p:<22} {d:>5}  {m}")
        print(f"{'':<22} {'':>5}  {n}")
    print("\nSelect with  --model <preset|model-id>  or the KB_MODEL env var.")
    print("A populated store cannot change models; use a fresh KB_HOME for a different one.")


def build_parser():
    p = argparse.ArgumentParser(prog="kb", description="Local knowledge base (RAG) CLI")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--kb-home",
        help="Override store location (default $KB_HOME or ~/.knowledgebase)",
    )
    common.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    common.add_argument(
        "--model",
        "-m",
        help="Embedding model id or preset (en, multilingual, multilingual-base, "
        "multilingual-e5-large). Overrides KB_MODEL; must match the store's model.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("add", parents=[common], help="Add files or directories")
    a.add_argument("path", nargs="+", help="File(s) or directory(ies) to index")
    a.add_argument("--collection", "-c", default="default", help="Collection tag (default: default)")
    a.add_argument("--force", action="store_true", help="Re-embed even if unchanged")
    a.set_defaults(func=cmd_add)

    r = sub.add_parser("remove", parents=[common], help="Remove sources from the KB")
    r.add_argument("--source", "-s", help="Path (or dir) to remove; exact or substring match")
    r.add_argument("--id", type=int, help="Remove the source owning this chunk id")
    r.add_argument("--all", action="store_true", help="Remove all sources (scoped by --collection)")
    r.add_argument("--collection", "-c", help="Restrict removal to a collection")
    r.set_defaults(func=cmd_remove)

    s = sub.add_parser("search", parents=[common], help="Semantic search")
    s.add_argument("query", help="Natural-language query")
    s.add_argument("-k", type=int, default=5, help="Number of results (default 5)")
    s.add_argument(
        "--collection", "-c", action="append",
        help="Include only this collection (repeatable; default: all collections)",
    )
    s.add_argument(
        "--exclude-collection", action="append", default=[],
        help="Exclude this collection (repeatable)",
    )
    s.set_defaults(func=cmd_search)

    l = sub.add_parser("list", parents=[common], help="List indexed sources")
    l.add_argument(
        "--collection", "-c", action="append",
        help="Include only this collection (repeatable; default: all collections)",
    )
    l.add_argument(
        "--exclude-collection", action="append", default=[],
        help="Exclude this collection (repeatable)",
    )
    l.set_defaults(func=cmd_list)

    sub.add_parser("collections", parents=[common], help="List collections").set_defaults(
        func=cmd_collections
    )
    sub.add_parser("stats", parents=[common], help="Show store stats").set_defaults(func=cmd_stats)
    sub.add_parser(
        "models", parents=[common], help="List recommended embedding models"
    ).set_defaults(func=cmd_models)
    return p


def main(argv=None):
    global MODEL_NAME
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    args = build_parser().parse_args(argv)
    MODEL_NAME = resolve_model(args.model or os.environ.get("KB_MODEL", DEFAULT_MODEL))
    if args.command == "models":
        cmd_models(args)
        return
    home = Path(args.kb_home).expanduser() if args.kb_home else kb_home()
    set_model_cache(home)
    store = Store(home)
    args.func(args, store)


if __name__ == "__main__":
    main()
