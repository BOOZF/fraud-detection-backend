"""Knowledge-base documents for the copilot: extract -> chunk -> embed (384-dim) -> store in Teradata.

Tables: policy_docs (one row per document), policy_chunks (text + citation label),
policy_emb (embedding per chunk, searched in-database with TD_VectorDistance)."""
import io
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

import pandas as pd
from pypdf import PdfReader
from teradataml import execute_sql, fastload

from .. import db
from . import rag

SUPPORTED = {".pdf": "pdf", ".md": "md", ".txt": "txt"}
MAX_BYTES = 25 * 1024 * 1024
CHUNK_CHARS = 1200
OVERLAP = 200
MIN_CHUNK_CHARS = 120
EMBED_BATCH = 64


class DocumentError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class Chunk:
    section: str  # citation label: "3.1" for numbered sections, "p.12" / "pp.12-13" for PDF pages
    text: str


# ---------- names and schema ----------

def safe_name(filename: str | None) -> str:
    name = PurePosixPath((filename or "").replace("\\", "/")).name.strip()
    name = re.sub(r"[^A-Za-z0-9._() -]", "_", name)[:200]
    if not name or name.startswith("."):
        raise DocumentError(400, "A file name is required")
    return name


def kind_of(name: str) -> str:
    ext = PurePosixPath(name).suffix.lower()
    if ext not in SUPPORTED:
        raise DocumentError(415, f"Unsupported file type '{ext or name}'. Upload a PDF, Markdown or text file.")
    return SUPPORTED[ext]


def _q(value: str) -> str:
    return value.replace("'", "''")


DDL = [
    "CREATE TABLE policy_docs (doc VARCHAR(255) CHARACTER SET UNICODE NOT NULL, kind VARCHAR(8), "
    "pages INTEGER, chunks INTEGER, file_bytes INTEGER, uploaded_at TIMESTAMP(0)) PRIMARY INDEX (doc)",
    "CREATE TABLE policy_chunks (chunk_id INTEGER NOT NULL, doc VARCHAR(255) CHARACTER SET UNICODE, "
    "section VARCHAR(64) CHARACTER SET UNICODE, text VARCHAR(4000) CHARACTER SET UNICODE) PRIMARY INDEX (chunk_id)",
    "CREATE TABLE policy_emb (chunk_id INTEGER NOT NULL, "
    + ", ".join(f"e{i} FLOAT" for i in range(rag.DIM)) + ") PRIMARY INDEX (chunk_id)",
]


def ensure_schema() -> None:
    for ddl in DDL:
        try:
            execute_sql(ddl)
        except Exception:
            pass  # table already exists


def reset_schema() -> None:
    for table in ("policy_emb", "policy_chunks", "policy_docs"):
        try:
            execute_sql(f"DROP TABLE {table}")
        except Exception:
            pass
    ensure_schema()


# ---------- extraction and chunking ----------

def _clean(text: str) -> str:
    text = re.sub(r"[^\x20-\x7E -￿\n]", " ", text)  # control characters
    text = text.replace("�", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def _readable(text: str) -> bool:
    """Skip OCR noise, blank pages and form boilerplate: needs enough mostly-alphabetic text."""
    if len(text) < MIN_CHUNK_CHARS:
        return False
    letters = sum(c.isalpha() for c in text)
    return letters / len(text) > 0.5


def _windows(text: str) -> list[str]:
    if len(text) <= CHUNK_CHARS:
        return [text]
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + CHUNK_CHARS])
        i += CHUNK_CHARS - OVERLAP
    return out


def _page_label(first: int, last: int) -> str:
    return f"p.{first}" if first == last else f"pp.{first}-{last}"


def chunk_pdf(data: bytes) -> tuple[list[Chunk], int]:
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [_clean(p.extract_text() or "") for p in reader.pages]
    except Exception as e:
        raise DocumentError(400, f"Could not read this PDF: {e}")
    chunks: list[Chunk] = []
    buf, first = "", 0
    for number, text in enumerate(pages, start=1):
        if not text:
            continue
        if buf and len(buf) + len(text) + 1 > CHUNK_CHARS:
            chunks.extend(_flush(buf, first, number - 1))
            buf = ""
        if not buf:
            first = number
        buf = f"{buf}\n{text}" if buf else text
    if buf:
        chunks.extend(_flush(buf, first, len(pages)))
    return chunks, len(pages)


def _flush(buf: str, first: int, last: int) -> list[Chunk]:
    return [Chunk(_page_label(first, last), w) for w in _windows(buf) if _readable(w)]


SECTION = re.compile(r"^#{2,3}\s+(\d+(?:\.\d+)*)\s+(.+)$", re.M)


def chunk_text(text: str) -> list[Chunk]:
    text = text.replace("\r\n", "\n")
    parts = SECTION.split(text)
    if len(parts) > 1:  # numbered headings: one chunk per section, label = section number
        out = []
        for i in range(1, len(parts), 3):
            body = f"{parts[i]} {parts[i + 1]}: {_clean(parts[i + 2])}"
            for n, w in enumerate(_windows(body)):
                out.append(Chunk(parts[i] if n == 0 else f"{parts[i]} (cont.)", w))
        return out
    cleaned = _clean(text)
    return [Chunk(f"part {n}", w) for n, w in enumerate(_windows(cleaned), start=1) if w.strip()] if cleaned else []


def extract(kind: str, data: bytes) -> tuple[list[Chunk], int | None]:
    if kind == "pdf":
        return chunk_pdf(data)
    try:
        return chunk_text(data.decode("utf-8")), None
    except UnicodeDecodeError:
        raise DocumentError(400, "Text files must be UTF-8 encoded")


# ---------- storage ----------

def list_documents() -> list[dict]:
    ensure_schema()
    df = db.query_df("SELECT doc, kind, pages, chunks, file_bytes, uploaded_at FROM policy_docs ORDER BY uploaded_at DESC")
    out = []
    for r in df.itertuples(index=False):
        out.append({"doc": r.doc, "kind": r.kind, "pages": None if pd.isna(r.pages) else int(r.pages),
                    "chunks": int(r.chunks), "bytes": int(r.file_bytes), "uploaded_at": str(r.uploaded_at)})
    return out


def delete_document(name: str) -> int:
    n = int(db.query_df(f"SELECT COUNT(*) AS n FROM policy_chunks WHERE doc = '{_q(name)}'")["n"].iloc[0])
    known = int(db.query_df(f"SELECT COUNT(*) AS n FROM policy_docs WHERE doc = '{_q(name)}'")["n"].iloc[0])
    if n == 0 and known == 0:
        raise DocumentError(404, f"Document '{name}' not found")
    execute_sql(f"DELETE FROM policy_emb WHERE chunk_id IN (SELECT chunk_id FROM policy_chunks WHERE doc = '{_q(name)}')")
    execute_sql(f"DELETE FROM policy_chunks WHERE doc = '{_q(name)}'")
    execute_sql(f"DELETE FROM policy_docs WHERE doc = '{_q(name)}'")
    rag.clear_chunk_cache()
    return n


def ingest(filename: str | None, data: bytes) -> dict:
    name = safe_name(filename)
    kind = kind_of(name)
    if len(data) > MAX_BYTES:
        raise DocumentError(413, f"File is larger than {MAX_BYTES // (1024 * 1024)} MB")
    if not data.strip():
        raise DocumentError(400, "The file is empty")
    chunks, pages = extract(kind, data)
    if not chunks:
        raise DocumentError(400, "No readable text found (a scanned image-only file cannot be indexed)")

    ensure_schema()
    try:
        delete_document(name)  # same name replaces the earlier upload
    except DocumentError:
        pass
    first_id = int(db.query_df("SELECT COALESCE(MAX(chunk_id), 0) + 1 AS n FROM policy_chunks")["n"].iloc[0])
    ids = list(range(first_id, first_id + len(chunks)))

    vectors: list[list[float]] = []
    for i in range(0, len(chunks), EMBED_BATCH):
        vectors += rag.embed_many([c.text for c in chunks[i:i + EMBED_BATCH]])

    frame = pd.DataFrame({"chunk_id": ids, "doc": name, "section": [c.section for c in chunks],
                          "text": [c.text for c in chunks]})
    emb = pd.DataFrame(vectors, columns=[f"e{i}" for i in range(rag.DIM)])
    emb.insert(0, "chunk_id", ids)
    fastload(df=frame, table_name="policy_chunks", if_exists="append")
    fastload(df=emb, table_name="policy_emb", if_exists="append")
    execute_sql(
        "INSERT INTO policy_docs VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP(0))",
        [(name, kind, pages, len(chunks), len(data))],
    )
    rag.clear_chunk_cache()
    return next(d for d in list_documents() if d["doc"] == name)
