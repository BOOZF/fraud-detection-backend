"""Knowledge-base documents for the copilot: extract -> chunk -> embed (384-dim) -> store in Teradata.

Tables: policy_docs (one row per document), policy_chunks (text + citation label),
policy_emb (embedding per chunk, searched in-database with TD_VectorDistance)."""
import collections
import difflib
import io
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pandas as pd
from pypdf import PdfReader
from teradataml import fastload

from .. import db
from . import rag

UPLOAD_DIR = Path(__file__).resolve().parents[2] / "uploads"  # original files, served back to the viewer
SUPPORTED = {".pdf": "pdf", ".md": "md", ".txt": "txt"}
MAX_BYTES = 25 * 1024 * 1024
MAX_CHUNKS = 2000  # ~2 MB of text; keeps one upload to about a minute and a few cents of embeddings
CHUNK_CHARS = 800  # measured: smaller chunks embed more precisely (1200: 0.60 hit@1, 800: 0.74, 500: 0.79)
OVERLAP = 150
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


def file_path(name: str) -> Path:
    return UPLOAD_DIR / safe_name(name)


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
            db.execute(ddl)
        except Exception:
            pass  # table already exists


def reset_schema() -> None:
    for table in ("policy_emb", "policy_chunks", "policy_docs"):
        try:
            db.execute(f"DROP TABLE {table}")
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


REDACTION = re.compile(r"\(\s*b\s*\)\s*\(\s*\d\s*\)(?:\s*\(\s*[a-zA-Z]\s*\))?")  # FOIA marks: (b)(7)(e), (b )(5) ...
REPEAT_SHARE = 0.2        # a line on >= 20% of the pages is a running header/footer
MIN_PAGES_FOR_REPEATS = 8  # too few pages to tell a header from repeated content
LOGO_JUNK_MAX_CHARS = 45   # short OCR garbage lines sitting above the running header (a garbled logo)


def _norm(line: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", line.lower()).strip()


JUNK_TOKEN = re.compile(r"[=$<>{}|\\^~\"]")  # symbols that real words never contain, e.g. U.S.CJtl=$hl.fc
SHORT_JUNK_CHARS = 60      # a line shorter than this that is mostly symbols/digits is OCR junk
MIN_LETTER_SHARE = 0.65
SHORT_LINE_WORDS = 12  # running headers/footers are short; long lines are body text and must match exactly


def _signature(line: str) -> str:
    """Short lines: words only (no digits, no 1-2 letter bits), so 'Version 3.0 ... M-2' and '... M-3' share one
    signature. Long lines keep every token, so body sentences that differ by a number are never merged."""
    words = _norm(line).split()
    if len(words) > SHORT_LINE_WORDS:
        return " ".join(words)
    return " ".join(w for w in words if w.isalpha() and len(w) >= 3)


FUZZY_RATIO = 0.88  # tolerates OCR typos such as 'Contems' for 'Contents'


def _reflow(lines: list[str]) -> str:
    """Rejoin hyphenated words and wrap-broken sentences; keep a line break after a finished sentence."""
    text = re.sub(r"(\w)-\n(?=[a-z])", r"\1", "\n".join(lines))
    parts = text.split("\n")
    out = parts[0] if parts else ""
    for prev, line in zip(parts, parts[1:]):
        starts_new = bool(re.match(r"[A-Z0-9\u2022\-\*]", line))
        out += ("\n" if prev.rstrip().endswith((".", "!", "?", ":")) and starts_new else " ") + line
    return re.sub(r"[ \t]+", " ", out).strip()


def clean_pdf_pages(raw_pages: list[str]) -> list[str]:
    """Strip what scanned/OCR'd PDFs add around the real text, so chunks embed as meaning, not as boilerplate:
    redaction marks, running headers/footers (found statistically: lines repeated across many pages), the
    garbled logo above the header, bare page numbers, hyphenated line breaks."""
    pages = []
    for raw in raw_pages:
        lines = [REDACTION.sub(" ", _clean(line)).strip() for line in raw.split("\n")]
        pages.append([line for line in lines if line])
    repeated: set[str] = set()
    if len(pages) >= MIN_PAGES_FOR_REPEATS:
        seen = collections.Counter(sig for lines in pages for sig in {_signature(l) for l in lines if len(_signature(l)) >= 4})
        repeated = {sig for sig, n in seen.items() if n / len(pages) >= REPEAT_SHARE}

    def is_running(line: str) -> bool:
        sig = _signature(line)
        if len(sig) < 4 or not repeated:
            return False
        if sig in repeated:
            return True
        if len(_norm(line).split()) > SHORT_LINE_WORDS:
            return False
        return any(abs(len(sig) - len(r)) <= 3 and difflib.SequenceMatcher(None, sig, r).ratio() >= FUZZY_RATIO
                   for r in repeated)

    # Boilerplate that OCR glued onto the end/start of a content line (e.g. a table-of-contents entry followed by the
    # page footer): remove the exact repeated strings wherever they occur inside a longer line.
    verbatim = collections.Counter(l for lines in pages for l in lines if len(l) >= 12 and is_running(l))
    boilerplate = sorted((l for l, n in verbatim.items() if n >= 3), key=len, reverse=True)

    def strip_glued(line: str) -> str:
        for b in boilerplate:
            if b in line and line != b:
                line = line.replace(b, " ")
        line = re.sub(r"\s+", " ", line).strip()
        words = line.split(" ")
        while len(words) > 3 and JUNK_TOKEN.search(words[0]):  # OCR junk glued to the start of a real line
            words.pop(0)
        return " ".join(words)

    def symbol_heavy(line: str) -> bool:
        """Short OCR junk such as U.S.CJtl=$hl.fc 'e' ": mostly symbols and stray letters, not words."""
        compact = re.sub(r"\s", "", line)
        if len(line) >= SHORT_JUNK_CHARS or not compact:
            return False
        words = line.split()
        if len(words) <= 4 and any(JUNK_TOKEN.search(w) for w in words):
            return True  # e.g. U.S.CJtl=$hl.fc: a symbol that real words never contain
        return sum(c.isalpha() for c in compact) / len(compact) < MIN_LETTER_SHARE

    cleaned = []
    for lines in pages:
        first = next((i for i, l in enumerate(lines) if is_running(l)), None)
        if first and all(len(l) < LOGO_JUNK_MAX_CHARS for l in lines[:first]):
            lines = lines[first:]  # drop the garbled logo above the running header
        lines = [strip_glued(l) for l in lines if not is_running(l)]
        lines = [l for l in lines if not l.isdigit() and sum(c.isalnum() for c in l) >= 2 and not symbol_heavy(l)]
        cleaned.append(_reflow(lines) if lines else "")
    return cleaned


def chunk_pdf(data: bytes) -> tuple[list[Chunk], int]:
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = clean_pdf_pages([p.extract_text() or "" for p in reader.pages])
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
    with db.locked():
        db.execute(f"DELETE FROM policy_emb WHERE chunk_id IN (SELECT chunk_id FROM policy_chunks WHERE doc = '{_q(name)}')")
        db.execute(f"DELETE FROM policy_chunks WHERE doc = '{_q(name)}'")
        db.execute(f"DELETE FROM policy_docs WHERE doc = '{_q(name)}'")
    file_path(name).unlink(missing_ok=True)
    rag.clear_chunk_cache()
    db.clear_cache()  # cached copilot briefs may cite this document
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
    if len(chunks) > MAX_CHUNKS:
        raise DocumentError(
            413, f"Document is too large to index ({len(chunks):,} chunks; the maximum is {MAX_CHUNKS:,}). "
                 "Upload a shorter document or split it.")

    ensure_schema()
    vectors: list[list[float]] = []  # network calls to OpenAI happen before taking the Teradata lock
    for i in range(0, len(chunks), EMBED_BATCH):
        vectors += rag.embed_many([c.text for c in chunks[i:i + EMBED_BATCH]])

    def write() -> None:  # id allocation and all inserts are one atomic unit; safe to repeat (replaces by name)
        try:
            delete_document(name)  # same name replaces the earlier upload
        except DocumentError:
            pass
        first_id = int(db.query_df("SELECT COALESCE(MAX(chunk_id), 0) + 1 AS n FROM policy_chunks")["n"].iloc[0])
        ids = list(range(first_id, first_id + len(chunks)))
        frame = pd.DataFrame({"chunk_id": ids, "doc": name, "section": [c.section for c in chunks],
                              "text": [c.text for c in chunks]})
        emb = pd.DataFrame(vectors, columns=[f"e{i}" for i in range(rag.DIM)])
        emb.insert(0, "chunk_id", ids)
        fastload(df=frame, table_name="policy_chunks", if_exists="append")
        fastload(df=emb, table_name="policy_emb", if_exists="append")
        db.execute(
            "INSERT INTO policy_docs VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP(0))",
            [(name, kind, pages, len(chunks), len(data))],
        )

    db.run(write)
    UPLOAD_DIR.mkdir(exist_ok=True)
    file_path(name).write_bytes(data)
    rag.clear_chunk_cache()
    db.clear_cache()
    return next(d for d in list_documents() if d["doc"] == name)
