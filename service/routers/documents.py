from fastapi import APIRouter, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse

from .. import db
from ..services import documents, highlight

router = APIRouter(prefix="/api")


def _fail(e: documents.DocumentError):
    raise HTTPException(status_code=e.status, detail=e.message)


@router.get("/documents")
def list_documents():
    return documents.list_documents()


@router.post("/documents")
def upload_document(file: UploadFile):
    data = file.file.read(documents.MAX_BYTES + 1)
    try:
        return documents.ingest(file.filename, data)
    except documents.DocumentError as e:
        _fail(e)


@router.delete("/documents/{name}")
def delete_document(name: str):
    try:
        return {"deleted": name, "chunks": documents.delete_document(documents.safe_name(name))}
    except documents.DocumentError as e:
        _fail(e)


MEDIA_TYPES = {"pdf": "application/pdf", "md": "text/markdown; charset=utf-8", "txt": "text/plain; charset=utf-8"}


@router.get("/documents/{name}/file")
def document_file(name: str, chunk: int | None = None, quote: str | None = None):
    """The original file. With ?chunk=<id> (a PDF citation) the cited paragraph is highlighted on its page; with &quote=<sentence>
    the key sentence inside it gets a stronger highlight."""
    try:
        path = documents.file_path(name)
        kind = documents.kind_of(path.name)
    except documents.DocumentError as e:
        _fail(e)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"File for '{name}' not found")
    headers = {"Content-Disposition": f'inline; filename="{path.name}"'}
    if chunk is not None and kind == "pdf":
        rows = db.query(f"SELECT section, text FROM policy_chunks WHERE chunk_id = {int(chunk)} AND doc = '{documents._q(path.name)}'")
        if not rows:
            raise HTTPException(status_code=404, detail=f"Chunk {chunk} does not belong to '{name}'")
        section, text = rows[0]
        return Response(highlight.highlighted_pdf(path, section, text, (quote or "")[:500]), media_type=MEDIA_TYPES[kind], headers=headers)
    return FileResponse(path, media_type=MEDIA_TYPES[kind], headers=headers)
