from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..services import documents

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
def document_file(name: str):
    try:
        path = documents.file_path(name)
        kind = documents.kind_of(path.name)
    except documents.DocumentError as e:
        _fail(e)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"File for '{name}' not found")
    return FileResponse(path, media_type=MEDIA_TYPES[kind], headers={"Content-Disposition": f'inline; filename="{path.name}"'})
