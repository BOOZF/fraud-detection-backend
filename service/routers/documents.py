from fastapi import APIRouter, HTTPException, UploadFile

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
