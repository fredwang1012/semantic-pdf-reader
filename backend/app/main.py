"""FastAPI app: upload PDFs, list the library, serve the PDF bytes, and search a document."""
import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import ROOT, settings
from .embedder import get_embedder
from .library import Library, NoTextLayer
from .retrieval import DEFAULT_K

app = FastAPI(title="Semantic PDF Reader")
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
                   allow_methods=["*"], allow_headers=["*"])

_library: Library | None = None


def library() -> Library:
    global _library
    if _library is None:
        _library = Library(settings.docs_dir, get_embedder())
    return _library


def _meta_or_404(id_: str):
    try:
        return library().get(id_)
    except KeyError:
        raise HTTPException(404, "no such document") from None


@app.get("/api/health")
def health():
    return {"ok": True, "search": "hybrid" if library().embedder is not None else "bm25"}


@app.get("/api/docs")
def list_docs():
    return [m.to_json() for m in library().list()]


@app.post("/api/docs", status_code=201)
async def upload(file: UploadFile):
    data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"PDF is larger than {settings.max_upload_mb} MB")
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "not a PDF file")
    try:
        meta = await asyncio.to_thread(library().add, data, file.filename or "document.pdf")
    except NoTextLayer as e:
        raise HTTPException(422, str(e)) from None
    except RuntimeError as e:  # PyMuPDF raises on a damaged file
        raise HTTPException(400, f"could not read the PDF: {e}") from None
    return meta.to_json()


@app.get("/api/docs/{doc_id}")
def get_doc(doc_id: str):
    return _meta_or_404(doc_id).to_json()


@app.get("/api/docs/{doc_id}/pdf")
def get_pdf(doc_id: str):
    meta = _meta_or_404(doc_id)
    return FileResponse(library().pdf_path(doc_id), media_type="application/pdf", filename=meta.filename)


@app.delete("/api/docs/{doc_id}", status_code=204)
def delete_doc(doc_id: str):
    _meta_or_404(doc_id)
    library().delete(doc_id)


@app.get("/api/docs/{doc_id}/search")
async def search(doc_id: str, q: str = Query(min_length=1, max_length=500), k: int = Query(DEFAULT_K, ge=1, le=50)):
    _meta_or_404(doc_id)
    index = library().index(doc_id)
    hits = await asyncio.to_thread(index.search, q, k)
    return {"mode": index.mode, "results": [{**h.chunk.to_json(), "score": round(h.score, 4)} for h in hits]}


_dist = ROOT / "frontend" / "dist"
if (_dist / "index.html").is_file():
    app.mount("/", StaticFiles(directory=Path(_dist), html=True), name="frontend")
