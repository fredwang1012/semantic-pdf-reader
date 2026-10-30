"""FastAPI app: upload PDFs, list the library, serve the PDF bytes, and search one document or the whole library."""
import asyncio
import mimetypes
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import ROOT, settings
from .embedder import get_embedder
from .library import Library, NoTextLayer
from .pdftext import PasswordProtected
from .reranker import get_reranker
from .retrieval import DEFAULT_K, Hit, match_terms

app = FastAPI(title="Semantic PDF Reader")
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
                   allow_methods=["*"], allow_headers=["*"])

_library: Library | None = None
_library_lock = threading.Lock()


def library() -> Library:
    global _library
    with _library_lock:
        if _library is None:
            _library = Library(settings.docs_dir, get_embedder(), get_reranker())
        return _library


def _meta_or_404(id_: str):
    try:
        return library().get(id_)
    except KeyError:
        raise HTTPException(404, "no such document") from None


@app.get("/api/health")
def health():
    lib = library()
    return {"ok": True, "search": "hybrid" if lib.embedder is not None else "bm25", "rerank": lib.reranker is not None}


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
    except (NoTextLayer, PasswordProtected) as e:
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


def _results(hits: list[Hit]) -> list[dict]:
    return [{**h.chunk.to_json(), "score": round(h.score, 4)} for h in hits]


def _terms(q: str) -> list[str]:
    """The query words the frontend should mark in snippets: the same ones keyword matching uses."""
    return sorted(match_terms(q))


@app.get("/api/search")
async def search_all(q: str = Query(min_length=1, max_length=500), k: int = Query(3, ge=1, le=20)):
    groups = await asyncio.to_thread(library().search_all, q, k)
    return {"terms": _terms(q),
            "groups": [{"doc": meta.to_json(), "mode": mode, "results": _results(hits)} for meta, hits, mode in groups]}


@app.get("/api/docs/{doc_id}/search")
async def search(doc_id: str, q: str = Query(min_length=1, max_length=500), k: int = Query(DEFAULT_K, ge=1, le=50)):
    def run():
        index = library().index(doc_id)
        return index.mode, index.search(q, k)

    try:
        mode, hits = await asyncio.to_thread(run)
    except KeyError:  # unknown id, or deleted while searching
        raise HTTPException(404, "no such document") from None
    return {"mode": mode, "terms": _terms(q), "results": _results(hits)}


# Windows maps .mjs to text/plain in the registry, and browsers refuse to run the pdf.js worker module under that type.
mimetypes.add_type("text/javascript", ".mjs")

_dist = ROOT / "frontend" / "dist"
if (_dist / "index.html").is_file():
    app.mount("/", StaticFiles(directory=Path(_dist), html=True), name="frontend")
