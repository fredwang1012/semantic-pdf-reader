import pytest
from fastapi.testclient import TestClient

from app.main import app

from .conftest import make_pdf


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _upload(client, data: bytes, name: str = "bio.pdf"):
    return client.post("/api/docs", files={"file": (name, data, "application/pdf")})


def test_upload_list_and_download(client, sample_pdf):
    r = _upload(client, sample_pdf)
    assert r.status_code == 201
    meta = r.json()
    assert meta["pages"] == 2 and meta["title"] == "bio"
    assert [d["id"] for d in client.get("/api/docs").json()] == [meta["id"]]
    assert client.get(f"/api/docs/{meta['id']}").json() == meta
    pdf = client.get(f"/api/docs/{meta['id']}/pdf")
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.content == sample_pdf


def test_search_returns_the_right_page_first(client, sample_pdf):
    id_ = _upload(client, sample_pdf).json()["id"]
    body = client.get(f"/api/docs/{id_}/search", params={"q": "Adam optimizer"}).json()
    assert body["mode"] == "bm25"
    assert body["results"][0]["page"] == 2
    assert len(body["results"][0]["rects"][0]) == 5


def test_health_reports_bm25_without_embeddings(client):
    assert client.get("/api/health").json() == {"ok": True, "search": "bm25"}


def test_upload_rejects_non_pdf_and_text_less_pdf(client):
    assert _upload(client, b"hello", "a.txt").status_code == 400
    import pymupdf
    doc = pymupdf.open()
    doc.new_page().draw_rect(pymupdf.Rect(100, 100, 300, 300))
    blank = doc.tobytes()
    doc.close()
    assert _upload(client, blank).status_code == 422


def test_unknown_and_malformed_ids_are_404(client):
    for id_ in ("0123456789abcdef", "not-an-id"):
        assert client.get(f"/api/docs/{id_}").status_code == 404
        assert client.get(f"/api/docs/{id_}/pdf").status_code == 404
        assert client.get(f"/api/docs/{id_}/search", params={"q": "x"}).status_code == 404
        assert client.delete(f"/api/docs/{id_}").status_code == 404


def test_search_requires_a_query(client, sample_pdf):
    id_ = _upload(client, sample_pdf).json()["id"]
    assert client.get(f"/api/docs/{id_}/search").status_code == 422
    assert client.get(f"/api/docs/{id_}/search", params={"q": ""}).status_code == 422


def test_delete_then_gone(client):
    id_ = _upload(client, make_pdf()).json()["id"]
    assert client.delete(f"/api/docs/{id_}").status_code == 204
    assert client.get(f"/api/docs/{id_}").status_code == 404
