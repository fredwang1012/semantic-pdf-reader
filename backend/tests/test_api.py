import time

import pytest
from fastapi.testclient import TestClient

from app.main import app

from .conftest import make_pdf


# Each page is padded past the chunk minimum so it becomes its own chunk.
def _two_pages(first: str, second: str = "Nothing relevant here.") -> bytes:
    pad = " The quick brown fox jumps over the lazy dog while the committee reviews the quarterly budget at length again and again." * 2
    return make_pdf([[first + pad], [second + pad]])


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


def test_library_search_groups_results_by_document(client):
    first = _upload(client, _two_pages("Photosynthesis lets plants turn sunlight into food."), "plants.pdf").json()
    second = _upload(client, _two_pages("Rockets burn fuel to reach orbit.", "Plants stay on the ground."),
                     "rockets.pdf").json()
    groups = client.get("/api/search", params={"q": "orbit"}).json()["groups"]
    assert [g["doc"]["id"] for g in groups] == [second["id"]]
    assert groups[0]["mode"] == "bm25" and len(groups[0]["results"][0]["rects"][0]) == 5
    both = client.get("/api/search", params={"q": "plants"}).json()["groups"]
    assert {g["doc"]["id"] for g in both} == {first["id"], second["id"]}


def test_library_search_ranks_fuller_matches_first(client):
    full = _upload(client, _two_pages("Rockets burn fuel to reach orbit."), "full.pdf").json()
    time.sleep(0.01)
    partial = _upload(client, _two_pages("Rockets are loud."), "partial.pdf").json()
    groups = client.get("/api/search", params={"q": "rockets orbit"}).json()["groups"]
    assert [g["doc"]["id"] for g in groups] == [full["id"], partial["id"]]


def test_library_search_empty_library_and_validation(client):
    assert client.get("/api/search", params={"q": "x"}).json() == {"groups": []}
    assert client.get("/api/search", params={"q": ""}).status_code == 422
    assert client.get("/api/search", params={"q": "x", "k": 21}).status_code == 422


def test_password_protected_upload_is_422(client):
    import pymupdf

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret text", fontsize=11)
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    doc.close()
    r = _upload(client, data)
    assert r.status_code == 422 and "password-protected" in r.json()["detail"]


def test_search_of_a_missing_document_is_404(client):
    assert client.get("/api/docs/0123456789abcdef/search", params={"q": "x"}).status_code == 404


def test_search_reports_mode_per_group(client, sample_pdf):
    _upload(client, sample_pdf)
    groups = client.get("/api/search", params={"q": "Calvin cycle"}).json()["groups"]
    assert groups and groups[0]["mode"] == "bm25"


def test_docs_still_served_when_the_embedder_cannot_load(client, monkeypatch):
    from app import embedder, main
    monkeypatch.setattr(main, "_library", None)
    monkeypatch.setattr(embedder.settings, "embed", "auto")
    monkeypatch.setattr(embedder, "_loaded", None)
    monkeypatch.setattr(embedder, "_failed", False)
    calls = []

    def boom():
        calls.append(1)
        raise OSError("offline")

    monkeypatch.setattr(embedder, "E5Embedder", boom)
    assert client.get("/api/docs").status_code == 200
    assert client.get("/api/health").json()["search"] == "bm25"
    assert embedder.get_embedder() is None
    assert len(calls) == 1
