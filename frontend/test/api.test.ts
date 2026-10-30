import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ApiError,
  deleteDoc,
  listDocs,
  pdfUrl,
  searchDoc,
  searchLibrary,
  uploadDoc,
} from "../src/api";

const fetchMock = vi.fn();
vi.stubGlobal("fetch", fetchMock);

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status });
}

afterEach(() => fetchMock.mockReset());

describe("api client", () => {
  it("lists documents", async () => {
    fetchMock.mockResolvedValue(json([{ id: "a" }]));
    expect(await listDocs()).toEqual([{ id: "a" }]);
    expect(fetchMock.mock.calls[0][0]).toBe("/api/docs");
  });

  it("uploads as multipart POST", async () => {
    fetchMock.mockResolvedValue(json({ id: "a" }, 201));
    await uploadDoc(new File(["x"], "a.pdf", { type: "application/pdf" }));
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/docs");
    expect(init.method).toBe("POST");
    expect((init.body as FormData).get("file")).toBeInstanceOf(File);
  });

  it("deletes with DELETE and tolerates 204", async () => {
    fetchMock.mockResolvedValue(new Response(null, { status: 204 }));
    await expect(deleteDoc("a")).resolves.toBeUndefined();
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/docs/a");
    expect(init.method).toBe("DELETE");
  });

  it("builds the pdf url", () => {
    expect(pdfUrl("a")).toBe("/api/docs/a/pdf");
  });

  it("searches one document and encodes the query", async () => {
    fetchMock.mockResolvedValue(json({ mode: "bm25", results: [] }));
    await searchDoc("a", "foo bar & baz", 5);
    const raw = fetchMock.mock.calls[0][0] as string;
    const url = new URL(raw, "http://x");
    expect(url.pathname).toBe("/api/docs/a/search");
    expect(url.searchParams.get("q")).toBe("foo bar & baz");
    expect(url.searchParams.get("k")).toBe("5");
    expect(raw).not.toContain("& baz");
  });

  it("omits k when not given and unwraps library groups", async () => {
    fetchMock.mockResolvedValue(json({ groups: [{ doc: { id: "a" }, mode: "hybrid", results: [] }] }));
    const groups = await searchLibrary("x y");
    expect(fetchMock.mock.calls[0][0]).toBe("/api/search?q=x+y");
    expect(groups[0].doc.id).toBe("a");
  });

  it("turns an error with a string detail into an ApiError", async () => {
    fetchMock.mockResolvedValue(json({ detail: "no text layer" }, 422));
    const err = await uploadDoc(new File(["x"], "a.pdf")).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(422);
    expect(err.message).toBe("no text layer");
  });

  it("uses the first msg of a validation-error list", async () => {
    fetchMock.mockResolvedValue(json({ detail: [{ msg: "field required" }, { msg: "other" }] }, 422));
    const err = await searchLibrary("").catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.message).toBe("field required");
  });

  it("still throws an ApiError for a non-JSON error body", async () => {
    fetchMock.mockResolvedValue(new Response("<html>boom</html>", { status: 502, statusText: "Bad Gateway" }));
    const err = await listDocs().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(502);
    expect(err.message).toBe("Bad Gateway");
  });
});
