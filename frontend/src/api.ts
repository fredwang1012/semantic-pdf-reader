/** Typed client for the backend's /api routes. */

/** [page, x0, y0, x1, y1]: page is 1-based, coordinates are PDF points from the top-left. */
export type Rect = [number, number, number, number, number];

export interface DocMeta {
  id: string;
  title: string;
  filename: string;
  pages: number;
  chunks: number;
  page_sizes: [number, number][];
  embedded: boolean;
  added: number;
}

export interface SearchResult {
  id: string;
  page: number;
  text: string;
  rects: Rect[];
  score: number;
}

export interface DocSearch {
  mode: "hybrid" | "bm25";
  results: SearchResult[];
}

export interface LibraryGroup extends DocSearch {
  doc: DocMeta;
}

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

// FastAPI sends {"detail": "text"}, or a list of {msg} objects for validation errors.
function detailMessage(body: unknown): string | undefined {
  if (typeof body !== "object" || body === null || !("detail" in body)) return undefined;
  const { detail } = body;
  if (typeof detail === "string") return detail || undefined;
  if (Array.isArray(detail) && typeof detail[0]?.msg === "string") return detail[0].msg;
  return undefined;
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let message: string | undefined;
    try {
      message = detailMessage(await res.json());
    } catch {
      // body was not JSON
    }
    throw new ApiError(res.status, message ?? (res.statusText || `HTTP ${res.status}`));
  }
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}

function searchQuery(q: string, k?: number): string {
  const params = new URLSearchParams({ q });
  if (k !== undefined) params.set("k", String(k));
  return params.toString();
}

export const listDocs = () => request<DocMeta[]>("/api/docs");

export function uploadDoc(file: File): Promise<DocMeta> {
  const form = new FormData();
  form.append("file", file);
  return request<DocMeta>("/api/docs", { method: "POST", body: form });
}

export const deleteDoc = (id: string) =>
  request<void>(`/api/docs/${encodeURIComponent(id)}`, { method: "DELETE" });

export const pdfUrl = (id: string) => `/api/docs/${encodeURIComponent(id)}/pdf`;

export const searchDoc = (id: string, q: string, k?: number) =>
  request<DocSearch>(`/api/docs/${encodeURIComponent(id)}/search?${searchQuery(q, k)}`);

export const searchLibrary = (q: string, k?: number) =>
  request<{ groups: LibraryGroup[] }>(`/api/search?${searchQuery(q, k)}`).then((r) => r.groups);
