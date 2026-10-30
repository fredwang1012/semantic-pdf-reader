/** Root component: library sidebar, search bar with results, and the PDF viewer. */
import { lazy, Suspense, useEffect, useRef, useState } from "react";
import { listDocs, pdfUrl, type DocMeta, type Rect } from "./api";
import Library from "./Library";
import SearchPanel, { type Scope } from "./SearchPanel";
import type { ViewerTarget } from "./viewer/PdfViewer";

// pdf.js is large; load it once a document is opened.
const PdfViewer = lazy(() => import("./viewer/PdfViewer"));

const DOC_KEY = "reader.doc";
const SCOPE_KEY = "reader.scope";

function load(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function save(key: string, value: string | null) {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch {
    // storage unavailable; the choice just isn't remembered
  }
}

export default function App() {
  const [docs, setDocs] = useState<DocMeta[]>([]);
  const [loading, setLoading] = useState(true);
  const [currentId, setCurrentId] = useState<string | null>(() => load(DOC_KEY));
  const [scope, setScope] = useState<Scope>(() => (load(SCOPE_KEY) === "library" ? "library" : "doc"));
  const [target, setTarget] = useState<ViewerTarget | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const targetKey = useRef(0);

  useEffect(() => {
    listDocs().then(
      (list) => {
        setDocs(list);
        setLoading(false);
        setCurrentId((id) => (list.some((d) => d.id === id) ? id : null));
      },
      () => {
        setLoading(false);
        setLoadError("Could not load the library");
      },
    );
  }, []);

  useEffect(() => save(DOC_KEY, currentId), [currentId]);
  useEffect(() => save(SCOPE_KEY, scope), [scope]);

  const current = docs.find((d) => d.id === currentId) ?? null;

  function select(id: string) {
    setCurrentId(id);
    setTarget(null);
  }

  function uploaded(doc: DocMeta) {
    setDocs((list) => [doc, ...list.filter((d) => d.id !== doc.id)]);
    select(doc.id);
  }

  function deleted(id: string) {
    const i = docs.findIndex((d) => d.id === id);
    const rest = docs.filter((d) => d.id !== id);
    setDocs(rest);
    if (id === currentId) {
      setCurrentId(rest[Math.min(i, rest.length - 1)]?.id ?? null);
      setTarget(null);
    }
  }

  // Document and target change in one update so the viewer loads the new file, then highlights.
  function open(docId: string, rects: Rect[]) {
    if (!docs.some((d) => d.id === docId)) return; // a result for a document deleted since the search
    setCurrentId(docId);
    setTarget({ rects, key: ++targetKey.current });
  }

  return (
    <div className="flex h-screen">
      <Library docs={docs} loading={loading} currentId={currentId} onSelect={select} onUploaded={uploaded} onDeleted={deleted} />
      <SearchPanel docs={docs} loading={loading} currentId={currentId} scope={scope} onScope={setScope} onOpen={open}>
        {current ? (
          <Suspense fallback={<p className="p-6 text-sm text-[var(--ink-soft)]">Loading…</p>}>
            <PdfViewer key={current.id} url={pdfUrl(current.id)} pageSizes={current.page_sizes} target={target} />
          </Suspense>
        ) : (
          <p className="p-6 text-sm text-[var(--ink-soft)]">
            {loadError ?? (loading ? "Loading…" : docs.length === 0 ? "Upload a PDF to start" : "Pick a PDF")}
          </p>
        )}
      </SearchPanel>
    </div>
  );
}
