/** Top bar with the search box and scope toggle, the results column, and a slot for the viewer. */
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { ApiError, searchDoc, searchLibrary, type DocMeta, type Rect, type SearchResult } from "./api";
import { markTerms, snippet } from "./results";

export type Scope = "doc" | "library";

interface Props {
  docs: DocMeta[];
  loading: boolean;
  currentId: string | null;
  scope: Scope;
  onScope: (s: Scope) => void;
  onOpen: (docId: string, rects: Rect[]) => void;
  children: ReactNode;
}

interface Item {
  docId: string;
  result: SearchResult;
}
interface Group {
  doc: DocMeta | null;
  items: Item[];
}
interface Found {
  query: string;
  terms: string[];
  mode: string;
  groups: Group[];
}

export default function SearchPanel({ docs, loading, currentId, scope, onScope, onOpen, children }: Props) {
  const [query, setQuery] = useState("");
  const [found, setFound] = useState<Found | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const list = useRef<HTMLElement>(null);
  const latest = useRef(0);

  const needsDoc = scope === "doc" && currentId === null;
  const hint = needsDoc ? (loading ? "Loading…" : docs.length === 0 ? "Upload a PDF to start" : "Pick a PDF") : null;

  // Results belong to the scope (and, for one PDF, the document) they were fetched for.
  const resultsKey = scope === "doc" ? currentId : null;
  useEffect(() => {
    latest.current++;
    setFound(null);
    setError(null);
    setBusy(false);
  }, [scope, resultsKey]);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement).tagName;
      if (e.key === "/" && tag !== "INPUT" && tag !== "TEXTAREA") {
        e.preventDefault();
        input.current?.focus();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    list.current?.querySelector('[aria-current="true"]')?.scrollIntoView({ block: "nearest" });
  }, [selected, found]);

  // Library results can outlive a delete: drop groups for documents that are gone.
  const groups = useMemo(
    () => found?.groups.filter((g) => g.doc === null || docs.some((d) => d.id === g.doc!.id)) ?? [],
    [found, docs],
  );
  const flat = useMemo(() => groups.flatMap((g) => g.items), [groups]);

  async function run() {
    const q = query.trim();
    if (!q || needsDoc) return;
    const id = ++latest.current;
    setBusy(true);
    setError(null);
    try {
      let groups: Group[];
      let mode: string;
      let terms: string[];
      if (scope === "doc") {
        const r = await searchDoc(currentId!, q, 10);
        mode = r.mode;
        terms = r.terms;
        groups = [{ doc: null, items: r.results.map((result) => ({ docId: currentId!, result })) }];
      } else {
        const r = await searchLibrary(q);
        mode = r.groups.some((g) => g.mode === "bm25") ? "bm25" : "hybrid";
        terms = r.terms;
        groups = r.groups
          .filter((g) => g.results.length > 0)
          .map((g) => ({ doc: g.doc, items: g.results.map((result) => ({ docId: g.doc.id, result })) }));
      }
      if (id !== latest.current) return;
      setFound({ query: q, terms, mode, groups });
      setSelected(0);
    } catch (e) {
      if (id === latest.current) {
        setFound(null);
        setError(e instanceof ApiError ? e.message : "Search failed");
      }
    } finally {
      if (id === latest.current) setBusy(false);
    }
  }

  function open(i: number) {
    const item = flat[i];
    if (!item) return;
    setSelected(i);
    onOpen(item.docId, item.result.rects);
  }

  // Enter opens the selection when the box still holds the query that produced the results.
  function onInputKey(e: React.KeyboardEvent) {
    if (e.key === "Enter") {
      if (found && found.query === query.trim() && flat.length > 0) open(selected);
      else void run();
    } else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      const step = e.key === "ArrowDown" ? 1 : -1;
      setSelected((s) => Math.min(flat.length - 1, Math.max(0, s + step)));
    } else if (e.key === "Escape") {
      input.current?.blur();
    }
  }

  let index = -1;
  return (
    <div className="flex min-w-0 flex-1 flex-col">
      <header className="flex items-center gap-3 border-b border-[var(--ground-edge)] p-3">
        <input
          ref={input}
          value={query}
          disabled={needsDoc}
          placeholder={hint ?? "Search (press / to focus)"}
          aria-label="Search"
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={onInputKey}
          className="min-w-0 flex-1 rounded border border-[var(--ground-edge)] bg-transparent px-3 py-1.5 text-sm disabled:opacity-50"
        />
        <div
          role="group"
          aria-label="Scope"
          className="flex overflow-hidden rounded border border-[var(--ground-edge)] text-sm"
        >
          {(["doc", "library"] as const).map((s) => (
            <button
              key={s}
              type="button"
              aria-pressed={scope === s}
              onClick={() => onScope(s)}
              className={`px-3 py-1.5 ${scope === s ? "bg-[var(--ground-edge)]" : ""}`}
            >
              {s === "doc" ? "This PDF" : "All PDFs"}
            </button>
          ))}
        </div>
      </header>
      <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
        <section
          ref={list}
          aria-label="Results"
          className="max-h-[40%] w-full shrink-0 overflow-y-auto border-b border-[var(--ground-edge)] lg:max-h-none lg:w-80 lg:border-r lg:border-b-0"
        >
          {busy && <p className="p-3 text-sm text-[var(--ink-soft)]">Searching…</p>}
          {error && (
            <p role="alert" className="p-3 text-sm text-red-600">
              {error}
            </p>
          )}
          {!busy && !error && !found && (
            <p className="p-3 text-sm text-[var(--ink-soft)]">
              {loading ? "Loading…" : docs.length === 0 ? "Upload a PDF to start" : (hint ?? "Type a query and press Enter")}
            </p>
          )}
          {!busy && found && flat.length === 0 && (
            <p className="p-3 text-sm text-[var(--ink-soft)]">No matches</p>
          )}
          {found && flat.length > 0 && (
            <>
              <p className="px-3 pt-2 text-xs text-[var(--ink-soft)]">
                {flat.length} {flat.length === 1 ? "result" : "results"} · {found.mode}
              </p>
              {groups.map((g) => (
                <div key={g.doc?.id ?? "current"}>
                  {g.doc && (
                    <h2 className="truncate px-3 pt-3 text-xs font-semibold tracking-wide uppercase">
                      {g.doc.title}
                    </h2>
                  )}
                  <ul>
                    {g.items.map((item) => {
                      const i = ++index;
                      return (
                        <li key={`${item.docId}:${item.result.id}`}>
                          <button
                            type="button"
                            aria-current={i === selected}
                            onClick={() => open(i)}
                            className={`block w-full px-3 py-2 text-left text-sm hover:bg-[var(--ground-edge)] ${i === selected ? "bg-[var(--ground-edge)]" : ""}`}
                          >
                            <span className="text-xs text-[var(--ink-soft)]">Page {item.result.page}</span>
                            <span className="mt-0.5 block">
                              {markTerms(snippet(item.result.text, 220), found.terms.join(" ")).map((p, n) =>
                                p.hit ? <strong key={n}>{p.text}</strong> : <span key={n}>{p.text}</span>,
                              )}
                            </span>
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              ))}
            </>
          )}
        </section>
        <div className="flex min-h-0 min-w-0 flex-1 flex-col">{children}</div>
      </div>
    </div>
  );
}
