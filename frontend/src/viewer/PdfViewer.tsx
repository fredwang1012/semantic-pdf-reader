/** Scrollable pdf.js viewer: lazy page canvases, fit-to-width layout and passage highlights. */
import { useEffect, useMemo, useRef, useState } from "react";
import { getDocument, GlobalWorkerOptions } from "pdfjs-dist";
import type { PDFDocumentProxy } from "pdfjs-dist";
import workerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import type { Rect } from "../api";
import { firstRect, fitScale, highlightBoxes } from "./geometry";

GlobalWorkerOptions.workerSrc = workerUrl;

export interface ViewerTarget {
  rects: Rect[];
  key: number;
}

interface Props {
  url: string;
  pageSizes: [number, number][];
  target: ViewerTarget | null;
}

export default function PdfViewer({ url, pageSizes, target }: Props) {
  const [scroller, setScroller] = useState<HTMLDivElement | null>(null);
  const [doc, setDoc] = useState<PDFDocumentProxy | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    if (!scroller) return;
    const observer = new ResizeObserver(() => setWidth(scroller.clientWidth));
    observer.observe(scroller);
    return () => observer.disconnect();
  }, [scroller]);

  useEffect(() => {
    setDoc(null);
    setError(null);
    const task = getDocument({ url });
    let cancelled = false;
    task.promise.then(
      (loaded) => {
        if (!cancelled) setDoc(loaded);
      },
      (e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      },
    );
    return () => {
      cancelled = true;
      void task.destroy(); // also destroys the document it loaded
    };
  }, [url]);

  const widest = Math.max(1, ...pageSizes.map((s) => s[0]));
  const scale = width > 0 ? fitScale(widest, width) : 0;

  const latest = useRef({ target, scale });
  latest.current = { target, scale };
  const ready = scale > 0;
  const key = target?.key;

  // Scroll on every new target key, even when the rects are identical to the last one.
  useEffect(() => {
    const { target: t, scale: s } = latest.current;
    const first = t ? firstRect(t.rects) : null;
    if (!scroller || !ready || !first) return;
    const pageEl = scroller.querySelector<HTMLElement>(`[data-page="${first[0]}"]`);
    if (!pageEl) return;
    const offset =
      pageEl.getBoundingClientRect().top - scroller.getBoundingClientRect().top + scroller.scrollTop;
    scroller.scrollTo({
      top: Math.max(0, offset + first[2] * s - scroller.clientHeight / 3),
      behavior: "smooth",
    });
  }, [scroller, ready, key]);

  return (
    <div ref={setScroller} className="relative h-full w-full overflow-auto">
      {error ? (
        <p role="alert" className="p-6 text-center text-[color:var(--ink-soft)]">
          Could not open the PDF: {error}
        </p>
      ) : (
        <>
          {!doc && <p className="p-6 text-center text-[color:var(--ink-soft)]">Loading PDF...</p>}
          {ready &&
            pageSizes.map((size, i) => (
              <PageView
                key={`${url}:${i}`}
                doc={doc}
                pageNumber={i + 1}
                size={size}
                scale={scale}
                root={scroller}
                target={target}
              />
            ))}
        </>
      )}
    </div>
  );
}

interface PageProps {
  doc: PDFDocumentProxy | null;
  pageNumber: number;
  size: [number, number];
  scale: number;
  root: HTMLElement | null;
  target: ViewerTarget | null;
}

function PageView({ doc, pageNumber, size, scale, root, target }: PageProps) {
  const holder = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [near, setNear] = useState(false);

  useEffect(() => {
    const el = holder.current;
    if (!el || !root) return;
    const observer = new IntersectionObserver(([entry]) => setNear(entry.isIntersecting), {
      root,
      rootMargin: `${root.clientHeight}px 0px`, // about one screen either side
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, [root]);

  useEffect(() => {
    const el = canvas.current;
    if (!el) return;
    if (!doc || !near) {
      el.width = 0; // release the bitmap of pages that scrolled far away
      el.height = 0;
      return;
    }
    let cancelled = false;
    let task: { cancel(): void } | null = null;
    const dpr = window.devicePixelRatio || 1;
    void doc.getPage(pageNumber).then((page) => {
      if (cancelled) return;
      const viewport = page.getViewport({ scale: scale * dpr });
      el.width = Math.floor(viewport.width);
      el.height = Math.floor(viewport.height);
      const render = page.render({ canvas: el, viewport });
      task = render;
      render.promise.catch(() => {
        // a cancelled render rejects; a real failure leaves the page blank
      });
    });
    return () => {
      cancelled = true;
      task?.cancel();
    };
  }, [doc, near, pageNumber, scale]);

  const boxes = useMemo(
    () => (target ? highlightBoxes(target.rects, pageNumber, scale) : []),
    [target, pageNumber, scale],
  );

  return (
    <div data-page={pageNumber} className="mx-auto mb-2 mt-4" style={{ width: size[0] * scale }}>
      <div
        ref={holder}
        className="relative shadow-md"
        style={{ width: size[0] * scale, height: size[1] * scale, background: "var(--paper)" }}
      >
        <canvas ref={canvas} className="absolute inset-0 h-full w-full" />
        {boxes.map((b, i) => (
          <div
            key={`${target?.key}:${i}`}
            className="pointer-events-none absolute rounded-sm"
            style={{
              ...b,
              background: "var(--highlight)",
              animation: "highlight-in 300ms ease-out",
            }}
          />
        ))}
      </div>
      <div className="py-1 text-center text-xs text-[color:var(--ink-soft)]">{pageNumber}</div>
    </div>
  );
}
