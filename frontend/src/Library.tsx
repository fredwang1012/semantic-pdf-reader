/** Sidebar listing the stored PDFs: select, upload (button or drop) and delete with an inline confirm. */
import { useRef, useState } from "react";
import { ApiError, deleteDoc, uploadDoc, type DocMeta } from "./api";

interface Props {
  docs: DocMeta[];
  currentId: string | null;
  onSelect: (id: string) => void;
  onUploaded: (doc: DocMeta) => void;
  onDeleted: (id: string) => void;
}

const message = (e: unknown) => (e instanceof ApiError ? e.message : "Something went wrong");

export default function Library({ docs, currentId, onSelect, onUploaded, onDeleted }: Props) {
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  async function upload(file: File) {
    setError(null);
    setStatus(`Uploading ${file.name}…`);
    try {
      onUploaded(await uploadDoc(file));
    } catch (e) {
      setError(message(e));
    } finally {
      setStatus(null);
    }
  }

  async function remove(id: string) {
    setConfirmId(null);
    setError(null);
    try {
      await deleteDoc(id);
      onDeleted(id);
    } catch (e) {
      setError(message(e));
    }
  }

  function drop(e: React.DragEvent) {
    e.preventDefault();
    setDragging(false);
    if (status !== null) return;
    const file = [...e.dataTransfer.files].find((f) => f.name.toLowerCase().endsWith(".pdf"));
    if (file) void upload(file);
    else setError("Drop a PDF file");
  }

  return (
    <aside
      className={`flex w-64 shrink-0 flex-col border-r border-[var(--ground-edge)] ${dragging ? "bg-[var(--ground-edge)]" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={drop}
    >
      <div className="p-3">
        <button
          type="button"
          disabled={status !== null}
          onClick={() => input.current?.click()}
          className="w-full rounded border border-[var(--ground-edge)] px-3 py-1.5 text-sm hover:bg-[var(--ground-edge)] disabled:opacity-50"
        >
          Upload PDF
        </button>
        <input
          ref={input}
          type="file"
          accept="application/pdf,.pdf"
          hidden
          onChange={(e) => {
            const file = e.target.files?.[0];
            e.target.value = "";
            if (file) void upload(file);
          }}
        />
        {status && <p className="mt-2 text-xs text-[var(--ink-soft)]">{status}</p>}
        {error && (
          <p role="alert" className="mt-2 text-xs text-red-600">
            {error}
          </p>
        )}
      </div>
      <ul className="min-h-0 flex-1 overflow-y-auto">
        {docs.length === 0 && (
          <li className="px-3 text-sm text-[var(--ink-soft)]">Drop a PDF here to add it</li>
        )}
        {docs.map((d) => (
          <li key={d.id} className={`px-3 py-2 ${d.id === currentId ? "bg-[var(--ground-edge)]" : ""}`}>
            <button type="button" onClick={() => onSelect(d.id)} className="block w-full text-left">
              <span className="block truncate text-sm font-medium">{d.title}</span>
              <span className="block text-xs text-[var(--ink-soft)]">
                {d.pages} {d.pages === 1 ? "page" : "pages"}
                {d.embedded ? " · semantic" : " · keyword only"}
              </span>
            </button>
            {confirmId === d.id ? (
              <div className="mt-1 flex gap-3 text-xs">
                <button type="button" onClick={() => void remove(d.id)} className="text-red-600">
                  Delete
                </button>
                <button type="button" onClick={() => setConfirmId(null)}>
                  Cancel
                </button>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => setConfirmId(d.id)}
                className="mt-1 text-xs text-[var(--ink-soft)] hover:underline"
              >
                Remove
              </button>
            )}
          </li>
        ))}
      </ul>
    </aside>
  );
}
