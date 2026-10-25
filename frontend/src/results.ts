/** Pure text helpers for search results: snippet trimming and query-term marking. */

/** Collapse whitespace and cut to about `max` characters at a word boundary. */
export function snippet(text: string, max: number): string {
  const flat = text.replace(/\s+/g, " ").trim();
  if (flat.length <= max) return flat;
  const cut = flat.slice(0, max);
  const space = cut.lastIndexOf(" ");
  return `${(space > 0 ? cut.slice(0, space) : cut).trimEnd()}…`;
}

export interface Part {
  text: string;
  hit: boolean;
}

/** Split text into parts, flagging case-insensitive matches of query words of 2+ characters. */
export function markTerms(text: string, query: string): Part[] {
  const words = [...new Set(query.toLowerCase().split(/\s+/).filter((w) => w.length >= 2))];
  if (!text) return [];
  if (words.length === 0) return [{ text, hit: false }];
  const escaped = words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const re = new RegExp(escaped.join("|"), "gi");
  const parts: Part[] = [];
  let last = 0;
  for (const m of text.matchAll(re)) {
    if (m.index > last) parts.push({ text: text.slice(last, m.index), hit: false });
    parts.push({ text: m[0], hit: true });
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push({ text: text.slice(last), hit: false });
  return parts;
}
