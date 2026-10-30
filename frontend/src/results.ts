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

/** Split text into parts, flagging whole-word, case-insensitive matches of the query's words of 2+ characters. */
export function markTerms(text: string, query: string): Part[] {
  const seen = new Set<string>();
  const words = (query.match(/[\p{L}\p{N}]+/gu) ?? []).filter((w) => {
    const key = w.toLowerCase();
    if (w.length < 2 || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  if (!text) return [];
  if (words.length === 0) return [{ text, hit: false }];
  words.sort((a, b) => b.length - a.length);
  const re = new RegExp(String.raw`(?<![\p{L}\p{N}])(${words.join("|")})(?![\p{L}\p{N}])`, "giu");
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
