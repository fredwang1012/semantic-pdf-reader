/** Pure layout maths for the PDF viewer: highlight boxes, scroll target and fit-to-width scale. */
import type { Rect } from "../api";

export interface Box {
  left: number;
  top: number;
  width: number;
  height: number;
}

const PAD = 2; // px of breathing room around each highlight
const MERGE_GAP = 6; // px: neighbouring words on one line fuse into one box
const PAGE_PADDING = 32; // px of horizontal padding reserved around a page

/** The box with x and y swapped. */
function transpose(b: Box): Box {
  return { left: b.top, top: b.left, width: b.height, height: b.width };
}

/**
 * Whether two boxes sit on one text line close enough to merge. A line is longer than it is tall in the direction it
 * is read, so two boxes taller than wide are lines running down a page rotated by 90 or 270 degrees: they are compared
 * with x and y swapped, or neighbouring lines would merge into one block.
 */
function mergeable(a: Box, b: Box): boolean {
  if (a.height > a.width && b.height > b.width) return mergeable(transpose(a), transpose(b));
  const overlap = Math.min(a.top + a.height, b.top + b.height) - Math.max(a.top, b.top);
  if (overlap < 0.5 * Math.min(a.height, b.height)) return false;
  const gap = Math.max(a.left, b.left) - Math.min(a.left + a.width, b.left + b.width);
  return gap <= MERGE_GAP;
}

function union(a: Box, b: Box): Box {
  const left = Math.min(a.left, b.left);
  const top = Math.min(a.top, b.top);
  return {
    left,
    top,
    width: Math.max(a.left + a.width, b.left + b.width) - left,
    height: Math.max(a.top + a.height, b.top + b.height) - top,
  };
}

/**
 * CSS-pixel highlight boxes for one page: scaled, same-line neighbours merged, then padded. The rects are already in
 * the page as displayed, after its /Rotate (see backend/app/pdftext.py), so scaling is the whole mapping.
 */
export function highlightBoxes(rects: Rect[], page: number, scale: number): Box[] {
  let boxes: Box[] = rects
    .filter((r) => r[0] === page)
    .map(([, x0, y0, x1, y1]) => ({
      left: x0 * scale,
      top: y0 * scale,
      width: (x1 - x0) * scale,
      height: (y1 - y0) * scale,
    }));

  for (let merged = true; merged; ) {
    merged = false;
    const next: Box[] = [];
    for (const box of boxes) {
      const i = next.findIndex((other) => mergeable(other, box));
      if (i >= 0) {
        next[i] = union(next[i], box);
        merged = true;
      } else {
        next.push(box);
      }
    }
    boxes = next;
  }

  return boxes.map((b) => {
    const left = Math.max(0, b.left - PAD);
    const top = Math.max(0, b.top - PAD);
    return {
      left,
      top,
      width: b.left + b.width + PAD - left,
      height: b.top + b.height + PAD - top,
    };
  });
}

/** The rect with the lowest page, then the lowest y0: where to scroll to. */
export function firstRect(rects: Rect[]): Rect | null {
  let best: Rect | null = null;
  for (const r of rects) {
    if (!best || r[0] < best[0] || (r[0] === best[0] && r[2] < best[2])) best = r;
  }
  return best;
}

/** Scale that fits a page into the container width minus padding, within [0.25, max]. */
export function fitScale(pageWidthPt: number, containerWidthPx: number, max = 2): number {
  const fit = (containerWidthPx - PAGE_PADDING) / pageWidthPt;
  return Math.min(max, Math.max(0.25, fit));
}
