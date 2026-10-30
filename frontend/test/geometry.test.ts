import { describe, expect, it } from "vitest";
import type { Rect } from "../src/api";
import { firstRect, fitScale, highlightBoxes } from "../src/viewer/geometry";

describe("highlightBoxes", () => {
  it("keeps only rects on the requested page", () => {
    const rects: Rect[] = [
      [1, 10, 100, 50, 112],
      [2, 10, 100, 50, 112],
    ];
    expect(highlightBoxes(rects, 2, 1)).toHaveLength(1);
    expect(highlightBoxes(rects, 3, 1)).toEqual([]);
  });

  it("scales and pads by 2px on every side", () => {
    const [box] = highlightBoxes([[1, 10, 20, 50, 30]], 1, 2);
    expect(box).toEqual({ left: 18, top: 38, width: 84, height: 24 });
  });

  it("clamps left and top at zero", () => {
    const [box] = highlightBoxes([[1, 0, 1, 10, 11]], 1, 1);
    expect(box.left).toBe(0);
    expect(box.top).toBe(0);
    expect(box.width).toBe(12);
    expect(box.height).toBe(13);
  });

  it("merges adjacent rects on the same line", () => {
    const boxes = highlightBoxes(
      [
        [1, 10, 100, 50, 112],
        [1, 53, 100, 90, 112],
      ],
      1,
      1,
    );
    expect(boxes).toEqual([{ left: 8, top: 98, width: 84, height: 16 }]);
  });

  it("does not merge across a wide gap", () => {
    const boxes = highlightBoxes(
      [
        [1, 10, 100, 50, 112],
        [1, 70, 100, 90, 112],
      ],
      1,
      1,
    );
    expect(boxes).toHaveLength(2);
  });

  it("does not merge rects on different lines", () => {
    const boxes = highlightBoxes(
      [
        [1, 10, 100, 50, 112],
        [1, 52, 114, 90, 126],
      ],
      1,
      1,
    );
    expect(boxes).toHaveLength(2);
  });
});

// Where a box in the unrotated 612 x 792 page is displayed once /Rotate turns the page clockwise: the space the
// backend reports rects in.
function turn([x0, y0, x1, y1]: number[], rotation: number, w = 612, h = 792): number[] {
  if (rotation === 90) return [h - y1, x0, h - y0, x1];
  if (rotation === 180) return [w - x1, h - y1, w - x0, h - y0];
  if (rotation === 270) return [y0, w - x1, y1, w - x0];
  return [x0, y0, x1, y1];
}

describe("highlightBoxes on rotated pages", () => {
  // Two lines of one passage, 2pt apart, in the unrotated page.
  const lines = [
    [72, 100, 400, 114],
    [72, 116, 300, 130],
  ];

  it.each([0, 90, 180, 270])("scales each line of a page rotated by %i into place, unmerged", (rotation) => {
    const rects = lines.map((l) => [1, ...turn(l, rotation)] as Rect);
    expect(highlightBoxes(rects, 1, 1.5)).toEqual(
      rects.map(([, x0, y0, x1, y1]) => ({
        left: x0 * 1.5 - 2,
        top: y0 * 1.5 - 2,
        width: (x1 - x0) * 1.5 + 4,
        height: (y1 - y0) * 1.5 + 4,
      })),
    );
  });

  it("puts a line of a page rotated by 90 at the right edge, running down", () => {
    const [box] = highlightBoxes([[1, ...turn(lines[0], 90)] as Rect], 1, 1);
    expect(box).toEqual({ left: 676, top: 70, width: 18, height: 332 });
  });

  it.each([90, 270])("merges two pieces of one line on a page rotated by %i", (rotation) => {
    const pieces = [
      [72, 100, 200, 114],
      [203, 100, 400, 114],
    ];
    const boxes = highlightBoxes(pieces.map((l) => [1, ...turn(l, rotation)] as Rect), 1, 1);
    expect(boxes).toEqual(highlightBoxes([[1, ...turn([72, 100, 400, 114], rotation)] as Rect], 1, 1));
  });
});

describe("firstRect", () => {
  it("picks the lowest page, then the lowest y0", () => {
    const rects: Rect[] = [
      [3, 0, 5, 10, 15],
      [2, 0, 300, 10, 310],
      [2, 0, 40, 10, 50],
    ];
    expect(firstRect(rects)).toEqual([2, 0, 40, 10, 50]);
  });

  it("returns null for no rects", () => {
    expect(firstRect([])).toBeNull();
  });
});

describe("fitScale", () => {
  it("fits the page into the container minus 32px padding", () => {
    expect(fitScale(500, 632)).toBeCloseTo(1.2);
  });

  it("never exceeds max", () => {
    expect(fitScale(100, 2000)).toBe(2);
    expect(fitScale(100, 2000, 1.5)).toBe(1.5);
  });

  it("never drops below 0.25", () => {
    expect(fitScale(1000, 100)).toBe(0.25);
  });
});
