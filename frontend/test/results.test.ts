import { describe, expect, it } from "vitest";
import { markTerms, snippet } from "../src/results";

describe("snippet", () => {
  it("returns short text unchanged, with whitespace collapsed", () => {
    expect(snippet("a  b\n c", 50)).toBe("a b c");
  });

  it("cuts at a word boundary and adds an ellipsis", () => {
    expect(snippet("alpha beta gamma delta", 13)).toBe("alpha beta…");
  });

  it("cuts a single long word hard", () => {
    expect(snippet("abcdefghij", 4)).toBe("abcd…");
  });
});

describe("markTerms", () => {
  it("marks matches case-insensitively and keeps the original text", () => {
    expect(markTerms("The Cat sat", "cat")).toEqual([
      { text: "The ", hit: false },
      { text: "Cat", hit: true },
      { text: " sat", hit: false },
    ]);
  });

  it("ignores single-character words", () => {
    expect(markTerms("a big cat", "a")).toEqual([{ text: "a big cat", hit: false }]);
  });

  it("escapes regex characters", () => {
    expect(markTerms("cost (usd) now", "(usd)")).toEqual([
      { text: "cost ", hit: false },
      { text: "(usd)", hit: true },
      { text: " now", hit: false },
    ]);
    expect(() => markTerms("x", "[ ** ")).not.toThrow();
  });

  it("marks several words and handles empty input", () => {
    const parts = markTerms("red fox, red hen", "fox hen");
    expect(parts.filter((p) => p.hit).map((p) => p.text)).toEqual(["fox", "hen"]);
    expect(markTerms("", "fox")).toEqual([]);
    expect(markTerms("text", "")).toEqual([{ text: "text", hit: false }]);
  });
});
