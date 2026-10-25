# Retrieval eval

Compares retrievers (BM25, hybrid BM25 + multilingual-e5, later a visual one) on a hand-labelled question set.

## Build the question set

1. Put the PDFs in this folder (`backend/eval/`). PDFs here are gitignored. The set targets the CSC2626 Week 5
   papers: NPI, NTP, TACO, GTI, Motion Grammar and Baker et al.
2. Write `questions.jsonl` next to them, one JSON object per line, about 30 in total:

   ```json
   {"question": "...", "file": "npi.pdf", "pages": [3, 4]}
   ```

   `file` is the PDF's file name in this folder. `pages` are the 1-based pages that answer the question (any one
   counts). Check the page numbers by hand in a PDF viewer.
3. Mix two kinds of question: exact-term ones (acronyms, equation names, a method's name) where BM25 should do well,
   and paraphrased ones that share few words with the passage, where embeddings should help.
   `questions.example.jsonl` shows the format.

## Run

From `backend/`:

```
py -3.12 -m uv run python -m app.evaluate eval eval/questions.jsonl --k 1 3 5 10 --retrievers bm25 hybrid --json eval/out.json
```

`hybrid` needs the embedding model (`py -3.12 -m uv sync --extra embed`, and the `EMBED` setting not `off`);
without it, it is skipped with a message and only BM25 is reported.

The output is a Markdown table with recall@k, MRR, median and p95 query latency, index build time and index size
(chunk count, embedding bytes). A question whose PDF is missing or has no text layer is reported as a warning and
counted as not found.
