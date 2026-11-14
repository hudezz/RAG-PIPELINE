# RAG Pipeline

Ask questions about a folder of documents and get an answer written only from
what those documents say, with the sources it used.

Everything runs locally by default: the embeddings come from a
sentence-transformers model on your machine, the vectors live in a local Chroma
database, and generation goes to Ollama. No API key is needed.

## The three stages

```
docs/*.txt, docs/*.pdf
        │
        │  ingestion_pipeline.py    load → split into overlapping chunks → embed
        ▼
db/chroma_db                        local vector store
        │
        │  retrieval_pipeline.py    embed the question → fetch the closest chunks
        ▼
   relevant chunks
        │
        │  rag.py                   write an answer from those chunks, and cite them
        ▼
    answer + sources
```

| File | Does |
|---|---|
| `ingestion_pipeline.py` | Builds the vector store from `docs/`. Run this first. |
| `retrieval_pipeline.py` | Prints the chunks that match a question. Useful for checking retrieval on its own. |
| `rag.py` | The whole thing: retrieve, then answer from the retrieved text. |

`docs/` currently holds company profiles for Google, Microsoft, Nvidia, SpaceX
and Tesla, plus the *Attention Is All You Need* paper as a PDF.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

For local generation, install [Ollama](https://ollama.com) and pull a model:

```bash
ollama pull llama3.2
```

To use OpenAI instead, put a key in `.env`:

```
OPENAI_API_KEY=sk-...
```

`rag.py` then picks OpenAI automatically. `--provider ollama` or
`--provider openai` overrides the choice. `OLLAMA_MODEL` and `OPENAI_MODEL`
change which model is used.

## Using it

```bash
# 1. Index the documents (first run downloads the embedding model, ~90 MB)
python3 ingestion_pipeline.py

# 2. Ask something
python3 rag.py "what legal trouble has Tesla faced?"
```

```
Question: what legal trouble has Tesla faced?
Generating with ollama...

Tesla has faced regulatory scrutiny over its driver-assistance marketing [1]
and several lawsuits brought by shareholders [2]. The documents don't cover
the outcomes.

Sources:
  [1] Tesla.txt
  [2] Tesla.txt
```

Re-index after changing `docs/`:

```bash
python3 ingestion_pipeline.py --rebuild
```

Look at retrieval by itself when an answer seems wrong:

```bash
python3 retrieval_pipeline.py -k 5 --threshold 0.3 "who founded SpaceX?"
```

### Options

| Flag | Means |
|---|---|
| `-k N` | How many chunks to retrieve (default 3) |
| `--threshold X` | Minimum cosine similarity, 0–1 (default 0.4). `0` disables it |
| `--provider` | `ollama` or `openai` |
| `--rebuild` | Ingestion only: discard the store and index again |

## Notes on the settings

**The similarity threshold is why you sometimes get nothing.** At 0.4, a
question whose wording is unlike anything in the documents returns no chunks,
and `rag.py` then says the documents don't cover it rather than inventing an
answer. That is the intended behaviour, but if a question you know is covered
comes back empty, lower it: `--threshold 0.2`.

**Chunking.** Chunks are 1000 characters with 150 of overlap, split on
paragraph, then line, then sentence boundaries. The overlap means a sentence
that straddles a boundary is still findable from either chunk.

**Embeddings.** `all-MiniLM-L6-v2`, downloaded once and then cached. The same
model must be used for ingestion and for querying, which is why both scripts
read it from one constant.

**Answers are constrained to the documents.** The prompt tells the model to use
only the supplied context and to say so when the answer isn't there. That is
not a hard guarantee — it is a prompt — but combined with the threshold it
keeps answers close to the sources.

## Limits

- `db/` is not committed; each clone builds its own index.
- PDFs are read as text, so tables and figures come out poorly.
- Retrieval is pure vector similarity: no keyword search, no reranking, and no
  conversation memory (each question is answered on its own).
