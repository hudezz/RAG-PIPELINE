"""Stage 1 of the pipeline: turn the files in docs/ into a vector store.

Loads every .txt and .pdf under docs/, splits them into overlapping chunks,
embeds each chunk locally with a sentence-transformers model, and saves the
result to a Chroma database in db/chroma_db.

    python3 ingestion_pipeline.py          # build it, or report what is there
    python3 ingestion_pipeline.py --rebuild  # throw it away and build again

Nothing here calls a paid API: the embedding model runs on this machine.
"""

import argparse
import os
import shutil

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DOCS_PATH = "docs"
PERSIST_DIRECTORY = "db/chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def get_embedding_model():
    """The same model has to be used for ingestion and for querying."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_documents(docs_path=DOCS_PATH):
    """Load every .txt and .pdf under ``docs_path``."""
    if not os.path.isdir(docs_path):
        raise FileNotFoundError(
            f"No {docs_path}/ directory. Create it and add the documents you want to search."
        )

    documents = []

    text_loader = DirectoryLoader(path=docs_path, glob="**/*.txt", loader_cls=TextLoader)
    documents.extend(text_loader.load())

    # The PDFs in docs/ used to be ignored: the only glob was "*.txt", so
    # attention-is-all-you-need.pdf was never indexed. PyPDFLoader gives one
    # document per page, each tagged with its page number.
    pdf_loader = DirectoryLoader(path=docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents.extend(pdf_loader.load())

    if not documents:
        raise FileNotFoundError(f"No .txt or .pdf files found in {docs_path}/.")

    total_chars = sum(len(doc.page_content) for doc in documents)
    sources = sorted({os.path.basename(doc.metadata.get("source", "?")) for doc in documents})
    print(f"Loaded {len(documents)} document(s), {total_chars:,} characters, from {len(sources)} file(s):")
    for source in sources:
        print(f"  - {source}")

    return documents


def split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split documents into overlapping chunks.

    Uses RecursiveCharacterTextSplitter rather than CharacterTextSplitter:
    the latter only splits on blank lines, so it happily emitted chunks far
    longer than chunk_size whenever a document had few paragraph breaks. The
    overlap keeps a sentence that straddles a boundary retrievable from either
    side.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    if chunks:
        lengths = [len(chunk.page_content) for chunk in chunks]
        print(
            f"Split into {len(chunks)} chunks "
            f"(shortest {min(lengths)}, longest {max(lengths)}, "
            f"average {sum(lengths) // len(lengths)} characters)."
        )

    return chunks


def create_vector_store(chunks, persist_directory=PERSIST_DIRECTORY):
    """Embed the chunks and write them to Chroma."""
    print(f"Embedding {len(chunks)} chunks with {EMBEDDING_MODEL} (first run downloads the model)...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )

    print(f"Vector store written to {persist_directory}/")
    return vectorstore


def load_vector_store(persist_directory=PERSIST_DIRECTORY):
    """Open an existing Chroma store."""
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=get_embedding_model(),
        collection_metadata={"hnsw:space": "cosine"},
    )


def count_chunks(vectorstore):
    """Number of stored chunks, or None if Chroma will not say."""
    try:
        return vectorstore._collection.count()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Build the vector store from docs/.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="delete the existing vector store and build it from scratch",
    )
    args = parser.parse_args()

    print("=== RAG document ingestion ===\n")

    if args.rebuild and os.path.exists(PERSIST_DIRECTORY):
        print(f"Removing {PERSIST_DIRECTORY}/ ...")
        shutil.rmtree(PERSIST_DIRECTORY)

    if os.path.exists(PERSIST_DIRECTORY):
        vectorstore = load_vector_store()
        count = count_chunks(vectorstore)
        print(
            f"Vector store already exists at {PERSIST_DIRECTORY}/"
            + (f" with {count} chunks." if count is not None else ".")
        )
        print("Add --rebuild to index the documents again.")
        return vectorstore

    documents = load_documents()
    chunks = split_documents(documents)
    vectorstore = create_vector_store(chunks)

    print("\nDone. Query it with:  python3 rag.py \"your question\"")
    return vectorstore


if __name__ == "__main__":
    main()
