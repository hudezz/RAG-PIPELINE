"""Stage 2 of the pipeline: fetch the chunks that match a question.

    python3 retrieval_pipeline.py "what legal trouble has Tesla faced?"
    python3 retrieval_pipeline.py -k 5 --threshold 0.3 "who founded SpaceX?"

This prints the raw context, which is useful for checking whether retrieval is
working at all. To get an answer written from that context, use rag.py.
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="langchain")

from ingestion_pipeline import PERSIST_DIRECTORY, load_vector_store  # noqa: E402

DEFAULT_K = 3
DEFAULT_THRESHOLD = 0.4


def get_retriever(k=DEFAULT_K, threshold=DEFAULT_THRESHOLD, persist_directory=PERSIST_DIRECTORY):
    """Return a retriever over the saved vector store.

    With ``threshold`` set, only chunks at or above that cosine similarity come
    back, so an unrelated question returns nothing rather than the three
    least-bad chunks in the database. Pass ``threshold=None`` to always get the
    top ``k``.
    """
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(
            f"No vector store at {persist_directory}/. Run: python3 ingestion_pipeline.py"
        )

    vectorstore = load_vector_store(persist_directory)

    if threshold is None:
        return vectorstore.as_retriever(search_kwargs={"k": k})

    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": threshold},
    )


def retrieve(query, k=DEFAULT_K, threshold=DEFAULT_THRESHOLD):
    """Return the chunks relevant to ``query``, best first."""
    return get_retriever(k=k, threshold=threshold).invoke(query)


def describe_source(document):
    """A short label for where a chunk came from, including the PDF page."""
    source = os.path.basename(document.metadata.get("source", "unknown"))
    page = document.metadata.get("page")
    return f"{source} (page {page + 1})" if isinstance(page, int) else source


def build_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("query", nargs="*", help="the question to search for")
    parser.add_argument("-k", type=int, default=DEFAULT_K, help=f"chunks to fetch (default {DEFAULT_K})")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"minimum cosine similarity, 0 to disable (default {DEFAULT_THRESHOLD})",
    )
    return parser


def read_query(args, parser):
    """The query from the command line, or typed in when none was given."""
    if args.query:
        return " ".join(args.query)
    try:
        return input("Question: ").strip()
    except EOFError:
        parser.error("no question given")


def main():
    parser = build_parser("Search the vector store and print the matching chunks.")
    args = parser.parse_args()
    query = read_query(args, parser)
    if not query:
        parser.error("the question is empty")

    threshold = None if args.threshold <= 0 else args.threshold

    print(f"Question: {query}")
    print("Searching...\n")

    documents = retrieve(query, k=args.k, threshold=threshold)

    if not documents:
        print("Nothing matched.")
        if threshold is not None:
            print(f"The similarity threshold is {threshold}; try --threshold 0.2, or 0 to turn it off.")
        return 1

    print(f"--- {len(documents)} chunk(s) ---")
    for i, document in enumerate(documents, 1):
        print(f"\n[{i}] {describe_source(document)}")
        print(document.page_content)
        print("-" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
