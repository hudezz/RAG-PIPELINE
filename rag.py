"""Stage 3 of the pipeline: answer a question from the retrieved chunks.

This is the step the repository was missing. Ingestion and retrieval were
here, but nothing ever wrote an answer, so the "G" of RAG was absent.

    python3 rag.py "what legal trouble has Tesla faced?"

By default it generates with a local Ollama model, so no API key and no
network are needed:

    ollama pull llama3.2

Set OPENAI_API_KEY (in a .env file or the environment) to use OpenAI instead,
or force either one with --provider.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="langchain")

from dotenv import load_dotenv  # noqa: E402

from retrieval_pipeline import (  # noqa: E402
    build_parser,
    describe_source,
    read_query,
    retrieve,
)

load_dotenv()

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

PROMPT_TEMPLATE = """Answer the question using only the context below.

If the context does not contain the answer, say "The documents don't cover
that" and stop. Do not use anything you know from outside the context. Cite
the sources you used by their number, like [1].

Context:
{context}

Question: {question}

Answer:"""


def choose_provider(requested=None):
    """Pick the generation backend: "ollama" (default) or "openai"."""
    if requested in ("ollama", "openai"):
        return requested
    return "openai" if os.environ.get("OPENAI_API_KEY") else "ollama"


def get_llm(provider):
    """Build the chat model, with an explanation if its package is missing."""
    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise SystemExit(
                "langchain-openai is not installed. Run: pip install langchain-openai"
            ) from exc
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set. Put it in .env, or use --provider ollama.")
        return ChatOpenAI(model=OPENAI_MODEL, temperature=0)

    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise SystemExit(
            "langchain-ollama is not installed. Run: pip install langchain-ollama"
        ) from exc
    return ChatOllama(model=OLLAMA_MODEL, temperature=0)


def format_context(documents):
    """Number the chunks so the model can cite them."""
    return "\n\n".join(
        f"[{i}] {describe_source(document)}\n{document.page_content}"
        for i, document in enumerate(documents, 1)
    )


def answer(query, k=3, threshold=0.4, provider=None):
    """Retrieve, then write an answer grounded in what came back.

    Returns ``(answer_text, documents)``. With no matching chunks, it says so
    rather than letting the model answer from memory.
    """
    documents = retrieve(query, k=k, threshold=threshold)
    if not documents:
        return "The documents don't cover that.", []

    prompt = PROMPT_TEMPLATE.format(context=format_context(documents), question=query)
    response = get_llm(choose_provider(provider)).invoke(prompt)

    # ChatModels return a message; .content is the text.
    text = getattr(response, "content", response)
    return str(text).strip(), documents


def main():
    parser = build_parser("Answer a question from the documents in docs/.")
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai"],
        help="which model to generate with (default: openai if OPENAI_API_KEY is set, else ollama)",
    )
    args = parser.parse_args()

    query = read_query(args, parser)
    if not query:
        parser.error("the question is empty")

    threshold = None if args.threshold <= 0 else args.threshold
    provider = choose_provider(args.provider)

    print(f"Question: {query}")
    print(f"Generating with {provider}...\n")

    text, documents = answer(query, k=args.k, threshold=threshold, provider=provider)

    print(text)

    if documents:
        print("\nSources:")
        for i, document in enumerate(documents, 1):
            print(f"  [{i}] {describe_source(document)}")
        return 0

    if threshold is not None:
        print(f"\n(Nothing cleared the {threshold} similarity threshold. Try --threshold 0.2, or 0 to turn it off.)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
