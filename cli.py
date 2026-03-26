"""
Missing Semester RAG Assistant — CLI

Usage:
  python cli.py                        # interactive mode
  python cli.py "your question here"   # single-question mode
  python cli.py --setup                # build vector store (first-time setup)
  python cli.py --rebuild              # force-rebuild vector store
"""

import argparse
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

CHROMA_DIR = Path(__file__).parent / "chroma_db"
BM25_CACHE = Path(__file__).parent / "bm25_texts.pkl"
SEPARATOR = "-" * 70


def check_setup() -> bool:
    return CHROMA_DIR.exists() and BM25_CACHE.exists()


def do_setup(force: bool = False) -> None:
    from load_data import load_and_chunk
    from embed_and_store import build_vectorstore

    if not force and check_setup():
        print("Vector store already exists. Use --rebuild to force a rebuild.")
        return

    if force and CHROMA_DIR.exists():
        print("Removing existing vector store...")
        shutil.rmtree(CHROMA_DIR)
        BM25_CACHE.unlink(missing_ok=True)

    print("Setting up the vector store (this may take a few minutes)...")
    chunks, raw_texts = load_and_chunk()
    build_vectorstore(chunks, raw_texts)
    print("\nSetup complete. Run `python cli.py` to start asking questions.")


def load_resources():
    from embed_and_store import load_vectorstore, load_bm25_data

    print("Loading resources...")
    vectorstore = load_vectorstore()
    chunks, raw_texts = load_bm25_data()
    print(f"Ready. Loaded {len(chunks)} chunks.\n")
    return vectorstore, chunks, raw_texts


def run_query(question: str, vectorstore, chunks, raw_texts) -> None:
    from pipeline import answer

    result = answer(question, chunks, raw_texts, vectorstore)

    # Deduplicate sources for display
    seen = []
    for s in result["sources"]:
        if s not in seen:
            seen.append(s)
    print(f"\n[Retrieved from: {', '.join(seen)}]\n")
    print(result["answer"])
    print()


def interactive_mode(vectorstore, chunks, raw_texts) -> None:
    print("Missing Semester RAG Assistant")
    print('Type your question, or "exit" / "quit" to quit.')
    print(SEPARATOR)

    while True:
        try:
            question = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        run_query(question, vectorstore, chunks, raw_texts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Missing Semester RAG Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python cli.py                        interactive mode
  python cli.py "How do I undo a git commit?"
  python cli.py --setup                first-time setup
  python cli.py --rebuild              force-rebuild index
        """,
    )
    parser.add_argument("question", nargs="?", help="question to answer (omit for interactive mode)")
    parser.add_argument("--setup", action="store_true", help="build the vector store")
    parser.add_argument("--rebuild", action="store_true", help="force-rebuild the vector store")
    args = parser.parse_args()

    if args.setup or args.rebuild:
        do_setup(force=args.rebuild)
        return

    if not check_setup():
        print("Vector store not found. Run `python cli.py --setup` first.")
        sys.exit(1)

    vectorstore, chunks, raw_texts = load_resources()

    if args.question:
        run_query(args.question, vectorstore, chunks, raw_texts)
    else:
        interactive_mode(vectorstore, chunks, raw_texts)


if __name__ == "__main__":
    main()
