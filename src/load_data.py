import os
import subprocess
from pathlib import Path
from typing import Tuple

from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

DATA_DIR = Path(__file__).parent.parent / "data"
POSTS_DIR = DATA_DIR / "_posts"
REPO_URL = "https://github.com/missing-semester/missing-semester.git"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def clone_repo_if_needed() -> None:
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        print(f"[load_data] Data directory already exists at {DATA_DIR}, skipping clone.")
        return
    print(f"[load_data] Cloning {REPO_URL} into {DATA_DIR} ...")
    subprocess.run(["git", "clone", REPO_URL, str(DATA_DIR)], check=True)
    print("[load_data] Clone complete.")


def load_and_chunk() -> Tuple[list[Document], list[str]]:
    clone_repo_if_needed()

    # The lecture files live in _posts/ as .md files
    md_dir = str(POSTS_DIR) if POSTS_DIR.exists() else str(DATA_DIR)
    print(f"[load_data] Loading Markdown files from: {md_dir}")

    loader = DirectoryLoader(
        md_dir,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True,
    )
    docs = loader.load()
    print(f"[load_data] Loaded {len(docs)} document(s).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"[load_data] Split into {len(chunks)} chunk(s).")

    raw_texts = [chunk.page_content for chunk in chunks]
    return chunks, raw_texts


if __name__ == "__main__":
    chunks, raw_texts = load_and_chunk()
    print(f"\nTotal chunks: {len(chunks)}")
    print("\n--- Sample chunk ---")
    print(f"Source: {chunks[0].metadata.get('source', 'unknown')}")
    print(chunks[0].page_content[:300])
