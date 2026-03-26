from rank_bm25 import BM25Okapi
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

RRF_K = 60  
TOP_K = 5   # final number of documents to return


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_bm25_index(raw_texts: list[str]) -> BM25Okapi:
    tokenized = [tokenize(t) for t in raw_texts]
    return BM25Okapi(tokenized)


def rrf_fusion(
    bm25_ranked: list[int],
    dense_ranked: list[int],
    k: int = RRF_K,
) -> list[int]:
    """
    Given two lists of document indices (ordered by rank),
    return a merged list of indices sorted by RRF score (descending).
    """
    scores: dict[int, float] = {}
    for rank, idx in enumerate(bm25_ranked):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rank + k)
    for rank, idx in enumerate(dense_ranked):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rank + k)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


def hybrid_retrieve(
    query: str,
    chunks: list[Document],
    raw_texts: list[str],
    vectorstore: Chroma,
    top_k: int = TOP_K,
) -> list[Document]:
    n = len(chunks)
    retrieve_n = min(n, max(top_k * 3, 20))  # retrieve more before fusion

    # BM25 retrieval 
    bm25 = build_bm25_index(raw_texts)
    bm25_scores = bm25.get_scores(tokenize(query))
    bm25_ranked = sorted(range(n), key=lambda i: bm25_scores[i], reverse=True)[:retrieve_n]

    # Dense retrieval 
    dense_docs = vectorstore.similarity_search(query, k=retrieve_n)
    content_to_idx = {chunk.page_content: i for i, chunk in enumerate(chunks)}
    dense_ranked = []
    for doc in dense_docs:
        idx = content_to_idx.get(doc.page_content)
        if idx is not None:
            dense_ranked.append(idx)

    # RRF fusion
    fused = rrf_fusion(bm25_ranked, dense_ranked)
    top_indices = fused[:top_k]

    return [chunks[i] for i in top_indices]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

    from embed_and_store import load_vectorstore, load_bm25_data

    query = "How do I use git to undo a commit?"
    print(f"Query: {query}\n")

    vectorstore = load_vectorstore()
    chunks, raw_texts = load_bm25_data()
    results = hybrid_retrieve(query, chunks, raw_texts, vectorstore)

    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        print(f"[{i}] Source: {source}")
        print(doc.page_content[:200])
        print()
