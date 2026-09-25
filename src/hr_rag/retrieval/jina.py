"""Jina AI embeddings and reranker over the public REST API."""

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

EMBED_URL = "https://api.jina.ai/v1/embeddings"
RERANK_URL = "https://api.jina.ai/v1/rerank"


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


_retry = retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, max=20),
    reraise=True,
)


class JinaClient:
    def __init__(
        self,
        api_key: str,
        embedding_model: str,
        embedding_dim: int,
        reranker_model: str,
        batch_size: int = 64,
        timeout: float = 30.0,
    ):
        if not api_key:
            raise ValueError("JINA_API_KEY is not set")
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.reranker_model = reranker_model
        self.batch_size = batch_size
        self._http = httpx.Client(
            timeout=timeout, headers={"Authorization": f"Bearer {api_key}"}
        )

    @_retry
    def _post(self, url: str, payload: dict) -> dict:
        response = self._http.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def _embed(self, texts: list[str], task: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            body = self._post(
                EMBED_URL,
                {
                    "model": self.embedding_model,
                    "task": task,
                    "dimensions": self.embedding_dim,
                    "normalized": True,
                    "input": batch,
                },
            )
            data = sorted(body["data"], key=lambda d: d["index"])
            vectors.extend(d["embedding"] for d in data)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, task="retrieval.passage")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], task="retrieval.query")[0]

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        """Return (document index, relevance score) pairs, best first."""
        if not documents:
            return []
        body = self._post(
            RERANK_URL,
            {
                "model": self.reranker_model,
                "query": query,
                "documents": documents,
                "top_n": min(top_n, len(documents)),
                "return_documents": False,
            },
        )
        return [(r["index"], float(r["relevance_score"])) for r in body["results"]]
