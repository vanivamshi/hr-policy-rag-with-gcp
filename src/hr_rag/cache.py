"""Semantic answer cache stored in its own Qdrant collection."""

import time
import uuid
from dataclasses import asdict, dataclass

from hr_rag.types import Citation


@dataclass(frozen=True)
class CachedAnswer:
    question: str
    answer: str
    citations: list[Citation]
    score: float


class NullCache:
    def lookup(self, question: str) -> CachedAnswer | None:
        return None

    def store(self, question: str, answer: str, citations: list[Citation]) -> None:
        return None


class SemanticCache:
    def __init__(self, client, collection: str, jina, threshold: float, ttl_seconds: int):
        self.client = client
        self.collection = collection
        self.jina = jina
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds

    def ensure_collection(self) -> None:
        from qdrant_client import models

        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(
                size=self.jina.embedding_dim, distance=models.Distance.COSINE
            ),
        )
        self.client.create_payload_index(
            self.collection, "created_at", field_schema=models.PayloadSchemaType.FLOAT
        )

    def clear(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.ensure_collection()

    def lookup(self, question: str) -> CachedAnswer | None:
        from qdrant_client import models

        result = self.client.query_points(
            self.collection,
            query=self.jina.embed_query(question),
            limit=1,
            score_threshold=self.threshold,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="created_at", range=models.Range(gte=time.time() - self.ttl_seconds)
                    )
                ]
            ),
            with_payload=True,
        )
        if not result.points:
            return None
        point = result.points[0]
        payload = point.payload
        return CachedAnswer(
            question=payload["question"],
            answer=payload["answer"],
            citations=[Citation(**c) for c in payload.get("citations", [])],
            score=point.score,
        )

    def store(self, question: str, answer: str, citations: list[Citation]) -> None:
        from qdrant_client import models

        normalized = " ".join(question.lower().split())
        self.client.upsert(
            self.collection,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"cache:{normalized}")),
                    vector=self.jina.embed_query(question),
                    payload={
                        "question": question,
                        "answer": answer,
                        "citations": [asdict(c) for c in citations],
                        "created_at": time.time(),
                    },
                )
            ],
        )
