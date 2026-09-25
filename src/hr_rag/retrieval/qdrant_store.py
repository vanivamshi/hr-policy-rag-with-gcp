"""Qdrant hybrid index: Jina dense vectors + BM25 sparse vectors, fused with RRF."""

from qdrant_client import QdrantClient, models

from hr_rag.ingestion.chunking import Chunk
from hr_rag.types import RetrievedChunk

DENSE = "dense"
SPARSE = "bm25"


class HybridStore:
    def __init__(self, client: QdrantClient, collection: str, jina, sparse_model):
        self.client = client
        self.collection = collection
        self.jina = jina
        self.sparse_model = sparse_model  # fastembed.SparseTextEmbedding

    def ensure_collection(self, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if exists:
            return
        self.client.create_collection(
            self.collection,
            vectors_config={
                DENSE: models.VectorParams(
                    size=self.jina.embedding_dim, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                # IDF is computed by Qdrant, which is what turns raw term counts into BM25.
                SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        self.client.create_payload_index(
            self.collection, "source", field_schema=models.PayloadSchemaType.KEYWORD
        )

    def delete_source(self, source: str) -> None:
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))]
                )
            ),
        )

    def upsert(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [c.embedding_text() for c in batch]
            dense = self.jina.embed_documents(texts)
            sparse = list(self.sparse_model.embed(texts))
            points = [
                models.PointStruct(
                    id=chunk.id,
                    vector={
                        DENSE: d,
                        SPARSE: models.SparseVector(
                            indices=s.indices.tolist(), values=s.values.tolist()
                        ),
                    },
                    payload=chunk.payload(),
                )
                for chunk, d, s in zip(batch, dense, sparse)
            ]
            self.client.upsert(self.collection, points=points, wait=True)

    def hybrid_search(self, query: str, limit: int) -> list[RetrievedChunk]:
        dense = self.jina.embed_query(query)
        sparse = next(iter(self.sparse_model.query_embed(query)))
        result = self.client.query_points(
            self.collection,
            prefetch=[
                models.Prefetch(query=dense, using=DENSE, limit=limit),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse.indices.tolist(), values=sparse.values.tolist()
                    ),
                    using=SPARSE,
                    limit=limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [
            RetrievedChunk(
                id=str(p.id),
                text=p.payload["text"],
                source=p.payload["source"],
                title=p.payload.get("title", p.payload["source"]),
                page=p.payload.get("page"),
                score=p.score,
            )
            for p in result.points
        ]
