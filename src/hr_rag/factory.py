"""Wire concrete clients together from Settings."""

from hr_rag.config import Settings


def build_qdrant_client(settings: Settings):
    from qdrant_client import QdrantClient

    if not settings.qdrant_url:
        raise ValueError("QDRANT_URL is not set")
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=30)


def build_jina(settings: Settings):
    from hr_rag.retrieval.jina import JinaClient

    return JinaClient(
        api_key=settings.jina_api_key,
        embedding_model=settings.jina_embedding_model,
        embedding_dim=settings.jina_embedding_dim,
        reranker_model=settings.jina_reranker_model,
    )


def build_sparse_model(settings: Settings):
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(settings.bm25_model, cache_dir=settings.fastembed_cache_dir)


def build_store(settings: Settings, client=None, jina=None):
    from hr_rag.retrieval.qdrant_store import HybridStore

    return HybridStore(
        client or build_qdrant_client(settings),
        settings.qdrant_collection,
        jina or build_jina(settings),
        build_sparse_model(settings),
    )


def build_cache(settings: Settings, client, jina):
    from hr_rag.cache import NullCache, SemanticCache

    if not settings.cache_enabled:
        return NullCache()
    cache = SemanticCache(
        client,
        settings.cache_collection,
        jina,
        threshold=settings.cache_similarity_threshold,
        ttl_seconds=settings.cache_ttl_seconds,
    )
    cache.ensure_collection()
    return cache


def build_guard(settings: Settings):
    from hr_rag.guardrails import ModelArmorGuard, NullGuard

    if not settings.model_armor_enabled:
        return NullGuard()
    if not settings.gcp_project:
        raise ValueError("GCP_PROJECT is required when MODEL_ARMOR_ENABLED=true")
    return ModelArmorGuard(
        project=settings.gcp_project,
        location=settings.model_armor_location,
        template=settings.model_armor_template,
        fail_open=settings.guardrail_fail_open,
    )


def build_agent(settings: Settings, store, jina):
    from hr_rag.agent.agent import HRAgent
    from hr_rag.agent.tools import GuardedSearchTool
    from hr_rag.llm import build_router
    from hr_rag.retrieval.retriever import Retriever

    retriever = Retriever(
        store,
        jina,
        prefetch_k=settings.retrieval_prefetch_k,
        top_n=settings.rerank_top_n,
        min_score=settings.rerank_min_score,
    )
    return HRAgent(
        build_router(settings),
        GuardedSearchTool(retriever),
        max_steps=settings.agent_max_steps,
        max_history_messages=settings.agent_max_history_messages,
    )


def build_assistant(settings: Settings):
    from hr_rag.pipeline import HRAssistant

    client = build_qdrant_client(settings)
    jina = build_jina(settings)
    store = build_store(settings, client, jina)
    return HRAssistant(
        guard=build_guard(settings),
        cache=build_cache(settings, client, jina),
        agent=build_agent(settings, store, jina),
    )
