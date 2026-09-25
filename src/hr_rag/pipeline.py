"""Request flow: input guardrail -> semantic cache -> agent -> output guardrail."""

import hashlib
import logging
import time
from dataclasses import dataclass, field

from langsmith import traceable

from hr_rag.types import Citation

logger = logging.getLogger(__name__)

BLOCKED_INPUT = (
    "I can't help with that request. Please rephrase your question about company HR policies."
)
BLOCKED_OUTPUT = (
    "My response didn't pass our safety checks, so I can't show it. "
    "Please rephrase your question or contact the HR team directly."
)


@dataclass
class ChatResponse:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    blocked: bool = False
    block_reasons: list[str] = field(default_factory=list)
    cache_hit: bool = False
    model: str | None = None
    standalone_question: str = ""
    contexts: list[str] = field(default_factory=list)
    latency_ms: int = 0


class HRAssistant:
    def __init__(self, guard, cache, agent):
        self.guard = guard
        self.cache = cache
        self.agent = agent

    @traceable(name="hr_assistant", run_type="chain")
    def answer(
        self, question: str, history: list[dict] | None = None, user_email: str | None = None
    ) -> ChatResponse:
        started = time.perf_counter()
        response = self._answer(question.strip(), history or [])
        response.latency_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "answered user=%s blocked=%s cache_hit=%s model=%s latency_ms=%s",
            _hash_user(user_email),
            response.blocked,
            response.cache_hit,
            response.model,
            response.latency_ms,
        )
        return response

    def _answer(self, question: str, history: list[dict]) -> ChatResponse:
        # 1. Input safety guardrail (prompt injection, jailbreak, harmful content, ...)
        input_check = self.guard.check_prompt(question)
        if not input_check.allowed:
            return ChatResponse(BLOCKED_INPUT, blocked=True, block_reasons=input_check.reasons)

        # 2. Resolve follow-ups so the cache and retriever see a standalone question.
        standalone = self.agent.condense(history, question)

        # 3. Semantic cache: a hit skips the agent entirely.
        cached = self.cache.lookup(standalone)
        if cached is not None:
            answer, citations, model, cache_hit, contexts = (
                cached.answer, cached.citations, None, True, [],
            )
        else:
            result = self.agent.run(standalone, history)
            answer, citations, model, cache_hit, contexts = (
                result.answer, result.citations, result.model, False, result.contexts,
            )

        # 4. Output safety guardrail (PII leakage, harmful content, malicious URLs, ...)
        output_check = self.guard.check_response(answer)
        if not output_check.allowed:
            return ChatResponse(
                BLOCKED_OUTPUT,
                blocked=True,
                block_reasons=output_check.reasons,
                model=model,
                standalone_question=standalone,
            )
        if output_check.sanitized_text:
            answer = output_check.sanitized_text

        # 5. Only cache grounded, cited, safe answers.
        if not cache_hit and citations:
            try:
                self.cache.store(standalone, answer, citations)
            except Exception:  # noqa: BLE001 - a cache write must never fail the request
                logger.exception("Failed to write semantic cache")

        return ChatResponse(
            answer,
            citations=citations,
            cache_hit=cache_hit,
            model=model,
            standalone_question=standalone,
            contexts=contexts,
        )


def _hash_user(email: str | None) -> str:
    if not email:
        return "anonymous"
    return hashlib.sha256(email.lower().encode()).hexdigest()[:12]
