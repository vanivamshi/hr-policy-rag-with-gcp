"""Guarded search tool exposed to the agent."""

import re

from hr_rag.types import Citation, RetrievedChunk

NO_RESULTS = (
    "NO_RELEVANT_POLICY_FOUND: The HR policy documents contain nothing relevant to this "
    "query. Do not answer from general knowledge."
)
MAX_QUERY_CHARS = 500

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_hr_policies",
        "description": (
            "Search the company's official HR policy documents (leave, benefits, payroll, "
            "conduct, remote work, travel, onboarding, etc.). Returns numbered excerpts. "
            "Call this before answering any policy question; call again with a different "
            "query if the first results are insufficient."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A concise, specific search query, e.g. 'parental leave duration'.",
                }
            },
            "required": ["query"],
        },
    },
}


class SourceRegistry:
    """Gives each retrieved chunk a stable [n] number across all tool calls in a turn."""

    def __init__(self):
        self._numbers: dict[str, int] = {}
        self._chunks: dict[int, RetrievedChunk] = {}

    def number(self, chunk: RetrievedChunk) -> int:
        if chunk.id not in self._numbers:
            n = len(self._numbers) + 1
            self._numbers[chunk.id] = n
            self._chunks[n] = chunk
        return self._numbers[chunk.id]

    @property
    def empty(self) -> bool:
        return not self._chunks

    def cited(self, answer: str) -> list[Citation]:
        numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
        return [
            Citation(
                number=n,
                source=self._chunks[n].source,
                title=self._chunks[n].title,
                page=self._chunks[n].page,
                excerpt=self._chunks[n].text[:500],
            )
            for n in numbers
            if n in self._chunks
        ]


class GuardedSearchTool:
    """Wraps the retriever with input limits, a relevance floor and injection-safe formatting."""

    name = "search_hr_policies"

    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, query: str, registry: SourceRegistry) -> str:
        query = " ".join(str(query).split())[:MAX_QUERY_CHARS]
        if not query:
            return "ERROR: empty query. Provide a specific search query."
        chunks = self.retriever.retrieve(query)
        if not chunks:
            return NO_RESULTS
        blocks = []
        for chunk in chunks:
            n = registry.number(chunk)
            location = f", page {chunk.page}" if chunk.page else ""
            # Excerpts are untrusted data; the tags let the system prompt tell the model so.
            blocks.append(
                f"[{n}] {chunk.title} ({chunk.source}{location})\n"
                f"<policy_excerpt>\n{chunk.text}\n</policy_excerpt>"
            )
        return "\n\n".join(blocks)
