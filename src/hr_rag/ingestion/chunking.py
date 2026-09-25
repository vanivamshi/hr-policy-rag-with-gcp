"""Recursive, separator-aware text chunking with overlap."""

import uuid
from dataclasses import dataclass
from pathlib import PurePosixPath

from hr_rag.ingestion.loaders import Section

_SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    title: str
    page: int | None
    chunk_index: int

    def embedding_text(self) -> str:
        """Text sent to the embedders: a short contextual header improves retrieval."""
        header = f"Document: {self.title}" + (f" (page {self.page})" if self.page else "")
        return f"{header}\n{self.text}"

    def payload(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "title": self.title,
            "page": self.page,
            "chunk_index": self.chunk_index,
        }


def _split_pieces(text: str, separators: list[str], size: int) -> list[str]:
    """Break text into pieces no longer than `size`, preferring coarse separators."""
    if len(text) <= size:
        return [text]
    if not separators:
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep, rest = separators[0], separators[1:]
    raw = text.split(sep)
    pieces: list[str] = []
    for i, part in enumerate(raw):
        if i < len(raw) - 1:
            part += sep
        if not part.strip():
            continue
        pieces.extend(_split_pieces(part, rest, size) if len(part) > size else [part])
    return pieces


def _tail(text: str, overlap: int) -> str:
    if overlap <= 0 or len(text) <= overlap:
        return ""
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1 :] if space != -1 else tail


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunks: list[str] = []
    current = ""
    for piece in _split_pieces(text, _SEPARATORS, chunk_size):
        if current and len(current) + len(piece) > chunk_size:
            chunks.append(current.strip())
            current = _tail(current, chunk_overlap)
            if len(current) + len(piece) > chunk_size:
                current = ""
        current += piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def title_from_source(source: str) -> str:
    return PurePosixPath(source).stem.replace("_", " ").replace("-", " ").strip().title()


def chunk_sections(
    sections: list[Section], source: str, chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    title = title_from_source(source)
    chunks: list[Chunk] = []
    for section in sections:
        for text in split_text(section.text, chunk_size, chunk_overlap):
            index = len(chunks)
            chunks.append(
                Chunk(
                    # Deterministic IDs make re-ingestion idempotent.
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}#{index}")),
                    text=text,
                    source=source,
                    title=title,
                    page=section.page,
                    chunk_index=index,
                )
            )
    return chunks
