from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    source: str
    title: str
    page: int | None
    score: float


@dataclass(frozen=True)
class Citation:
    number: int
    source: str
    title: str
    page: int | None
    excerpt: str

    @property
    def label(self) -> str:
        return f"{self.title} (p. {self.page})" if self.page else self.title
