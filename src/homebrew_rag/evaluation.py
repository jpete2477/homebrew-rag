"""Retrieval evaluation.

The test most RAG projects skip. Plumbing tests tell you the pipeline runs;
this tells you it retrieves the right thing. Run it after any change to
chunking, the embedding model, or top_k — otherwise you have no way to know
whether a "smarter" tweak actually improved retrieval or just felt like it did.

No LLM calls happen here, so it is fast and free to run in a loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .pipeline import RagPipeline


@dataclass
class GoldenCase:
    query: str
    expected_sources: list[str]
    source_tag: str | None = None
    note: str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> GoldenCase:
        expected = raw.get("expected_sources") or raw.get("expected_source")
        if isinstance(expected, str):
            expected = [expected]
        if not expected:
            raise ValueError(f"case {raw.get('query')!r} has no expected_source(s)")
        return cls(
            query=raw["query"],
            expected_sources=list(expected),
            source_tag=raw.get("source_tag"),
            note=raw.get("note", ""),
        )


@dataclass
class CaseResult:
    case: GoldenCase
    retrieved: list[str]
    rank: int | None  # 1-based rank of the first expected source, None if absent

    @property
    def hit(self) -> bool:
        return self.rank is not None

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.rank if self.rank else 0.0


@dataclass
class EvalReport:
    top_k: int
    results: list[CaseResult] = field(default_factory=list)

    @property
    def recall(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.hit for r in self.results) / len(self.results)

    @property
    def mrr(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.reciprocal_rank for r in self.results) / len(self.results)

    def render(self) -> str:
        lines = []
        for result in self.results:
            mark = "PASS" if result.hit else "FAIL"
            rank = f"rank {result.rank}" if result.rank else "not retrieved"
            lines.append(f"[{mark}] {result.case.query}")
            lines.append(f"        expected: {', '.join(result.case.expected_sources)} ({rank})")
            lines.append(f"        got:      {', '.join(result.retrieved) or '(nothing)'}")
        hits = sum(r.hit for r in self.results)
        lines.append("")
        lines.append(
            f"Recall@{self.top_k}: {hits}/{len(self.results)} = {self.recall:.1%}   "
            f"MRR: {self.mrr:.3f}"
        )
        return "\n".join(lines)


def load_golden_set(path: str | Path) -> list[GoldenCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = raw["cases"] if isinstance(raw, dict) else raw
    return [GoldenCase.from_dict(case) for case in cases]


def evaluate(
    cases: list[GoldenCase],
    pipeline: RagPipeline,
    top_k: int = 5,
) -> EvalReport:
    report = EvalReport(top_k=top_k)
    for case in cases:
        hits = pipeline.retrieve(case.query, top_k=top_k, source_tag=case.source_tag)
        retrieved = [hit.source for hit in hits]
        rank = next(
            (i + 1 for i, source in enumerate(retrieved) if source in case.expected_sources),
            None,
        )
        report.results.append(CaseResult(case=case, retrieved=retrieved, rank=rank))
    return report
