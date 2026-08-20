"""Answer generation with Claude.

Kept behind a narrow `Generator` protocol so the retrieval half of the system
can be tested, evaluated and tuned without spending a single API call — which
matters, because retrieval is where most of the quality work happens and none
of it needs an LLM in the loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import anthropic

from .config import Settings, get_settings
from .store import SearchHit

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You answer questions about a specific document collection.

Rules:
- Use ONLY the provided context. Do not fall back on general knowledge.
- If the context does not contain enough information to answer, say so plainly \
and name what is missing. A clear "the documents don't cover this" is a correct \
answer; a plausible guess is not.
- Cite the source file for every claim, inline, as [source: filename].
- Quote exact figures, dates and clause numbers rather than paraphrasing them."""

# Beta flag for server-side refusal fallbacks. Claude API only — remove it (set
# RAG_ENABLE_REFUSAL_FALLBACK=false) when pointing at Bedrock, Vertex or Foundry.
REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class GenerationError(RuntimeError):
    """The generation backend was unreachable, unauthenticated, or errored.

    Raised instead of leaking SDK exceptions upward, so the API can turn a bad
    API key into a 502 with a readable message rather than a bare 500.
    """


@dataclass
class Answer:
    text: str
    model: str
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class Generator(Protocol):
    def generate(self, question: str, hits: list[SearchHit]) -> Answer: ...


def build_prompt(question: str, hits: list[SearchHit]) -> str:
    """Render retrieved chunks plus the question into a single user turn."""
    blocks = []
    for hit in hits:
        header = f"[source: {hit.source}]"
        if hit.section:
            header += f" [section: {hit.section}]"
        blocks.append(f"{header}\n{hit.text}")
    context = "\n\n---\n\n".join(blocks)
    return f"<context>\n{context}\n</context>\n\nQuestion: {question}"


class ClaudeGenerator:
    """Anthropic-backed generator. The client is built lazily so that importing
    this module (or running retrieval-only evals) never requires an API key."""

    def __init__(
        self,
        model: str,
        max_tokens: int,
        api_key: str | None = None,
        enable_refusal_fallback: bool = True,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._api_key = api_key
        self.enable_refusal_fallback = enable_refusal_fallback
        self._client = None

    def _get_client(self):
        if self._client is None:
            # A bare client also resolves ANTHROPIC_AUTH_TOKEN or an `ant auth
            # login` profile, so only pass a key when we actually have one.
            try:
                self._client = (
                    anthropic.Anthropic(api_key=self._api_key)
                    if self._api_key
                    else anthropic.Anthropic()
                )
            except anthropic.AnthropicError as exc:
                raise GenerationError(
                    f"Could not build an Anthropic client: {exc}. Set ANTHROPIC_API_KEY in .env."
                ) from exc
        return self._client

    def generate(self, question: str, hits: list[SearchHit]) -> Answer:
        client = self._get_client()
        prompt = build_prompt(question, hits)

        request = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            if self.enable_refusal_fallback:
                # If a safety classifier declines the request, the server routes it
                # to a fallback model by category instead of returning nothing.
                response = client.beta.messages.create(
                    **request,
                    betas=[REFUSAL_FALLBACK_BETA],
                    fallbacks="default",
                )
            else:
                response = client.messages.create(**request)
        except anthropic.APIStatusError as exc:
            hint = " Check ANTHROPIC_API_KEY." if exc.status_code == 401 else ""
            raise GenerationError(f"Claude API returned {exc.status_code}.{hint}") from exc
        except anthropic.APIConnectionError as exc:
            raise GenerationError(f"Could not reach the Claude API: {exc}") from exc

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None)
            logger.warning("Generation refused (category=%s)", category)
            return Answer(
                text=(
                    "The model declined to answer this request"
                    + (f" (category: {category})." if category else ".")
                ),
                model=response.model,
                stop_reason=response.stop_reason,
            )

        # response.content holds a list of blocks — thinking blocks come first on
        # thinking-enabled models, so never reach for content[0].text.
        text = "\n".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()

        return Answer(
            text=text,
            model=response.model,
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


def build_generator(settings: Settings | None = None) -> Generator:
    settings = settings or get_settings()
    return ClaudeGenerator(
        model=settings.claude_model,
        max_tokens=settings.max_tokens,
        api_key=settings.anthropic_api_key,
        enable_refusal_fallback=settings.enable_refusal_fallback,
    )
