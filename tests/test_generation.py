"""Generation tests run against a stubbed Anthropic client — no API key, no calls."""

from __future__ import annotations

from types import SimpleNamespace

import anthropic
import pytest

from homebrew_rag.generation import (
    REFUSAL_FALLBACK_BETA,
    ClaudeGenerator,
    GenerationError,
)
from homebrew_rag.store import SearchHit


class StubMessages:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class StubClient:
    def __init__(self, response):
        self.messages = StubMessages(response)
        self.beta = SimpleNamespace(messages=StubMessages(response))


def make_response(blocks, stop_reason="end_turn", stop_details=None):
    return SimpleNamespace(
        content=blocks,
        model="claude-opus-5",
        stop_reason=stop_reason,
        stop_details=stop_details,
        usage=SimpleNamespace(input_tokens=42, output_tokens=7),
    )


def text_block(text):
    return SimpleNamespace(type="text", text=text)


@pytest.fixture
def hits():
    return [SearchHit(text="The audit is fixed-fee.", source="pricing.md", section="", score=0.9)]


def _generator(response, **kwargs):
    generator = ClaudeGenerator(model="claude-opus-5", max_tokens=1024, **kwargs)
    generator._client = StubClient(response)
    return generator


def test_text_blocks_are_joined_and_thinking_blocks_ignored(hits):
    response = make_response(
        [
            SimpleNamespace(type="thinking", thinking=""),
            text_block("Fixed fee."),
            text_block("[source: pricing.md]"),
        ]
    )
    generator = _generator(response)
    answer = generator.generate("cost?", hits)

    assert answer.text == "Fixed fee.\n[source: pricing.md]"
    assert answer.input_tokens == 42 and answer.output_tokens == 7


def test_refusal_fallback_uses_the_beta_endpoint(hits):
    response = make_response([text_block("ok")])
    generator = _generator(response, enable_refusal_fallback=True)
    generator.generate("q", hits)

    call = generator._client.beta.messages.calls[0]
    assert call["betas"] == [REFUSAL_FALLBACK_BETA]
    assert call["fallbacks"] == "default"
    assert generator._client.messages.calls == []


def test_fallback_can_be_disabled_for_non_first_party_endpoints(hits):
    response = make_response([text_block("ok")])
    generator = _generator(response, enable_refusal_fallback=False)
    generator.generate("q", hits)

    assert generator._client.messages.calls[0]["model"] == "claude-opus-5"
    assert "betas" not in generator._client.messages.calls[0]
    assert generator._client.beta.messages.calls == []


def test_refusal_is_surfaced_not_crashed_on(hits):
    response = make_response(
        [],
        stop_reason="refusal",
        stop_details=SimpleNamespace(type="refusal", category="cyber"),
    )
    answer = _generator(response).generate("q", hits)

    assert answer.stop_reason == "refusal"
    assert "declined" in answer.text and "cyber" in answer.text


def test_prompt_and_system_prompt_are_sent(hits):
    generator = _generator(make_response([text_block("ok")]))
    generator.generate("What does it cost?", hits)

    call = generator._client.beta.messages.calls[0]
    assert "ONLY the provided context" in call["system"]
    assert "pricing.md" in call["messages"][0]["content"]
    assert call["max_tokens"] == 1024


def test_api_status_errors_become_generation_errors(hits):
    import httpx

    class Failing:
        def create(self, **kwargs):
            raise anthropic.AuthenticationError(
                "bad key",
                response=httpx.Response(401, request=httpx.Request("POST", "https://x")),
                body=None,
            )

    generator = ClaudeGenerator(model="claude-opus-5", max_tokens=100)
    generator._client = SimpleNamespace(beta=SimpleNamespace(messages=Failing()))

    with pytest.raises(GenerationError, match="401.*ANTHROPIC_API_KEY"):
        generator.generate("q", hits)
