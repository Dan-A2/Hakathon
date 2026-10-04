"""One chat() interface over three backends.

    vllm   - Gemma 4 26B-A4B-it served by vLLM on Modal (OpenAI-compatible; exposes log-probs)
    claude - Claude Haiku 4.5 through the Anthropic SDK (fallback; no log-probs)
    mock   - deterministic offline stand-in used for tests and for building the trainer
             against a cache before any real LLM calls exist

Backend selection: KWTC_LLM_BACKEND, else vllm if KWTC_VLLM_URL is set, else claude if an
Anthropic credential is set, else mock (with a warning).
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from common import config as C


class LLMError(RuntimeError):
    pass


class LLMRefusal(LLMError):
    pass


@dataclass
class ChatResult:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    logprobs: list[tuple[str, float]] | None = None   # output tokens with log-probs (vLLM only)
    model: str = ""
    meta: dict = field(default_factory=dict)


class BaseLLM:
    backend = "base"
    model_id = ""
    revision = ""
    supports_logprobs = False

    def identity(self) -> dict:
        return {"backend": self.backend, "model": self.model_id, "revision": self.revision}

    def chat(self, messages: list[dict], *, temperature: float = 0.7, seed: int | None = None,
             max_tokens: int = 700, json_schema: dict | None = None, want_logprobs: bool = False,
             tag: str | None = None) -> ChatResult:
        raise NotImplementedError


def _split_system(messages: list[dict]) -> tuple[str | None, list[dict]]:
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or None
    rest = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
    return system, rest


class VLLMClient(BaseLLM):
    backend = "vllm"
    supports_logprobs = True

    def __init__(self, url: str | None = None, served_name: str | None = None,
                 model_id: str | None = None, revision: str | None = None):
        from openai import OpenAI

        self.url = (url or C.VLLM_URL).rstrip("/")
        if not self.url:
            raise LLMError("KWTC_VLLM_URL is not set")
        base = self.url if self.url.endswith("/v1") else self.url + "/v1"
        self.served_name = served_name or C.VLLM_MODEL
        self.model_id = model_id or C.VLLM_MODEL_ID
        self.revision = revision or C.VLLM_REVISION
        self.client = OpenAI(base_url=base, api_key=os.environ.get("KWTC_VLLM_API_KEY", "EMPTY"),
                             max_retries=3, timeout=180.0)

    def chat(self, messages, *, temperature=0.7, seed=None, max_tokens=700, json_schema=None,
             want_logprobs=False, tag=None) -> ChatResult:
        kwargs: dict[str, Any] = dict(
            model=self.served_name, messages=messages, temperature=temperature, max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},   # thinking off
        )
        if seed is not None:
            kwargs["seed"] = int(seed)
        if json_schema is not None:   # vLLM guided decoding
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "kwtc_output", "schema": json_schema}}
        if want_logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = 1
        t0 = time.time()
        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        text = choice.message.content or ""
        lps = None
        if want_logprobs and choice.logprobs and choice.logprobs.content:
            lps = [(t.token, float(t.logprob)) for t in choice.logprobs.content]
        usage = resp.usage
        return ChatResult(text=text, tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
                          tokens_out=getattr(usage, "completion_tokens", 0) or 0,
                          latency_s=time.time() - t0, logprobs=lps, model=resp.model or self.served_name)


class ClaudeClient(BaseLLM):
    backend = "claude"
    supports_logprobs = False

    def __init__(self, model: str | None = None):
        import anthropic

        self._anthropic = anthropic
        self.model_id = model or C.CLAUDE_MODEL
        self.revision = "api"
        self.client = anthropic.Anthropic(max_retries=3, timeout=180.0)

    def chat(self, messages, *, temperature=0.7, seed=None, max_tokens=700, json_schema=None,
             want_logprobs=False, tag=None) -> ChatResult:
        system, msgs = _split_system(messages)
        kwargs: dict[str, Any] = dict(model=self.model_id, max_tokens=max_tokens, messages=msgs,
                                      temperature=temperature)
        if system:
            kwargs["system"] = system
        if json_schema is not None:   # structured outputs: the first text block is schema-valid JSON
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": json_schema}}
        t0 = time.time()
        try:
            resp = self.client.messages.create(**kwargs)
        except self._anthropic.BadRequestError as e:
            if json_schema is not None:          # model/platform without structured outputs: fall back to prompt JSON
                kwargs.pop("output_config")
                resp = self.client.messages.create(**kwargs)
            else:
                raise LLMError(str(e)) from e
        if resp.stop_reason == "refusal":
            raise LLMRefusal("model refused the request")
        text = "".join(b.text for b in resp.content if b.type == "text")
        return ChatResult(text=text, tokens_in=resp.usage.input_tokens, tokens_out=resp.usage.output_tokens,
                          latency_s=time.time() - t0, model=resp.model)


def get_llm(backend: str | None = None) -> BaseLLM:
    backend = (backend or C.llm_backend()).lower()
    if backend == "vllm":
        return VLLMClient()
    if backend == "claude":
        return ClaudeClient()
    if backend == "mock":
        from agent.mock_llm import MockLLM

        if not os.environ.get("KWTC_LLM_BACKEND"):
            print("[llm] no LLM credentials found; using the deterministic MOCK backend "
                  "(set KWTC_LLM_BACKEND=claude|vllm for real runs)", file=sys.stderr)
        return MockLLM()
    raise LLMError(f"unknown backend {backend!r}")
