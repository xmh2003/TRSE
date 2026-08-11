from __future__ import annotations

import os
import time
from dataclasses import dataclass

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)


@dataclass(frozen=True)
class ChatResult:
    text: str
    response_id: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_ms: float
    attempts: int


class OpenAICompatibleChatClient:
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_output_tokens: int = 32,
        max_retries: int = 4,
        retry_base_seconds: float = 1.0,
    ) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.client = OpenAI()

    def complete(self, system: str, user: str) -> ChatResult:
        retryable = (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        )
        started = time.perf_counter()
        for attempt in range(1, self.max_retries + 2):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_output_tokens,
                )
                message = response.choices[0].message.content or ""
                usage = response.usage
                return ChatResult(
                    text=message,
                    response_id=getattr(response, "id", None),
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                    total_tokens=getattr(usage, "total_tokens", None),
                    latency_ms=1000.0 * (time.perf_counter() - started),
                    attempts=attempt,
                )
            except retryable:
                if attempt > self.max_retries:
                    raise
                time.sleep(self.retry_base_seconds * (2 ** (attempt - 1)))
        raise AssertionError("unreachable")
