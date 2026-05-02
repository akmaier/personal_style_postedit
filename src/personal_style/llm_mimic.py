"""Style-mimicking LLM generators for the post-editing comparison experiment.

For each treatment task we have:
  - participant's planning details (the same the original GPT-o4-mini draft saw)
  - participant's 2 control texts (unassisted writing on other tasks)
  - the original LLM draft
  - the human-edited final text

This module asks GPT 5.5 / Claude Opus 4.7 to produce a "style-mimicking" draft
that is conditioned on the planning details *and* the control texts as style
examples. The result is then comparable -- via LUAR self-similarity -- to the
human-edited text in the original study.

The module abstracts over three back-ends:
  - openai (`gpt-5.5` via Responses API)
  - anthropic (`claude-opus-4-7` via Messages API)
  - stub (deterministic, no API calls -- used for smoke tests and offline runs)
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Task -> instruction template (taken verbatim from the paper §4.1, "wedding" prompt
# generalised across the eight scenarios actually used in the study).
TASK_INSTRUCTIONS = {
    "wedding": "Please help me write a draft of wedding vows.",
    "thank": "Please help me write a draft of a thank-you letter.",
    "apology": "Please help me write a draft of an apology letter.",
    "letter": "Please help me write a draft of a personal letter.",
    "condolence": "Please help me write a draft of a condolence letter.",
    "reassurance": "Please help me write a draft of a reassurance letter to a loved one.",
    "speech": "Please help me write a draft of a personal speech.",
    "eulogy": "Please help me write a draft of a eulogy.",
}

PROMPT_TEMPLATE = """{instruction} Only respond with the draft with no additional content or explanation. Please do not include any placeholder text such as "[name]". The final draft should be about 150 words.

Here are some details to help you write:
{details}

Below are two short writing samples written by the same author, on different topics, in their natural unassisted style. Mimic this author's personal writing style as closely as possible -- their typical sentence length, punctuation habits, vocabulary register, and tone:

--- STYLE SAMPLE 1 ---
{sample_1}

--- STYLE SAMPLE 2 ---
{sample_2}
"""


@dataclass(frozen=True)
class MimicRequest:
    pid: str
    task_idx: int
    scenario: str
    details: str
    style_samples: tuple[str, ...]

    def prompt(self) -> str:
        instruction = TASK_INSTRUCTIONS.get(
            self.scenario,
            f"Please help me write a draft of a personal {self.scenario}.",
        )
        s1 = self.style_samples[0] if self.style_samples else ""
        s2 = self.style_samples[1] if len(self.style_samples) > 1 else s1
        return PROMPT_TEMPLATE.format(
            instruction=instruction,
            details=self.details.strip() or "(no details provided)",
            sample_1=s1.strip() or "(no sample available)",
            sample_2=s2.strip() or "(no sample available)",
        )


class Generator(ABC):
    name: str

    @abstractmethod
    def generate(self, req: MimicRequest) -> str: ...


class StubGenerator(Generator):
    """Deterministic offline generator used for smoke tests.

    Returns a "draft" that is the participant's first style sample lightly
    interleaved with the planning details. Not meant to be a meaningful baseline,
    only to exercise the rest of the pipeline without API calls.
    """

    name = "stub"

    def generate(self, req: MimicRequest) -> str:
        rng = random.Random(int(hashlib.sha256(req.pid.encode()).hexdigest(), 16) % (2**32))
        bits = []
        if req.style_samples:
            sample = req.style_samples[0].strip().split()
            chunk = " ".join(sample[: rng.randint(20, 40)])
            bits.append(chunk)
        if req.details:
            bits.append("Some thoughts:")
            for line in req.details.split("\n"):
                line = line.strip()
                if line:
                    bits.append(f"- {line}")
        return "\n".join(bits) or "Stub draft."


class OpenAIMimicGenerator(Generator):
    name = "gpt-5.5"
    model_id = "gpt-5.5"

    def __init__(self, api_key: str | None = None, model: str | None = None, max_retries: int = 4):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        if model:
            self.model_id = model
        self.max_retries = max_retries

    def generate(self, req: MimicRequest) -> str:
        prompt = req.prompt()
        for attempt in range(self.max_retries):
            try:
                resp = self.client.responses.create(
                    model=self.model_id,
                    input=prompt,
                    reasoning={"effort": "medium"},
                )
                # Responses API returns output_text helper.
                text = getattr(resp, "output_text", None)
                if not text:
                    # Fallback: walk the structured output.
                    chunks = []
                    for item in getattr(resp, "output", []) or []:
                        for content in getattr(item, "content", []) or []:
                            t = getattr(content, "text", None)
                            if t:
                                chunks.append(t)
                    text = "\n".join(chunks)
                return (text or "").strip()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2**attempt)
        return ""


class AnthropicMimicGenerator(Generator):
    name = "claude-opus-4-7"
    model_id = "claude-opus-4-7"

    def __init__(self, api_key: str | None = None, model: str | None = None, max_retries: int = 4):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        if model:
            self.model_id = model
        self.max_retries = max_retries

    def generate(self, req: MimicRequest) -> str:
        prompt = req.prompt()
        for attempt in range(self.max_retries):
            try:
                resp = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                chunks = []
                for block in resp.content:
                    if getattr(block, "type", None) == "text":
                        chunks.append(block.text)
                return ("\n".join(chunks)).strip()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2**attempt)
        return ""


def make_generator(name: str) -> Generator:
    name = name.lower()
    if name in {"stub", "offline", "test"}:
        return StubGenerator()
    if name in {"gpt-5.5", "gpt5.5", "openai"}:
        return OpenAIMimicGenerator()
    if name in {"claude-opus-4-7", "claude", "anthropic"}:
        return AnthropicMimicGenerator()
    raise ValueError(f"unknown generator name: {name!r}")


@dataclass
class MimicCache:
    path: Path

    def load(self) -> dict[str, dict]:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {}

    def save(self, data: dict[str, dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    @staticmethod
    def key(req: MimicRequest, generator_name: str) -> str:
        h = hashlib.sha256()
        h.update(generator_name.encode())
        h.update(req.pid.encode())
        h.update(str(req.task_idx).encode())
        h.update(req.scenario.encode())
        h.update(req.details.encode())
        for s in req.style_samples:
            h.update(s.encode())
        return h.hexdigest()[:16]
