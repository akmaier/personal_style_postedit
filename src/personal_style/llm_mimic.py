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

_STYLE_BLOCK_SINGULAR = (
    "Below is a short writing sample written by the author, on a different topic, "
    "in their natural unassisted style. Mimic this author's personal writing style "
    "as closely as possible -- their typical sentence length, punctuation habits, "
    "vocabulary register, capitalization, and tone. If the author writes in lowercase "
    "or with idiosyncratic spelling, preserve those tendencies."
)
_STYLE_BLOCK_PLURAL = (
    "Below are short writing samples written by the same author, on different topics, "
    "in their natural unassisted style. Mimic this author's personal writing style "
    "as closely as possible -- their typical sentence length, punctuation habits, "
    "vocabulary register, capitalization, and tone. If the author writes in lowercase "
    "or with idiosyncratic spelling, preserve those tendencies."
)


def _build_prompt(instruction: str, details: str, style_samples: tuple[str, ...]) -> str:
    """Build a prompt that adapts to 1 or 2 style samples."""
    head = (
        f"{instruction} Only respond with the draft with no additional content or "
        f'explanation. Please do not include any placeholder text such as "[name]". '
        f"The final draft should be about 150 words.\n\n"
        f"Here are some details to help you write:\n{details.strip() or '(no details provided)'}\n\n"
    )
    samples = [s.strip() for s in style_samples if s and s.strip()]
    if not samples:
        return head
    if len(samples) == 1:
        return (
            head
            + _STYLE_BLOCK_SINGULAR
            + "\n\n--- STYLE SAMPLE ---\n"
            + samples[0]
            + "\n"
        )
    body = _STYLE_BLOCK_PLURAL + "\n"
    for i, s in enumerate(samples, start=1):
        body += f"\n--- STYLE SAMPLE {i} ---\n{s}\n"
    return head + body


@dataclass(frozen=True)
class MimicRequest:
    """A request to a style-mimicking generator.

    `style_samples` are the participant's unassisted writing samples shown to
    the model as style demonstrations. Under the leakage-free held-out protocol
    we pass exactly one sample (the "demo control"); the held-out control is
    not shown to the model and is used downstream as the evaluation target.

    `held_out_task_idx` is the task_idx of the participant's other control text
    (the one *not* shown to the model). It is recorded only as metadata so the
    evaluation script can look up the correct held-out vector; it is **not**
    embedded in the prompt and does not affect the model's input.
    """

    pid: str
    task_idx: int
    scenario: str
    details: str
    style_samples: tuple[str, ...]
    held_out_task_idx: int | None = None

    def prompt(self) -> str:
        instruction = TASK_INSTRUCTIONS.get(
            self.scenario,
            f"Please help me write a draft of a personal {self.scenario}.",
        )
        return _build_prompt(instruction, self.details, self.style_samples)


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
        # Salt the cache key with a leakage-protocol tag so cached entries from
        # the original 2-sample protocol cannot collide with the new 1-sample
        # held-out protocol. This is set whenever held_out_task_idx is provided.
        if req.held_out_task_idx is not None:
            h.update(b"|held_out_protocol_v1|")
            h.update(str(req.held_out_task_idx).encode())
        for s in req.style_samples:
            h.update(s.encode())
        return h.hexdigest()[:16]
