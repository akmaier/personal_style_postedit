"""Unit tests for the style-mimicking experiment scaffolding."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from personal_style.llm_mimic import (
    MimicCache,
    MimicRequest,
    PROMPT_TEMPLATE,
    StubGenerator,
    make_generator,
)


def _req() -> MimicRequest:
    return MimicRequest(
        pid="abc",
        task_idx=2,
        scenario="thank",
        details="thank Alice for helping me move",
        style_samples=("Hey Alice, you're the best.", "Honestly, I owe you one."),
    )


def test_prompt_includes_style_samples_and_details():
    req = _req()
    p = req.prompt()
    assert "thank-you letter" in p
    assert "Alice" in p
    assert "STYLE SAMPLE 1" in p and "STYLE SAMPLE 2" in p
    assert "you're the best" in p


def test_stub_generator_is_deterministic():
    gen = StubGenerator()
    a = gen.generate(_req())
    b = gen.generate(_req())
    assert a == b
    assert len(a) > 0


def test_make_generator_dispatch():
    assert isinstance(make_generator("stub"), StubGenerator)


def test_cache_key_is_stable():
    req = _req()
    k1 = MimicCache.key(req, "stub")
    k2 = MimicCache.key(req, "stub")
    assert k1 == k2
    # Different generator => different key
    assert MimicCache.key(req, "gpt-5.5") != k1
