"""Unit tests for the style-mimicking experiment scaffolding."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from personal_style.llm_mimic import (
    MimicCache,
    MimicRequest,
    StubGenerator,
    make_generator,
)


def _req(samples=("Hey Alice, you're the best.",), held_out_idx: int | None = 4) -> MimicRequest:
    return MimicRequest(
        pid="abc",
        task_idx=2,
        scenario="thank",
        details="thank Alice for helping me move",
        style_samples=samples,
        held_out_task_idx=held_out_idx,
    )


def test_prompt_includes_single_style_sample_and_details():
    req = _req()
    p = req.prompt()
    assert "thank-you letter" in p
    assert "Alice" in p
    # Single-sample prompt should NOT advertise multiple samples
    assert "STYLE SAMPLE 1" not in p
    assert "STYLE SAMPLE 2" not in p
    assert "STYLE SAMPLE" in p
    assert "you're the best" in p


def test_prompt_supports_two_style_samples_for_compatibility():
    req = _req(samples=("first sample text.", "second sample text."), held_out_idx=None)
    p = req.prompt()
    assert "STYLE SAMPLE 1" in p and "STYLE SAMPLE 2" in p
    assert "first sample text" in p
    assert "second sample text" in p


def test_stub_generator_is_deterministic():
    gen = StubGenerator()
    a = gen.generate(_req())
    b = gen.generate(_req())
    assert a == b
    assert len(a) > 0


def test_make_generator_dispatch():
    assert isinstance(make_generator("stub"), StubGenerator)


def test_cache_key_is_stable_and_protocol_aware():
    req = _req()
    k1 = MimicCache.key(req, "stub")
    k2 = MimicCache.key(req, "stub")
    assert k1 == k2
    # Different generator => different key
    assert MimicCache.key(req, "gpt-5.5") != k1
    # Held-out protocol salt: same content, no held_out_task_idx => different key
    req_no_protocol = MimicRequest(
        pid=req.pid, task_idx=req.task_idx, scenario=req.scenario,
        details=req.details, style_samples=req.style_samples, held_out_task_idx=None,
    )
    assert MimicCache.key(req_no_protocol, "stub") != k1
