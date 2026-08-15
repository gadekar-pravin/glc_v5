"""Offline proof that GLC v3 keeps Gemini keys as independently-metered
providers.  The live-graph scheduler must use this gateway seam, never keys."""

from __future__ import annotations

import os
from collections import defaultdict

from glc import providers
from glc.routing import Router
from glc.routing.core import RateState, rotate_pools


def _providers(monkeypatch, keys: dict[str, str]):
    for name in list(os.environ):
        if name.startswith("GEMINI_API_KEY"):
            monkeypatch.delenv(name, raising=False)
    for name, value in keys.items():
        monkeypatch.setenv(name, value)
    return providers.build_providers(cache_store=object())


def test_numbered_keys_are_real_providers_and_logical_gemini_expands(monkeypatch):
    pool = _providers(
        monkeypatch,
        {
            "GEMINI_API_KEY_1": "one",
            "GEMINI_API_KEY_2": "two",
            "GEMINI_API_KEY_3": "three",
        },
    )
    assert {name for name in pool if name.startswith("gemini_")} == {"gemini_1", "gemini_2", "gemini_3"}
    assert [pool[f"gemini_{i}"].api_key for i in (1, 2, 3)] == ["one", "two", "three"]

    router = Router(pool, ["gemini"])
    assert router.candidates() == ["gemini_1", "gemini_2", "gemini_3"]

    first, _ = router.pick(100, router.candidates())
    assert first == "gemini_1"
    router.state[first].record(0)
    second, _ = router.pick(100, router.candidates())
    assert second == "gemini_2"
    router.state[second].mark_unavailable(60, "test quota")
    third, _ = router.pick(100, router.candidates())
    assert third == "gemini_3"


def test_legacy_single_key_is_a_one_member_pool(monkeypatch):
    pool = _providers(monkeypatch, {"GEMINI_API_KEY": "legacy"})
    assert {name for name in pool if name.startswith("gemini_")} == {"gemini_1"}
    assert Router(pool, ["gemini"]).candidates() == ["gemini_1"]


def test_the_pool_rotates_without_relying_on_a_cooldown(monkeypatch):
    """Every key gets used even when none is ever unavailable.

    Rotation used to be a side effect of the per-key `cooldown` making the
    previous key briefly unusable, which meant load spreading and rate limiting
    shared one dial: raise it and concurrent callers get "all providers
    unavailable", drop it and the pool collapses onto its first member. With
    LIMITS["gemini"]["cooldown"] now 0, nothing here is ever unavailable, so this
    only passes if `pick` itself rotates.
    """
    pool = _providers(
        monkeypatch,
        {"GEMINI_API_KEY_1": "one", "GEMINI_API_KEY_2": "two", "GEMINI_API_KEY_3": "three"},
    )
    router = Router(pool, ["gemini"])
    served = []
    for _ in range(6):
        name, _ = router.pick(100, router.candidates())
        assert name is not None, "a zero cooldown must never make the pool unavailable"
        router.state[name].record(0)
        served.append(name)
    assert served == ["gemini_1", "gemini_2", "gemini_3"] * 2, served


def test_rotation_does_not_reorder_distinct_providers():
    """A pool keeps the slot its first member held.

    The rings in routing.yaml express PREFERENCE — cheapest or best first — so
    rotating within a pool must not promote a different provider up the order.
    """
    state = defaultdict(RateState)
    state["gemini_1"].last_call = 100.0
    ordered = rotate_pools(["gemini_1", "gemini_2", "groq", "ollama"], state)
    assert ordered == ["gemini_2", "gemini_1", "groq", "ollama"], ordered
    # A provider that is not part of any pool is never moved.
    assert rotate_pools(["groq", "gemini_1"], state) == ["groq", "gemini_1"]
