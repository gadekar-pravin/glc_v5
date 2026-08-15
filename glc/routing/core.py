"""Capability-aware router. Same RPM/RPD bookkeeping as V1, but now it can
skip providers that lack a requested capability (tools/reasoning/structured/caching)."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque

LIMITS = {
    "ollama": {"rpm": 9999, "rpd": 9999999, "tpm": 99999999, "cooldown": 0, "max_ctx": 32000},
    "cerebras": {
        "rpm": 30,
        "rpd": 9999,
        "tpm": 60000,
        "cooldown": 2,
        "max_ctx": 8000,
        "tokens_per_day": 1_000_000,
    },
    "groq": {"rpm": 30, "rpd": 1000, "tpm": 6000, "cooldown": 2, "max_ctx": 100000},
    "nvidia": {"rpm": 40, "rpd": 9999, "tpm": 100000, "cooldown": 2, "max_ctx": 100000},
    # PAID key. {rpm 15, rpd 1000, tpm 250000} are the AI Studio FREE-tier numbers
    # for a flash model; a billed project is entitled to far more. Same reasoning
    # as openrouter below — these are loose local bounds, not an entitlement model.
    #
    # `cooldown` was the expensive one here, and it is per KEY: at 4 s, with a
    # two-key pool, five calls in as many seconds exhausted both and the gateway
    # answered "all providers unavailable" while neither key was near a real quota
    # (rpd_used was 6 of 1000 at the time). It reads as an outage.
    #
    # Setting it to 0 was only safe once `rotate_pools` existed. The cooldown was
    # doing a second, undocumented job: `pick()` takes the first AVAILABLE
    # candidate, so the only thing spreading calls across GEMINI_API_KEY_1..N was
    # the previous key being briefly unavailable. That left no good value — high
    # enough to rotate and concurrent callers get "all providers unavailable"
    # (measured: 4 of 6 parallel requests failed at 0.25 s), low enough not to
    # throttle and gemini_2 never serves anything. Rotation now lives in
    # rotate_pools() where it costs nothing, so this can be 0.
    #
    # NB this does NOT address the other Gemini failure seen on 2026-08-15: HTTP
    # 503 "this model is currently experiencing high demand" on gemini-3.7-flash is
    # an upstream capacity signal, not a local gate, and no value here can fix it.
    # That is why S17's frontier rung reaches that model through OpenRouter.
    # Related, and deliberately left alone: routing.yaml's `backoff.timeout: 600`
    # benches a key for ten minutes after one slow call, which is what took the
    # whole pool down when 3.7-flash answered in 34 s and 43 s.
    "gemini": {"rpm": 2000, "rpd": 200000, "tpm": 4000000, "cooldown": 0, "max_ctx": 1000000},
    # PAID key. The old {rpm 20, rpd 50, cooldown 3} described a free OpenRouter
    # account, and rpd 50 is a hard local gate: `can_use` refuses the provider
    # outright at 50 calls in a day, whatever the account is actually entitled to.
    # Fifty calls is one afternoon of testing, and because the counter is per
    # PROVIDER rather than per model it took every openrouter-routed model down
    # with it, paid ones included.
    #
    # A gateway cannot know an account's real entitlement — on OpenRouter it moves
    # with the credit balance — so guessing low here fails closed on spend the user
    # has already paid for. These bounds are deliberately loose: upstream is the
    # authority, a real 429 is caught by `_backoff_seconds("rate_limited")` (30 s),
    # and the caller's own ceilings (S17's max_calls_per_run) bound a runaway loop.
    #
    # NB `:free` model variants stay rate-limited by OpenRouter no matter how the
    # account is funded, so a rung pinned to a `...:free` model can still 429.
    # max_ctx is left at 100000 on purpose: it is a routing-correctness knob, not a
    # rate limit, and OpenRouter serves models from 8k to 2M behind one name.
    "openrouter": {"rpm": 500, "rpd": 200000, "tpm": 99999999, "cooldown": 0, "max_ctx": 100000},
    "github": {"rpm": 10, "rpd": 50, "tpm": 99999999, "cooldown": 6, "max_ctx": 8000},
}

# One Google AI Studio key is one independently-metered provider.  The graph
# scheduler requests the logical name ``gemini``; this router owns expansion,
# cooldown and failover so no caller can accidentally build a second key pool.
MAX_GEMINI_KEYS = 16

SHORTCUTS = {
    "g": "gemini",
    "gem": "gemini",
    "gemini": "gemini",
    "n": "nvidia",
    "nv": "nvidia",
    "nvidia": "nvidia",
    "o": "ollama",
    "oll": "ollama",
    "ollama": "ollama",
    "gr": "groq",
    "groq": "groq",
    "c": "cerebras",
    "cer": "cerebras",
    "cerebras": "cerebras",
    "or": "openrouter",
    "opr": "openrouter",
    "openrouter": "openrouter",
    "gh": "github",
    "ghb": "github",
    "github": "github",
}


def resolve(name):
    if not name:
        return None
    return SHORTCUTS.get(name.lower())


class RateState:
    def __init__(self):
        self.calls_minute = deque()
        self.tokens_minute = deque()
        self.calls_today = 0
        self.tokens_today = 0
        self.day_start = self._day_start()
        self.last_call = 0.0
        self.unavailable_until = 0.0
        self.unavailable_reason = ""

    @staticmethod
    def _day_start():
        now = time.time()
        return now - (now % 86400)

    def gc(self):
        now = time.time()
        if now - self.day_start >= 86400:
            self.calls_today = 0
            self.tokens_today = 0
            self.day_start = self._day_start()
        cutoff = now - 60
        while self.calls_minute and self.calls_minute[0] < cutoff:
            self.calls_minute.popleft()
        while self.tokens_minute and self.tokens_minute[0][0] < cutoff:
            self.tokens_minute.popleft()

    def can_use(self, limits, est_tokens=0):
        self.gc()
        now = time.time()
        if now < self.unavailable_until:
            return False, f"backoff: {self.unavailable_reason} ({self.unavailable_until - now:.0f}s left)"
        wait = limits["cooldown"] - (now - self.last_call)
        if wait > 0:
            return False, f"cooldown ({wait:.1f}s)"
        if len(self.calls_minute) >= limits["rpm"]:
            return False, "RPM limit"
        if self.calls_today >= limits["rpd"]:
            return False, "RPD limit"
        tpm = sum(t for _, t in self.tokens_minute)
        if tpm + est_tokens > limits["tpm"]:
            return False, "TPM limit"
        if "tokens_per_day" in limits and self.tokens_today + est_tokens > limits["tokens_per_day"]:
            return False, "daily token cap"
        return True, None

    def record(self, tokens):
        now = time.time()
        self.calls_minute.append(now)
        self.tokens_minute.append((now, tokens))
        self.calls_today += 1
        self.tokens_today += tokens
        self.last_call = now

    def mark_unavailable(self, seconds: float, reason: str):
        self.unavailable_until = time.time() + seconds
        self.unavailable_reason = reason

    def snapshot(self, limits):
        self.gc()
        now = time.time()
        tpm = sum(t for _, t in self.tokens_minute)
        return {
            "rpm_used": len(self.calls_minute),
            "rpm_limit": limits["rpm"],
            "rpd_used": self.calls_today,
            "rpd_limit": limits["rpd"],
            "tpm_used": tpm,
            "tpm_limit": limits["tpm"],
            "tokens_today": self.tokens_today,
            "tokens_per_day": limits.get("tokens_per_day"),
            "cooldown_remaining": max(0, limits["cooldown"] - (now - self.last_call))
            if self.last_call
            else 0,
            "last_call": self.last_call,
            "backoff_remaining": max(0, self.unavailable_until - now),
            "backoff_reason": self.unavailable_reason if now < self.unavailable_until else "",
        }


def _pool_base(name: str) -> str:
    """``gemini_2`` -> ``gemini``; anything else is its own pool of one."""
    base, sep, tail = name.rpartition("_")
    return base if sep and tail.isdigit() else name


def rotate_pools(candidates: list[str], state) -> list[str]:
    """Order the members of one key pool least-recently-used first.

    ``pick`` takes the first AVAILABLE candidate, so a pool like
    ``gemini_1, gemini_2`` serves everything from the first member unless
    something makes that member unavailable. That something used to be the
    per-key ``cooldown``, which left load spreading riding on a rate-limit knob
    and no good value to set: high enough to rotate and concurrent callers get
    "all providers unavailable"; low enough not to throttle and the rest of the
    pool is never touched at all.

    Rotation belongs here instead, where it costs nothing. Only members of the
    SAME pool are reordered, and a pool keeps the slot its first member held, so
    the preference order BETWEEN providers — which is the whole meaning of the
    rings in routing.yaml — is left exactly as the caller supplied it.
    """
    grouped: dict[str, list[str]] = {}
    for name in candidates:
        grouped.setdefault(_pool_base(name), []).append(name)
    out: list[str] = []
    for base in dict.fromkeys(_pool_base(name) for name in candidates):
        members = grouped[base]
        if len(members) > 1:
            members = sorted(members, key=lambda name: state[name].last_call)
        out.extend(members)
    return out


class Router:
    def __init__(self, providers: dict, order: list[str]):
        self.providers = providers
        self.order = [p for p in order if p in providers or self._pool_of(p)]
        self.state = defaultdict(RateState)
        self.lock = asyncio.Lock()

    def _pool_of(self, base: str) -> list[str]:
        return [p for p in self.providers if p.startswith(base + "_")]

    def expand(self, names: list[str]) -> list[str]:
        """Expand a logical provider (for example ``gemini``) to its live,
        individually-metered instances while preserving order and removing
        duplicates."""
        out: list[str] = []
        seen: set[str] = set()
        for name in names:
            base = resolve(name) or name
            instances = [base] if base in self.providers else self._pool_of(base)
            for instance in instances:
                if instance not in seen:
                    seen.add(instance)
                    out.append(instance)
        return out

    def candidates(self, override=None):
        if override:
            return self.expand([override])
        return self.expand(self.order)

    def pick(self, est_tokens, candidates, required_caps: list[str] | None = None):
        attempts = []
        for name in rotate_pools(candidates, self.state):
            limits = LIMITS[name]
            prov = self.providers[name]
            caps = getattr(prov, "capabilities", {})
            if required_caps:
                missing = [c for c in required_caps if not caps.get(c)]
                if missing:
                    attempts.append({"provider": name, "reason": f"skipped:no_{missing[0]}"})
                    continue
            if est_tokens > limits["max_ctx"]:
                attempts.append(
                    {"provider": name, "reason": f"prompt {est_tokens} > max_ctx {limits['max_ctx']}"}
                )
                continue
            ok, why = self.state[name].can_use(limits, est_tokens)
            if ok:
                return name, attempts
            attempts.append({"provider": name, "reason": why})
        return None, attempts

    def all_status(self):
        out = {}
        for name in self.providers:
            out[name] = self.state[name].snapshot(LIMITS[name])
            out[name]["model"] = self.providers[name].model
            out[name]["capabilities"] = getattr(self.providers[name], "capabilities", {})
        return out


# -----------------------------------------------------------------------------
# V3 Router pool — separate failover ring for routing-decision LLM calls.
# Same rate-state machinery, separate state dict so router quotas never compete
# with worker quotas (provider keys are shared but providers meter per-model).
# -----------------------------------------------------------------------------

DEFAULT_ROUTER_ORDER = ["cerebras", "groq", "nvidia", "github"]


class RouterPool:
    """Failover ring for router-LLM calls. Mirrors `Router` but for the
    Perception/Memory/Decision routing classifiers. Each call is logged with
    a call_role marker (router_perception | router_memory | router_decision)
    so the dashboard can show router activity separately from worker activity.
    """

    def __init__(self, providers: dict, order: list[str]):
        self.providers = providers
        self.order = [p for p in order if p in providers]
        self.state = defaultdict(RateState)
        self.lock = asyncio.Lock()

    def candidates(self):
        return list(self.order)

    def pick(self, est_tokens=400):
        """Pick first available router provider. Caps require nothing — router
        LLMs only need to emit one word, no tools/reasoning/structured needed."""
        attempts = []
        for name in rotate_pools(self.candidates(), self.state):
            limits = LIMITS[name]
            ok, why = self.state[name].can_use(limits, est_tokens)
            if ok:
                return name, attempts
            attempts.append({"provider": name, "reason": why})
        return None, attempts

    def all_status(self):
        out = {}
        for name in self.providers:
            out[name] = self.state[name].snapshot(LIMITS[name])
            out[name]["model"] = self.providers[name].model
        return out
