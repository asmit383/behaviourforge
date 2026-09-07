"""forge(seed) -> Persona. The whole library, in one function.

BrowserForge samples a coherent FINGERPRINT from a Bayesian network trained on real device
records. BehaviourForge is the same recipe one layer up: sample a coherent MOTOR IDENTITY
from real human keystroke data, so a persona's fields fit together the way a real person's
do instead of being drawn independently from guessed ranges.

    BrowserForge:   device corpus ──► per-device records ──► Bayes net ──► FingerprintGenerator
    BehaviourForge: human corpus  ──► per-person vectors  ──► bootstrap ──► forge(seed)

Why coherence is the point: a sensor scoring one signal in isolation is easy to satisfy.
A sensor FUSING signals catches the combinations that no real person produces. Draw nine
motor parameters independently and you will eventually emit a persona that types in the 95th
percentile for speed while holding keys in the 5th — each value ordinary, the bundle
impossible. Resampling a real person's vector makes that unrepresentable, because the
combination came from someone who existed.

Why a seed is the entire scaling story: a persona is a pure function of an integer. Ten
thousand agents need ten thousand integers — no server to mint identities, no shared state
to contend on, no coordination. Agent 4,712 forges its own identity in microseconds and gets
the same one back tomorrow.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

from behaviourforge import corpus as _corpus
from behaviourforge import mouse as _mouse
from behaviourforge.keystroke import RANGES, Key, Keyboard
from behaviourforge.mouse import Move, Wheel
from behaviourforge.sampler import KEY_FIELDS, get_copula

# US-QWERTY finger physics is baked into the digraph model. A French AZERTY typist has
# genuinely different same-finger pairs, so claiming to support them would be silently
# wrong rather than approximately right.
_QWERTY_LOCALES = ("en", "es", "it", "pt", "nl", "sv", "da", "no", "fi", "pl", "tr", "id")


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass(frozen=True)
class Motor:
    """One synthetic person's motor parameters. Plain data — JSON-serializable, hashable
    identity, safe to persist and replay."""
    # keystroke — bootstrapped from a real participant, then lightly jittered
    base_ms: float          # median inter-key flight for a neutral digraph
    flight_sigma: float     # log-normal spread of flight (the right skew)
    dwell_ms: float         # median key hold, press->release
    dwell_sigma: float
    error_rate: float       # per-key typo probability
    tempo_theta: float      # OU mean-reversion — how fast tempo returns to baseline
    tempo_sigma: float      # OU volatility — how far tempo wanders
    think_ms: float         # median read-the-field / new-word pause
    shift_ms: float         # extra reach time for a shifted key
    # pointer — from SapiMouse, speed-linked to typing (see sampler.CROSS_MODAL_RHO)
    mouse_speed: float          # movement-time multiplier (LOWER is faster)
    mouse_curve: float          # arc bow magnitude
    mouse_overshoot: float      # probability of a corrective overshoot
    mouse_overshoot_px: float   # how far past the target that correction goes
    mouse_tremor: float         # per-sample jitter amplitude, px
    mouse_settle_ms: float      # pause on arrival before acting
    # scroll + idle — from Balabit, which logs wheel events and captures ordinary work
    scroll_gap_ms: float        # median gap between wheel notches
    scroll_gap_sigma: float     # log-spread of that gap (the real one is bimodal)
    scroll_reversal: float      # probability of reversing direction to re-read
    idle_pause_ms: float        # median stillness while resting
    idle_pause_sigma: float     # log-spread; the real tail runs to several seconds
    idle_drift_px: float        # typical displacement of a resting micro-movement
    # measured per-bigram flight multipliers for this person's common pairs
    bigram_mult: dict[str, float] = field(default_factory=dict)
    seed: int = 0
    locale: str = "en-US"
    device: str = "desktop"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Motor":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})


class Persona:
    """A forged identity. Emits timed plans; never touches a browser.

    Stateful within a session: the keystroke tempo drifts continuously across every call, so
    a persona filling three fields carries its rhythm through all three instead of resetting
    to neutral at each one. Re-forge from the same seed to rewind to the start."""

    def __init__(self, motor: Motor):
        self.motor = motor
        # Split RNGs so the streams can't desync each other. Without this, adding one mouse
        # move between two fields would shift every subsequent keystroke — and a persona
        # that types differently depending on where you moved the pointer is not replayable.
        self._kb = Keyboard(motor, random.Random(motor.seed))
        self._mrng = random.Random(motor.seed ^ 0x9E3779B9)

    # ── keyboard ─────────────────────────────────────────────────────────────
    def keystrokes(self, text: str, *, first_field: bool = False,
                   typos: bool = True) -> list[Key]:
        """Timed key events for `text`, typos and their corrections inlined.

        `first_field=True` opens with a read-the-field pause rather than a normal flight.
        `typos=False` for passwords and any field where a stray character could submit or be
        echoed somewhere you cannot correct it."""
        return self._kb.plan(text, first_field=first_field, typos=typos)

    # ── pointer ──────────────────────────────────────────────────────────────
    def path(self, x0: float, y0: float, x1: float, y1: float) -> list[Move]:
        """Timed pointer samples from (x0,y0) to (x1,y1)."""
        return _mouse.path(self.motor, x0, y0, x1, y1, self._mrng)

    def point_in(self, box) -> tuple[float, float]:
        """A click point inside `box` — never dead centre. Takes a dict with x/y/width/height
        or an (x, y, w, h) tuple, so it fits whatever shape your driver returns."""
        return _mouse.point_in(box, self._mrng)

    def scroll(self, notches: int | None = None) -> list[Wheel]:
        """Wheel notches with measured timing and occasional direction reversal.

        Counted in notches rather than abstract "steps" because that is what the hardware and
        the source data both emit; `mouse.PX_PER_NOTCH` converts to pixels."""
        return _mouse.scroll(self.motor, self._mrng, notches=notches)

    def idle(self, ms: float, x: float, y: float, *,
             width: int = 1280, height: int = 800) -> list[Move]:
        """Fill a `ms` wait with idle pointer behaviour instead of a frozen cursor.

        Call this around an LLM round-trip. An agent's observe -> think -> act loop parks the
        cursor perfectly still for seconds on every single turn, and that pattern is regular,
        repeated, and structural to the loop rather than incidental."""
        return _mouse.idle(self.motor, self._mrng, ms, x, y, width=width, height=height)

    # ── identity ─────────────────────────────────────────────────────────────
    @property
    def seed(self) -> int:
        return self.motor.seed

    def __repr__(self) -> str:
        m = self.motor
        return (f"Persona(seed={m.seed}, base_ms={m.base_ms:.0f}, dwell_ms={m.dwell_ms:.0f}, "
                f"error_rate={m.error_rate:.3f}, mouse_speed={m.mouse_speed:.2f})")


def _bootstrap(rng: random.Random, vectors: list[dict], fields, ranges) -> dict:
    """Resample one real participant, then jitter slightly.

    Coherent by construction: the parameter combination is real because the person was.
    Jitter displaces the draw so agents are not exact copies, and is kept small on purpose —
    the corpus already carries the true correlation structure and heavy jitter would
    overwrite it with ours. Fields the chosen participant lacks are filled from another
    participant who has them, rather than defaulted to a constant."""
    v = rng.choice(vectors)
    pace = rng.gauss(1.0, 0.05)                  # shared factor keeps timings moving together
    out = {}
    for f in fields:
        x = v.get(f)
        if x is None:                            # borrow from someone who supplied it
            pool = [w[f] for w in vectors if w.get(f) is not None]
            if not pool:
                lo, hi = ranges[f]
                out[f] = (lo + hi) / 2
                continue
            x = pool[rng.randrange(len(pool))]
        scale = pace if f in ("base_ms", "dwell_ms", "think_ms") else 1.0
        out[f] = _clamp(x * scale * rng.gauss(1.0, 0.04), *ranges[f])
    return out


def forge(seed: int | None = None, *, locale: str = "en-US", device: str = "desktop",
          corpus=None, mouse_corpus=None, scroll_corpus=None,
          model: str = "copula") -> Persona:
    """Forge a coherent motor identity from a seed.

        p = forge(seed=agent_id)
        for k in p.keystrokes("hello@example.com"):
            ...

    `seed=None` draws a random one and records it on `p.motor.seed`, so an identity forged by
    accident is still reproducible.

    `model="copula"` (default) fits empirical marginals plus the corpus correlation structure
    and draws unlimited distinct vectors from it. `model="bootstrap"` resamples a real
    participant and jitters them — simpler, and every persona is provably one real human, but
    it cannot produce more distinct identities than the corpus has rows. At 10,000 agents
    against 6,000 participants that means reusing people, so copula is the default.

    Both paths are coherent. Draw nine motor parameters independently and you will eventually
    emit a persona that types in the 95th percentile for speed while holding keys in the 5th
    — each value ordinary, the bundle impossible. A sensor scoring one signal in isolation is
    easy to satisfy; a sensor fusing signals catches exactly those combinations."""
    if device != "desktop":
        raise NotImplementedError(
            f"device={device!r}: only 'desktop' is modelled. Touch input has no hover, no "
            "pointer path, and different error dynamics — a desktop persona on mobile would "
            "be less realistic than none.")
    if not locale.split("-")[0].lower().startswith(_QWERTY_LOCALES):
        raise NotImplementedError(
            f"locale={locale!r}: the digraph model uses US-QWERTY finger physics. AZERTY and "
            "QWERTZ have different same-finger pairs, so this would be wrong rather than "
            "approximate. Supply a corpus from that layout to add support.")
    if model not in ("copula", "bootstrap"):
        raise ValueError(f"model={model!r}: expected 'copula' or 'bootstrap'")

    if seed is None:
        seed = random.getrandbits(32)

    ks = corpus if isinstance(corpus, dict) else _corpus.load(corpus)
    mc = mouse_corpus if isinstance(mouse_corpus, dict) else _corpus.load_mouse(mouse_corpus)
    sc = scroll_corpus if isinstance(scroll_corpus, dict) else _corpus.load_scroll(scroll_corpus)
    rng = random.Random(seed)

    if model == "copula":
        kv = get_copula(ks, "key").draw(rng)
        anchor = kv.pop("_z0")
        if mc:
            mv = get_copula(mc, "mouse").draw(rng, anchor=anchor)
            mouse_anchor = mv.pop("_z0", 0.0)
        else:
            mv, mouse_anchor = dict(zip(_mouse.FIELDS, _mouse._FALLBACK)), 0.0
        if sc:
            # Anchored to POINTER speed rather than typing speed: scroll and pointer are the
            # same hand on the same device, so that is the more defensible of two links we
            # cannot measure directly (Balabit and SapiMouse share no participants).
            sv = get_copula(sc, "scroll").draw(rng, anchor=mouse_anchor)
            sv.pop("_z0", None)
        else:
            sv = dict(zip(_mouse.SCROLL_FIELDS, _mouse._SCROLL_FALLBACK))
        # A copula vector belongs to no participant, so it has no measured bigram table.
        # Borrow one and individualize it: the table is a ratio to the person's own base, so
        # it transfers, and the jitter stops 10,000 agents sharing 6,000 tables verbatim.
        src = ks["vectors"][rng.randrange(len(ks["vectors"]))]
        bigram = {k: round(v * rng.uniform(0.92, 1.08), 4)
                  for k, v in (src.get("bigram_mult") or {}).items()}
    else:
        kv = _bootstrap(rng, ks["vectors"], KEY_FIELDS, RANGES)
        mv = (_bootstrap(rng, mc["vectors"], _mouse.FIELDS, _mouse.MOUSE_RANGES) if mc
              else dict(zip(_mouse.FIELDS, _mouse._FALLBACK)))
        sv = (_bootstrap(rng, sc["vectors"], _mouse.SCROLL_FIELDS, _mouse.SCROLL_RANGES) if sc
              else dict(zip(_mouse.SCROLL_FIELDS, _mouse._SCROLL_FALLBACK)))
        src = rng.choice(ks["vectors"])
        bigram = dict(src.get("bigram_mult") or {})

    return Persona(Motor(bigram_mult=bigram, seed=seed, locale=locale, device=device,
                         **kv, **mv, **sv))


__all__ = ["forge", "Persona", "Motor"]
