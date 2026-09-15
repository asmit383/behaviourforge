"""Keystroke timing — pure. Text in, timed key events out. No browser, no I/O.

Sensors don't score the marginal histogram of your delays; they score the CORRELATION
STRUCTURE of the time series. Random delays fail because they're IID, and IID has zero
autocorrelation — which is itself the tell. So this reproduces the correlations a real
typist has:

  1. digraph latency  — flight time depends on the KEY PAIR. Measured from 3000 Aalto
                        typists: hand-alternation 0.80x, same-hand 0.92x, same-finger
                        1.06x. (The intuitive guess is ~1.6x for same-finger; the real
                        effect is far gentler. Measured, not invented.)
  2. per-bigram       — on top of the physiology bucket, THIS typist's "th" has its own
     individuality      stable speed, distinct from "he". Taken from the corpus's measured
                        per-bigram table when present, else a stable synthetic offset.
  3. autocorrelated   — tempo drifts as an Ornstein-Uhlenbeck process, so consecutive
     tempo              keys are correlated in speed rather than independently drawn.
  4. log-normal       — right-skewed draws. Uniform is as detectable as constant.
  5. boundary pauses  — longer gaps at word boundaries and before shifted keys.
  6. errors           — typo -> notice -> backspace -> retype, the strongest human tell.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

# ── QWERTY hand/finger map (US layout) — the basis of digraph latency ──────────
# finger id: 0-4 left (pinky->index), 5-9 right (index->pinky). hand = id < 5.
_ROWS = {
    "1qaz": 0, "2wsx": 1, "3edc": 2, "4rfv5tgb": 3,           # left  pinky->index
    "6yhn7ujm": 6, "8ik,": 7, "9ol.": 8, "0p;/-=[]'": 9,      # right index->pinky
}
_FINGER: dict[str, int] = {c: fid for keys, fid in _ROWS.items() for c in keys}

_SHIFT_SYMBOLS = set('~!@#$%^&*()_+{}|:"<>?')

# The corpus stores tempo_sigma as fitted by a single AR(1)-plus-noise decomposition, which
# reproduces the variance SPLIT correctly but does not reproduce the OBSERVABLE it exists to
# produce: real Aalto participants show a lag-1 autocorrelation of +0.056 on raw flight times,
# and the fitted parameters generate -0.005 on the same sentences.
#
# The cause is model misspecification rather than a bad estimate. Real typing tempo varies on
# several timescales at once — within a word, within a sentence, across a session — and
# extrapolating one exponential decay back to lag 0 underestimates the total tempo variance.
# Fitting on high lags only changes theta by 9%, so it is not fast-component contamination.
#
# A latent parameter and a measurable observable disagreed, and the observable is what a
# detector actually reads, so this is calibrated against the observable: 1.6 reproduces
# +0.061 against a real +0.056, on the SAME 400 Aalto sentences the participants typed.
# (Comparing on a different text would confound the digraph sequence with the tempo, which
# is why the benchmark replays real sentences rather than a fixed pangram.)
#
# TASK-DEPENDENT, and this is not a caveat to skim. Cross-dataset validation against KeyRecs
# (99 people, disjoint from Aalto, different protocol) measures a lag-1 autocorrelation of
# +0.001 where Aalto gives +0.056. Tempo drift is a property of CONTINUOUS typing: Aalto
# participants transcribe whole sentences, so a rhythm builds and wanders, while a
# fixed-phrase protocol restarts every trial and the drift never accumulates.
#
# 1.6 is therefore right for prose — a message, a comment, a long field — and probably too
# high for short form entries typed one at a time. Set TEMPO_SIGMA_SCALE_FORM for those.
# See examples/validate_crossdataset.py.
TEMPO_SIGMA_SCALE = 1.6
TEMPO_SIGMA_SCALE_FORM = 0.4      # short isolated fields; reproduces KeyRecs' ~0 drift

# Tempo bounds have to be wide enough for that spread. At scale 2.5 the stationary sd is
# ~0.43, so the old [0.6, 1.7] would have clamped roughly a fifth of all draws — recreating
# the clamp-saturation failure this project has already hit four times.
TEMPO_MIN, TEMPO_MAX = 0.25, 3.0

# Plausible bounds for each motor parameter — the single source of truth, used to clamp
# during extraction, to clamp jitter during sampling, and to draw a field the corpus turned
# out not to have measured. Widths matter: bounds tight enough to bite pin many personas to
# the same value, which is a fleet-diversity leak dressed up as a safety rail.
RANGES: dict[str, tuple[float, float]] = {
    "base_ms": (60.0, 320.0),
    "flight_sigma": (0.05, 0.80),     # trial noise only — tempo variance is separate now
    "dwell_ms": (40.0, 170.0),
    "dwell_sigma": (0.08, 0.60),
    "error_rate": (0.0, 0.20),
    "think_ms": (0.0, 1200.0),        # the word-boundary pause; may genuinely be ~0
    "shift_ms": (0.0, 500.0),         # measured p10 3ms, p90 311ms — the old cap was 120
    "tempo_theta": (0.02, 0.98),      # measured median 0.44 — the old cap was 0.15
    "tempo_sigma": (0.01, 0.80),      # measured median 0.17 — the old cap was 0.09
}


def _finger(ch: str) -> int | None:
    """Finger id for a character (lowercased), or None for space/unknown."""
    return _FINGER.get((ch or "").lower())


def digraph_mult(prev: str, cur: str) -> float:
    """Flight-time multiplier for a key pair, as a fraction of the typist's own base.

    MEASURED across 3000 Aalto typists (median bigram flight / that typist's base). The
    ordering that intuition predicts holds — alternation < same-hand < same-finger — but
    the magnitudes are much gentler than the intuitive guess."""
    a, b = _finger(prev), _finger(cur)
    if a is None or b is None:
        return 1.0
    if a == b:
        return 1.06                      # same finger — slowest
    if (a < 5) == (b < 5):
        return 0.92                      # same hand, different finger
    return 0.80                          # hand alternation — fastest


def needs_shift(ch: str) -> bool:
    return ch.isupper() or ch in _SHIFT_SYMBOLS


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass(frozen=True)
class Key:
    """One keyboard action: wait `flight_ms`, then hold `key` for `dwell_ms`.

    `kind` is "key" | "typo" | "backspace" — a typo is a wrong character that the plan
    then corrects, so a driver executes all three verbatim and the correction is visible
    to anyone watching the field."""
    key: str
    flight_ms: float
    dwell_ms: float
    kind: str = "key"

    @property
    def total_ms(self) -> float:
        return self.flight_ms + self.dwell_ms


class Keyboard:
    """Stateful keystroke engine for ONE identity.

    Stateful on purpose: the OU tempo and the per-bigram offsets persist across calls, so
    a persona that types three fields in a session drifts continuously through all three
    rather than resetting to a neutral tempo at each one. That continuity is a real signal.
    """

    def __init__(self, motor, rng: random.Random):
        self.motor = motor
        self._rng = rng
        self._tempo = 1.0                     # OU state — persists across plan() calls
        self._offsets: dict[str, float] = {}  # lazily-fixed synthetic per-bigram offsets

    # ── draws ────────────────────────────────────────────────────────────────
    def _lognorm(self, target_ms: float, sigma: float) -> float:
        return _clamp(self._rng.lognormvariate(math.log(max(target_ms, 1.0)), sigma), 15, 3000)

    def _dwell(self) -> float:
        # Bounds match what the corpus extractor accepts as a valid hold, so anything counted
        # as data can also be emitted. A tighter ceiling looks like a safety rail and behaves
        # like a defect: dwell_ms alone reaches 170 and the log-normal tail runs well past it,
        # so a 180ms cap pinned one keystroke in nineteen to exactly 180.000 — a hard spike in
        # the dwell histogram, which is precisely what a sensor scoring the distribution reads.
        # Real holds exceed 180ms 3.8% of the time, with a smooth tail to ~300ms at p99.9.
        m = self.motor
        return _clamp(self._rng.lognormvariate(math.log(m.dwell_ms), m.dwell_sigma), 10, 1000)

    def _tempo_step(self) -> float:
        """Ornstein-Uhlenbeck: mean-reverting drift around 1.0, so consecutive keys are
        correlated in speed. This is the property that IID random delays cannot fake."""
        m = self.motor
        self._tempo += (m.tempo_theta * (1.0 - self._tempo)
                        + m.tempo_sigma * TEMPO_SIGMA_SCALE * self._rng.gauss(0, 1))
        self._tempo = _clamp(self._tempo, TEMPO_MIN, TEMPO_MAX)
        return self._tempo

    def _bigram_mult(self, prev: str, cur: str) -> float:
        """This typist's multiplier for a specific key pair.

        Prefers the MEASURED per-bigram table carried on the motor vector (extracted from
        the corpus, normalized to that person's base). Falls back to the physiology bucket
        times a stable synthetic offset for pairs the corpus never observed — which is most
        of them, since a typist's measured table covers only their common bigrams."""
        measured = self.motor.bigram_mult.get(prev.lower() + cur.lower())
        if measured is not None:
            return measured
        off = self._offsets.get(prev + cur)
        if off is None:
            # seeded by string -> deterministic across processes (Random(str) hashes via sha512)
            off = random.Random(f"{self.motor.seed}:{prev}{cur}").uniform(-0.15, 0.15)
            self._offsets[prev + cur] = off
        return digraph_mult(prev, cur) * (1.0 + off)

    def _neighbor(self, ch: str) -> str:
        """A plausible wrong key: a different key on the SAME hand. A real slip is a
        mis-assigned finger, not a random jump across the keyboard."""
        fid = _finger(ch)
        pool = [c for c, f in _FINGER.items()
                if c.isalpha() and fid is not None and (f < 5) == (fid < 5) and c != ch.lower()]
        return self._rng.choice(pool) if pool else "e"

    # ── the plan ─────────────────────────────────────────────────────────────
    def plan(self, text: str, *, first_field: bool = False,
             typos: bool = True) -> list[Key]:
        """Plan the keystroke sequence for `text`, with typos and corrections inlined.

        `first_field` opens with a read-the-field think pause instead of a normal flight.
        `typos=False` for passwords and anything where a stray character would submit or
        be echoed somewhere you can't correct it."""
        m = self.motor
        out: list[Key] = []
        prev: str | None = None

        for ch in text:
            tempo = self._tempo_step()
            if prev is None:
                # Both branches are "first key of a field"; arriving at a form costs the
                # read-the-field pause ON TOP of switching between fields. Adding think_ms
                # rather than substituting it keeps that ordering true for every persona —
                # think_ms is a measured word-boundary increment and is legitimately smaller
                # than base_ms for plenty of people, so substituting made those personas
                # open a field faster than they moved between fields.
                base = m.base_ms * 1.5 + (m.think_ms if first_field else 0.0)
            else:
                base = m.base_ms * self._bigram_mult(prev, ch)

            flight = self._lognorm(base, m.flight_sigma) * tempo
            if prev == " ":                               # new-word think pause
                flight += self._lognorm(m.think_ms * 0.5, m.flight_sigma)
            if needs_shift(ch):                           # reach for shift
                flight += self._lognorm(m.shift_ms, m.flight_sigma)

            if (typos and prev is not None and ch.isalnum()
                    and self._rng.random() < m.error_rate):
                out.append(Key(self._neighbor(ch), flight, self._dwell(), "typo"))
                notice = self._lognorm(m.base_ms * 2.5, m.flight_sigma)   # see it, react
                out.append(Key("Backspace", notice, self._dwell(), "backspace"))
                flight = self._lognorm(m.base_ms * 0.8, m.flight_sigma)   # retype, primed

            out.append(Key(ch, flight, self._dwell(), "key"))
            prev = ch

        return out


__all__ = ["Key", "Keyboard", "digraph_mult", "needs_shift"]
