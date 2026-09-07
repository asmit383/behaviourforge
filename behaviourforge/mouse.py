"""Mouse and scroll dynamics — pure. Geometry in, timed points out. No browser.

The scale problem, stated plainly: a humanizer that ships ONE path algorithm gives every
agent the same Bezier signature. At 10,000 agents that is a fleet signature — the sessions
are individually plausible and collectively impossible. So the trajectory must be generated
FROM the persona (velocity, curvature, overshoot, tremor all vary per identity), not applied
uniformly on top of it.

Emitted points are fractional on purpose. Integer-only coordinates are a tell: real pointer
hardware reports sub-pixel deltas, and a path that lands on whole numbers every sample did
not come from a hand.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass(frozen=True)
class Move:
    """One pointer sample: move to (x, y), then wait `dt_ms` before the next."""
    x: float
    y: float
    dt_ms: float


@dataclass(frozen=True)
class Wheel:
    """One wheel tick: scroll `dy` pixels, then wait `dt_ms`."""
    dy: int
    dt_ms: float


FIELDS = ("mouse_speed", "mouse_curve", "mouse_overshoot", "mouse_overshoot_px",
          "mouse_tremor", "mouse_settle_ms")

# Scroll and idle come from a different dataset (Balabit) than the pointer path (SapiMouse),
# because SapiMouse logs no wheel events and its participants were instructed to perform as
# many operations as possible — which makes its pauses useless as a model of resting.
SCROLL_FIELDS = ("scroll_gap_ms", "scroll_gap_sigma", "scroll_reversal",
                 "idle_pause_ms", "idle_pause_sigma", "idle_drift_px")

SCROLL_RANGES: dict[str, tuple[float, float]] = {
    "scroll_gap_ms": (8.0, 600.0),      # measured per-user medians span 16-249ms
    "scroll_gap_sigma": (0.3, 2.5),     # wide: real gaps are bimodal (fast spin vs notches)
    "scroll_reversal": (0.005, 0.25),   # measured 2.9-8.9%; the old hardcoded 0.2 was ~3.4x high
    "idle_pause_ms": (200.0, 6000.0),
    "idle_pause_sigma": (0.4, 2.2),
    "idle_drift_px": (3.0, 200.0),
}

# One wheel notch in pixels. This is a browser/OS setting (typically 3 lines), NOT a property
# of the human, so it is a constant here rather than a sampled persona parameter. Balabit logs
# notch events without deltas, so the human part is the TIMING and the direction changes.
PX_PER_NOTCH = 100

# Bounds measured from SapiMouse (120 users), widened past the p05-p95 range to leave the
# tails room. These replaced four hand-authored archetypes, and the data moved two of them a
# long way: `tremor` from a guessed 0.8-1.6 to a measured 1.12-3.33, and `speed` from an
# effective median of 1.05 to 2.74 — 2.6x, which made every generated pointer that much too
# fast. A 500px reach took 230ms where the data says 602ms. (The old sampler never used the
# archetype's own speed field; it derived speed from typing speed, so the 1.05 is that
# formula's median against the real base_ms distribution, not the archetype value.)
# The 2.6x is robust to segmentation: counting only reaches that end in a real click, rather
# than a pause that might be idle drift, moves the measured median by 1%.
MOUSE_RANGES: dict[str, tuple[float, float]] = {
    "mouse_speed": (0.8, 6.0),
    "mouse_curve": (0.05, 1.30),
    "mouse_overshoot": (0.0, 0.50),
    "mouse_overshoot_px": (2.0, 120.0),   # measured median 17.8px; was a guessed uniform(5,14)
    "mouse_tremor": (0.2, 5.0),
    "mouse_settle_ms": (5.0, 500.0),
}

# Fallback only, used when no mouse corpus is present. Centred on the SapiMouse medians so
# the fallback is at least in the right place; the real corpus supersedes it.
_FALLBACK = (2.74, 0.40, 0.10, 17.9, 2.03, 61.0)
_SCROLL_FALLBACK = (179.0, 1.24, 0.06, 780.0, 1.01, 55.8)

# Converts the stored (measured) tremor statistic into the per-axis sigma the path generator
# adds: sqrt(1.5) from differencing against the neighbour midpoint, times sqrt(2*ln 2) for
# the median of the resulting 2D Rayleigh magnitude.
TREMOR_TO_SIGMA = math.sqrt(1.5) * math.sqrt(2 * math.log(2))

# A movement counts as an overshoot when it travels this many pixels past its endpoint and
# comes back. Absolute, not a fraction of distance: a corrective submovement is a fixed
# 5-15px regardless of how far the reach was, so a relative threshold goes blind on long
# moves and undercounts overshoot by roughly half.
OVERSHOOT_PX = 4.0


def _ease(t: float) -> float:
    """Ease-in-out: speed rises then falls. The measured shape of human pointer velocity —
    a hand accelerates off the mark and decelerates onto the target, it does not translate
    at constant speed."""
    return 2 * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 2 / 2


def path(motor, x0: float, y0: float, x1: float, y1: float,
         rng: random.Random) -> list[Move]:
    """Humanized path (x0,y0) -> (x1,y1) as timed Move samples.

    Ease-in-out velocity, a sine-arc bow (0 at both ends, max in the middle — the shape of
    a real sweep), small per-sample tremor, and a persona-probability overshoot that aims
    slightly past the target and corrects back.

    Sampled at ~1 point per 9px, capped at 60. That cap matters for a driver: each point is
    typically one IPC round-trip, so point COUNT sets wall-clock, not pixel density. 60 keeps
    the path smooth (below frame rate) without making a long sweep feel sluggish."""
    dx, dy = x1 - x0, y1 - y0
    if math.hypot(dx, dy) < 1:
        return [Move(x1, y1, motor.mouse_settle_ms)]

    # mouse_tremor is stored in MEASURED units: the median distance of a sample from the
    # midpoint of its neighbours. That statistic is a 2D magnitude, and it inflates the
    # underlying per-axis sigma by sqrt(1.5) (the midpoint subtraction) times the Rayleigh
    # median factor sqrt(2 ln 2). Dividing it out here is what makes extraction and
    # generation agree, instead of two plausible numbers in different units.
    trem = motor.mouse_tremor / TREMOR_TO_SIGMA
    dist = math.hypot(dx, dy)
    over = rng.random() < motor.mouse_overshoot and dist > 40

    tx, ty = x1, y1
    if over:                                # aim PAST the target, then correct back
        # Magnitude is measured, not guessed: real corrective overshoots have a median of
        # ~18px with a long tail, where the old uniform(5,14) topped out below the median.
        past = _clamp(rng.lognormvariate(math.log(motor.mouse_overshoot_px), 0.7), 2, 200)
        tx = x1 + dx / dist * past
        ty = y1 + dy / dist * past

    pts: list[Move] = []

    def seg(ax: float, ay: float, bx: float, by: float) -> None:
        sdx, sdy = bx - ax, by - ay
        sd = math.hypot(sdx, sdy) or 1.0
        nx, ny = -sdy / sd, sdx / sd                     # perpendicular -> bow direction
        # Bow scales with distance and nothing else. A fixed pixel ceiling here used to clip
        # every long sweep flat: measured curve wants ~16% of the distance as perpendicular
        # deviation, which is 200px on a 1300px move, far past any constant cap worth having.
        bow = (rng.random() - 0.5) * sd * motor.mouse_curve
        steps = max(8, min(60, round(sd / 9)))
        dt = (motor.mouse_speed * (50 + sd * 0.34) * rng.uniform(0.9, 1.1)) / steps
        for i in range(1, steps + 1):
            t = i / steps
            e = _ease(t)
            arc = math.sin(t * math.pi) * bow
            pts.append(Move(ax + sdx * e + nx * arc + rng.gauss(0, trem),
                            ay + sdy * e + ny * arc + rng.gauss(0, trem), dt))

    seg(x0, y0, tx, ty)
    if over:
        seg(tx, ty, x1, y1)                              # smooth correction onto the target

    # Land exactly, and make the final wait the settle. Tremor on the last sample would leave
    # the pointer a pixel or two off, which on a small element is a missed click — and a hand
    # settles onto its target rather than jittering on it. Baking the settle into the last
    # Move also makes the plan self-contained: execute every move with its wait, then click.
    # Leaving settle as a separate field the caller must remember to honour is an easy way to
    # emit a click the instant the pointer arrives, which no human does.
    pts[-1] = Move(x1, y1, motor.mouse_settle_ms)
    return pts


def scroll(motor, rng: random.Random, *, notches: int | None = None) -> list[Wheel]:
    """Scrolling as a stream of wheel notches with measured timing.

    Modelled directly on what Balabit logs: a human emits discrete wheel events, and what
    varies between people is the RHYTHM of those events and how often they reverse direction.
    Both are strongly individual — measured per-user median gaps span 16ms to 249ms, a 15x
    spread, so a fixed scroll cadence is a fleet signature the same way a fixed tempo was.

    The gap distribution is deliberately log-normal with a wide sigma because the real one is
    bimodal: a fast spin fires events ~12ms apart, deliberate notches land nearer 90-200ms,
    and the tail runs into multi-second reading pauses. One heavy-tailed draw covers all
    three without hardcoding a burst structure that the data does not actually separate."""
    n = notches if notches is not None else rng.randint(12, 40)
    out: list[Wheel] = []
    direction = 1
    for i in range(n):
        if i and rng.random() < motor.scroll_reversal:    # re-read / correct
            direction = -direction
        gap = _clamp(rng.lognormvariate(math.log(motor.scroll_gap_ms), motor.scroll_gap_sigma),
                     4, 30_000)
        out.append(Wheel(direction * PX_PER_NOTCH, gap))
    return out


def idle(motor, rng: random.Random, ms: float, x: float, y: float, *,
         width: int = 1280, height: int = 800) -> list[Move]:
    """Fill a wait with idle pointer behaviour instead of a frozen cursor.

    This exists for one specific failure mode: an AI agent's observe -> think -> act loop
    leaves the cursor dead still for seconds on every turn, and that stillness is regular,
    repeated, and inherent to the loop. A human waiting is mostly still but never perfectly
    still. So: long pauses with occasional SMALL local drift — a hand at rest does not fling
    the pointer across the page. Always clamped on-screen."""
    out: list[Move] = []
    spent = 0.0
    cx, cy = x, y

    while spent < ms:
        # Pause lengths are log-normal with a long tail: measured across real work sessions
        # at p50 780ms, p75 1.4s, p90 4.2s. A bounded uniform draw has no tail at all, which
        # makes every rest period suspiciously similar in length.
        pause = min(_clamp(rng.lognormvariate(math.log(motor.idle_pause_ms),
                                              motor.idle_pause_sigma), 80, 60_000),
                    ms - spent)
        if pause <= 0:
            break
        out.append(Move(cx, cy, pause))                  # mostly STILL...
        spent += pause
        if spent < ms:                                   # ...broken by a small local drift
            d = _clamp(rng.lognormvariate(math.log(motor.idle_drift_px), 0.8), 2, 400)
            ang = rng.uniform(0, 2 * math.pi)
            nx = _clamp(cx + d * math.cos(ang), 2, width - 2)
            ny = _clamp(cy + d * math.sin(ang), 2, height - 2)
            # Clamp every emitted sample, not just the destination: the path bows and jitters
            # around the straight line, so a target inside the viewport can still produce
            # intermediate points outside it near a corner.
            hop = [Move(_clamp(h.x, 0, width), _clamp(h.y, 0, height), h.dt_ms)
                   for h in path(motor, cx, cy, nx, ny, rng)]
            out.extend(hop)
            spent += sum(p.dt_ms for p in hop)
            cx, cy = nx, ny

    return out


def point_in(box, rng: random.Random) -> tuple[float, float]:
    """A random interior point of a box — never dead centre, which is a tell no hand
    produces. Accepts a dict with x/y/width/height or a 4-tuple, so it works with whatever
    your driver's bounding-box shape happens to be."""
    if isinstance(box, dict):
        bx, by, bw, bh = box["x"], box["y"], box["width"], box["height"]
    else:
        bx, by, bw, bh = box
    return (bx + bw * rng.uniform(0.25, 0.75),
            by + bh * rng.uniform(0.30, 0.70))


__all__ = ["Move", "Wheel", "path", "scroll", "idle", "point_in",
           "FIELDS", "MOUSE_RANGES", "TREMOR_TO_SIGMA", "OVERSHOOT_PX"]
