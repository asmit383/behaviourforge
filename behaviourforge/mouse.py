"""Mouse and scroll dynamics — pure. Geometry in, timed points out. No browser.

The scale problem, stated plainly: a humanizer that ships ONE path algorithm gives every
agent the same Bezier signature. At 10,000 agents that is a fleet signature — the sessions
are individually plausible and collectively impossible. So the trajectory must be generated
FROM the persona (velocity, curvature, overshoot, tremor all vary per identity), not applied
uniformly on top of it.

Emitted points are fractional on purpose, but be precise about why. `clientX`/`clientY` are
integers, so a page cannot observe the fraction directly — verified by driving Camoufox and
reading the events back. What fractional arithmetic buys is the SEQUENCE of integers the
rounding produces: a path computed in floats and rounded lands on a different, less regular
set of pixels than one computed in integer steps.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass(frozen=True)
class Move:
    """One pointer sample: move to (x, y), then wait `dt_ms` before the next.

    `coalesce` marks a sample that should be DISPATCHED without waiting for a paint, so the
    browser merges it with its neighbours into one `pointermove` carrying the rest in
    `getCoalescedEvents()`. `dt_ms` still holds the sample's true time — a driver pipelines the
    flagged run and then pays back the accumulated wait, so no timing statistic changes."""
    x: float
    y: float
    dt_ms: float
    coalesce: bool = False


@dataclass(frozen=True)
class Wheel:
    """One wheel tick: scroll `dy` pixels, then wait `dt_ms`."""
    dy: int
    dt_ms: float


FIELDS = ("mouse_speed", "mouse_curve", "mouse_overshoot", "mouse_overshoot_px",
          "mouse_tremor", "mouse_settle_ms", "mouse_click_ms")

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

# One wheel tick in pixels — a platform constant, not a human parameter (40 on macOS).
#
# The value matters more than it looks. `wheelDeltaY` is derived from the TICK COUNT, not from
# `deltaY`, so a driver can only ever express whole ticks: emitting 100px reports a
# deltaY/wheelDeltaY pair that no mouse or trackpad can generate. Distance is therefore built
# from whole ticks rather than chosen freely.
PX_PER_NOTCH = 40

# Probability that a notch carries two ticks instead of one. A wheel spun quickly accumulates
# more than one detent between reports, so the delta is a small multiple rather than a
# constant — which also stops every wheel event being byte-identical.
DOUBLE_TICK = 0.14

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
    # How long the button is held down. Measured from 34,989 SapiMouse press/release pairs:
    # per-user medians run 70-121ms. Previously absent entirely, so the driver fired
    # down-and-up with no hold at all and a page measured a 0.1ms click — which is not a
    # subtle statistical tell, it is a value no hand can produce.
    "mouse_click_ms": (20.0, 400.0),
}

# Within-person spread of click hold. Measured WITHIN each person and excluding holds past
# 600ms, which are press-and-hold — a different motor act that inflates the spread. The pooled
# figure across all users and all holds is 0.78, but that folds in between-person variance and
# generated a 197ms mean click against a human 100ms. Trimmed and within-person it is 0.48,
# and the resulting distribution lands at p50 83 / p90 162 against a human p50 100 / p90 170.
CLICK_LOG_SD = 0.48
CLICK_MAX_MS = 600.0

# Fallback only, used when no mouse corpus is present. Centred on the SapiMouse medians so
# the fallback is at least in the right place; the real corpus supersedes it.
_FALLBACK = (2.74, 0.40, 0.10, 17.9, 2.03, 61.0, 87.0)
_SCROLL_FALLBACK = (179.0, 1.24, 0.06, 780.0, 1.01, 55.8)

# ── path shape, all four constants measured from SapiMouse (24,451 strokes, 120 users) ──
#
# Sample rate. Real pointer input arrives at ~59Hz; we were emitting 79Hz and 55 samples per
# stroke against a real 34. Over-sampling is both a tell and a cost — each sample is an IPC
# round-trip — so the sample count now follows from the duration instead of pixel density.
SAMPLE_HZ = 59.0

# Velocity peaks EARLY, at 0.26 of the stroke. Any symmetric easing peaks at exactly 0.50,
# which is a fixed checkable signature. (agenthands reports 0.40 from a smaller capture; our
# 120-user measurement says 0.26. Both agree it is early; we use our own number.) The profile
# v(t) ~ t^a (1-t)^b peaks at a/(a+b). The measured position is of the GLOBAL maximum
# over a two-segment stroke, so the profile's own peak sits later than the observed one.
_VEL_A, _VEL_B = 1.6, 3.7

# Human reaching is BALLISTIC-PLUS-CORRECTIONS. The opening thrust misses, and one or two
# homing submovements close the gap. This is not cosmetic: real strokes reach a path/straight
# ratio of 2.15 at p90 while their maximum PERPENDICULAR deviation stays at 0.37 of the
# distance, and no single arc satisfies both. The extra length comes from backtracking, which
# only a multi-segment stroke produces. A single bow tuned to match the p90 ratio would have
# to bulge far past the measured deviation, trading one wrong number for another.
_MISS_FRAC = 0.06       # median ballistic miss, as a fraction of the distance
_MISS_LOG_SD = 0.95     # its spread
LAT_FRAC = 0.25         # lateral component of the miss, as a fraction of it

# Per-stroke directness. Measured within-person log-sd of 4*perp/dist is 1.03, and tightly
# held across users (p10 0.87, p90 1.18), so it is a constant rather than a persona field.
# This is what our old uniform bow got wrong: most human strokes are nearly direct and a few
# wander a long way, giving path/straight p90 of 2.15. A uniform draw has no tail and gave
# 1.21 — every stroke equally direct, which is exactly the synthetic signature.
# Fitted to the measured quantiles, not assumed: real click-terminated reaches have
# path/straight p50 1.11 and p90 2.17, with max perpendicular deviation p50 0.09 of the
# distance. CURVE_DIV sets the centre (the round-trip requires median perp/dist == curve/4,
# and the lateral miss and wobble also contribute perpendicular deviation, so the bow itself
# must sit below that) and CURVE_LOG_SD sets the tail.
CURVE_DIV = 6.0
CURVE_LOG_SD = 2.0

# Tremor is a CORRELATED wander, not white noise. This is the key correction: 2.03px of
# measured midpoint-deviation is equally true of white noise and of a smooth wobble, but white
# noise reverses direction every sample and manufactures a velocity peak each time. Real
# strokes have ~2 velocity peaks; per-sample white noise gave us 7.
TREMOR_PHI = 0.86

# Midpoint deviation d_i = w_i - (w_{i-1} + w_{i+1})/2 for an AR(1) series of marginal sd s
# has Var(d) = s^2 (1.5 - 2*phi + 0.5*phi^2); times sqrt(2 ln 2) converts the per-axis sd to
# the median of the 2D magnitude, which is what the extractor measures. At phi=0 this reduces
# to the previous white-noise factor, so the calibration carries over rather than being reset.
TREMOR_TO_SIGMA = (math.sqrt(1.5 - 2 * TREMOR_PHI + 0.5 * TREMOR_PHI ** 2)
                   * math.sqrt(2 * math.log(2)))

# ── coalesced pointer samples ─────────────────────────────────────────────────
# Real input occasionally produces a `pointermove` carrying several samples, because hardware
# reports independently of the display: when samples arrive faster than the compositor paints,
# the browser merges them. Injected input never does this on its own — every dispatch gets its
# own frame, so `getCoalescedEvents()` returns exactly 1 forever, which is itself the tell.
#
# This is derived rather than copied. Our sample rate is a measured 59Hz against a 60Hz
# display, so samples do NOT normally outpace paints and coalescing only happens when a frame
# is missed. A miss rate of ~1% with a 2-3 sample burst yields a mean near 1.01, which is what
# real sessions show. Overcorrecting is its own tell: pipelining every move gives a mean near
# 1.8, and a display dropping most of its frames is not a plausible client either.
FRAME_DROP_CHANCE = 0.008
FRAME_BURST = (2, 4)          # samples merged when a frame is missed

# A movement counts as an overshoot when it travels this many pixels past its endpoint and
# comes back. Absolute, not a fraction of distance: a corrective submovement is a fixed
# 5-15px regardless of how far the reach was, so a relative threshold goes blind on long
# moves and undercounts overshoot by roughly half.
OVERSHOOT_PX = 4.0


def _profile(steps: int) -> list[float]:
    """Cumulative displacement fractions for an asymmetric velocity profile.

    Integrating v(t) = t^a (1-t)^b numerically and normalising is simpler and more obviously
    correct than inverting a regularised incomplete beta, and `steps` never exceeds 60."""
    v = [((i / steps) ** _VEL_A) * ((1 - i / steps) ** _VEL_B) for i in range(1, steps + 1)]
    total = sum(v) or 1.0
    out, acc = [], 0.0
    for x in v:
        acc += x / total
        out.append(acc)
    return out


def _wobble(steps: int, sigma: float, rng: random.Random) -> list[float]:
    """Correlated perpendicular jitter — a smooth wander rather than white noise.

    An AR(1) walk reproduces the measured deviation amplitude with the correlation structure a
    hand actually has, so the SapiMouse calibration carries over intact while the spurious
    velocity reversals disappear."""
    innov = sigma * math.sqrt(1 - TREMOR_PHI * TREMOR_PHI)
    out, w = [], rng.gauss(0, sigma)
    for _ in range(steps):
        w = TREMOR_PHI * w + rng.gauss(0, innov)
        out.append(w)
    return out


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

    ux, uy = dx / dist, dy / dist
    if over:
        # Overshoot: aim PAST the target, then correct back. Magnitude is measured — real
        # corrective overshoots have a median of ~18px with a long tail, where the old
        # uniform(5,14) topped out below the median.
        past = _clamp(rng.lognormvariate(math.log(motor.mouse_overshoot_px), 0.7), 2, 200)
        tx, ty = x1 + ux * past, y1 + uy * past
    else:
        # Undershoot: the ordinary case. The thrust falls short and drifts off-axis, and a
        # homing submovement finishes the reach.
        miss = dist * min(rng.lognormvariate(math.log(_MISS_FRAC), _MISS_LOG_SD), 0.75)
        lat = rng.gauss(0, miss * LAT_FRAC)
        tx, ty = x1 - ux * miss - uy * lat, y1 - uy * miss + ux * lat

    pts: list[Move] = []

    # Total duration is a property of the whole reach, not of each piece. Charging the 50ms
    # constant once per segment inflated both the duration and the sample count — 42 samples
    # against a real 34 — because a two-segment stroke paid it twice.
    total_dur = motor.mouse_speed * (50 + dist * 0.34) * rng.uniform(0.9, 1.1)

    def seg(ax: float, ay: float, bx: float, by: float, share: float) -> None:
        sdx, sdy = bx - ax, by - ay
        sd = math.hypot(sdx, sdy) or 1.0
        nx, ny = -sdy / sd, sdx / sd                     # perpendicular -> bow direction

        # Per-stroke bow, log-normal around the persona's own directness. The extractor takes
        # the MEDIAN of 4*perp/dist per user, and a log-normal's median is its scale, so the
        # calibration round-trips while the tail this adds is what produces the p90 of 2.15.
        frac = rng.lognormvariate(math.log(max(motor.mouse_curve, 1e-3) / CURVE_DIV), CURVE_LOG_SD)
        bow = sd * min(frac, 1.2) * (1 if rng.random() < 0.5 else -1)

        # Duration first, then sample count from the real pointer rate — not from pixels.
        dur = total_dur * share
        steps = max(3, min(60, round(dur * SAMPLE_HZ / 1000.0)))
        dt = dur / steps
        disp = _profile(steps)
        # Isotropic, not purely perpendicular: a hand's tremor has no preferred axis, and the
        # 2-D form is what TREMOR_TO_SIGMA's Rayleigh factor assumes. Projecting one scalar
        # onto the perpendicular made the recovered amplitude 20% low.
        wx = _wobble(steps, trem, rng)
        wy = _wobble(steps, trem, rng)
        burst = 0
        for i in range(steps):
            e = disp[i]
            arc = math.sin((i + 1) / steps * math.pi) * bow
            if burst == 0 and i + FRAME_BURST[1] < steps and rng.random() < FRAME_DROP_CHANCE:
                burst = rng.randint(*FRAME_BURST)      # a missed compositor frame
            pts.append(Move(ax + sdx * e + nx * arc + wx[i],
                            ay + sdy * e + ny * arc + wy[i], dt, burst > 0))
            if burst:
                burst -= 1

    d1 = math.hypot(tx - x0, ty - y0)
    d2 = math.hypot(x1 - tx, y1 - ty)
    tot = (d1 + d2) or 1.0
    seg(x0, y0, tx, ty, d1 / tot)
    seg(tx, ty, x1, y1, d2 / tot)        # the homing correction, always present now

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
        ticks = 2 if rng.random() < DOUBLE_TICK else 1
        out.append(Wheel(direction * PX_PER_NOTCH * ticks, gap))
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
