"""Pointer geometry, and the round-trip that keeps extraction honest."""
import math
import random
import statistics
import statistics as st

from behaviourforge import forge
from behaviourforge.corpus import _movement_features
from behaviourforge.mouse import path


def test_path_lands_exactly_on_target():
    """Tremor on the final sample leaves the pointer a pixel or two off, which on a small
    element is a missed click."""
    for seed in range(40):
        p = forge(seed)
        pts = p.path(10, 10, 640, 480)
        assert (pts[-1].x, pts[-1].y) == (640, 480)


def test_final_wait_is_the_settle():
    """Settle is baked into the plan so a driver cannot forget it and fire a click the
    instant the pointer arrives, which no human does."""
    p = forge(3)
    pts = p.path(0, 0, 400, 400)
    assert pts[-1].dt_ms == p.motor.mouse_settle_ms


def test_path_is_not_a_straight_line():
    p = forge(6)
    pts = p.path(0, 0, 800, 0)
    assert max(abs(m.y) for m in pts) > 1.0, "path has no curvature at all"


def test_coordinates_are_fractional():
    """Integer-only coordinates are a tell: real pointer hardware reports sub-pixel deltas."""
    pts = forge(9).path(0, 0, 500, 300)
    assert any(m.x != int(m.x) for m in pts[:-1])


def test_short_moves_collapse_to_a_single_settle():
    pts = forge(1).path(100, 100, 100.2, 100.2)
    assert len(pts) == 1


def test_sample_count_is_bounded():
    """Each sample is typically one IPC round-trip, so point count sets wall-clock."""
    pts = forge(2).path(0, 0, 3000, 2000)
    assert len(pts) <= 130                       # <=60 per segment, at most two segments


def test_scroll_emits_notches_at_measured_cadence():
    """One 800px wheel delta teleports the page in a frame; hardware emits discrete notches.
    Balabit's per-user median gaps span 16-249ms, so cadence is a persona trait."""
    from behaviourforge.mouse import PX_PER_NOTCH
    w = forge(4).scroll(30)
    assert len(w) == 30
    # Whole ticks only: wheelDeltaY derives from the tick count, so any other delta reports a
    # pair no real device can produce.
    assert all(abs(x.dy) % PX_PER_NOTCH == 0 for x in w)
    assert all(abs(x.dy) >= PX_PER_NOTCH for x in w)
    assert all(4 <= x.dt_ms <= 30_000 for x in w)


def test_scroll_reversal_rate_matches_the_corpus():
    """Measured 2.9-8.9% across 10 users. The old hardcoded 0.2 was ~3.4x too high, which
    made every agent look like it kept losing its place."""
    rev = tot = 0
    for seed in range(60):
        w = forge(seed).scroll(200)
        rev += sum(1 for a, b in zip(w, w[1:]) if (a.dy > 0) != (b.dy > 0))
        tot += len(w) - 1
    assert 0.01 < rev / tot < 0.15, f"reversal rate {rev / tot:.1%} is outside the measured range"


def test_scroll_cadence_varies_across_the_fleet():
    gaps = [statistics.median([x.dt_ms for x in forge(s).scroll(60)]) for s in range(50)]
    assert max(gaps) / min(gaps) > 2.0, "every agent scrolls at the same speed"


def test_idle_fills_the_requested_duration():
    """An agent's observe-think-act loop parks the cursor dead still on every turn; idle
    exists to fill that, so it must actually cover the wait."""
    p = forge(5)
    moves = p.idle(4000, 300, 300)
    total = sum(m.dt_ms for m in moves)
    assert 3900 <= total <= 4400


def test_idle_pauses_have_a_long_tail():
    """Real still-gaps in work sessions: p50 780ms, p75 1.4s, p90 4.2s. A bounded uniform
    draw has no tail, which makes every rest period suspiciously similar in length."""
    pauses = sorted(m.dt_ms for s in range(40) for m in forge(s).idle(60_000, 400, 300)
                    if m.dt_ms > 200)
    q = lambda p: pauses[min(len(pauses) - 1, int(p * len(pauses)))]
    assert q(0.90) / q(0.50) > 2.5, "idle pause distribution has no tail"


def test_idle_stays_on_screen():
    p = forge(7)
    for m in p.idle(8000, 5, 5, width=1280, height=800):
        assert -1 <= m.x <= 1281 and -1 <= m.y <= 801


def test_point_in_box_is_never_dead_centre():
    p = forge(8)
    box = {"x": 0, "y": 0, "width": 100, "height": 40}
    pts = [p.point_in(box) for _ in range(50)]
    assert all(0 < x < 100 and 0 < y < 40 for x, y in pts)
    assert not all(abs(x - 50) < 0.5 and abs(y - 20) < 0.5 for x, y in pts)


def test_generation_inverts_extraction():
    """Round-trip: generate with known parameters, re-measure with the SapiMouse extractor,
    and recover the inputs.

    This is the test that keeps the corpus meaningful. Without it, extraction can measure
    something plausible but adjacent to what the generator consumes, and the two drift apart
    while both look fine. It caught a bow cap that flattened every long sweep, a relative
    overshoot threshold that went blind on long moves, and a tremor unit mismatch that was
    passing only because two errors cancelled."""
    p = forge(1)
    m = p.motor
    rng = random.Random(99)
    rows = []
    for _ in range(500):
        x0, y0 = rng.uniform(0, 1200), rng.uniform(0, 700)
        x1, y1 = rng.uniform(0, 1200), rng.uniform(0, 700)
        t, seq = 0.0, []
        for pt in path(m, x0, y0, x1, y1, rng):
            seq.append((t, pt.x, pt.y))
            t += pt.dt_ms
        f = _movement_features(seq, t)
        if f:
            rows.append(f)

    assert len(rows) > 300
    got = {
        "mouse_speed": st.median([r["speed"] for r in rows]),
        "mouse_curve": st.median([r["curve"] for r in rows]),
        "mouse_tremor": st.median([r["tremor"] for r in rows]),
        "mouse_settle_ms": st.median([r["settle"] for r in rows if r["settle"] is not None]),
    }
    for field, recovered in got.items():
        truth = getattr(m, field)
        assert 0.8 <= recovered / truth <= 1.25, (
            f"{field}: generated {truth:.3f} but re-measured {recovered:.3f} — extraction "
            f"and generation have drifted apart")


# ── path SHAPE, measured against SapiMouse (24,451 strokes, 120 users) ────────
# These guard the externally observable geometry of a stroke, which is what a behavioural
# sensor reads. Every target below was measured from the raw dataset with this same estimator,
# not taken from another library's published figure — one of those figures (peak velocity at
# 0.40) did not reproduce against our data, which said 0.26.

def _stroke_metrics(pts):
    sp = []
    for a, b in zip(pts, pts[1:]):
        sp.append(math.hypot(b.x - a.x, b.y - a.y) / (a.dt_ms or 1))
    plen = sum(math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(pts, pts[1:]))
    straight = math.hypot(pts[-1].x - pts[0].x, pts[-1].y - pts[0].y) or 1
    return sp, plen / straight


def _smooth(v, k=3):
    return [st.mean(v[max(0, i - k // 2):i + k // 2 + 1]) for i in range(len(v))] if len(v) >= k else v


def _peaks(v, prom):
    if len(v) < 3:
        return 0
    mx = max(v) or 1.0
    n = 0
    for i in range(1, len(v) - 1):
        if v[i] >= v[i - 1] and v[i] >= v[i + 1]:
            lo = min(min(v[:i] or [v[i]]), min(v[i + 1:] or [v[i]]))
            if (v[i] - lo) / mx > prom:
                n += 1
    return n


def _sample_strokes(n=500, seed=5):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        p = forge(i)
        a = (rng.uniform(0, 1200), rng.uniform(0, 700))
        b = (rng.uniform(0, 1200), rng.uniform(0, 700))
        pts = p.path(*a, *b)
        if len(pts) >= 8 and math.hypot(b[0] - a[0], b[1] - a[1]) >= 40:
            out.append(pts)
    return out


def test_velocity_peaks_early_not_at_the_midpoint():
    """Any symmetric easing peaks at exactly 0.50, which is a fixed checkable signature.
    Real strokes peak at 0.26 — the thrust is front-loaded and the tail is a slow homing."""
    pos = []
    for pts in _sample_strokes():
        sp, _ = _stroke_metrics(pts)
        if len(sp) >= 6 and max(sp) > 0:
            pos.append(sp.index(max(sp)) / len(sp))
    assert 0.18 <= st.median(pos) <= 0.34, f"peak velocity at {st.median(pos):.2f}, real is 0.26"


def test_stroke_has_few_velocity_peaks():
    """A stroke is a thrust plus a correction — about 2 peaks. Per-sample white-noise tremor
    produced 7, because white noise reverses direction every sample and each reversal is a
    peak. Correlated wobble keeps the measured amplitude without the spurious reversals."""
    counts = [_peaks(_smooth(_stroke_metrics(pts)[0]), 0.15) for pts in _sample_strokes()]
    assert st.median(counts) <= 3, f"{st.median(counts)} velocity peaks per stroke, real is 2"


def test_stroke_directness_has_a_heavy_tail():
    """Most reaches are nearly direct and a few wander a long way: real p50 1.11, p90 2.15.
    A uniform bow gave p90 1.21 — every stroke equally direct, which is the synthetic tell."""
    ratios = sorted(_stroke_metrics(pts)[1] for pts in _sample_strokes())
    p50 = st.median(ratios)
    p90 = ratios[min(len(ratios) - 1, int(0.9 * len(ratios)))]
    assert 1.05 <= p50 <= 1.25, f"median path/straight {p50:.2f}, real is 1.11"
    assert p90 >= 1.8, f"p90 path/straight {p90:.2f}, real is 2.15 — strokes are too uniform"


def test_pointer_sample_rate_matches_real_hardware():
    """Real pointer input arrives at ~59Hz. Over-sampling is both a tell and a cost, since
    every sample is an IPC round-trip in the driver."""
    hz = []
    for pts in _sample_strokes():
        dur = sum(m.dt_ms for m in pts[:-1])
        if dur > 0:
            hz.append(1000 * len(pts) / dur)
    assert 45 <= st.median(hz) <= 80, f"{st.median(hz):.0f}Hz, real hardware is 59Hz"
