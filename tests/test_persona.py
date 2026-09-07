"""Identity, determinism, and the guarantees a fleet depends on."""
import random

import pytest

from behaviourforge import Motor, forge
from behaviourforge.keystroke import RANGES
from behaviourforge.mouse import FIELDS as MOUSE_FIELDS


def _stream(p, text="the quick brown fox"):
    return [(k.key, round(k.flight_ms, 9), round(k.dwell_ms, 9), k.kind)
            for k in p.keystrokes(text)]


@pytest.mark.parametrize("model", ["copula", "bootstrap"])
def test_same_seed_same_identity(model):
    a, b = forge(4712, model=model), forge(4712, model=model)
    assert a.motor == b.motor
    assert _stream(a) == _stream(b)


@pytest.mark.parametrize("model", ["copula", "bootstrap"])
def test_different_seeds_differ(model):
    assert forge(1, model=model).motor != forge(2, model=model).motor
    assert _stream(forge(1, model=model)) != _stream(forge(2, model=model))


def test_seed_none_is_recorded_and_replayable():
    """An identity forged without a seed must still be reproducible, or a session that
    turned out to matter cannot be replayed."""
    p = forge()
    assert isinstance(p.motor.seed, int)
    assert forge(p.motor.seed).motor == p.motor


def test_mouse_calls_do_not_shift_keystroke_timings():
    """The RNG split matters: a persona that types differently depending on where you moved
    the pointer is not replayable, and the bug would be near-impossible to trace."""
    a = forge(7)
    before = _stream(a, "hello")
    b = forge(7)
    b.path(0, 0, 500, 300)
    b.scroll(3)
    b.idle(500, 100, 100)
    assert _stream(b, "hello") == before


def test_tempo_persists_across_calls():
    """One session is one continuous rhythm. Two fields typed in sequence must not produce
    the same timings as the same field typed twice from scratch."""
    p = forge(11)
    first = [round(k.flight_ms, 9) for k in p.keystrokes("abcdefgh")]
    second = [round(k.flight_ms, 9) for k in p.keystrokes("abcdefgh")]
    assert first != second


def test_motor_serialization_round_trip():
    m = forge(99).motor
    assert Motor.from_dict(m.to_dict()) == m


def test_all_parameters_within_range():
    for i in range(200):
        m = forge(i).motor
        for f, (lo, hi) in RANGES.items():
            assert lo <= getattr(m, f) <= hi, f"{f} out of range at seed {i}"


def test_mouse_parameters_present():
    m = forge(3).motor
    for f in MOUSE_FIELDS:
        assert getattr(m, f) > 0


def test_rejects_unmodelled_device_and_locale():
    """Silently producing a desktop persona for a phone, or US-QWERTY finger physics for
    AZERTY, would be worse than refusing: the caller would believe it was modelled."""
    with pytest.raises(NotImplementedError, match="device"):
        forge(1, device="mobile")
    with pytest.raises(NotImplementedError, match="locale"):
        forge(1, locale="fr-FR")
    with pytest.raises(ValueError, match="model"):
        forge(1, model="bayes-net")


def test_forging_is_cheap_enough_for_a_fleet():
    """10,000 agents is the design point; identity generation must not be a bottleneck."""
    import time
    t = time.time()
    seeds = [forge(i).motor.seed for i in range(2000)]
    assert time.time() - t < 5.0
    assert len(set(seeds)) == 2000


def test_no_parameter_sits_on_a_clamp_more_than_the_corpus_does():
    """Clamp saturation is this project's recurring bug: a bound tight enough to bite turns a
    parameter into a constant while still looking like data. tempo_theta had ONE distinct
    value across 3000 people for exactly this reason.

    Real distributions do contain atoms, so the test is relative — the fleet may inherit the
    corpus's own pile-up, but it may not invent its own."""
    from behaviourforge import corpus
    from behaviourforge.mouse import MOUSE_RANGES
    ks, mouse = corpus.load(), corpus.load_mouse()
    n = 2000
    motors = [forge(i).motor for i in range(n)]
    for field, (lo, hi) in {**RANGES, **MOUSE_RANGES}.items():
        src = ks if field in RANGES else mouse
        pool = [v[field] for v in src["vectors"] if v.get(field) is not None]
        for bound in (lo, hi):
            corpus_share = sum(1 for v in pool if abs(v - bound) < 1e-9) / len(pool)
            fleet_share = sum(1 for m in motors
                              if abs(getattr(m, field) - bound) < 1e-9) / n
            assert fleet_share <= corpus_share * 1.4 + 0.01, (
                f"{field} piles up at {bound}: {fleet_share:.2%} of the fleet vs "
                f"{corpus_share:.2%} of the corpus — the sampler invented this spike")


def test_emitted_timings_do_not_saturate_their_bounds():
    """The parameters can be in range while the EVENTS are not. A dwell ceiling of 180ms once
    pinned 5.35% of keystrokes to exactly 180.000 — a hard spike in the hold-time histogram,
    which is what a sensor scoring the distribution actually reads."""
    dwell, flight = [], []
    for i in range(150):
        for k in forge(i).keystrokes("the quick brown fox jumps over the lazy dog"):
            dwell.append(k.dwell_ms)
            flight.append(k.flight_ms)
    for name, vals in (("dwell", dwell), ("flight", flight)):
        counts = {}
        for v in vals:
            counts[round(v, 3)] = counts.get(round(v, 3), 0) + 1
        top_value, top_n = max(counts.items(), key=lambda kv: kv[1])
        assert top_n / len(vals) < 0.005, (
            f"{name} spikes: {top_n / len(vals):.2%} of events land on exactly {top_value}")


def test_generated_holds_match_the_real_distribution():
    """Aalto's own hold times: p50 104ms, p90 156, p99 215, p99.9 298."""
    dwell = sorted(k.dwell_ms for i in range(200)
                   for k in forge(i).keystrokes("the quick brown fox jumps over", typos=False))
    q = lambda p: dwell[min(len(dwell) - 1, int(p * len(dwell)))]
    assert 90 <= q(0.50) <= 120, f"median hold {q(0.50):.0f}ms, real is 104ms"
    assert 130 <= q(0.90) <= 185, f"p90 hold {q(0.90):.0f}ms, real is 156ms"
    assert 180 <= q(0.99) <= 280, f"p99 hold {q(0.99):.0f}ms, real is 215ms"
