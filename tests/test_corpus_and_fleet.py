"""The corpus is real, and the fleet it produces is actually a fleet."""
import pathlib

import pytest

from behaviourforge import audit, corpus, forge
from behaviourforge.audit import ALL_FIELDS


@pytest.fixture(scope="module")
def ks():
    return corpus.load()


def test_bundled_corpora_are_present_and_large(ks):
    assert ks["meta"]["n"] >= 5000
    mouse = corpus.load_mouse()
    assert mouse and mouse["meta"]["n"] >= 100
    scroll = corpus.load_scroll()
    assert scroll and scroll["meta"]["n"] >= 10        # Balabit is only 10 users — thin


def test_every_generator_parameter_comes_from_a_corpus():
    """The point of the whole exercise: no hand-authored motor constant survives. Every field
    the generators consume must be backed by observations in one of the three corpora."""
    from behaviourforge.audit import ALL_FIELDS
    backed = set()
    for c in (corpus.load(), corpus.load_mouse(), corpus.load_scroll()):
        for v in c["vectors"]:
            backed |= {k for k, x in v.items() if x is not None}
    missing = [f for f in ALL_FIELDS if f not in backed]
    assert not missing, f"parameters with no data behind them: {missing}"


def test_every_field_is_actually_measured(ks):
    """Guards the exact class of bug that shipped in the original: a column that looks like
    data but is one imputed constant repeated, or a clamp ceiling every participant
    saturated. shift_ms was 97% a hardcoded 55.0; tempo_theta had ONE distinct value across
    3000 people."""
    d = corpus.describe(ks)
    for field, s in d.items():
        assert s["measured"], (
            f"{field} is not measured: {s['mode_share']:.1%} of present values are identical, "
            f"coverage {s['coverage']:.1%}")
        assert s["distinct"] > 50, f"{field} has only {s['distinct']} distinct values"


def test_reliability_is_reported_for_every_field(ks):
    rel = ks["meta"]["reliability"]
    for f in ("base_ms", "dwell_ms", "flight_sigma", "think_ms"):
        assert rel.get(f, 0) > 0.7, f"{f} reliability {rel.get(f)} — expected a stable trait"


def test_base_dwell_coupling_is_weak_like_real_humans(ks):
    """'Fast typist, short key hold' is intuitive and essentially false: flight time and hold
    time are separately controlled, r ~ 0.15. Hand-authored archetypes encode the intuition
    and produce r ~ 0.96, which is how you can tell they are invented."""
    from behaviourforge.audit import _pearson
    v = ks["vectors"]
    r = _pearson([x["base_ms"] for x in v], [x["dwell_ms"] for x in v])
    assert 0.0 < r < 0.4, f"base~dwell correlation {r:.3f} does not look like real data"


@pytest.mark.parametrize("model", ["copula", "bootstrap"])
def test_fleet_audit_passes(model):
    r = audit(1500, model=model)
    assert r.passed, "\n" + str(r)


def test_copula_beats_the_corpus_size_ceiling():
    """Bootstrap cannot emit more distinct identities than the corpus has rows; at fleet
    scale that means reusing real people, ~1.7 times each at 10k agents against 6k
    participants, with one participant reused a dozen times. The copula samples the
    distribution instead, so distinctness does not depend on corpus size."""
    n = 8000
    sigs = {tuple(getattr(forge(i, model="copula").motor, f) for f in ALL_FIELDS)
            for i in range(n)}
    assert len(sigs) == n


def test_no_constant_parameter_across_a_large_fleet():
    """The specific bug: every agent sharing an identical tempo. Individually invisible,
    collectively a signature."""
    motors = [forge(i).motor for i in range(3000)]
    for f in ALL_FIELDS:
        assert len({getattr(m, f) for m in motors}) > 100, f"{f} is near-constant across 3000"


def test_core_never_imports_a_browser():
    """Browser-agnostic is a structural property, not an intention. The core emits plans;
    executing them is the caller's job."""
    banned = ("playwright", "selenium", "pyppeteer", "requests", "httpx")
    root = pathlib.Path(__file__).resolve().parent.parent / "behaviourforge"
    for py in root.rglob("*.py"):
        text = py.read_text()
        for b in banned:
            assert f"import {b}" not in text, f"{py.name} imports {b}"


def test_library_has_no_third_party_dependencies():
    pyproject = (pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
    assert "dependencies = []" in pyproject
