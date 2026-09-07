"""The timing model — the correlations, not just the spread."""
import math
import statistics as st

from behaviourforge import forge
from behaviourforge.keystroke import digraph_mult, needs_shift


def test_digraph_ordering_matches_measurement():
    """Hand alternation is fastest, same finger slowest. Measured across 3000 Aalto typists;
    the magnitudes are far gentler than intuition suggests (1.06x for same finger, not 1.6x),
    so this guards the ordering rather than inviting a re-guess of the values."""
    assert digraph_mult("f", "j") < digraph_mult("f", "d") < digraph_mult("f", "f")
    assert digraph_mult("f", "j") == 0.80
    assert digraph_mult("f", "f") == 1.06
    assert digraph_mult(" ", "a") == 1.0            # unknown pair -> neutral


def test_needs_shift():
    assert needs_shift("A") and needs_shift("?") and needs_shift("$")
    assert not needs_shift("a") and not needs_shift(" ") and not needs_shift("1")


def test_plan_covers_every_character():
    p = forge(5)
    keys = [k for k in p.keystrokes("hello world", typos=False) if k.kind == "key"]
    assert "".join(k.key for k in keys) == "hello world"


def test_typos_are_corrected_immediately():
    """A typo must be followed by a backspace and then the intended character. A wrong
    character left standing is not a human error, it is a broken field."""
    p = forge(2)
    found = False
    for seed in range(300):
        q = forge(seed)
        if q.motor.error_rate < 0.05:
            continue
        keys = q.keystrokes("the quick brown fox jumps over the lazy dog" * 2)
        for i, k in enumerate(keys):
            if k.kind == "typo":
                assert keys[i + 1].kind == "backspace"
                assert keys[i + 1].key == "Backspace"
                assert keys[i + 2].kind == "key"
                found = True
        if found:
            break
    assert found, "no typo produced across 300 personas — error_rate may be broken"


def test_no_typos_when_disabled():
    for seed in range(50):
        keys = forge(seed).keystrokes("password123!" * 4, typos=False)
        assert all(k.kind == "key" for k in keys)


def test_timings_are_positive_and_bounded():
    for seed in range(50):
        for k in forge(seed).keystrokes("Hello, World! 42"):
            assert 0 < k.flight_ms < 5000
            assert 0 < k.dwell_ms < 500


def test_flight_times_are_not_iid():
    """Autocorrelated tempo is the property random delays cannot fake, and the reason the
    constant-tempo bug mattered: IID timings have zero autocorrelation, which is itself the
    giveaway to anything reading the time series.

    Pooled across personas on purpose. The effect is genuinely small — per-keystroke trial
    noise swamps the slow tempo drift, which is exactly why estimating tempo from a RAW
    lag-1 autocorrelation fails and needed the AR(1)-plus-noise decomposition. A single
    persona over a few hundred keys has a standard error around 0.05, so a one-persona
    assertion here would be a coin flip."""
    accs = []
    for seed in range(30):
        f = [k.flight_ms for k in forge(seed).keystrokes("a" * 600, typos=False)]
        logs = [math.log(x) for x in f]
        m = sum(logs) / len(logs)
        num = sum((logs[i] - m) * (logs[i + 1] - m) for i in range(len(logs) - 1))
        den = sum((x - m) ** 2 for x in logs)
        accs.append(num / den)
    mean = st.mean(accs)
    assert mean > 0.02, f"pooled lag-1 acf {mean:+.4f} — tempo drift is not reaching output"


def test_distribution_is_right_skewed():
    """Log-normal, never uniform. Uniform timing is as detectable as constant timing."""
    f = [k.flight_ms for k in forge(33).keystrokes("the quick brown fox " * 30, typos=False)]
    assert st.mean(f) > st.median(f), "flight distribution is not right-skewed"


def test_shifted_keys_cost_more():
    p = forge(8)
    lower = st.median([k.flight_ms for k in p.keystrokes("aaaa" * 40, typos=False)])
    upper = st.median([k.flight_ms for k in p.keystrokes("AAAA" * 40, typos=False)])
    assert upper > lower


def test_first_field_opens_with_a_pause():
    p = forge(12)
    cold = p.keystrokes("hello", first_field=True)[0].flight_ms
    warm = forge(12).keystrokes("hello", first_field=False)[0].flight_ms
    assert cold > warm


def test_measured_bigram_table_is_used():
    """The corpus ships a per-person bigram table; using it is the whole point of extracting
    it. Hydra extracted the same table and never consumed it."""
    p = forge(4)
    assert p.motor.bigram_mult, "persona carries no measured bigram table"
    common = {"th", "he", "in", "er", "an"} & set(p.motor.bigram_mult)
    assert common, "bigram table has none of the most frequent English pairs"
