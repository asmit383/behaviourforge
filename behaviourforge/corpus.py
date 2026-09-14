"""The corpus — real human motor vectors in, sampling material out.

This is the part that matters most and gets talked about least. BrowserForge is only as good
as Apify's device corpus; BehaviourForge is only as good as its human corpus. Modelling is
the easy half — sourcing and cleaning real data is the work, and it is also the moat.

    raw keydown/keyup ──► per-participant vector ──► corpus.json ──► forge(seed)

Bundled: 3000+ participants from the Aalto "136M Keystrokes" dataset (Dhakal et al. 2018),
extracted by `build_from_aalto` below, so the blob is reproducible rather than magic.

── WHY THIS FILE IS PARANOID ─────────────────────────────────────────────────────────────
A corpus can carry a column that looks like data and is not, and nothing downstream can tell.
Three ways it happened here, all found by auditing the output rather than reading the code:

  1. Imputation.       `shift_ms` needs enough shifted keystrokes to estimate. 97% of
                       participants never typed enough capitals, so they got a default
                       constant. One number, 2,916 times, indistinguishable from data.
  2. Clamp saturation. `tempo_theta` was derived as 1 - lag1_autocorrelation and clamped to
                       [0.05, 0.15]. The derivation actually lands near 0.99, because trial
                       noise attenuates the raw autocorrelation to ~0. Every participant
                       saturated the ceiling. The clamp silently turned a broken estimator
                       into a constant.
  3. Unreliability.    Even with a correct estimator, ~500 keystrokes is not enough to pin
                       one person's tempo. Split-half reliability is 0.08 — the spread
                       across participants is estimation noise, not human variation.

So this module ships `reliability` alongside the vectors. A field's spread is only worth
propagating to the extent it is reproducible, and the sampler shrinks accordingly. Measuring
that honestly is worth more than another decimal place on any single estimate.

Licensing: Aalto is released for research use. Verify your rights before redistributing a
derived corpus commercially.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import statistics as st
from collections import Counter

from behaviourforge.keystroke import RANGES, digraph_mult
from behaviourforge.mouse import CLICK_MAX_MS, MOUSE_RANGES, OVERSHOOT_PX

_HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLED = os.path.join(_HERE, "data", "keystroke_aalto.json")
BUNDLED_MOUSE = os.path.join(_HERE, "data", "mouse_sapimouse.json")
BUNDLED_SCROLL = os.path.join(_HERE, "data", "scroll_balabit.json")

# Column map for the Aalto TSVs. Adjust if your copy's schema differs.
COLUMNS = {
    "participant": "PARTICIPANT_ID", "section": "TEST_SECTION_ID",
    "press": "PRESS_TIME", "release": "RELEASE_TIME",
    "letter": "LETTER", "keycode": "KEYCODE",
}
_BACKSPACE = {"8", "backspace", "bksp"}
_SHIFT_SYM = set('~!@#$%^&*()_+{}|:"<>?')

# Fields that can legitimately be absent from a participant's vector: not every person
# supplies enough evidence to estimate them, and a default would be a lie.
OPTIONAL = ("shift_ms", "think_ms", "tempo_theta", "tempo_sigma")

_cache: dict[str, dict] = {}


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


# ── loading ───────────────────────────────────────────────────────────────────
def load(path: str | None = None) -> dict:
    """Load a corpus. Returns {"vectors": [...], "meta": {...}}.

    Normalizes the measured per-bigram table from absolute milliseconds into multipliers of
    that person's own base. Absolute values cannot survive resampling — once a vector is
    speed-jittered or a copula draws a new base, a raw 96ms "as" contradicts the new base.
    As a ratio it rides along correctly.

    Cached by path, so forging 10,000 personas parses the file once."""
    p = path or BUNDLED
    hit = _cache.get(p)
    if hit is not None:
        return hit

    with open(p, encoding="utf-8") as f:
        raw = json.load(f)

    # tolerate the legacy flat-list format as well as the current {meta, vectors} one
    vectors_in = raw["vectors"] if isinstance(raw, dict) else raw
    meta = dict(raw.get("meta", {})) if isinstance(raw, dict) else {}

    out: list[dict] = []
    for v in vectors_in:
        required = [f for f in RANGES if f not in OPTIONAL]
        if not all(v.get(f) is not None for f in required):
            continue                                  # ragged entry — skip, don't guess
        base = v["base_ms"] or 1.0
        vec = {f: v.get(f) for f in RANGES}           # optional fields may be None
        vec["bigram_mult"] = {
            bg.lower(): round(ms / base, 4)
            for bg, ms in (v.get("bigram_ms") or v.get("bigram_mult_raw") or {}).items()
            if len(bg) == 2 and 0.3 <= ms / base <= 3.0   # drop afk-contaminated outliers
        }
        out.append(vec)

    if not out:
        raise ValueError(f"corpus at {p!r} yielded no usable vectors")

    meta.setdefault("n", len(out))
    meta.setdefault("reliability", {})
    result = {"vectors": out, "meta": meta}
    _cache[p] = result
    return result


# Above this share of participants sitting on one value, a "measured" field is really an
# imputed constant or a clamp ceiling wearing measured data's clothes.
IMPUTED_MODE_SHARE = 0.5


def describe(corpus: dict | list) -> dict[str, dict]:
    """Per-field report: is this actually measured, how reliable is it, how much is missing.

    `forge()` reads this and refuses to propagate a constant to a whole fleet — 10,000 agents
    sharing an exact parameter value is the correlatable signature the library exists to
    avoid. Guessed-and-diverse beats guessed-and-identical, because the value is guessed
    either way and only the fleet property differs."""
    vectors = corpus["vectors"] if isinstance(corpus, dict) else corpus
    rel = (corpus.get("meta", {}) if isinstance(corpus, dict) else {}).get("reliability", {})
    n = len(vectors) or 1
    out: dict[str, dict] = {}
    for f in RANGES:
        present = [v[f] for v in vectors if v.get(f) is not None]
        counts = Counter(present)
        # Mode share is computed AMONG PRESENT VALUES. Missing data and fake data are
        # different failures: a field estimable for only a third of participants still
        # carries a real distribution for that third, and fitting a marginal to it beats
        # falling back to a guessed range. A field where most PRESENT values are the same
        # number is the actual problem, because that number was imputed.
        mode_share = (counts.most_common(1)[0][1] / len(present)) if present else 1.0
        coverage = len(present) / n
        out[f] = {
            "measured": mode_share < IMPUTED_MODE_SHARE and coverage >= 0.02,
            "distinct": len(counts),
            "mode_share": round(mode_share, 4),
            "coverage": round(coverage, 4),
            "reliability": rel.get(f),
        }
    return out


# ── estimators ────────────────────────────────────────────────────────────────
def _log_std(xs: list[float]) -> float:
    logs = [math.log(x) for x in xs if x > 0]
    return st.pstdev(logs) if len(logs) > 1 else 0.25


def _acov(xs: list[float], k: int) -> float | None:
    """Autocovariance at lag k."""
    n = len(xs)
    if n <= k + 2:
        return None
    m = sum(xs) / n
    return sum((xs[i] - m) * (xs[i + k] - m) for i in range(n - k)) / n


def _tempo_fit(residual_sections: list[list[float]]) -> tuple[float, float, float] | None:
    """Decompose flight residuals into a slow drifting tempo plus fast trial noise.

    The model is flight = base * digraph * tempo_t * noise, where tempo follows an
    Ornstein-Uhlenbeck process. In logs that is an AR(1) signal buried in white noise, and
    the standard identification applies: for x = s + n with s AR(1) of coefficient phi,

        gamma_x(k) = var_s * phi^k   for all k >= 1      (the noise only touches lag 0)

    so phi = gamma(2)/gamma(1), immune to the noise that flattens the raw lag-1
    autocorrelation to ~0 and produced the original constant-tempo bug.

    Returns (theta, sigma, trial_noise_var), or None when the participant shows no
    detectable tempo structure — in which case the honest output is a missing value, not a
    default. Two payoffs from one fit: tempo parameters, and a trial-noise variance that is
    NOT contaminated by tempo drift. The old flight_sigma took the total log-spread, so it
    double-counted the very variance tempo re-adds at generation time."""
    LAGS = 5
    tot = [0.0] * (LAGS + 1)
    cnt = [0] * (LAGS + 1)
    for r in residual_sections:
        for k in range(LAGS + 1):
            v = _acov(r, k)
            if v is not None:
                tot[k] += v
                cnt[k] += 1
    if not all(cnt[:3]):
        return None
    g = [tot[k] / cnt[k] if cnt[k] else 0.0 for k in range(LAGS + 1)]
    g0 = g[0]
    if g0 <= 0 or g[1] <= 1e-9:
        return None

    # Fit phi across every usable lag rather than the single ratio gamma(2)/gamma(1).
    # log gamma(k) = log(var_s) + k*log(phi) for k >= 1, so a least-squares line through the
    # positive lags gives the slope. Lag-2 alone is a small, noisy difference and frequently
    # comes out negative, which threw away three quarters of participants.
    pts = [(k, math.log(g[k])) for k in range(1, LAGS + 1) if g[k] > 1e-12]
    if len(pts) < 2:
        return None
    mk = sum(k for k, _ in pts) / len(pts)
    mv = sum(v for _, v in pts) / len(pts)
    den = sum((k - mk) ** 2 for k, _ in pts)
    if den <= 0:
        return None
    slope = sum((k - mk) * (v - mv) for k, v in pts) / den
    phi = math.exp(slope)
    if not (0.02 < phi < 0.98):
        return None
    var_s = math.exp(mv - slope * mk)         # intercept at k=0 is log(var_s)
    var_noise = g0 - var_s                 # what is left is per-keystroke trial noise
    if var_noise <= 1e-6 or var_s <= 1e-9:
        return None
    theta = 1.0 - phi
    sigma = math.sqrt(var_s * (1 - phi * phi))     # OU innovation from stationary variance
    return theta, sigma, var_noise


def vector_from_sections(sections: list[list[tuple]], *, min_keys: int = 150) -> dict | None:
    """PURE: one participant's keystrokes (grouped by typed sentence) -> one motor vector.

    Each keystroke is (press_ms, release_ms, letter, keycode). Returns None if the person
    typed too little to estimate from. Optional fields are None when unsupported by evidence.

    Data-hygiene rules that matter more than they look:
      * Flights are press->press WITHIN a section, never across. The gap between sentences is
        the participant reading the next prompt, which is not a typing latency.
      * Modifier keys are skipped but do NOT break the chain, so flights stay letter->letter.
        Counting SHIFT as its own keystroke halves every measured flight around a capital.
      * A backspace DOES break the chain — the flight across a correction is a different
        motor act — and it drives error_rate.
      * Shift cost is measured only against non-space predecessors. Capitals cluster at word
        starts, so comparing all-shifted against all-unshifted silently absorbs the
        word-boundary pause: it reads 184ms that way and 124ms done properly."""
    flights: list[float] = []
    dwells: list[float] = []
    bigrams: dict[str, list[float]] = {}
    resid_sections: list[list[float]] = []
    # Four buckets keyed by (predecessor was a space, target needs shift). Comparing only
    # within a predecessor class is what keeps the word-boundary pause out of the shift
    # estimate, and pooling BOTH matched comparisons is what keeps the yield up — capitals
    # overwhelmingly follow spaces, so the non-space bucket alone is thin.
    buckets: dict[tuple[bool, bool], list[float]] = {
        (False, False): [], (False, True): [], (True, False): [], (True, True): []}
    n_keys = n_back = 0

    raw_sections: list[list[tuple]] = []
    for sec in sections:
        prev_press = prev_letter = None
        seq: list[tuple] = []
        for press, release, letter, keycode in sec:
            kc, letter = str(keycode), str(letter)
            if kc in _BACKSPACE or letter.lower() in _BACKSPACE:
                n_keys += 1
                n_back += 1
                prev_press = prev_letter = None       # a correction breaks the chain
                continue
            if len(letter) != 1:                      # SHIFT/CTRL/ENTER — skip, keep chain
                continue
            n_keys += 1
            d = release - press
            if 10 <= d <= 1000:
                dwells.append(d)
            if prev_press is not None:
                f = press - prev_press
                if 0 < f <= 2000:                     # drop afk gaps and non-positive
                    flights.append(f)
                    seq.append((f, prev_letter, letter))
                    bigrams.setdefault(prev_letter + letter, []).append(f)
                    buckets[(prev_letter == " ",
                             letter.isupper() or letter in _SHIFT_SYM)].append(f)
            prev_press, prev_letter = press, letter
        if len(seq) >= 8:
            raw_sections.append(seq)

    if n_keys < min_keys or len(flights) < 50:
        return None

    base_ms = st.median(flights)

    # residual = log(flight) - log(base * digraph) = log(tempo) + trial noise
    for seq in raw_sections:
        resid_sections.append([math.log(f) - math.log(base_ms * digraph_mult(a, b))
                               for f, a, b in seq])

    fit = _tempo_fit(resid_sections)
    if fit is not None:
        theta, tsigma, var_noise = fit
        flight_sigma = math.sqrt(var_noise)
    else:
        theta = tsigma = None                          # no evidence -> no value
        flight_sigma = _log_std(flights)

    plain, sp_plain = buckets[(False, False)], buckets[(True, False)]
    sh, sp_sh = buckets[(False, True)], buckets[(True, True)]

    # word-boundary pause: the model adds lognorm(think_ms * 0.5) after a space, so the
    # measured extra is half of think_ms
    think = None
    if len(sp_plain) >= 5 and len(plain) >= 10:
        think = 2.0 * (st.median(sp_plain) - st.median(plain))

    # shift reach: extra flight into a shifted key. Two independent matched comparisons —
    # within non-space predecessors and within space predecessors — pooled by sample count.
    est: list[tuple[float, int]] = []
    if len(sh) >= 4 and len(plain) >= 10:
        est.append((st.median(sh) - st.median(plain), len(sh)))
    if len(sp_sh) >= 4 and len(sp_plain) >= 10:
        est.append((st.median(sp_sh) - st.median(sp_plain), len(sp_sh)))
    shift = (sum(v * w for v, w in est) / sum(w for _, w in est)) if est else None

    def _opt(x, field):
        return None if x is None else round(_clamp(x, *RANGES[field]), 4)

    return {
        "base_ms": round(_clamp(base_ms, *RANGES["base_ms"]), 1),
        "flight_sigma": round(_clamp(flight_sigma, *RANGES["flight_sigma"]), 4),
        "dwell_ms": round(_clamp(st.median(dwells) if dwells else 85, *RANGES["dwell_ms"]), 1),
        "dwell_sigma": round(_clamp(_log_std(dwells) if dwells else 0.2,
                                    *RANGES["dwell_sigma"]), 4),
        "error_rate": round(_clamp(n_back / max(n_keys, 1), *RANGES["error_rate"]), 4),
        "think_ms": _opt(think, "think_ms"),
        "shift_ms": _opt(shift, "shift_ms"),
        "tempo_theta": _opt(theta, "tempo_theta"),
        "tempo_sigma": _opt(tsigma, "tempo_sigma"),
        "bigram_ms": {bg: round(st.median(fs), 1) for bg, fs in bigrams.items() if len(fs) >= 5},
    }


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 5:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else 0.0


def _split_half(halves: list[tuple[dict, dict]], fields, *, min_pairs: int = 30) -> dict:
    """Split-half reliability per field, Spearman-Brown corrected. None when there are too
    few paired estimates to say anything — which is NOT the same as a measured zero, and the
    sampler treats the two differently."""
    out = {}
    for f in fields:
        pairs = [(x[f], y[f]) for x, y in halves
                 if x.get(f) is not None and y.get(f) is not None]
        if len(pairs) < min_pairs:
            out[f] = None
            continue
        r = _pearson([p[0] for p in pairs], [p[1] for p in pairs])
        out[f] = round(max(0.0, min(1.0, 2 * r / (1 + r) if r > 0 else 0.0)), 4)
    return out


def reliability(halves: list[tuple[dict, dict]]) -> dict[str, float]:
    """Split-half reliability per field, Spearman-Brown corrected.

    Each participant is estimated twice, on odd and on even sentences. If the two estimates
    agree across people the field measures a stable trait; if they don't, the spread across
    participants is estimation noise and propagating it would manufacture fake diversity.
    Spearman-Brown corrects the half-length penalty back to full-length reliability.

    This is what tells `base_ms` (reliable, hundreds of samples per person) apart from
    `tempo_theta` (0.08 — real structure, but not resolvable per person from ~500 keys)."""
    out: dict[str, float] = {}
    for f in RANGES:
        a = [(x[f], y[f]) for x, y in halves if x.get(f) is not None and y.get(f) is not None]
        if len(a) < 30:
            out[f] = None          # unknown, NOT zero — the distinction changes what the
            continue               # sampler does with the field
        r = _pearson([p[0] for p in a], [p[1] for p in a])
        out[f] = round(max(0.0, min(1.0, 2 * r / (1 + r) if r > 0 else 0.0)), 4)
    return out


# ── extraction driver ─────────────────────────────────────────────────────────
# Aalto carries the full prompt and the full typed response on EVERY keystroke row, and a
# handful of participants pasted something enormous into the input box. The default 128KB
# field cap aborts the whole read on those.
csv.field_size_limit(16_000_000)


def _parse_tsv(fileobj):
    c = COLUMNS
    reader = csv.DictReader(fileobj, delimiter="\t")
    while True:
        try:
            row = next(reader)
        except StopIteration:
            return
        except csv.Error:            # unparseable line — abandon this file, keep the rest
            return
        try:
            yield (row[c["participant"]], row[c["section"]],
                   float(row[c["press"]]), float(row[c["release"]]),
                   row[c["letter"]], row[c["keycode"]])
        except (KeyError, ValueError, TypeError):     # ragged or blank row
            continue


def build_from_aalto(inputs: list[str], out_path: str, *, limit: int | None = None,
                     min_keys: int = 150, from_zip: bool = False,
                     progress=None) -> int:
    """Extract a corpus from the Aalto dataset. Returns the number of vectors written.

    `from_zip` reads participant files straight out of Keystrokes.zip one at a time, so peak
    disk stays at the 1.4GB archive instead of the ~16GB the unpacked set needs.

    Every participant is vectorized three times — full, odd sentences, even sentences — so
    the corpus can ship measured reliability alongside the values themselves."""
    vectors: list[dict] = []
    halves: list[tuple[dict, dict]] = []

    def ingest(secs: list[list[tuple]]) -> None:
        v = vector_from_sections(secs, min_keys=min_keys)
        if v is None:
            return
        vectors.append(v)
        if len(secs) >= 6:
            a = vector_from_sections(secs[0::2], min_keys=min_keys // 2)
            b = vector_from_sections(secs[1::2], min_keys=min_keys // 2)
            if a and b:
                halves.append((a, b))

    def group(rows) -> dict[str, dict[str, list[tuple]]]:
        people: dict[str, dict[str, list[tuple]]] = {}
        for pid, sec, press, rel, letter, kc in rows:
            people.setdefault(pid, {}).setdefault(sec, []).append((press, rel, letter, kc))
        return people

    if from_zip:
        import zipfile
        with zipfile.ZipFile(inputs[0]) as z:
            members = [m for m in z.namelist()
                       if m.lower().endswith(".txt") and "readme" not in m.lower()
                       and not m.endswith("/")]
            for i, m in enumerate(members):
                if limit and len(vectors) >= limit:
                    break
                if progress and i % 500 == 0:
                    progress(len(vectors), i, len(members))
                with z.open(m) as fh:
                    stream = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
                    for secs in group(_parse_tsv(stream)).values():
                        ingest(list(secs.values()))
    else:
        people: dict[str, dict[str, list[tuple]]] = {}
        for path in inputs:
            with open(path, newline="", encoding="utf-8", errors="replace") as f:
                for pid, sec, press, rel, letter, kc in _parse_tsv(f):
                    if limit and pid not in people and len(people) >= limit:
                        continue
                    people.setdefault(pid, {}).setdefault(sec, []).append(
                        (press, rel, letter, kc))
        for secs in people.values():
            ingest(list(secs.values()))

    payload = {
        "meta": {
            "source": "Aalto 136M Keystrokes (Dhakal et al. 2018)",
            "n": len(vectors),
            "min_keys": min_keys,
            "reliability": reliability(halves),
            "reliability_n": len(halves),
        },
        "vectors": vectors,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return len(vectors)


# ── mouse: SapiMouse -> pointer vectors ───────────────────────────────────────
# SapiMouse (Antal, Fejer & Buza 2020) — 120 subjects, two sessions each, event-driven
# sampling of [timestamp, button, state, x, y]. Public, and the raw logs carry timestamps
# and true coordinates, which the preprocessed |dx|,|dy| release on GitHub does not: absolute
# first differences destroy direction, so curvature and overshoot are unrecoverable from it.
_PAUSE_MS = 250.0     # a gap this long ends a movement (arrival, or a new intention)
_MIN_DIST = 40.0      # below this it is a micro-adjustment, not a reach
_MIN_PTS = 5          # too few samples to have a measurable shape


def _segment(events: list[tuple]) -> list[tuple[list, float | None]]:
    """Split an event log into pointer movements, each optionally ending at a click.

    A movement is a run of Move samples terminated by a button press (so the click is the
    target and the gap before it is the settle), by a long pause, or by a drag. Drags are
    excluded — dragging is a different motor act with the button held, and its dynamics do
    not describe a reach-and-click."""
    segs: list[tuple[list, float | None]] = []
    cur: list[tuple] = []
    for t, x, y, state in events:
        if state == "Move":
            if cur and t - cur[-1][0] > _PAUSE_MS:
                segs.append((cur, None))
                cur = []
            cur.append((t, x, y))
        elif state == "Pressed":
            if cur:
                segs.append((cur, t))
                cur = []
        else:                                    # Drag / Released — end without a target
            if cur:
                segs.append((cur, None))
                cur = []
    if cur:
        segs.append((cur, None))
    return segs


def _movement_features(pts: list[tuple], click_t: float | None) -> dict | None:
    """One movement -> the parameters `mouse.trajectory()` consumes.

    Each formula INVERTS the generator rather than measuring something adjacent, so feeding
    the results back in reproduces the measured motion:
      speed     inverts the time model  dt_total = speed * (50 + dist*0.34)
      curve     inverts the sine-arc bow, whose mean absolute magnitude is dist*curve/4
      overshoot is the rate at which the path projects past the endpoint and comes back
      tremor    is the median deviation from the local midpoint (high-frequency jitter)
      settle_ms is the pause between arriving and pressing"""
    (t0, x0, y0), (t1, x1, y1) = pts[0], pts[-1]
    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy)
    if dist < _MIN_DIST or len(pts) < _MIN_PTS:
        return None
    move_ms = t1 - t0
    if not (30 <= move_ms <= 8000):              # stalled or absurd — not one reach
        return None

    ux, uy = dx / dist, dy / dist
    perp = 0.0
    projs = []
    for _, x, y in pts:
        vx, vy = x - x0, y - y0
        projs.append(vx * ux + vy * uy)
        perp = max(perp, abs(vx * -uy + vy * ux))
    max_proj = max(projs)
    # An overshoot is a CORRECTIVE submovement, so the furthest point has to occur near the
    # end of the path. Without that constraint the measure also counts a path that merely
    # wandered through a further point mid-flight, which is a different motor event: it
    # reports a 32.6% rate at 42px median, versus a genuine 10.0% at 18px. The generator
    # produces corrections, so the estimator has to measure corrections.
    late = projs.index(max_proj) >= 0.7 * len(projs)

    tremor = [math.hypot(pts[i][1] - (pts[i - 1][1] + pts[i + 1][1]) / 2,
                         pts[i][2] - (pts[i - 1][2] + pts[i + 1][2]) / 2)
              for i in range(1, len(pts) - 1)]

    settle = click_t - t1 if click_t is not None else None
    return {
        "speed": move_ms / (50 + dist * 0.34),
        "curve": 4 * perp / dist,
        "overshoot": 1.0 if (late and max_proj > dist + OVERSHOOT_PX) else 0.0,
        "overshoot_px": (max_proj - dist) if (late and max_proj > dist + OVERSHOOT_PX) else None,
        "tremor": st.median(tremor) if tremor else 0.0,
        "settle": settle if (settle is not None and 0 <= settle <= 1500) else None,
    }


def _mouse_vector(rows: list[dict], *, min_moves: int = 15) -> dict | None:
    """Aggregate one user's movements into a single pointer vector."""
    if len(rows) < min_moves:
        return None
    settles = [r["settle"] for r in rows if r["settle"] is not None]
    overs = [r["overshoot_px"] for r in rows if r["overshoot_px"] is not None]
    v = {
        "mouse_speed": st.median([r["speed"] for r in rows]),
        "mouse_curve": st.median([r["curve"] for r in rows]),
        "mouse_overshoot": sum(r["overshoot"] for r in rows) / len(rows),
        "mouse_overshoot_px": st.median(overs) if len(overs) >= 3 else None,
        "mouse_tremor": st.median([r["tremor"] for r in rows]),
        "mouse_settle_ms": st.median(settles) if len(settles) >= 3 else None,
    }
    return {k: (None if x is None else round(_clamp(x, *MOUSE_RANGES[k]), 4))
            for k, x in v.items()}


def build_from_sapimouse(zip_path: str, out_path: str, *, min_moves: int = 15) -> int:
    """Extract a pointer corpus from the SapiMouse raw archive. Returns users written.

    Download: https://ms.sapientia.ro/~manyi/sapimouse/sapimouse.zip"""
    import zipfile
    from behaviourforge.mouse import FIELDS as MOUSE_FIELDS

    per_user: dict[str, list[dict]] = {}
    clicks: dict[str, list[float]] = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.lower().endswith(".csv"):
                continue
            parts = [p for p in name.split("/") if p]
            user = parts[1] if len(parts) > 2 else parts[0]
            with z.open(name) as fh:
                rd = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"))
                events = []
                for r in rd:
                    try:
                        events.append((float(r["client timestamp"]), float(r["x"]),
                                       float(r["y"]), r["state"]))
                    except (KeyError, ValueError, TypeError):
                        continue
            # Button hold time, straight from the press/release pairs. This is a separate
            # pass from the movement segmentation because a click is not a trajectory.
            down = None
            for t, _x, _y, state in events:
                if state == "Pressed":
                    down = t
                elif state == "Released" and down is not None:
                    held = t - down
                    # A hold past CLICK_MAX_MS is a press-and-hold, not a click.
                    if 10 <= held <= CLICK_MAX_MS:
                        clicks.setdefault(user, []).append(held)
                    down = None

            for pts, click_t in _segment(events):
                f = _movement_features(pts, click_t)
                if f:
                    per_user.setdefault(user, []).append(f)

    vectors, halves = [], []
    for user, rows in per_user.items():
        v = _mouse_vector(rows, min_moves=min_moves)
        if v is None:
            continue
        held = clicks.get(user, [])
        v["mouse_click_ms"] = (round(_clamp(st.median(held), *MOUSE_RANGES["mouse_click_ms"]), 4)
                               if len(held) >= 10 else None)
        vectors.append(v)
        a = _mouse_vector(rows[0::2], min_moves=min_moves // 2)
        b = _mouse_vector(rows[1::2], min_moves=min_moves // 2)
        if a and b and len(held) >= 20:
            a["mouse_click_ms"] = round(st.median(held[0::2]), 4)
            b["mouse_click_ms"] = round(st.median(held[1::2]), 4)
        if a and b:
            halves.append((a, b))

    rel = _split_half(halves, MOUSE_FIELDS)

    payload = {"meta": {"source": "SapiMouse (Antal, Fejer & Buza 2020)", "n": len(vectors),
                        "reliability": rel, "reliability_n": len(halves)},
               "vectors": vectors}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return len(vectors)


def build_from_balabit(zip_path: str, out_path: str, *, min_events: int = 200) -> int:
    """Extract scroll and idle behaviour from the Balabit Mouse Dynamics Challenge.

    Balabit is used here rather than SapiMouse for two specific reasons. It logs wheel events
    (`button=Scroll`, state Down/Up), which SapiMouse does not record at all. And it captures
    people doing their ordinary work, whereas SapiMouse participants were instructed to
    perform as many operations as possible — which makes SapiMouse pauses a model of hurrying
    rather than of resting, and idle behaviour is precisely about resting.

    The wheel log carries no pixel delta, only notch events, so what is extracted is the part
    that belongs to the human: the rhythm of the notches and how often direction reverses.
    Pixels-per-notch is a browser setting and lives in `mouse.PX_PER_NOTCH`.

    Caveat worth stating plainly: 10 users. That is thin for a population model — far thinner
    than Aalto's 6,000 or SapiMouse's 120 — so scroll and idle parameters are real but their
    between-person spread is poorly pinned. `describe()` reports the coverage."""
    import zipfile
    from behaviourforge.mouse import SCROLL_FIELDS, SCROLL_RANGES

    STILL_MS = 400.0        # a gap this long means the hand came to rest
    DRIFT_BREAK = 250.0     # gap that ends a micro-movement run

    per_user: dict[str, list[dict]] = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if "/session_" not in name or name.endswith("/"):
                continue
            user = name.split("/")[-2]
            # Accumulate PER SESSION, not per user. The split-half below then divides whole
            # sessions between the halves. Splitting alternating events instead puts two
            # halves of the same scroll burst on opposite sides, which correlates trivially
            # and reports a reliability near 1.0 for everything — measuring the estimator's
            # self-agreement rather than whether the trait is stable across occasions.
            acc = {"gaps": [], "rev": [], "pause": [], "drift": []}
            per_user.setdefault(user, []).append(acc)
            with z.open(name) as fh:
                ev, scroll_ev = [], []
                rd = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"))
                for r in rd:
                    try:
                        t = float(r["client timestamp"])
                        if r.get("button") == "Scroll":
                            scroll_ev.append((t, r.get("state")))
                        ev.append((t, float(r["x"]), float(r["y"]), r.get("state")))
                    except (KeyError, ValueError, TypeError):
                        continue

            for a, b in zip(scroll_ev, scroll_ev[1:]):
                g = (b[0] - a[0]) * 1000
                if 0 < g < 30_000:
                    acc["gaps"].append(g)
                    acc["rev"].append(1.0 if a[1] != b[1] else 0.0)

            for a, b in zip(ev, ev[1:]):
                g = (b[0] - a[0]) * 1000
                if STILL_MS < g < 300_000:
                    acc["pause"].append(g)

            run: list = []
            for e in ev:
                if e[3] != "Move":
                    run = []
                    continue
                if run and (e[0] - run[-1][0]) * 1000 > DRIFT_BREAK:
                    if len(run) >= 2:
                        d = math.hypot(run[-1][1] - run[0][1], run[-1][2] - run[0][2])
                        if 2 < d < 200:
                            acc["drift"].append(d)
                    run = []
                run.append(e)

    vectors, halves = [], []

    def vec(gaps, rev, pause, drift) -> dict | None:
        if len(gaps) < min_events or len(pause) < 30:
            return None
        v = {
            "scroll_gap_ms": st.median(gaps),
            "scroll_gap_sigma": _log_std(gaps),
            "scroll_reversal": sum(rev) / len(rev),
            "idle_pause_ms": st.median(pause),
            "idle_pause_sigma": _log_std(pause),
            "idle_drift_px": st.median(drift) if len(drift) >= 10 else None,
        }
        return {k: (None if x is None else round(_clamp(x, *SCROLL_RANGES[k]), 4))
                for k, x in v.items()}

    def merge(sessions):
        out = {"gaps": [], "rev": [], "pause": [], "drift": []}
        for s_ in sessions:
            for k in out:
                out[k].extend(s_[k])
        return out

    for sessions in per_user.values():
        allm = merge(sessions)
        v = vec(allm["gaps"], allm["rev"], allm["pause"], allm["drift"])
        if v is None:
            continue
        vectors.append(v)
        if len(sessions) >= 4:                     # split whole sessions between the halves
            ha, hb = merge(sessions[0::2]), merge(sessions[1::2])
            a = vec(ha["gaps"], ha["rev"], ha["pause"], ha["drift"])
            b = vec(hb["gaps"], hb["rev"], hb["pause"], hb["drift"])
            if a and b:
                halves.append((a, b))

    rel = _split_half(halves, SCROLL_FIELDS, min_pairs=8)  # n=10 users: noisy

    payload = {"meta": {"source": "Balabit Mouse Dynamics Challenge", "n": len(vectors),
                        "reliability": rel, "reliability_n": len(halves)},
               "vectors": vectors}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return len(vectors)


def load_scroll(path: str | None = None) -> dict | None:
    """Load the scroll/idle corpus, or None if absent."""
    return load_mouse(path or BUNDLED_SCROLL)


def load_mouse(path: str | None = None) -> dict | None:
    """Load the pointer corpus, or None if absent (callers fall back to a fixed vector)."""
    p = path or BUNDLED_MOUSE
    hit = _cache.get(p)
    if hit is not None:
        return hit
    try:
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return None
    result = raw if isinstance(raw, dict) else {"vectors": raw, "meta": {}}
    _cache[p] = result
    return result


__all__ = ["load", "load_mouse", "load_scroll", "build_from_balabit", "describe", "reliability", "build_from_aalto",
           "build_from_sapimouse", "vector_from_sections", "BUNDLED", "BUNDLED_MOUSE",
           "COLUMNS", "OPTIONAL"]
