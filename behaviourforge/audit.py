"""audit(n) — measure whether a fleet of n personas is actually a fleet.

The failure this exists to catch is invisible from any single session. Each persona can be
individually flawless while the POPULATION gives the whole thing away, in four ways:

  1. Collision   — two agents forged the same identity. Two sessions with identical motor
                   parameters are provably one operator.
  2. Degeneracy  — a single PARAMETER is constant across the fleet. Subtler than a full
                   collision and just as fatal: 10,000 agents sharing an exact tempo is a
                   correlatable signature even when every other field differs.
  3. Clustering  — personas pile onto a handful of attractors. A finite set of presets is
                   correlatable, and the discreteness itself is something real traffic cannot
                   produce: humans do not come in five flavours.
  4. Drift       — the fleet's aggregate distribution stops matching the population it was
                   sampled from. Every persona passes alone; the histogram of ten thousand
                   does not.

This is not decoration. The degeneracy check is here because the first version of this file
did not have one, and it therefore missed that every persona in the fleet was being handed an
identical `tempo_theta` — the autocorrelation structure, which is the one property IID random
delays cannot fake, was a constant. Sampling correctly is a claim; this turns it into a
number, so "10,000 agents don't move alike" is something you check rather than assert.
"""
from __future__ import annotations

import math
import statistics as st
from collections import Counter
from dataclasses import dataclass, field

from behaviourforge import corpus as _corpus
from behaviourforge.keystroke import RANGES
from behaviourforge.mouse import FIELDS as MOUSE_FIELDS
from behaviourforge.mouse import SCROLL_FIELDS
from behaviourforge.persona import forge

ALL_FIELDS = tuple(RANGES) + MOUSE_FIELDS + SCROLL_FIELDS

# Pairwise nearest-neighbour is O(n^2), so it runs on a subsample. Clustering is a property
# of the sampler, not of the sample size — if personas collapse onto attractors, a few
# hundred draws reveal it. The report states the subsample size rather than implying
# full coverage.
_NN_SUBSAMPLE = 400

# A field is degenerate if the fleet concentrates on one value MORE than the source
# population does. The comparison has to be relative: real distributions contain genuine
# atoms, and reproducing them is correct rather than a defect. 5.7% of Aalto participants
# have think_ms of exactly 0 (no word-boundary pause at all), and SapiMouse's integer
# coordinates quantize tremor medians onto repeated values. A fleet matching those is
# faithful; a fleet inventing its own spike is not.
MAX_MODE_SHARE = 0.05
MODE_TOLERANCE = 1.5      # allowed multiple of the corpus's own concentration
MIN_DISTINCT = 20         # absolute backstop: below this the field has collapsed
# Only KS-test fields reliable enough that shrinkage barely narrows them; below this the
# forged spread is intentionally tighter than the corpus and a KS gap is the design working.
KS_RELIABILITY_FLOOR = 0.80


def _ks(a: list[float], b: list[float]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: the largest gap between two empirical CDFs.
    0 is identical, 1 is disjoint. Distribution-free, which is what we want — nobody knows
    the true shape of the human population, only the sample."""
    a, b = sorted(a), sorted(b)
    na, nb = len(a), len(b)
    if not na or not nb:
        return 1.0
    i = j = 0
    fa = fb = d = 0.0
    while i < na and j < nb:
        if a[i] <= b[j]:
            i += 1
            fa = i / na
        else:
            j += 1
            fb = j / nb
        d = max(d, abs(fa - fb))
    return d


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else 0.0


@dataclass
class FieldReport:
    name: str
    distinct: int
    mode_share: float
    reliability: float | None
    ks: float | None          # None when not checked (see KS_RELIABILITY_FLOOR)


@dataclass
class AuditReport:
    n: int
    model: str
    collisions: int
    nn_min: float
    nn_p01: float
    nn_median: float
    nn_sampled: int
    fields: list[FieldReport]
    corr_corpus: float
    corr_forged: float
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def __str__(self) -> str:
        checked = [f for f in self.fields if f.ks is not None]
        worst = max(checked, key=lambda f: f.ks) if checked else None
        lines = [
            f"BehaviourForge fleet audit — n={self.n}, model={self.model}",
            "",
            f"  collisions        {self.collisions}  (identical motor vectors)",
            f"  nearest-neighbour min {self.nn_min:.4f}  p01 {self.nn_p01:.4f}  "
            f"median {self.nn_median:.4f}   [sd units, {self.nn_sampled} sampled]",
        ]
        if worst:
            lines.append(f"  distribution      worst KS {worst.ks:.4f} on {worst.name}")
        lines += [
            f"  correlation       base~dwell  corpus {self.corr_corpus:+.3f}  "
            f"forged {self.corr_forged:+.3f}  (delta {self.corr_forged - self.corr_corpus:+.3f})",
            "",
            f"    {'field':<17}{'distinct':>9}{'mode':>8}{'reliab':>9}{'KS':>9}",
        ]
        for f in self.fields:
            rel = "  n/a" if f.reliability is None else f"{f.reliability:.3f}"
            ks = "  skip" if f.ks is None else f"{f.ks:.4f}"
            lines.append(f"    {f.name:<17}{f.distinct:>9}{f.mode_share:>8.1%}{rel:>9}{ks:>9}")
        lines.append("")
        if self.passed:
            lines.append("  PASS — no collisions, no constant parameters, distribution and "
                         "correlation preserved.")
        else:
            lines.append("  FAIL")
            lines.extend(f"    - {r}" for r in self.failures)
        return "\n".join(lines)


def audit(n: int = 10_000, *, model: str = "copula", corpus=None, mouse_corpus=None,
          scroll_corpus=None, seed_offset: int = 0, max_ks: float = 0.10,
          max_corr_delta: float = 0.15) -> AuditReport:
    """Forge n personas and measure the population they form.

    Seeds are `seed_offset + i`, deliberately adjacent, because that is the realistic worst
    case: a fleet keyed by agent index. Sequential seeds are where a weak sampler leaks — if
    consecutive integers produce neighbouring personas, agents 1 and 2 are visibly related.
    Random seeds would flatter the result."""
    ks_corpus = corpus if isinstance(corpus, dict) else _corpus.load(corpus)
    mouse = mouse_corpus if isinstance(mouse_corpus, dict) else _corpus.load_mouse(mouse_corpus)
    scr = scroll_corpus if isinstance(scroll_corpus, dict) else _corpus.load_scroll(scroll_corpus)
    rel = dict(ks_corpus.get("meta", {}).get("reliability", {}))
    for extra in (mouse, scr):
        if extra:
            rel.update(extra.get("meta", {}).get("reliability", {}))

    motors = [forge(seed_offset + i, corpus=ks_corpus, mouse_corpus=mouse, scroll_corpus=scr,
                    model=model).motor for i in range(n)]
    failures: list[str] = []

    # 1. collisions — exact repeats of the full parameter vector
    sigs = {tuple(round(getattr(m, f), 6) for f in ALL_FIELDS) for m in motors}
    collisions = n - len(sigs)
    if collisions:
        failures.append(f"{collisions} identical personas across {n} seeds")

    # 2. degeneracy + drift, per field
    src = {f: [v[f] for v in ks_corpus["vectors"] if v.get(f) is not None] for f in RANGES}
    if mouse:
        src.update({f: [v[f] for v in mouse["vectors"] if v.get(f) is not None]
                    for f in MOUSE_FIELDS})
    if scr:
        src.update({f: [v[f] for v in scr["vectors"] if v.get(f) is not None]
                    for f in SCROLL_FIELDS})

    reports: list[FieldReport] = []
    for f in ALL_FIELDS:
        vals = [getattr(m, f) for m in motors]
        counts = Counter(vals)
        mode_share = counts.most_common(1)[0][1] / n
        r = rel.get(f)
        do_ks = bool(src.get(f)) and r is not None and r >= KS_RELIABILITY_FLOOR
        ks = _ks(vals, src[f]) if do_ks else None
        # The KS threshold has to scale with the size of the REFERENCE corpus. Comparing a
        # continuous distribution against the empirical CDF of n samples cannot do better
        # than the Kolmogorov critical value 1.36/sqrt(n): against Aalto's 6000 that is 0.018
        # and the fixed 0.10 binds, but against Balabit's 10 users it is 0.43, and a fixed
        # 0.10 would fail every scroll field for being small rather than for being wrong.
        ks_limit = max(max_ks, 1.36 / math.sqrt(len(src[f]))) if do_ks else max_ks
        reports.append(FieldReport(f, len(counts), mode_share, r, ks))

        pool = src.get(f) or []
        corpus_mode = (Counter(pool).most_common(1)[0][1] / len(pool)) if pool else 0.0
        limit = max(MAX_MODE_SHARE, corpus_mode * MODE_TOLERANCE)
        if mode_share > limit or len(counts) < min(MIN_DISTINCT, n):
            failures.append(
                f"{f} is degenerate: {mode_share:.1%} of the fleet shares one exact value "
                f"(corpus itself: {corpus_mode:.1%}, {len(counts)} distinct)")
        if ks is not None and ks > ks_limit:
            failures.append(f"{f} drifted from the corpus (KS {ks:.3f} > {ks_limit:.3f})")

    # 3. clustering — nearest-neighbour distance in corpus-normalized space, so each
    #    parameter contributes on its own scale rather than whichever has the biggest units
    scale = {f: (st.pstdev(src[f]) or 1.0) if src.get(f) else 1.0 for f in ALL_FIELDS}
    pts = [[getattr(m, f) / scale[f] for f in ALL_FIELDS] for m in motors[:_NN_SUBSAMPLE]]
    nn: list[float] = []
    for i, a in enumerate(pts):
        best = float("inf")
        for j, b in enumerate(pts):
            if i == j:
                continue
            d = 0.0
            for k in range(len(a)):
                d += (a[k] - b[k]) ** 2
                if d >= best:
                    break                       # early exit — most pairs lose fast
            best = min(best, d)
        nn.append(math.sqrt(best))
    nn.sort()

    # 4. correlation structure — does sampling distort the couplings the corpus encodes?
    #    base~dwell is the diagnostic pair: weak in real humans (~0.15), and exactly the
    #    coupling intuition wants to make strong ("fast typist, short hold" is largely false).
    corr_corpus = _pearson([v["base_ms"] for v in ks_corpus["vectors"]],
                           [v["dwell_ms"] for v in ks_corpus["vectors"]])
    corr_forged = _pearson([m.base_ms for m in motors], [m.dwell_ms for m in motors])
    if abs(corr_forged - corr_corpus) > max_corr_delta:
        failures.append(
            f"base~dwell correlation distorted {corr_corpus:+.3f} -> {corr_forged:+.3f}")

    return AuditReport(
        n=n, model=model, collisions=collisions,
        nn_min=nn[0] if nn else 0.0,
        nn_p01=nn[max(0, len(nn) // 100)] if nn else 0.0,
        nn_median=st.median(nn) if nn else 0.0,
        nn_sampled=len(pts), fields=reports,
        corr_corpus=corr_corpus, corr_forged=corr_forged, failures=failures)


__all__ = ["audit", "AuditReport", "FieldReport", "ALL_FIELDS"]
