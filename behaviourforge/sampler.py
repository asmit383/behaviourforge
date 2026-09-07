"""The Gaussian copula — unlimited coherent identities from a finite corpus.

Bootstrap resampling (pick a real participant, jitter them slightly) is coherent by
construction and is the right MVP, but it has a hard ceiling exactly where this library
needs headroom. Drawing 10,000 agents from 6,000 participants reuses each real person ~1.7
times on average and one of them a dozen times, and jitter only smears those clones a few
percent apart. You cannot get more distinct identities out of resampling than the corpus has
rows.

A copula separates the two things a joint distribution is made of:

    marginals    — what values each parameter takes, one field at a time
    dependence   — how the fields move together

Fit the marginals empirically (so every value the sampler emits is one a real human
produced), capture the dependence as a correlation matrix in normal-score space, and you can
draw unlimited vectors that are new but structurally identical to the population. No agent is
a copy of a participant, and no agent is an impossible combination either.

── RELIABILITY SHRINKAGE ─────────────────────────────────────────────────────────────────
The observed spread of a field is true human variation PLUS estimation error, and those want
opposite treatment. Propagating the whole spread manufactures diversity that isn't real;
collapsing it to a constant manufactures a fleet signature. Classical test theory gives the
split: observed variance = true variance / reliability, so the honest spread to sample is
sqrt(reliability) of the observed one.

That is a genuine difference here, not a formality. `base_ms` has reliability 0.99 and keeps
essentially all its spread. `tempo_theta` has 0.14 — ~500 keystrokes cannot pin one person's
tempo drift — so most of its apparent between-person variation is noise and gets shrunk away.

── CROSS-MODAL DEPENDENCE ────────────────────────────────────────────────────────────────
No public dataset records the same person typing AND moving a pointer. Aalto has keystrokes
from 168k people, SapiMouse has pointer traces from 120, and they are disjoint. So the
keystroke-to-pointer correlation cannot be measured, only assumed — and an identity that
types in the 90th percentile for speed while mousing in the 10th is incoherent in exactly the
way a sensor fusing both would notice. `CROSS_MODAL_RHO` is that assumption, isolated in one
named constant rather than smuggled into a formula.
"""
from __future__ import annotations

import math
import random

from behaviourforge.keystroke import RANGES
from behaviourforge.mouse import FIELDS as MOUSE_FIELDS
from behaviourforge.mouse import MOUSE_RANGES, SCROLL_FIELDS, SCROLL_RANGES

KEY_FIELDS = tuple(RANGES)

# Assumed correlation between typing speed and pointer speed, in normal-score space. Not
# measured — see the module note. Modest on purpose: enough to keep an identity coherent,
# not so much that the two modalities become one signal.
CROSS_MODAL_RHO = 0.35

# Never shrink a field's spread below this fraction. Reliability is itself an estimate with
# error, and a zero-variance parameter is the one outcome known to be wrong: it hands every
# agent in the fleet an identical value.
MIN_SPREAD = 0.35

# Minimum observations before a field can be fitted at all. Low because the scroll corpus has
# only 10 users — thin, but real, and 10 interpolated marginal points still beat a hardcoded
# constant. Correlations are a different matter: a 6x6 matrix from 10 points is not estimable,
# so below MIN_CORR_N the fields are drawn independently rather than pretending to a
# dependence structure. Independent-but-real beats confidently-wrong.
MIN_FIT_N = 8
MIN_CORR_N = 20


# ── normal distribution helpers (stdlib only) ─────────────────────────────────
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation, ~1e-9 accurate)."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
               ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
                ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / \
           (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1)


def _cholesky(m: list[list[float]]) -> list[list[float]] | None:
    """Lower-triangular Cholesky factor, or None if the matrix is not positive definite."""
    n = len(m)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                d = m[i][i] - s
                if d <= 1e-12:
                    return None
                L[i][j] = math.sqrt(d)
            else:
                L[i][j] = (m[i][j] - s) / L[j][j]
    return L


def _quantile(sorted_vals: list[float], p: float) -> float:
    """Empirical inverse CDF with linear interpolation between order statistics."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = p * (n - 1)
    lo = int(pos)
    if lo >= n - 1:
        return sorted_vals[-1]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[lo + 1] * frac


class Copula:
    """Empirical marginals plus a normal-score correlation structure.

    Fitting is done once per corpus and cached, so forging is a handful of multiplications
    per agent — cheap enough that 10,000 personas cost milliseconds in total."""

    def __init__(self, fields, marginals, chol, spread, ranges):
        self.fields = fields
        self.marginals = marginals        # field -> sorted list of observed values
        self.chol = chol                  # lower-triangular factor of the correlation matrix
        self.spread = spread              # field -> sqrt(reliability), the retained fraction
        self.ranges = ranges

    @classmethod
    def fit(cls, vectors: list[dict], fields, reliability: dict, ranges) -> "Copula":
        fields = [f for f in fields
                  if sum(1 for v in vectors if v.get(f) is not None) >= MIN_FIT_N]
        if not fields:
            raise ValueError(
                f"corpus has fewer than {MIN_FIT_N} observations for every field — "
                "too thin to fit; supply a larger corpus")
        marginals = {f: sorted(v[f] for v in vectors if v.get(f) is not None) for f in fields}

        # Normal scores: replace each value by the normal quantile of its rank. This is what
        # makes the dependence structure independent of the marginal shapes.
        scores: dict[str, dict[int, float]] = {}
        for f in fields:
            present = [(v[f], i) for i, v in enumerate(vectors) if v.get(f) is not None]
            present.sort()
            n = len(present)
            scores[f] = {idx: _norm_ppf((r + 0.5) / n) for r, (_, idx) in enumerate(present)}

        k = len(fields)
        corr = [[1.0] * k for _ in range(k)]
        for a in range(k):
            for b in range(a + 1, k):
                sa, sb = scores[fields[a]], scores[fields[b]]
                both = [i for i in sa if i in sb]          # complete cases only
                if len(both) < MIN_CORR_N:
                    r = 0.0                                # not estimable — treat as independent
                else:
                    xs = [sa[i] for i in both]
                    ys = [sb[i] for i in both]
                    n = len(both)
                    mx, my = sum(xs) / n, sum(ys) / n
                    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
                    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
                    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
                    r = num / (dx * dy) if dx and dy else 0.0
                corr[a][b] = corr[b][a] = max(-0.99, min(0.99, r))

        # Pairwise correlations from differing complete-case subsets need not form a positive
        # definite matrix. Shrink the off-diagonals toward identity until they do — standard
        # practice, and it degrades gracefully toward independence rather than failing.
        chol = _cholesky(corr)
        w = 1.0
        while chol is None and w > 0.01:
            w *= 0.8
            shrunk = [[corr[i][j] * (w if i != j else 1.0) for j in range(k)] for i in range(k)]
            chol = _cholesky(shrunk)
        if chol is None:
            chol = [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]

        spread = {}
        for f in fields:
            rel = reliability.get(f)
            # unknown reliability -> no shrinkage; we have no evidence the spread is inflated
            s = 1.0 if rel is None else math.sqrt(max(0.0, min(1.0, rel)))
            spread[f] = max(MIN_SPREAD, s)
        return cls(fields, marginals, chol, spread, ranges)

    def draw(self, rng: random.Random, *, anchor: float | None = None) -> dict:
        """One vector. `anchor` is a normal score to correlate the first field with, used to
        tie pointer speed to typing speed across two datasets that share no participants."""
        k = len(self.fields)
        u = [rng.gauss(0, 1) for _ in range(k)]
        z = [sum(self.chol[i][j] * u[j] for j in range(i + 1)) for i in range(k)]
        if anchor is not None and k:
            z[0] = CROSS_MODAL_RHO * anchor + math.sqrt(1 - CROSS_MODAL_RHO ** 2) * z[0]

        out = {}
        for i, f in enumerate(self.fields):
            zs = z[i] * self.spread[f]                    # reliability shrinkage
            lo, hi = self.ranges[f]
            out[f] = min(hi, max(lo, _quantile(self.marginals[f], _norm_cdf(zs))))
        out["_z0"] = z[0] if k else 0.0                   # exposed so mouse can anchor to it
        return out


_fitted: dict[tuple, Copula] = {}


def get_copula(corpus: dict, kind: str) -> Copula:
    """Fit (and cache) the copula for a corpus. Keyed by identity, so repeated forges reuse
    the fit rather than re-deriving a correlation matrix per agent."""
    key = (id(corpus), kind)
    hit = _fitted.get(key)
    if hit is not None:
        return hit
    fields, ranges = {"key": (KEY_FIELDS, RANGES),
                      "mouse": (MOUSE_FIELDS, MOUSE_RANGES),
                      "scroll": (SCROLL_FIELDS, SCROLL_RANGES)}[kind]
    c = Copula.fit(corpus["vectors"], fields,
                   corpus.get("meta", {}).get("reliability", {}), ranges)
    _fitted[key] = c
    return c


__all__ = ["Copula", "get_copula", "CROSS_MODAL_RHO", "MIN_SPREAD", "KEY_FIELDS"]
