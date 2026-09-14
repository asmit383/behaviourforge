# behaviourforge

Human keyboard, mouse and scroll input for browser agents, with every timing constant
**measured from real human sessions** instead of guessed — and a **fleet** that stays diverse
when you run ten thousand of them.

```bash
pip install behaviourforge
```

```python
from behaviourforge import forge

p = forge(seed=4712)                 # one agent, one stable identity

for k in p.keystrokes("someone@example.com", first_field=True):
    ...                              # key, flight_ms, dwell_ms
for m in p.path(x0, y0, *p.point_in(button_box)):
    ...                              # x, y, dt_ms
```

The library emits timed **plans** and never imports a browser. Executing them is ~90 lines —
see [`examples/camoufox_driver.py`](examples/camoufox_driver.py) for Playwright/Camoufox.

## Why an agent needs this

An agent that reasons perfectly and clicks like a robot is still a robot. Every number below
is a surface a detector reads, and the defaults of every automation library are wrong on most
of them. Playwright clicks the exact centre of a bounding box and releases the button in
0.1ms. No hand does either, ever.

## Why this exists

The second problem is the one nobody models: **a fleet**. A persona that is individually
flawless can still give you away in aggregate — ten thousand agents sharing one exact tempo
value is correlatable even when every other field differs, and the *discreteness* of a preset
set is something real traffic cannot produce. So identities here are drawn from a fitted
population rather than resampled from a list, and `audit()` measures whether the fleet is
actually a fleet.

Every number in the table was measured by us, from the raw datasets or from a real browser.
Several are the opposite of what the intuition says.

| Measurement | Human | Naive bot | Note |
|---|---|---|---|
| Click dwell | med **86ms**, p90 162 | **0.1ms** | `mouse.click()` presses and releases with no hold |
| Click landing offset | sd **15.0** | 0.0, sd 0.0 | Playwright clicks bbox centre, every time |
| Key dwell | mean 109ms, sd 37 | — | |
| Key flight | mean 217ms, sd **218** | fixed delay | Typing needs *more* variance, not less |
| Flight autocorrelation | **+0.056** | 0.000 | IID delays have zero autocorrelation — that *is* the tell |
| Peak-velocity position | **0.26** | 0.50 | Any eased curve peaks dead centre; human velocity is asymmetric |
| Sub-movements per stroke | **2** | **17** with per-sample tremor | Noise added "to look organic" is the loudest tell |
| Path / straight line | 1.11, p90 **2.15** | 1.09, p90 **1.21** | Synthetic strokes are all equally direct |
| Pointer sample rate | **59Hz** | 79Hz | Over-sampling is a tell *and* an IPC cost |
| Scroll `deltaY` | whole ticks (40px) | any value | `wheelDeltaY` derives from the tick count, not from `deltaY` |
| Scroll direction reversal | **5.8%** | 20% | Bots re-read far more than people do |
| Idle pause | p50 **780ms**, p90 **4.2s** | `uniform(700, 2200)` | A bounded draw has no tail; every rest looks alike |
| `tempo_theta` across a fleet | thousands of values | **1** | A clamp turned a broken estimator into a constant |
| `getCoalescedEvents()` | mean 1.01, max 7 | hard 1, no variance | **Not modelled here yet** — see Known gaps |

Findings worth calling out because they are counter-intuitive:

- **Tremor is correlated, not white noise.** 2.03px of measured sample-to-sample deviation is
  equally true of white noise and of a smooth wobble — but white noise reverses direction
  every sample, and each reversal registers as a velocity peak. Real strokes have 2; ours had
  17. Switching to an AR(1) wander fixed it *without* recalibrating anything, because the
  amplitude is identical.
- **`base_ms` and `dwell_ms` are nearly independent** (r ≈ 0.16). "Fast typist, short key
  hold" is intuitive and essentially false. Hand-authored archetypes encode the intuition and
  produce r ≈ 0.96, which is how you can tell they were invented.
- **The digraph effect is far gentler than intuition.** Same-finger is **1.06×**, not the
  ~1.6× that obviously follows from finger reuse. The *ordering* (alternation < same-hand <
  same-finger) holds across 3,000 typists; the magnitudes did not.
- **Raw lag-1 autocorrelation cannot estimate tempo.** Trial noise attenuates it to ≈ 0.009,
  so `theta = 1 - acf` lands near 0.99 and saturates any sane clamp. Recovering it needs an
  AR(1)-plus-white-noise decomposition.
- **A clamp tight enough to bite turns a parameter into a constant.** This happened four
  separate times — `tempo_theta` had *one* distinct value across 3,000 people, and a 180ms
  dwell ceiling pinned one keystroke in nineteen to exactly 180.000ms. Both look like safety
  rails and behave like defects. Regression tests now guard the pattern.
- **Most published wheel tells are Chromium-specific.** Driving Firefox, the
  `wheelDeltaY`/`deltaY` ratio is a consistent −3.00, with no cancelable events and no
  per-event `scrollend`. Those are artefacts of `Input.dispatchMouseWheel`, not of automation.

## Why a seed is the whole scaling story

A persona is a pure function of an integer. Ten thousand agents need ten thousand integers —
no server minting identities, no shared state, no coordination. Agent 4,712 forges its own in
**44µs** and gets the same one back tomorrow.

```
$ behaviourforge audit -n 10000

  collisions        0
  nearest-neighbour min 1.6709  p01 1.9283  median 2.8773   [sd units]
  distribution      worst KS 0.0606 on mouse_tremor
  correlation       base~dwell  corpus +0.161  forged +0.135  (delta -0.026)

  PASS — no collisions, no constant parameters, distribution and correlation preserved.
```

`audit()` checks four population failures invisible from any single session: collisions,
per-field degeneracy, clustering, and distribution drift. It is not decoration — it exists
because the first version shipped a constant `tempo_theta` across the whole fleet, and
nothing else would have caught it.

Two samplers, both coherent. `copula` (default) fits empirical marginals plus the corpus
correlation matrix and draws unlimited distinct vectors. `bootstrap` resamples one real
participant. Bootstrap cannot emit more distinct identities than the corpus has rows — at
10,000 agents against 6,000 participants, one participant gets reused **12 times** — which is
why copula is the default.

## Data

| corpus | source | n | licence |
|---|---|---|---|
| keystroke | Aalto *136M Keystrokes* (Dhakal et al. 2018) | **6,000** | research use |
| pointer | SapiMouse (Antal, Fejer & Buza 2020) | **120** | Apache-2.0 |
| scroll + idle | Balabit Mouse Dynamics Challenge | **10** | research use |

Only extracted per-participant parameter vectors are redistributed, never raw files. The
extraction pipelines are in `corpus.py`, so every bundled corpus is reproducible.

Each field ships **split-half reliability**, Spearman-Brown corrected — each participant
estimated twice, on odd and even occasions, then correlated across people. It answers the
question that decides what to do with a field: is this spread real human variation, or my
estimation error?

| field | reliability | | field | reliability |
|---|---|---|---|---|
| `base_ms`, `dwell_ms` | **0.992** | | `mouse_click_ms` | **0.990** |
| `mouse_tremor` | **0.983** | | `idle_drift_px` | **0.977** |
| `think_ms` | **0.970** | | `scroll_gap_ms` | **0.974** |
| `mouse_speed` | **0.949** | | `idle_pause_ms` | **0.952** |
| `flight_sigma` | **0.898** | | `scroll_reversal` | **0.947** |
| `error_rate` | 0.809 | | `mouse_overshoot` | 0.746 |
| `shift_ms` | 0.552 | | `scroll_gap_sigma` | 0.377 |
| `tempo_sigma` | 0.221 | | `tempo_theta` | 0.143 |

The sampler shrinks each field's spread by `sqrt(reliability)`. Propagating the whole spread
manufactures diversity that isn't real; collapsing it to a constant manufactures a fleet
signature. `base_ms` keeps essentially all of its spread; `tempo_theta` keeps ~38%.

## Known gaps

Stated plainly, because a library that hides these is worse than one that has them.

- **`getCoalescedEvents()` is not modelled.** We emit a hard 1 where humans average 1.01.
  Confirmed achievable — pipelining moves so they arrive faster than a paint gives a measured
  mean of 6.0 — but not implemented.
- **Click landing offset is guessed.** `point_in` uses an interior band rather than a measured
  distribution. It is the last hand-authored motor constant in the library.
- **One fitted constant.** `TEMPO_SIGMA_SCALE` is calibrated against the observable rather
  than the latent estimate, because the fitted `tempo_sigma` reproduces the variance split but
  generates −0.004 autocorrelation where real participants show +0.056 on identical text. Two
  alternative explanations were tested and rejected first.
- **Scroll rests on 10 users**, and no `scrollend` fires where a real gesture produces one.
- **Cross-modal correlation is assumed.** No public dataset records the same person typing and
  mousing, so `CROSS_MODAL_RHO` is an assumption isolated in one named constant.
- **Validation is against the corpora we fitted to.** A held-out split is the first genuinely
  external check and has not been done.
- **Nothing here has been tested against a real detector.** Every number is
  distribution-matching, which is necessary and not sufficient.

## API

```python
forge(seed=None, *, locale="en-US", device="desktop", model="copula") -> Persona

p.keystrokes(text, *, first_field=False, typos=True)   -> list[Key]    # key, flight_ms, dwell_ms
p.path(x0, y0, x1, y1)                                 -> list[Move]   # x, y, dt_ms
p.scroll(notches=None)                                 -> list[Wheel]  # dy, dt_ms
p.idle(ms, x, y, *, width=1280, height=800)            -> list[Move]
p.point_in(box) -> (x, y)          p.click_hold_ms() -> float
p.motor.to_dict()                                                      # persist an identity

audit(n=10_000, *, model="copula") -> AuditReport
```

A persona is **stateful within a session**: keystroke tempo drifts continuously across calls,
so filling three fields carries one rhythm through all three. Keyboard and pointer use split
RNGs, so adding a mouse move never shifts keystroke timings — otherwise a session would not be
replayable.

```bash
behaviourforge show 42 --text "hello@example.com"   # inspect a persona and its timings
behaviourforge audit -n 10000                       # fleet report; exits 1 on FAIL
behaviourforge build-corpus Keystrokes.zip --zip --limit 6000
```

## Demos

```bash
python examples/demo_capture.py            # drive a real browser, measure what the page got
python examples/demo_store.py --headful    # log in, browse, scroll, add to cart
python examples/bench_probe.py             # head-to-head on agenthands' probe harness
```

Two things the driver encodes that are easy to get wrong: pass `humanize=False` to Camoufox
(its own humanizer re-curves every sample and reintroduces the fleet-identical signature), and
schedule events against an **absolute clock**. Sleeping between events inflates press-to-press
intervals by 69% and turns IPC jitter into *negative* autocorrelation, inverting the one
property random delays cannot fake.

## Install

```bash
pip install -e ".[dev]"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q     # 55 tests
```

Python ≥ 3.10. **No runtime dependencies** — stdlib only, including the normal-quantile
function and the Cholesky decomposition.

## The honest ceiling

Behaviour is **necessary, not sufficient**. A detector fuses it with IP reputation,
fingerprint and challenge response; perfect timing from a datacenter IP still flags. The
server-side model is a black box you can match on every client-visible feature and still be
scored on aggregates you cannot see. And it retrains.

The defensible claim is *"statistically indistinguishable from human on the features I can
measure, and verifiably diverse across the fleet"* — never *"beats the sensor"*.

## Licence

MIT. See [Data](#data) for the corpora, and [`ROADMAP.md`](ROADMAP.md) for what is next.
