<div align="center">

# BehaviourForge

**Coherent synthetic motor identities from a seed.**
*The behavioural BrowserForge — browser-agnostic, zero dependencies, one function.*

</div>

---

BrowserForge samples a coherent **fingerprint** from a Bayesian network trained on real
device records. BehaviourForge is the same recipe one layer up: it samples a coherent
**motor identity** — keystroke timing, pointer dynamics, scroll rhythm, idle behaviour —
from three real human datasets.

```python
from behaviourforge import forge

p = forge(seed=42)

for k in p.keystrokes("hello@example.com", first_field=True):
    driver.press(k.key, hold_ms=k.dwell_ms, wait_before_ms=k.flight_ms)

for m in p.path(cursor_x, cursor_y, *p.point_in(button_box)):
    driver.move(m.x, m.y); driver.wait(m.dt_ms)
driver.click()
```

The library emits timed **plans** and never touches a browser. Executing them is a handful of
lines against Playwright, CDP, Selenium, or anything else that can press a key — see
[Drivers](#drivers).

---

## Why a seed is the whole scaling story

A persona is a pure function of an integer. Ten thousand agents need ten thousand integers:
no server minting identities, no shared state to contend on, no coordination. Agent 4,712
forges its own identity in **44µs** and gets the same one back tomorrow.

That matters because the failure mode at fleet scale is invisible from any single session.
Each persona can be flawless while the *population* gives you away — 10,000 agents sharing
one exact tempo value is a correlatable signature even when every other field differs, and
the discreteness of a preset set is something real traffic can never produce.

```
$ behaviourforge audit -n 10000

  collisions        0
  nearest-neighbour min 1.6709  p01 1.9283  median 2.8773   [sd units]
  distribution      worst KS 0.0606 on mouse_tremor
  correlation       base~dwell  corpus +0.161  forged +0.135  (delta -0.026)

  PASS — no collisions, no constant parameters, distribution and correlation preserved.
```

`audit()` is not decoration. It exists because the first version of this library shipped a
constant `tempo_theta` across the entire fleet, and nothing else would have caught it.

---

## Coherence

Draw twenty-one motor parameters independently and you will eventually emit a persona that
types in the 95th percentile for speed while holding keys in the 5th — every value ordinary, the
bundle impossible. A sensor scoring one signal in isolation is easy to satisfy; a sensor
**fusing** signals catches exactly those combinations.

Two sampling models, both coherent:

| model | how | when |
|---|---|---|
| `copula` *(default)* | empirical marginals + the corpus correlation matrix in normal-score space | unlimited distinct identities; correct at fleet scale |
| `bootstrap` | resample one real participant's whole vector, jitter slightly | every persona is provably one real human; capped by corpus size |

Bootstrap cannot produce more distinct identities than the corpus has rows. At 10,000 agents
against 6,000 participants each real person is reused ~1.7 times, one of them **12 times**,
and jitter only smears those clones a few percent apart. Hence copula by default.

---

## What is measured, and what is not

Every parameter comes from a public dataset. `reliability` is **split-half, Spearman-Brown
corrected** — each participant estimated twice, on odd and even sentences, then correlated
across people. It answers the question that matters: is this field's spread real human
variation, or my estimation error?

### Keystroke — Aalto *136M Keystrokes* (Dhakal et al. 2018), 6,000 participants

| field | reliability | coverage | notes |
|---|---|---|---|
| `base_ms` | **0.992** | 100% | median inter-key flight |
| `dwell_ms` | **0.992** | 100% | key hold time |
| `think_ms` | **0.970** | 100% | word-boundary pause |
| `dwell_sigma` | **0.939** | 100% | |
| `flight_sigma` | **0.898** | 100% | trial noise only — tempo variance is separated out |
| `error_rate` | **0.809** | 100% | backspaces / keystrokes |
| `shift_ms` | 0.552 | 89% | matched on predecessor class |
| `tempo_sigma` | 0.221 | 34% | OU innovation |
| `tempo_theta` | 0.143 | 34% | OU mean-reversion |
| `bigram_mult` | — | 100% | per-person measured bigram table |

`tempo_*` is real but weakly identified: ~500 keystrokes is not enough to pin one person's
tempo drift, so most of its apparent between-person spread is noise. The sampler **shrinks**
it accordingly (see below) rather than propagating fake diversity or collapsing to a
constant.

### Pointer — SapiMouse (Antal, Fejer & Buza 2020), 120 subjects, 245 raw sessions

| field | reliability | notes |
|---|---|---|
| `mouse_tremor` | **0.983** | |
| `mouse_settle_ms` | **0.979** | |
| `mouse_speed` | **0.949** | |
| `mouse_curve` | **0.918** | |
| `mouse_overshoot` | **0.746** | probability of a corrective overshoot |
| `mouse_overshoot_px` | 0.146 | its magnitude — real, but not a stable personal trait |

Pointer dynamics are strongly individual and reliably measurable — better than keystroke
tempo. ~196 movements per user is ample.

### Scroll + idle — Balabit Mouse Dynamics Challenge, 10 users, 1,676 sessions

| field | reliability |
|---|---|
| `scroll_gap_ms` | **0.974** |
| `idle_drift_px` | **0.977** |
| `idle_pause_ms` | **0.952** |
| `scroll_reversal` | **0.947** |
| `idle_pause_sigma` | 0.795 |
| `scroll_gap_sigma` | 0.377 |

Balabit rather than SapiMouse for two specific reasons: it logs wheel events, which SapiMouse
does not record at all, and it captures people doing ordinary work, whereas SapiMouse
participants were told to perform as many operations as possible — which makes its pauses a
model of hurrying rather than of resting.

**Only 10 users.** That is thin, and the between-person spread is correspondingly
poorly pinned. `audit()` scales its KS threshold by the Kolmogorov critical value `1.36/√n`
so a small reference corpus is not mistaken for drift.

### Reliability shrinkage

Observed spread is true human variation **plus** estimation error, and those want opposite
treatment. Propagating the whole spread manufactures diversity that isn't real; collapsing it
to a constant manufactures a fleet signature. Classical test theory gives the split — the
honest spread to sample is `sqrt(reliability)` of the observed one. So `base_ms` (0.99) keeps
essentially all its spread and `tempo_theta` (0.14) keeps ~38% of it.

### Not measured

- **Cross-modal correlation.** No public dataset records the same person typing *and* moving
  a pointer; Aalto and SapiMouse are disjoint populations. Typing-to-pointer speed coupling is
  an assumption, isolated in one named constant (`sampler.CROSS_MODAL_RHO = 0.35`) rather than
  smuggled into a formula.
- **Non-QWERTY layouts and touch devices.** `forge()` raises rather than silently returning
  US-QWERTY finger physics for AZERTY, or a mouse persona for a phone.
- **Where inside a target a human clicks.** `point_in` still uses a guessed interior band
  (25–75% horizontally, 30–70% vertically). Measuring it needs click coordinates paired with
  element geometry, which none of these datasets record.
- **Trial-to-trial jitter on movement time** (`±10%`), and the default notch count when a
  caller does not say how far to scroll — a usage default, not a human parameter.

### Path shape — verified against the raw dataset

Measured with one estimator applied to both sides: 24,451 SapiMouse strokes from 120 users,
and 500 generated strokes.

| metric | before | now | SapiMouse |
|---|---|---|---|
| peak-velocity position | 0.47 | **0.25** | 0.26 |
| velocity peaks per stroke | 7 | **2** | 2 |
| path/straight (p50) | 1.09 | **1.16** | 1.11 |
| path/straight (p90) | 1.21 | **2.13** | 2.15 |
| samples per stroke | 55 | **34** | 34 |
| pointer sample rate | 79 Hz | **62 Hz** | 59 Hz |

Four corrections, each from the data rather than from a published constant:

- **Velocity peaks early, at 0.26 of the stroke.** Any symmetric easing peaks at exactly 0.50,
  which is a fixed, checkable signature.
- **Tremor is a correlated wander, not white noise.** This was the important one: 2.03px of
  measured midpoint-deviation is equally true of white noise and of a smooth wobble, but white
  noise reverses direction every sample and each reversal registers as a velocity peak. An
  AR(1) walk keeps the measured amplitude and drops the count from 7 to 2 — no recalibration
  needed.
- **Directness needs a heavy tail.** Real reaches are mostly direct with a few that wander
  (p90 2.15). A uniform bow made every stroke equally direct, which is itself the tell.
- **Sample rate follows duration, not pixel density.** We emitted 79Hz against a real 59Hz.
  Fixing it also cut IPC round-trips by a third.

A cross-check worth recording: `agenthands` reports peak velocity at 0.40 from a smaller
capture. Against our 24,451 strokes it is **0.26**. Both agree it is early rather than
symmetric; the magnitude did not reproduce, so we use our own measurement. Their
path/straight p90 of 2.13 **did** reproduce independently (we measure 2.15).

### Keystroke — verified against raw Aalto

Generated output against the raw dataset, same estimator, replaying the **same sentences**
the participants typed (comparing on different text confounds the digraph sequence with the
tempo):

| metric | before | now | Aalto |
|---|---|---|---|
| flight mean / sd | 206 / 191 | **208 / 212** | 217 / 218 |
| flight CV | 0.93 | **0.99** | 1.00 |
| dwell mean / sd | 111 / 39 | **111 / 39** | 109 / 37 |
| gaps over 500ms | 6.45% | **7.5%** | 7.77% |
| lag-1 autocorrelation | −0.004 | **+0.066** | +0.056 |

The autocorrelation was the real defect, and the fix is worth stating plainly because it is a
departure from "everything measured". The corpus stores `tempo_sigma` from a single
AR(1)-plus-noise fit, which reproduces the variance SPLIT correctly but generated −0.004
where real participants show +0.056 on identical text. Two hypotheses were tested and both
failed: fitting on high lags only moves theta by 9% (so it is not fast-component
contamination), and replaying the participants' own sentences does not close it (so it is not
a text confound). The remaining explanation is model misspecification — real typing tempo
varies on several timescales at once, and extrapolating one exponential decay back to lag 0
underestimates total tempo variance.

So `TEMPO_SIGMA_SCALE` is calibrated against the **observable** rather than the latent
estimate, because the observable is what a detector reads. It is the one fitted constant in
the library, and it is named and isolated rather than folded into the corpus.

### Scroll — most of the known wheel tells are Chromium-specific

`agenthands` documents four wheel signatures, all measured against Chromium's
`Input.dispatchMouseWheel`. Driving Firefox through Camoufox, three of them do not apply —
measured, not assumed:

| signature | Chromium/CDP | this library, Firefox |
|---|---|---|
| `wheelDeltaY`/`deltaY` ratio | any ratio; impossible pairs | **−3.00, fixed and consistent** |
| cancelable wheel events per gesture | 199 of 199 | **0** |
| `scrollend` events per gesture | 136 (human: 1) | **0** (human: 1) |
| `deltaY` variance | — | **sd 0.0 — a constant 100.0** |

The remaining two are ours. `deltaY` is literally constant, where a real device varies it, and
no `scrollend` fires at all where a real gesture produces one. Neither is fixable from the
corpora we have: Balabit logs wheel notch events without pixel deltas, so the delta
distribution needs a probe capture of a real human scrolling rather than another dataset.

---

## What the data corrected

The model was built on assumptions, then measured. Several assumptions were wrong, and that
is the entire argument for using real data:

- **`base_ms` and `dwell_ms` are nearly independent** (r ≈ 0.16). "Fast typist, short key
  hold" is intuitive and essentially false — flight time and hold time are separately
  controlled. Hand-authored archetypes encode the intuition and produce r ≈ 0.96, which is
  how you can tell they were invented.
- **The digraph effect is far gentler than intuition.** Same-finger is **1.06×**, not the
  ~1.6× that obviously follows from finger reuse. The *ordering* (alternation < same-hand <
  same-finger) holds across 3,000 typists; the magnitudes did not.
- **Pointer speed was off by 2.6×.** The old sampler made a 500px reach take **230ms**;
  SapiMouse says **602ms**. (Fitts' law with textbook constants and a 50px target predicts
  ~700ms — corroborating, though the constants are an assumption, so the measurement is the
  evidence.) The error is robust to how a movement is segmented: restricting to reaches that
  end in an actual click, rather than a pause, moves the median by 1%.
- **The scroll re-read rate was 3.4× too high.** Hardcoded at `0.2`; measured direction-
  reversal is **5.8%** (range 2.9–8.9% across users). Scroll cadence is also strongly
  individual — per-user median gaps span 16ms to 249ms, a **15× spread** — where the old code
  gave every agent an identical rhythm.
- **Idle pauses had the wrong centre and no tail.** `uniform(700, 2200)` against a measured
  p50 of **780ms** running out to p90 **4.2s**. A bounded uniform draw makes every rest period
  suspiciously similar in length.
- **Overshoot was measuring the wrong event.** Counting any excursion past the endpoint gives
  32.6% at 42px median — but that includes paths that merely wandered through a further point
  mid-flight. Requiring the excursion to occur in the last 30% of the path, which is what a
  corrective submovement actually is, gives **10.0% at 18px**. The generator produces
  corrections, so the estimator has to measure corrections.
- **Raw lag-1 autocorrelation cannot estimate tempo.** Trial noise attenuates it to ≈ 0.009,
  so `theta = 1 - acf` lands near 0.99 and saturates any sane clamp. Recovering it needs an
  AR(1)-plus-white-noise decomposition — `gamma(k) = var_s * phi^k` for k ≥ 1, fit across
  lags. The same fit also separates trial noise from tempo variance, which the naive
  `flight_sigma` was double-counting.

---

## Drivers

The core is browser-agnostic by construction — a test asserts no module imports a browser.

<details>
<summary><b>Playwright</b></summary>

```python
def type_into(page, selector, text, p, first_field=False):
    page.locator(selector).click()
    for k in p.keystrokes(text, first_field=first_field):
        page.wait_for_timeout(k.flight_ms)
        page.keyboard.press(k.key if len(k.key) > 1 else k.key, delay=k.dwell_ms)

def click(page, box, p, cursor):
    for m in p.path(*cursor, *p.point_in(box)):
        page.mouse.move(m.x, m.y)
        page.wait_for_timeout(m.dt_ms)      # final dt IS the settle
    page.mouse.click(*cursor)
```
</details>

<details>
<summary><b>Selenium</b></summary>

```python
from selenium.webdriver import ActionChains
def scroll(driver, p, steps=5):
    for w in p.scroll(steps):
        driver.execute_script("window.scrollBy(0, arguments[0])", w.dy)
        time.sleep(w.dt_ms / 1000)
```
</details>

<details>
<summary><b>Idle — for agent loops</b></summary>

An AI agent's observe → think → act loop parks the cursor perfectly still for seconds on
every turn, and that stillness is regular, repeated, and structural to the loop rather than
incidental. Call this instead of `sleep` around an LLM round-trip.

```python
for m in p.idle(llm_latency_ms, cursor_x, cursor_y):
    driver.move(m.x, m.y); driver.wait(m.dt_ms)
```
</details>

---

## API

```python
forge(seed=None, *, locale="en-US", device="desktop",
      corpus=None, mouse_corpus=None, model="copula") -> Persona

p.keystrokes(text, *, first_field=False, typos=True) -> list[Key]   # key, flight_ms, dwell_ms, kind
p.path(x0, y0, x1, y1)                                -> list[Move]  # x, y, dt_ms
p.scroll(notches=None)                                -> list[Wheel] # dy, dt_ms
p.idle(ms, x, y, *, width=1280, height=800)           -> list[Move]
p.point_in(box)                                       -> (x, y)
p.motor.to_dict() / Motor.from_dict(d)                             # persist an identity

audit(n=10_000, *, model="copula") -> AuditReport
```

A persona is **stateful within a session**: keystroke tempo drifts continuously across every
call, so filling three fields carries one rhythm through all three. Re-forge from the same
seed to rewind. Keyboard and pointer use split RNGs, so adding a mouse move never shifts
keystroke timings — otherwise a session would not be replayable.

## CLI

```bash
behaviourforge show 42 --text "hello@example.com"   # inspect a persona and its timings
behaviourforge audit -n 10000                       # fleet diversity report; exits 1 on FAIL
behaviourforge build-corpus Keystrokes.zip --zip --out corpus.json --limit 6000
```

`python -m behaviourforge.cli ...` works identically if the console script is not on your
PATH.

```
$ behaviourforge show 42 --text "hi there"
persona  seed=42  locale=en-US  device=desktop
  typing   base 152ms  dwell 101ms  think 30ms  errors 4.6%
  pointer  speed 2.74  curve 0.41  overshoot 0.43  tremor 2.5px
  measured bigrams: 29

  "hi there"  ->  8 events, 3.03s (158 cpm)
    'h'          flight  198.2ms  dwell 146.6ms
    'i'          flight  133.3ms  dwell 101.3ms
    ' '          flight  163.2ms  dwell 128.2ms
    't'          flight  256.9ms  dwell 157.9ms
    'h'          flight  908.7ms  dwell 118.0ms      <- tempo drift, not a fixed pause
```

## Rebuilding the corpora

Both bundled corpora are reproducible, not magic blobs:

```bash
# Aalto — reads straight from the 1.4GB archive, no 16GB unpack (~20s for 6000 participants)
behaviourforge build-corpus /path/to/Keystrokes.zip --zip --limit 6000 \
    --out behaviourforge/data/keystroke_aalto.json

# SapiMouse — https://ms.sapientia.ro/~manyi/sapimouse/sapimouse.zip
python -c "from behaviourforge.corpus import build_from_sapimouse as b; \
           b('sapimouse.zip', 'behaviourforge/data/mouse_sapimouse.json')"

# Balabit — github.com/balabit/Mouse-Dynamics-Challenge (scroll + idle)
python -c "from behaviourforge.corpus import build_from_balabit as b; \
           b('balabit.zip', 'behaviourforge/data/scroll_balabit.json')"
```

A round-trip test keeps extraction honest: generate trajectories from known parameters,
re-measure them with the SapiMouse extractor, and recover the inputs within 3%. Without it,
extraction can measure something plausible but *adjacent* to what the generator consumes, and
the two drift apart while both look fine. It caught a bow cap flattening long sweeps, a
relative overshoot threshold going blind on long moves, and a tremor unit mismatch that was
passing only because two errors cancelled.

## Install

```bash
pip install -e ".[dev]"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q     # 51 tests
```

Python ≥ 3.10. **No runtime dependencies** — stdlib only, including the normal-quantile
function and the Cholesky decomposition.

---

## The honest ceiling

Behaviour is **necessary, not sufficient**. A detector fuses it with IP reputation,
fingerprint, and challenge response; perfect timing from a datacenter IP still flags. The
server-side model is a black box you can match on every client-visible feature and still be
scored on aggregates you cannot see. And it retrains.

The defensible claim is *"statistically indistinguishable from human on the features I can
measure, and verifiably diverse across the fleet"* — never *"beats the sensor"*.

## Licence & data

Library: MIT.

The bundled corpora are **derived** from three datasets — no raw files are redistributed here,
only extracted per-participant parameter vectors (medians, log-spreads, correlations):

| dataset | licence | used for |
|---|---|---|
| SapiMouse | Apache-2.0 | pointer dynamics |
| Aalto 136M Keystrokes | research use | keystroke timing |
| Balabit Mouse Dynamics Challenge | released to researchers for behavioural-biometrics work | scroll + idle |

Aalto and Balabit are both research-oriented releases. Verify your rights before
redistributing the derived corpora commercially, or rebuild them yourself from the sources —
the extraction pipelines are in `corpus.py` for exactly that reason.

## Demos

```bash
python examples/demo_capture.py            # drive a real browser, measure what the page got
python examples/demo_store.py --headful    # log in, browse, scroll, add to cart
```

`examples/camoufox_driver.py` is the ~90-line executor. Two things it encodes that are easy
to get wrong: pass `humanize=False` to Camoufox (its own humanizer would re-curve every
sample and reintroduce the fleet-identical signature), and schedule events against an
absolute clock rather than sleeping between them.
