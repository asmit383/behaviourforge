# Roadmap — closing the gap with agenthands

## The strategic point

The gap is not a list of wrong constants. It is a **difference in methodology**, and patching
numbers one at a time will always leave us a step behind.

We measured **human motor behaviour** — how people type and move. They measured **what a
detector actually reads** — running a human and a bot through the same probe page, on the same
machine, in the same browser, and diffing. That method finds signals neither of us would think
to look for: `getCoalescedEvents()`, `wheelDeltaY`/`deltaY` consistency, gesture phase. We
would never have derived those from Aalto or SapiMouse, because they are artefacts of the
*injection path*, not of the hand.

So the real goal is to adopt their methodology while keeping our advantages (6,130
participants, reliability shrinkage, copula sampling, the fleet audit) — not to chase their
numbers.

---

## Step 0 — verify their claims against our own data (do this first, it is cheap)

We have 120 users of raw SapiMouse trajectories. Their headline path findings are testable
against it directly, and their own base profile was **25 clicks and 83 keystrokes from one
person** (they say so, and corrected part of it with Balabit).

| claim | their number | check against |
|---|---|---|
| peak velocity at 0.40 of the stroke | 0.40 | SapiMouse, 120 users |
| ~3 submovements per stroke | 3 | SapiMouse, 120 users |
| path/straight p90 | 2.13 | SapiMouse, 120 users |
| click dwell right-skewed, p90 170ms | 170 | Balabit (we have it) |

If SapiMouse agrees, rebuild with confidence. If it disagrees, that is a finding in itself and
we should publish the discrepancy rather than quietly adopt their constant. **Do not rebuild
the path model on numbers we have not reproduced.**

---

## Tier 1 — path shape (we measurably lose here today)

Measured on our current output, 400 strokes:

| metric | us | human |
|---|---|---|
| peak-velocity position | 0.47 | 0.40 |
| sub-movements per stroke | **17** | **3** |
| path/straight (median) | 1.10 | 1.07 ✓ |
| path/straight (p90) | **1.23** | **2.13** |

1. **Ballistic + corrective submovements, asymmetric velocity.** One opening thrust covering
   ~80–95% of the distance, then 1–2 homing corrections, each with a velocity profile peaking
   at 0.40 (`v(t) ~ t^2 (1-t)^3`). Fixes the peak position and the submovement count together.
2. **Correlated tremor instead of white noise.** This is the key insight and it does *not*
   cost us the SapiMouse calibration: 2.03px of midpoint deviation is true of white noise AND
   of a smooth wobble. White noise reverses direction every sample, manufacturing a velocity
   peak each time; an AR(1) wander reproduces the same amplitude with a hand's correlation
   structure.
3. **Per-stroke directness variance.** Our `curve` is fixed per persona, so every stroke is
   equally direct — the exact tell they name. Needs a heavy-tailed per-stroke deviation draw.

**Risk:** all three change the geometry that `_movement_features` inverts. The round-trip test
(`test_generation_inverts_extraction`) must still pass, or the corpus stops meaning anything.
Re-extract if the estimators need to change.

---

## Tier 2 — observable surface we currently fail outright

4. **Scroll is unphysical.** `PX_PER_NOTCH = 100` emits a `deltaY`/`wheelDeltaY` pair no real
   device can produce, because `wheelDeltaY` derives from tick count. Must be whole ticks at
   the platform's px-per-tick (40 on macOS).
5. **Gesture phase.** A trackpad gives 1 cancelable wheel event and 1 `scrollend` *per
   gesture*; `Input.dispatchMouseWheel` stamps every event `began`, so we emit one per notch.
   They claim this is not expressible over stock CDP. **Verify that ourselves** — if true, it
   is a documented ceiling for everyone, not a defect unique to us.
6. **`getCoalescedEvents()`.** Human mean 1.01; injected input returns a hard 1 forever.
   Reproduce by modelling the cause (an occasional missed frame), not by faking the API.

---

## Tier 3 — replace our last guesses

7. **Click landing offset.** Currently a guessed 25–75% band, and the one motor parameter with
   no data behind it. Their SERP source (Zenodo, CC-BY-4.0) has it.
8. **The autocorrelation gap.** Real humans +0.058 on real text; we produce −0.004. Likely
   `tempo_theta` (median 0.44 = fast mean reversion) being biased high by the AR(1)+noise fit.

---

## Tier 4 — defend the lead

9. **Held-out validation.** All our KS checks compare against the corpora we fitted to. Reserve
   1,000 Aalto participants, never fit on them, KS generated output against them. This is the
   first genuinely external check and it is ~an hour.
10. **Extend the audit** to the new observable metrics, so submovement count and velocity
    asymmetry are regression-tested at fleet scale rather than checked once by hand.

---

## Where we are already ahead — do not regress these

- **6,000 keystroke participants** vs their 99 (KeyRecs). Their mouse corpus is 47 people.
- **Fleet diversity.** Their persona indexes a *recorded individual* (`mouse: number` into 47
  people), so 10,000 agents share ~213 to a person. Our copula has no such ceiling. This is
  their clearest structural weakness and it is exactly what `audit()` was built to catch.
- **Reliability-based shrinkage.** No equivalent on their side.
- **Browser-agnostic pure core.** Theirs is Playwright-coupled.

They do have **cleaner licensing** — CC-BY-4.0 on both corpora, against our research-use Aalto
and Balabit. Worth fixing if this is ever distributed commercially.

---

## Decisions needed

- **Build a probe harness?** Recording a real human (you) and diffing against our bot on the
  same machine is the method that found most of their wins. `examples/demo_capture.py` is
  already half of it. Highest leverage, but needs you to sit and record sessions.
- **Adopt their datasets alongside ours?** SERP (47) and KeyRecs (99) are CC-BY-4.0 and would
  give us click-offset data plus a cleaner licensing story.
- **Target: parity or superset?** Parity on observable metrics is the smaller job. Superset —
  their surface fidelity plus our population scale — is the defensible position, and nobody
  currently occupies it.
