"""Demo 1 — drive a real browser, then measure what the page actually received.

"It looks human" is a vibe. This turns it into numbers: a local page records every
keydown/keyup/mousemove exactly as a behavioural sensor would, BehaviourForge drives Camoufox
through it, and we compare the captured timings against the Aalto corpus the persona was
sampled from.

Worth being clear about what this does and does not prove. It is an end-to-end check that the
model survives the trip through Playwright, Firefox, and the DOM event loop — which is not
guaranteed, since a driver that rounds or coalesces events would flatten the distribution.
It is NOT evidence that any particular detector is fooled.

    python examples/demo_capture.py [--headful]
"""
from __future__ import annotations

import math
import statistics as st
import sys
import tempfile
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from behaviourforge import corpus, forge                       # noqa: E402
from examples.camoufox_driver import Driver                    # noqa: E402

PAGE = """<!doctype html><html><body style="font:14px system-ui;margin:40px">
<h2>BehaviourForge capture harness</h2>
<input id="field" style="width:420px;padding:8px;font-size:15px" placeholder="type here">
<div id="pad" style="width:640px;height:320px;border:1px solid #999;margin-top:16px"></div>
</body></html>"""

# Installed via add_init_script rather than an inline <script>: Camoufox does not execute
# inline scripts on file:// pages, so the hooks silently never registered and window.__cap
# came back undefined. An init script runs in the page context before anything else and works
# regardless of the page's own script handling.
CAPTURE_JS = """window.__cap = {keys: [], moves: []};
const down = {};
addEventListener('keydown', e => { if (!(e.key in down)) down[e.key] = e.timeStamp; });
addEventListener('keyup',   e => {
  if (e.key in down) { window.__cap.keys.push({k: e.key, t: down[e.key],
                                               hold: e.timeStamp - down[e.key]});
                       delete down[e.key]; } });
addEventListener('mousemove', e => window.__cap.moves.push({x: e.clientX, y: e.clientY,
                                                            t: e.timeStamp}));"""

TEXT = ("the quick brown fox jumps over the lazy dog and then keeps typing for a while "
        "so there are enough samples to estimate a distribution from")


def _ks(a, b):
    a, b = sorted(a), sorted(b)
    i = j = 0
    fa = fb = d = 0.0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            i += 1; fa = i / len(a)
        else:
            j += 1; fb = j / len(b)
        d = max(d, abs(fa - fb))
    return d


def main(headful: bool = False) -> int:
    from camoufox.sync_api import Camoufox

    seed = 4712
    p = forge(seed)
    print(f"persona seed={seed}")
    print(f"  intended  base_ms {p.motor.base_ms:.0f}  dwell_ms {p.motor.dwell_ms:.0f}  "
          f"error_rate {p.motor.error_rate:.1%}")

    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as fh:
        fh.write(PAGE)
        path = fh.name

    # humanize=False is essential: Camoufox's own humanizer would re-curve every sample.
    with Camoufox(headless=not headful, humanize=False) as browser:
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.add_init_script(CAPTURE_JS)
        d = Driver(page, p)
        d.goto("file://" + path)
        page.wait_for_selector("#field")

        d.type("#field", TEXT, first_field=True)
        pad = page.locator("#pad").bounding_box()
        for _ in range(6):                      # a few reaches across the pad
            d.move_to(*p.point_in(pad))
        d.idle(3000)

        cap = page.evaluate("window.__cap")
        typed = page.input_value("#field")

    os.unlink(path)

    keys = [k for k in cap["keys"] if len(k["k"]) == 1]
    holds = [k["hold"] for k in keys if 0 < k["hold"] < 1000]
    flights = [b["t"] - a["t"] for a, b in zip(keys, keys[1:]) if 0 < b["t"] - a["t"] < 3000]
    moves = cap["moves"]

    print(f"\ncaptured from the DOM: {len(keys)} keys, {len(moves)} mousemove events")
    print(f"  field contents match intended text: {typed == TEXT}")
    if typed != TEXT:
        print(f"    (typos self-corrected: {len(typed)} vs {len(TEXT)} chars)")

    corp = corpus.load()["vectors"]
    print(f"\n{'':16}{'captured':>10}{'persona':>10}{'corpus p50':>12}")
    print(f"  {'hold ms':<14}{st.median(holds):>10.0f}{p.motor.dwell_ms:>10.0f}"
          f"{st.median([v['dwell_ms'] for v in corp]):>12.0f}")
    print(f"  {'flight ms':<14}{st.median(flights):>10.0f}{p.motor.base_ms:>10.0f}"
          f"{st.median([v['base_ms'] for v in corp]):>12.0f}")

    # right-skew and autocorrelation must survive the trip through the browser
    skew = st.mean(flights) > st.median(flights)
    lg = [math.log(f) for f in flights]
    m = sum(lg) / len(lg)
    acf = (sum((lg[i] - m) * (lg[i + 1] - m) for i in range(len(lg) - 1))
           / sum((x - m) ** 2 for x in lg))
    print(f"\n  right-skewed (not uniform): {skew}")
    print(f"  lag-1 autocorrelation of captured flights: {acf:+.3f}")

    # do the captured holds look like the population, or like a machine?
    ks_corpus = _ks(holds, [v["dwell_ms"] for v in corp])
    print(f"  KS(captured holds vs 6000-participant corpus medians): {ks_corpus:.3f}")

    dt = [b["t"] - a["t"] for a, b in zip(moves, moves[1:]) if 0 < b["t"] - a["t"] < 500]
    print(f"\n  mousemove samples: {len(moves)}, median gap {st.median(dt):.0f}ms")

    # The path is not straight: measure how far the captured points deviate from the line
    # between successive resting positions. A scripted move_to lands in one jump.
    devs = []
    for a, b in zip(moves, moves[2:]):
        mid = moves[moves.index(a) + 1]
        ax, ay, bx, by = a["x"], a["y"], b["x"], b["y"]
        L = math.hypot(bx - ax, by - ay)
        if L > 8:
            devs.append(abs((bx - ax) * (ay - mid["y"]) - (ax - mid["x"]) * (by - ay)) / L)
    if devs:
        print(f"  median deviation from a straight line: {st.median(devs):.2f}px")
    print("\n  note: clientX/clientY are integers, so the sub-pixel coordinates we emit are")
    print("  rounded by the DOM. They still change WHICH integers get reported, but a page")
    print("  reading clientX cannot observe them directly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--headful" in sys.argv))
