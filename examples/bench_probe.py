"""Head-to-head on agenthands' own probe harness.

The fairest comparison available: their measurement apparatus, our output, on the browser
they built it against (Chromium, not Camoufox — otherwise the browser is a confound and the
wheel results in particular are not comparable at all).

The probe records what a page can actually see — coalesced pointer counts, click offset from
the target's centre, wheel delta consistency, keystroke dwell and flight — and returns a
report. Both libraries drive the same three targets, the same text, and the same scrollbox.

    python examples/bench_probe.py            # BehaviourForge
    node   /tmp/ah/run_agenthands.mjs         # agenthands, same page
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from behaviourforge import forge                                # noqa: E402
from examples.camoufox_driver import Driver                     # noqa: E402

PROBE = os.environ.get("PROBE", "/tmp/ah/probe.html")
TEXT = "the quick brown fox jumps over the lazy dog"


def main() -> int:
    from playwright.sync_api import sync_playwright

    p = forge(seed=4712)
    with sync_playwright() as pw:
        exe = os.environ.get("CHROMIUM")          # same binary as the node runner
        browser = pw.chromium.launch(headless=True,
                                     **({"executable_path": exe} if exe else {}))
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto("file://" + PROBE)
        page.wait_for_selector("#t1")

        d = Driver(page, p)
        d._x, d._y = 640, 450
        for _ in range(5):                  # several passes: 4 clicks is too few to judge
            for sel in ("#t1", "#t2", "#t3"):
                d.click(sel)
                d.idle(300)
        d.type("#txt", TEXT, first_field=True)

        page.locator("#scrollbox").hover()
        for _ in range(4):                      # a few bursts, with pauses between
            d.scroll(8)
            time.sleep(0.4)

        rep = page.evaluate("() => report()")
        browser.close()

    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
