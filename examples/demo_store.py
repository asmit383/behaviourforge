"""Demo 2 — a shopping flow on a live site: log in, browse, scroll, add to cart.

Target is Sauce Labs' demo storefront, which exists for automation practice. The interactions
are the same ones a real retail flow needs (login, product browse, scroll, add to cart), so
swapping SITE for another target is a one-line change — but pointing a behavioural-realism
demo at a real retailer's live bot protection is a different activity from demonstrating the
library, and this is the latter.

    python examples/demo_store.py [--headful]

Note for headful runs: do not touch the physical mouse while it is running. Moving the real
pointer steals focus from the input and the synthetic keystrokes go nowhere.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from behaviourforge import forge                                # noqa: E402
from examples.camoufox_driver import Driver                     # noqa: E402

SITE = "https://www.saucedemo.com"
USER, PASSWORD = "standard_user", "secret_sauce"


def main(headful: bool = False) -> int:
    from camoufox.sync_api import Camoufox

    seed = 20260907
    p = forge(seed)
    m = p.motor
    print(f"persona seed={seed}")
    print(f"  typing   base {m.base_ms:.0f}ms   dwell {m.dwell_ms:.0f}ms   "
          f"errors {m.error_rate:.1%}")
    print(f"  pointer  speed {m.mouse_speed:.2f}   curve {m.mouse_curve:.2f}   "
          f"overshoot {m.mouse_overshoot:.0%} @ {m.mouse_overshoot_px:.0f}px")
    print(f"  scroll   gap {m.scroll_gap_ms:.0f}ms   reversal {m.scroll_reversal:.1%}")
    print()

    t0 = time.time()
    # humanize=False: Camoufox's own humanizer would re-curve every sample we emit.
    with Camoufox(headless=not headful, humanize=False) as browser:
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        d = Driver(page, p, trace=True)

        print("  [1] navigate")
        d.goto(SITE)
        page.wait_for_selector("#user-name")

        print("  [2] log in")
        d.type("#user-name", USER, first_field=True)
        d.type("#password", PASSWORD, secret=True)      # no typos in a password field
        d.click("#login-button")
        page.wait_for_selector(".inventory_list", timeout=15000)

        print("  [3] read the catalogue")
        d.scroll(18)
        d.idle(2500)                                    # an agent would be thinking here
        d.scroll(10)

        print("  [4] open a product")
        d.click("#item_4_title_link")
        page.wait_for_selector("button[data-test^='add-to-cart']", timeout=15000)
        d.idle(1800)

        print("  [5] add to cart")
        d.click("button[data-test^='add-to-cart']")
        page.wait_for_selector(".shopping_cart_badge", timeout=10000)
        badge = page.text_content(".shopping_cart_badge")

        print("  [6] open the cart")
        d.click(".shopping_cart_link")
        page.wait_for_selector(".cart_item", timeout=15000)
        item = page.text_content(".inventory_item_name")

        if headful:
            time.sleep(3)

    print(f"\n  cart badge: {badge}   item: {item!r}")
    print(f"  completed in {time.time() - t0:.1f}s of humanized interaction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--headful" in sys.argv))
