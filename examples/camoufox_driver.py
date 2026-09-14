"""Camoufox / Playwright adapter — turns BehaviourForge plans into real browser input.

BehaviourForge emits timed plans and never touches a browser. This is the other half: a
~90-line executor. It is deliberately not part of the library, because the whole point is
that the model has no browser dependency — swap this file for a CDP or Selenium equivalent
and nothing upstream changes.

    from behaviourforge import forge
    from camoufox.sync_api import Camoufox

    with Camoufox(headless=False, humanize=False) as browser:
        page = browser.new_page()
        d = Driver(page, forge(seed=4712))
        d.goto("https://example.com")
        d.type("#user", "standard_user", first_field=True)
        d.click("#login")

ONE GOTCHA THAT MATTERS: pass `humanize=False` to Camoufox. Its built-in humanizer applies
its own Bezier curve to every mouse move, so leaving it on re-humanizes each of our samples
and reintroduces the single fleet-identical signature that per-persona trajectories exist to
avoid. Owning the path is the point; two humanizers stacked is worse than either alone.
"""
from __future__ import annotations

import time

# Playwright key tokens for the characters that are not literal keys.
_TOK = {" ": "Space", "\n": "Enter", "\t": "Tab"}


class _Clock:
    """Absolute scheduling. This is not a detail — a relative-sleep driver corrupts the model.

    Two failures, both measured against a real page rather than reasoned about:

    1. Sleeping `flight_ms` between keyup and the next keydown makes the observable
       press-to-press interval flight + dwell. The corpus defines flight as press-to-press,
       so a 144ms persona typed at 243ms — 69% slow, and every derived statistic with it.
    2. Sleeping relative to the previous event turns IPC jitter into a compensation artifact:
       a delayed event lengthens one gap and shortens the next, which produces NEGATIVE
       autocorrelation. Captured lag-1 came out at -0.165 while the model generates positive
       drift. That inverts the single property random delays cannot fake.

    Scheduling every event against one absolute origin fixes both: latency is absorbed
    instead of propagating, and each timestamp lands where the model intended."""

    def __init__(self):
        self.t = time.perf_counter()

    def wait(self, ms: float) -> None:
        """Advance the schedule by `ms` and block until that instant."""
        self.t += ms / 1000.0
        delay = self.t - time.perf_counter()
        if delay > 0:
            time.sleep(delay)


class Driver:
    """Executes a Persona's plans against a Playwright-style page."""

    def __init__(self, page, persona, *, trace: bool = False):
        self.page = page
        self.p = persona
        self.trace = trace
        self._x, self._y = 0.0, 0.0          # tracked cursor, so paths start where we left off

    # ── keyboard ─────────────────────────────────────────────────────────────
    def type(self, selector: str, text: str, *, first_field: bool = False,
             secret: bool = False) -> None:
        """Type with real key-down/key-up separation.

        Using down()/up() rather than press(delay=) is what makes the hold time observable:
        a sensor reading keyup-minus-keydown gets our actual dwell, not a synthetic one."""
        self.click(selector)
        clock = _Clock()
        prev_dwell = 0.0
        for k in self.p.keystrokes(text, first_field=first_field, typos=not secret):
            # flight is press-to-press, and the previous hold already consumed part of it
            clock.wait(max(k.flight_ms - prev_dwell, 0.0))
            tok = _TOK.get(k.key, k.key)
            self.page.keyboard.down(tok)
            clock.wait(k.dwell_ms)
            self.page.keyboard.up(tok)
            prev_dwell = k.dwell_ms
        if self.trace:
            print(f"    typed {text!r} ({len(text)} chars)")

    # ── pointer ──────────────────────────────────────────────────────────────
    def move_to(self, x: float, y: float) -> None:
        clock = _Clock()
        for m in self.p.path(self._x, self._y, x, y):
            self.page.mouse.move(m.x, m.y)
            clock.wait(m.dt_ms)              # the final dt IS the settle — see mouse.path
        self._x, self._y = x, y

    def click(self, selector: str) -> None:
        loc = self.page.locator(selector).first
        loc.scroll_into_view_if_needed(timeout=5000)
        box = loc.bounding_box()
        if not box:
            raise RuntimeError(f"no bounding box for {selector!r}")
        x, y = self.p.point_in(box)
        self.move_to(x, y)
        # Hold the button. page.mouse.click() presses and releases with no delay, which a
        # page measures as a 0.1ms click — a duration no hand can produce, and the single
        # loudest tell this driver had.
        self.page.mouse.move(x, y)
        self.page.mouse.down()
        time.sleep(self.p.click_hold_ms() / 1000.0)
        self.page.mouse.up()
        if self.trace:
            print(f"    clicked {selector} at ({x:.0f}, {y:.0f})")

    # ── scroll / idle ────────────────────────────────────────────────────────
    def scroll(self, notches: int = 20) -> None:
        clock = _Clock()
        for w in self.p.scroll(notches):
            self.page.mouse.wheel(0, w.dy)
            clock.wait(w.dt_ms)

    def idle(self, ms: float) -> None:
        """Fill an LLM round-trip with resting behaviour instead of a frozen cursor."""
        vp = self.page.viewport_size or {"width": 1280, "height": 800}
        clock = _Clock()
        for m in self.p.idle(ms, self._x, self._y, width=vp["width"], height=vp["height"]):
            self.page.mouse.move(m.x, m.y)
            clock.wait(m.dt_ms)
            self._x, self._y = m.x, m.y

    def goto(self, url: str) -> None:
        self.page.goto(url, wait_until="domcontentloaded")
        vp = self.page.viewport_size or {"width": 1280, "height": 800}
        self._x, self._y = vp["width"] / 2, vp["height"] / 2
