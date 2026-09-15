"""Async Playwright driver — the same executor, plus real coalesced pointer samples.

The sync driver cannot produce coalescing, and the reason is structural rather than a missing
feature: every `mouse.move` is a blocking IPC round-trip, so each sample gets its own
compositor frame and `getCoalescedEvents()` returns exactly 1 forever. That constant is itself
a signal — real input occasionally merges samples, because hardware reports independently of
the display.

Dispatching a marked run WITHOUT awaiting lets the samples arrive faster than the browser can
paint, so the browser genuinely coalesces them. The coalescing comes out of its own input
pipeline; nothing patches `getCoalescedEvents`.

    import asyncio
    from playwright.async_api import async_playwright
    from behaviourforge import forge

    async def main():
        async with async_playwright() as pw:
            b = await pw.chromium.launch()
            page = await b.new_page()
            d = AsyncDriver(page, forge(seed=4712))
            await d.goto("https://example.com")
            await d.click("#login")
"""
from __future__ import annotations

import asyncio
import time

_TOK = {" ": "Space", "\n": "Enter", "\t": "Tab"}


class _Clock:
    """Absolute scheduling — see the note in camoufox_driver. Sleeping between events
    inflates press-to-press intervals and turns IPC jitter into negative autocorrelation."""

    def __init__(self):
        self.t = time.perf_counter()

    async def wait(self, ms: float) -> None:
        self.t += ms / 1000.0
        delay = self.t - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)


class AsyncDriver:
    def __init__(self, page, persona, *, trace: bool = False):
        self.page = page
        self.p = persona
        self.trace = trace
        self._x, self._y = 0.0, 0.0

    # ── pointer ──────────────────────────────────────────────────────────────
    async def move_to(self, x: float, y: float) -> None:
        plan = self.p.path(self._x, self._y, x, y)
        clock = _Clock()
        i = 0
        while i < len(plan):
            if not plan[i].coalesce:
                await self.page.mouse.move(plan[i].x, plan[i].y)
                await clock.wait(plan[i].dt_ms)
                i += 1
                continue
            # A missed compositor frame: fire the whole run concurrently so they land in one
            # paint, then pay back the accumulated time so the emitted rate stays at 59Hz.
            run = []
            while i < len(plan) and plan[i].coalesce:
                run.append(plan[i])
                i += 1
            await asyncio.gather(*[self.page.mouse.move(m.x, m.y) for m in run])
            await clock.wait(sum(m.dt_ms for m in run))
        self._x, self._y = x, y

    async def click(self, selector: str) -> None:
        loc = self.page.locator(selector).first
        await loc.scroll_into_view_if_needed(timeout=5000)
        box = await loc.bounding_box()
        if not box:
            raise RuntimeError(f"no bounding box for {selector!r}")
        x, y = self.p.point_in(box)
        await self.move_to(x, y)
        await self.page.mouse.move(x, y)
        await self.page.mouse.down()
        await asyncio.sleep(self.p.click_hold_ms() / 1000.0)
        await self.page.mouse.up()

    # ── keyboard ─────────────────────────────────────────────────────────────
    async def type(self, selector: str, text: str, *, first_field: bool = False,
                   secret: bool = False) -> None:
        await self.click(selector)
        clock = _Clock()
        prev_dwell = 0.0
        for k in self.p.keystrokes(text, first_field=first_field, typos=not secret):
            await clock.wait(max(k.flight_ms - prev_dwell, 0.0))
            tok = _TOK.get(k.key, k.key)
            await self.page.keyboard.down(tok)
            await clock.wait(k.dwell_ms)
            await self.page.keyboard.up(tok)
            prev_dwell = k.dwell_ms

    # ── scroll / idle ────────────────────────────────────────────────────────
    async def scroll(self, notches: int = 20) -> None:
        clock = _Clock()
        for w in self.p.scroll(notches):
            await self.page.mouse.wheel(0, w.dy)
            await clock.wait(w.dt_ms)

    async def idle(self, ms: float) -> None:
        vp = self.page.viewport_size or {"width": 1280, "height": 800}
        clock = _Clock()
        for m in self.p.idle(ms, self._x, self._y, width=vp["width"], height=vp["height"]):
            await self.page.mouse.move(m.x, m.y)
            await clock.wait(m.dt_ms)
            self._x, self._y = m.x, m.y

    async def goto(self, url: str) -> None:
        await self.page.goto(url, wait_until="domcontentloaded")
        vp = self.page.viewport_size or {"width": 1280, "height": 800}
        self._x, self._y = vp["width"] / 2, vp["height"] / 2
