"""BehaviourForge — coherent synthetic motor identities from a seed.

    from behaviourforge import forge

    p = forge(seed=42)
    for k in p.keystrokes("hello@example.com"):
        driver.press(k.key, hold_ms=k.dwell_ms, after_wait_ms=k.flight_ms)

The library emits timed PLANS and never touches a browser. Executing them is a handful of
lines against Playwright, CDP, Selenium, or anything else that can press a key. Keeping the
model on the pure side of that line is what makes it testable without a browser, portable
across drivers, and cheap enough to run ten thousand times.

See README.md for what is measured and what is still guessed.
"""
from behaviourforge.audit import AuditReport, audit
from behaviourforge.keystroke import Key
from behaviourforge.mouse import Move, Wheel
from behaviourforge.persona import Motor, Persona, forge

__version__ = "0.1.0"
__all__ = ["forge", "Persona", "Motor", "Key", "Move", "Wheel", "audit", "AuditReport"]
