"""CLI — inspect a persona, audit a fleet, rebuild the corpus."""
from __future__ import annotations

import argparse
import json
import sys


def _cmd_show(args) -> int:
    from behaviourforge import forge
    p = forge(args.seed, corpus=args.corpus)
    if args.json:
        print(json.dumps(p.motor.to_dict(), indent=2))
        return 0

    m = p.motor
    print(f"persona  seed={m.seed}  locale={m.locale}  device={m.device}")
    print(f"  typing   base {m.base_ms:.0f}ms  dwell {m.dwell_ms:.0f}ms  "
          f"think {m.think_ms:.0f}ms  errors {m.error_rate * 100:.1f}%")
    print(f"  pointer  speed {m.mouse_speed:.2f}  curve {m.mouse_curve:.2f}  "
          f"overshoot {m.mouse_overshoot:.2f}  tremor {m.mouse_tremor:.1f}px")
    print(f"  measured bigrams: {len(m.bigram_mult)}")

    if args.text:
        keys = p.keystrokes(args.text, first_field=True)
        total = sum(k.total_ms for k in keys)
        print(f'\n  "{args.text}"  ->  {len(keys)} events, {total / 1000:.2f}s '
              f"({len(args.text) / (total / 60000):.0f} cpm)")
        for k in keys:
            mark = {"typo": " <- typo", "backspace": " <- correct"}.get(k.kind, "")
            print(f"    {k.key!r:<12} flight {k.flight_ms:6.1f}ms  "
                  f"dwell {k.dwell_ms:5.1f}ms{mark}")
    return 0


def _cmd_audit(args) -> int:
    from behaviourforge import audit
    r = audit(args.n, corpus=args.corpus)
    print(r)
    return 0 if r.passed else 1


def _cmd_build(args) -> int:
    from behaviourforge.corpus import build_from_aalto
    n = build_from_aalto(args.inputs, args.out, limit=args.limit,
                         min_keys=args.min_keys, from_zip=args.zip)
    print(f"wrote {n} motor vectors -> {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="behaviourforge",
                                 description="Coherent synthetic motor identities from a seed.")
    ap.add_argument("--corpus", default=None, help="corpus JSON path (default: bundled Aalto)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("show", help="inspect one persona, optionally typing a string")
    s.add_argument("seed", type=int)
    s.add_argument("--text", help="plan this text and print the keystroke timings")
    s.add_argument("--json", action="store_true", help="dump the motor vector as JSON")
    s.set_defaults(fn=_cmd_show)

    a = sub.add_parser("audit", help="forge a fleet and measure its diversity")
    a.add_argument("-n", type=int, default=10_000)
    a.set_defaults(fn=_cmd_audit)

    b = sub.add_parser("build-corpus", help="extract a corpus from the Aalto dataset")
    b.add_argument("inputs", nargs="+", help="TSV file(s), or Keystrokes.zip with --zip")
    b.add_argument("--zip", action="store_true", help="read straight from the zip, no unpack")
    b.add_argument("--out", default="corpus.json")
    b.add_argument("--limit", type=int, default=3000, help="cap participants")
    b.add_argument("--min-keys", type=int, default=150)
    b.set_defaults(fn=_cmd_build)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
