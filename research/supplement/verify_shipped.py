#!/usr/bin/env python3
"""Check every shipped file against the SHA-256 values in the evidence ledger.

Run from anywhere:
    python3 verify_shipped.py

Exit code 0 iff every file present under this package that is listed in
paper/results/evidence_ledger.json hashes to its recorded value.
Sources that are intentionally not shipped (pinned in README.md) are reported
separately and do not fail the check.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LEDGER = ROOT / "paper" / "results" / "evidence_ledger.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    ok, pinned, bad = [], [], []
    for src in ledger["sources"]:
        rel, want = src["path"], src["sha256"]
        f = ROOT / rel
        if not f.exists():
            pinned.append(rel)
            continue
        if sha256(f) == want:
            ok.append(rel)
        else:
            bad.append(rel)
    print(f"shipped and verified : {len(ok)}")
    print(f"pinned, not shipped  : {len(pinned)}")
    for rel in pinned:
        print(f"    {rel}")
    print(f"hash mismatches      : {len(bad)}")
    for rel in bad:
        print(f"    MISMATCH {rel}")
    if bad:
        print("FAIL")
        return 1
    print("OK — every shipped file matches the ledger")
    return 0


if __name__ == "__main__":
    sys.exit(main())
