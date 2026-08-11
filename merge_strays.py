"""
One-off recovery: merge the stray root-level ledgers into the real ones.

Today's run used the wrong paths and wrote a fresh ledger at the repo root
containing only today's verdicts. Those verdicts were logged at this
morning's prices, which is exactly what we want to keep, so they're merged
into data/ledger/ rather than discarded and re-logged later at moved lines.

Existing records always win: this only ADDS games the real ledger is
missing. Nothing already in data/ledger/ is modified.

Run from the repo root:   python3 merge_strays.py
Then delete the strays:   rm f5_forward_log.json f5r_forward_log.json
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

PAIRS = [
    (Path("f5_forward_log.json"),  Path("data/ledger/f5_forward_log.json")),
    (Path("f5r_forward_log.json"), Path("data/ledger/f5r_forward_log.json")),
]


def main():
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for stray, real in PAIRS:
        if not stray.exists():
            print(f"  {stray} — not present, skipping")
            continue
        if not real.exists():
            print(f"  !! {real} missing — not touching {stray}. "
                  f"Investigate before proceeding.")
            continue

        stray_log = json.loads(stray.read_text())
        real_log = json.loads(real.read_text())

        # back up the good ledger before writing anything
        backup = real.with_suffix(f".backup-{stamp}.json")
        shutil.copy2(real, backup)

        added, already = 0, 0
        for key, rec in stray_log.items():
            if key in real_log:
                already += 1
                continue
            real_log[key] = rec
            added += 1

        if added:
            real.write_text(json.dumps(real_log, indent=1))
        print(f"  {real.name}: {len(real_log) - added} existing "
              f"+ {added} recovered from stray ({already} already present) "
              f"-> {len(real_log)} total   [backup: {backup.name}]")

    print("\nDone. Verify, then remove the strays:")
    print("  rm f5_forward_log.json f5r_forward_log.json")


if __name__ == "__main__":
    main()
