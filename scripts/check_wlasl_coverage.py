"""
Diagnose *why* download counts came out low: reads the WLASL metadata that
download_wlasl_subset.py already cached locally (data/WLASL_v0.3.json) and
reports, for each of your target signs, the TOTAL number of instances listed
in the dataset -- regardless of whether the video actually downloaded.

This tells you which of two different problems you have, which matters
because the fix is different for each:

  - Total instances is also low (e.g. 3-5)  -> WLASL just doesn't have much
    footage of that word. Lowering min_clips_per_sign formalizes a real
    ceiling; better fix is swapping that word for a better-covered one, or
    supplementing with your own recorded clips.

  - Total instances is high (e.g. 15-30) but you only downloaded 3 -> most
    of the failures were dead video links, which is normal for this
    scraped, multi-year-old dataset. Re-running the download script won't
    magically fix dead links, but it's worth knowing the data exists in
    principle (e.g. you could try again later, or accept a lower success
    rate as expected rather than a data-scarcity problem).

Also prints the best-covered signs in the *entire* WLASL vocabulary, as
candidates if you want to swap out any chronically low-coverage words.

Usage:
    python scripts/check_wlasl_coverage.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WLASL_JSON_LOCAL = ROOT / "data" / "WLASL_v0.3.json"


def load_vocab():
    with open(ROOT / "vocab.json") as f:
        return json.load(f)


def main():
    if not WLASL_JSON_LOCAL.exists():
        print(f"{WLASL_JSON_LOCAL} not found -- run download_wlasl_subset.py first "
              f"(it caches the metadata there even if individual clip downloads fail).")
        return

    with open(WLASL_JSON_LOCAL) as f:
        metadata = json.load(f)

    vocab = load_vocab()
    target_signs = {s.lower() for s in vocab["target_signs"]}
    min_needed = vocab.get("min_clips_per_sign", 8)

    counts = {entry["gloss"].lower(): len(entry["instances"]) for entry in metadata}

    print("=== Total WLASL instances for your target signs (before download attempts) ===")
    for gloss in sorted(target_signs):
        total = counts.get(gloss)
        if total is None:
            print(f"  {gloss:15s} NOT FOUND in WLASL -- check spelling/gloss variant")
        elif total < min_needed:
            print(f"  {gloss:15s} {total:3d} total instances  <- dataset ceiling is below "
                  f"min_clips_per_sign ({min_needed}); more downloading attempts won't help")
        else:
            print(f"  {gloss:15s} {total:3d} total instances  <- data exists; low download "
                  f"count is likely dead links, not scarcity")

    print(f"\n=== Best-covered signs in all of WLASL (candidates to swap in) ===")
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    for gloss, total in ranked[:30]:
        flag = "  <- already in your vocab" if gloss in target_signs else ""
        print(f"  {gloss:20s} {total:3d} instances{flag}")


if __name__ == "__main__":
    main()
