"""
Checks whether the "Google - Isolated Sign Language Recognition" dataset on
Kaggle (internally called GISLR, competition slug "asl-signs") covers your
vocab.json target signs, and whether per-file selective download works for
it (avoiding pulling the full ~40GB dataset).

Why this dataset, given the WLASL/YouTube problems: GISLR is landmark data,
not video -- contributed by roughly a hundred different real signers via
Google's PopSign project, hosted directly on Kaggle. No YouTube dependency
at all, so none of the dead-link or bot-blocking problems apply. It's built
on the MacArthur-Bates CDI early-vocabulary word list, so common functional
words (eat, drink, more, help, water, finish, book) are very likely
included; more formal/social words (thank you, sorry, hello, name, friend)
are less certain -- which is exactly what this script checks, rather than
guessing.

One-time setup before running this (about 5 minutes):
  1. Create a free account at https://www.kaggle.com
  2. Go to https://www.kaggle.com/settings -> API -> "Create New Token"
     (downloads kaggle.json to your Downloads folder)
  3. Move it into place and lock down its permissions:
       mkdir -p ~/.kaggle
       mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
       chmod 600 ~/.kaggle/kaggle.json
  4. pip install kaggle pyarrow   (already in requirements.txt)
  5. Visit https://www.kaggle.com/competitions/asl-signs/rules and click
     "I Understand and Accept" -- Kaggle requires this before its API will
     let you download, even for a long-closed competition.

This version prints everything it does (raw kaggle CLI output, directory
listings after each step) rather than trying to silently guess -- if
something's off about how Kaggle names or nests files, that's the fastest
way to see it and adjust, instead of a bare "it didn't work."

Usage:
    python scripts/check_gislr_vocab.py
"""
import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GISLR_DIR = ROOT / "data" / "gislr"
COMPETITION = "asl-signs"


def load_vocab():
    with open(ROOT / "vocab.json") as f:
        return json.load(f)


def run(cmd) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    out = (result.stdout or "").strip()
    err = (result.stderr or "").strip()
    if out:
        print(out)
    if err:
        print(err)
    print(f"  (exit code {result.returncode})\n")
    return result


def list_dir(label: str, d: Path):
    print(f"--- {label}: {d} ---")
    if not d.exists():
        print("  (does not exist)")
    else:
        entries = sorted(d.rglob("*"))
        if not entries:
            print("  (empty)")
        for p in entries[:40]:
            print(f"  {p.relative_to(d)}" + ("/" if p.is_dir() else f"  ({p.stat().st_size} bytes)"))
        if len(entries) > 40:
            print(f"  ... and {len(entries) - 40} more")
    print()


def kaggle_download_file(remote_path: str, dest_dir: Path) -> bool:
    dest_dir.mkdir(parents=True, exist_ok=True)
    result = run(["kaggle", "competitions", "download", "-c", COMPETITION,
                  "-f", remote_path, "-p", str(dest_dir), "--force"])
    return result.returncode == 0


def unzip_everything(dest_dir: Path):
    for zip_path in list(dest_dir.glob("*.zip")):
        print(f"Unzipping {zip_path.name}...")
        subprocess.run(["unzip", "-o", "-q", str(zip_path), "-d", str(dest_dir)], check=True)
        zip_path.unlink()


def find_file(dest_dir: Path, filename: str) -> Path | None:
    matches = list(dest_dir.rglob(filename))
    return matches[0] if matches else None


def main():
    vocab = load_vocab()
    target_signs = {s.lower() for s in vocab["target_signs"]}

    print("=== Step 1: what files does Kaggle expose for this competition? ===\n")
    run(["kaggle", "competitions", "files", "-c", COMPETITION])

    print("=== Step 2: download train.csv ===\n")
    kaggle_download_file("train.csv", GISLR_DIR)
    unzip_everything(GISLR_DIR)
    list_dir("data/gislr after download+unzip", GISLR_DIR)

    train_csv_path = find_file(GISLR_DIR, "train.csv")
    if train_csv_path is None:
        print("Still no train.csv anywhere under data/gislr. Paste this entire output back -- "
              "the 'kaggle competitions files' listing above will show the exact filename/path "
              "Kaggle actually uses, which may not be literally 'train.csv'.")
        return

    df = pd.read_csv(train_csv_path)
    print(f"Loaded {train_csv_path}: {len(df)} total sequences across {df['sign'].nunique()} signs\n")

    counts = df["sign"].str.lower().value_counts()

    print("=== Your target signs: found in GISLR? ===")
    found, missing = [], []
    for gloss in sorted(target_signs):
        n = counts.get(gloss, 0)
        if n > 0:
            print(f"  {gloss:15s} FOUND -- {n} samples available")
            found.append(gloss)
        else:
            print(f"  {gloss:15s} not found in GISLR's 250-word vocabulary")
            missing.append(gloss)

    print(f"\n{len(found)}/{len(target_signs)} of your target signs are in GISLR.")
    if missing:
        print(f"Missing: {missing}")
        print("These will still need WLASL, self-recording, or swapping for a replacement word.")

    if found:
        test_gloss = found[0]
        sample_path = df[df["sign"].str.lower() == test_gloss].iloc[0]["path"]
        print(f"\n=== Step 3: test single-file selective download using '{test_gloss}' ({sample_path}) ===\n")
        test_dir = GISLR_DIR / "_test"
        success = kaggle_download_file(sample_path, test_dir)
        unzip_everything(test_dir)
        list_dir("data/gislr/_test after download+unzip", test_dir)
        if success:
            print("Selective per-file download WORKS -- we can pull just the samples we need "
                  "instead of the full dataset.")
        else:
            print("Selective per-file download did NOT work for this competition. We'd need the "
                  "full dataset instead (~40GB), via the Kaggle website's 'Download All' button, "
                  "or `kaggle competitions download -c asl-signs` with no -f flag. Report back "
                  "before doing that -- 40GB is a lot for a proof-of-concept and we should "
                  "reconsider whether that's worth it.")


if __name__ == "__main__":
    main()
