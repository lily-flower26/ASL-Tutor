"""
Download a small subset of the WLASL dataset for the signs listed in vocab.json.

WLASL (Word-Level American Sign Language) is a public research dataset: a JSON
file maps each gloss (word) to a list of "instances" -- clips of someone signing
that word, each with a source URL and a frame_start/frame_end range to trim to.
Videos live on YouTube and a handful of other sites, not in the JSON itself, so
this script downloads each clip individually. Expect some links to be dead --
that's normal for a multi-year-old scraped dataset. The script skips failures
and reports a summary at the end instead of crashing.

Run this on your own machine (real internet access), not in the cloud sandbox.

Usage:
    python scripts/download_wlasl_subset.py

Requires: requests, yt-dlp, ffmpeg (`brew install ffmpeg` on Mac) on PATH.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
WLASL_JSON_URL = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
WLASL_JSON_LOCAL = ROOT / "data" / "WLASL_v0.3.json"


def load_vocab():
    with open(ROOT / "vocab.json") as f:
        return json.load(f)


def fetch_wlasl_metadata():
    if WLASL_JSON_LOCAL.exists():
        print(f"Using cached metadata at {WLASL_JSON_LOCAL}")
        with open(WLASL_JSON_LOCAL) as f:
            return json.load(f)

    print(f"Downloading WLASL metadata from {WLASL_JSON_URL} ...")
    resp = requests.get(WLASL_JSON_URL, timeout=60)
    resp.raise_for_status()
    WLASL_JSON_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    with open(WLASL_JSON_LOCAL, "w") as f:
        f.write(resp.text)
    return json.loads(resp.text)


def download_direct(url: str, dest: Path) -> bool:
    try:
        resp = requests.get(url, timeout=30, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"    direct download failed: {e}")
        return False



# If yt-dlp keeps failing with "Sign in to confirm you're not a bot", set
# this to "chrome" or "safari" (whichever browser you're logged into YouTube
# with, and close it first -- yt-dlp needs to read its cookie file, which
# some browsers lock while running). This is a known, ongoing YouTube
# anti-scraping measure as of 2026, not specific to this dataset -- see
# https://github.com/yt-dlp/yt-dlp/issues/12264
COOKIES_FROM_BROWSER = None  # e.g. "chrome" or "safari"


def download_youtube(url: str, dest: Path) -> bool:
    cmd = ["yt-dlp", "-f", "mp4", "-o", str(dest)]
    if COOKIES_FROM_BROWSER:
        cmd += ["--cookies-from-browser", COOKIES_FROM_BROWSER]
    cmd.append(url)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0 or not dest.exists():
            # Show the real reason (dead video, bot-check, unsupported format, etc.)
            # instead of swallowing it -- these come back different enough
            # that the fix differs (retry vs. cookies vs. give up on this clip).
            tail = (result.stderr or result.stdout or "").strip().splitlines()
            reason = tail[-1] if tail else f"exit code {result.returncode}"
            print(f"    yt-dlp failed: {reason}")
            return False
        return True
    except Exception as e:
        print(f"    yt-dlp failed: {e}")
        return False


def trim_clip(src: Path, dest: Path, frame_start: int, frame_end: int, fps_hint: float = 25.0):
    """Trim src to [frame_start, frame_end] frames using ffmpeg, saved to dest."""
    if frame_start in (-1, 0) and frame_end == -1:
        # No trimming needed/possible with metadata; just copy.
        src.rename(dest)
        return
    start_sec = max(frame_start, 0) / fps_hint
    # frame_end == -1 means "to the end of the clip"
    args = ["ffmpeg", "-y", "-i", str(src), "-ss", str(start_sec)]
    if frame_end != -1:
        duration = max(frame_end - max(frame_start, 0), 1) / fps_hint
        args += ["-t", str(duration)]
    args += ["-c", "copy", str(dest)]
    subprocess.run(args, check=True, capture_output=True)
    src.unlink(missing_ok=True)


def main():
    vocab = load_vocab()
    target_signs = {s.lower() for s in vocab["target_signs"]}
    metadata = fetch_wlasl_metadata()

    found_glosses = {entry["gloss"].lower() for entry in metadata}
    missing = target_signs - found_glosses
    if missing:
        print(f"WARNING: these glosses aren't in WLASL metadata (check spelling): {sorted(missing)}")

    summary = {}
    for entry in metadata:
        gloss = entry["gloss"].lower()
        if gloss not in target_signs:
            continue

        gloss_dir = RAW_DIR / gloss
        gloss_dir.mkdir(parents=True, exist_ok=True)
        ok_count = 0

        for inst in entry["instances"]:
            video_id = inst["video_id"]
            url = inst["url"]
            frame_start = inst.get("frame_start", 0)
            frame_end = inst.get("frame_end", -1)

            raw_tmp = gloss_dir / f"{video_id}_raw.mp4"
            final_path = gloss_dir / f"{video_id}.mp4"
            if final_path.exists():
                ok_count += 1
                continue

            print(f"[{gloss}] fetching {video_id} from {url}")
            if "youtube" in url or "youtu.be" in url:
                success = download_youtube(url, raw_tmp)
            else:
                success = download_direct(url, raw_tmp)

            if not success:
                continue

            try:
                trim_clip(raw_tmp, final_path, frame_start, frame_end)
                ok_count += 1
            except Exception as e:
                print(f"    trim failed for {video_id}: {e}")
                raw_tmp.unlink(missing_ok=True)

        summary[gloss] = ok_count
        print(f"[{gloss}] {ok_count} clips downloaded")

    print("\n=== Download summary ===")
    min_needed = vocab.get("min_clips_per_sign", 8)
    for gloss in sorted(target_signs):
        count = summary.get(gloss, 0)
        flag = "OK" if count >= min_needed else "LOW (consider dropping or re-running)"
        print(f"  {gloss:15s} {count:3d} clips   {flag}")


if __name__ == "__main__":
    main()
