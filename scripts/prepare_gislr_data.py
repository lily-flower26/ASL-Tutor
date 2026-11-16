"""
Downloads a sample of GISLR (Kaggle) landmark sequences for each vocab.json
target sign that GISLR has, converts each from GISLR's parquet format into
our standard 258-dim-per-frame format, and saves them straight into
data/landmarks/<gloss>/gislr_<sequence_id>.npy -- ready for
prepare_dataset.py. No video, no mediapipe extraction needed for this data
source; it's already landmarks.

Requires data/gislr/train.csv to already exist (run check_gislr_vocab.py
first -- it downloads this as part of checking vocabulary coverage).

Usage:
    python scripts/prepare_gislr_data.py
"""
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GISLR_DIR = ROOT / "data" / "gislr"
LANDMARK_CACHE = GISLR_DIR / "train_landmark_files"
LANDMARKS_OUT = ROOT / "data" / "landmarks"
COMPETITION = "asl-signs"

POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
SAMPLES_PER_SIGN = 150  # GISLR has 350-400 per sign; using more real signers helps generalization


def load_vocab():
    with open(ROOT / "vocab.json") as f:
        return json.load(f)


def kaggle_download_file(remote_path: str, dest_path: Path) -> bool:
    if dest_path.exists():
        return True  # already cached from a previous run
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", COMPETITION,
         "-f", remote_path, "-p", str(dest_path.parent), "--force"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"    download failed for {remote_path}: {(result.stderr or result.stdout).strip()[-200:]}")
        return False
    # Kaggle sometimes wraps a single file in a zip anyway -- handle both.
    zip_path = dest_path.parent / f"{dest_path.stem}.zip"
    if zip_path.exists():
        subprocess.run(["unzip", "-o", "-q", str(zip_path), "-d", str(dest_path.parent)], check=True)
        zip_path.unlink()
    return dest_path.exists()


def parquet_to_vector_sequence(parquet_path: Path) -> np.ndarray:
    """Converts a GISLR parquet file (long format: one row per landmark per
    frame, columns frame/type/landmark_index/x/y/z) into our (num_frames,
    258) array: pose (33 x [x,y,z,visibility]) + left_hand (21 x [x,y,z]) +
    right_hand (21 x [x,y,z]). GISLR has no visibility/presence column, so
    visibility is set to 1.0 where a landmark was detected (non-NaN) and 0.0
    where it wasn't -- a reasonable stand-in for the same meaning."""
    df = pd.read_parquet(parquet_path)

    landmark_col = "landmark_index" if "landmark_index" in df.columns else "landmark_idx"

    frames = sorted(df["frame"].unique())
    sequence = np.zeros((len(frames), POSE_LANDMARKS * 4 + HAND_LANDMARKS * 3 + HAND_LANDMARKS * 3), dtype=np.float32)

    for i, frame in enumerate(frames):
        frame_df = df[df["frame"] == frame]

        pose_rows = frame_df[frame_df["type"] == "pose"]
        for _, row in pose_rows.iterrows():
            idx = int(row[landmark_col])
            if idx < POSE_LANDMARKS and not np.isnan(row["x"]):
                base = idx * 4
                sequence[i, base:base + 3] = [row["x"], row["y"], row["z"]]
                sequence[i, base + 3] = 1.0  # visibility stand-in

        left_rows = frame_df[frame_df["type"] == "left_hand"]
        left_base = POSE_LANDMARKS * 4
        for _, row in left_rows.iterrows():
            idx = int(row[landmark_col])
            if idx < HAND_LANDMARKS and not np.isnan(row["x"]):
                base = left_base + idx * 3
                sequence[i, base:base + 3] = [row["x"], row["y"], row["z"]]

        right_rows = frame_df[frame_df["type"] == "right_hand"]
        right_base = POSE_LANDMARKS * 4 + HAND_LANDMARKS * 3
        for _, row in right_rows.iterrows():
            idx = int(row[landmark_col])
            if idx < HAND_LANDMARKS and not np.isnan(row["x"]):
                base = right_base + idx * 3
                sequence[i, base:base + 3] = [row["x"], row["y"], row["z"]]

    return np.nan_to_num(sequence, nan=0.0).astype(np.float32)


def main():
    train_csv = GISLR_DIR / "train.csv"
    if not train_csv.exists():
        print(f"{train_csv} not found. Run scripts/check_gislr_vocab.py first.")
        return

    vocab = load_vocab()
    target_signs = {s.lower() for s in vocab["target_signs"]}

    df = pd.read_csv(train_csv)
    df["sign_lower"] = df["sign"].str.lower()
    available = target_signs & set(df["sign_lower"].unique())
    skipped = target_signs - available
    if skipped:
        print(f"Not in GISLR, skipping (need another source for these): {sorted(skipped)}")

    for gloss in sorted(available):
        gloss_rows = df[df["sign_lower"] == gloss]
        sample = gloss_rows.sample(n=min(SAMPLES_PER_SIGN, len(gloss_rows)), random_state=42)

        out_dir = LANDMARKS_OUT / gloss
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{gloss}] preparing {len(sample)} sequences...")
        converted = 0
        for _, row in sample.iterrows():
            remote_path = row["path"]  # e.g. train_landmark_files/16069/100015657.parquet
            sequence_id = row["sequence_id"]
            out_path = out_dir / f"gislr_{sequence_id}.npy"
            if out_path.exists():
                converted += 1
                continue

            local_parquet = GISLR_DIR / remote_path
            if not kaggle_download_file(remote_path, local_parquet):
                continue

            try:
                seq = parquet_to_vector_sequence(local_parquet)
                if seq.shape[0] == 0:
                    print(f"    no frames in {remote_path}, skipping")
                    continue
                np.save(out_path, seq)
                converted += 1
            except Exception as e:
                print(f"    failed to convert {remote_path}: {e}")

        print(f"[{gloss}] {converted}/{len(sample)} sequences ready in {out_dir}")

    print("\nDone. Run scripts/prepare_dataset.py next.")


if __name__ == "__main__":
    main()
