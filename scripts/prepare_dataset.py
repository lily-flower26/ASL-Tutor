"""
Turn data/landmarks/<gloss>/*.npy (variable-length landmark sequences) into a
fixed-length, model-ready dataset: resamples every clip to vocab.json's
sequence_length, builds train/test splits, and saves everything to
data/prepared/dataset.npz plus a label map JSON.

This step only needs numpy/pandas/sklearn (no mediapipe/tensorflow), so it can
run anywhere, including the cloud sandbox, once landmark .npy files exist.

Usage:
    python scripts/prepare_dataset.py
"""
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
LANDMARKS_DIR = ROOT / "data" / "landmarks"
PREPARED_DIR = ROOT / "data" / "prepared"


def load_vocab():
    with open(ROOT / "vocab.json") as f:
        return json.load(f)


def resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Uniformly resample (via nearest-index picking) a (T, D) sequence to (target_len, D).

    Works whether the clip is longer or shorter than target_len. Zero-length
    clips are turned into an all-zero sequence rather than raising.
    """
    num_frames, feature_dim = seq.shape if seq.ndim == 2 else (0, 0)
    if num_frames == 0:
        return np.zeros((target_len, feature_dim or 1), dtype=np.float32)
    if num_frames == target_len:
        return seq
    indices = np.linspace(0, num_frames - 1, target_len).round().astype(int)
    return seq[indices]


def main():
    vocab = load_vocab()
    seq_len = vocab["sequence_length"]
    min_clips = vocab.get("min_clips_per_sign", 8)

    gloss_dirs = sorted([d for d in LANDMARKS_DIR.iterdir() if d.is_dir()]) if LANDMARKS_DIR.exists() else []
    if not gloss_dirs:
        print(f"No landmark data found at {LANDMARKS_DIR}. Run extract_landmarks.py first.")
        return

    X, y_labels = [], []
    kept_glosses = []
    for gloss_dir in gloss_dirs:
        clips = sorted(gloss_dir.glob("*.npy"))
        if len(clips) < min_clips:
            print(f"Skipping '{gloss_dir.name}': only {len(clips)} clips (need >= {min_clips})")
            continue
        kept_glosses.append(gloss_dir.name)
        for clip_path in clips:
            seq = np.load(clip_path)
            X.append(resample_sequence(seq, seq_len))
            y_labels.append(gloss_dir.name)

    if not kept_glosses:
        print("No sign had enough clips to train on. Lower min_clips_per_sign in vocab.json, "
              "or gather more data for the low-count signs.")
        return

    X = np.stack(X).astype(np.float32)  # (N, seq_len, feature_dim)
    label_list = sorted(set(y_labels))
    label_to_idx = {label: i for i, label in enumerate(label_list)}
    y = np.array([label_to_idx[label] for label in y_labels], dtype=np.int64)

    # Guard against classes too small to stratify (need >= 2 samples per class in each split).
    class_counts = np.bincount(y)
    stratify = y if class_counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        PREPARED_DIR / "dataset.npz",
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
    )
    with open(PREPARED_DIR / "label_map.json", "w") as f:
        json.dump(label_list, f, indent=2)

    print(f"Prepared dataset: {X.shape[0]} clips across {len(label_list)} signs "
          f"(train={X_train.shape[0]}, test={X_test.shape[0]})")
    print(f"Signs included: {label_list}")
    print(f"Saved to {PREPARED_DIR}")


if __name__ == "__main__":
    main()
