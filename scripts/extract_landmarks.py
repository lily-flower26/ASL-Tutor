"""
Walk data/raw/<gloss>/*.mp4, run MediaPipe (hand + pose landmarkers) over
every frame, and save a per-clip landmark sequence to
data/landmarks/<gloss>/<clip_id>.npy as a (num_frames, 258) float32 array.

This is the exact same feature representation the live webcam demo
(live_demo.py) uses, which is what lets a model trained on these clips
generalize to a live camera feed. See landmark_extractor.py for how the
258-dim vector is built.

Run this on your own machine, not in the cloud sandbox (needs normal internet
access the first time, to download and cache the MediaPipe model files).

Usage:
    python scripts/extract_landmarks.py
"""
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from landmark_extractor import LandmarkExtractor, FEATURE_DIM

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
LANDMARKS_DIR = ROOT / "data" / "landmarks"


def extract_from_video(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0:
        fps = 25.0

    frames = []
    with LandmarkExtractor() as extractor:
        frame_index = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(extractor.process_video_frame(frame_rgb, frame_index, fps))
            frame_index += 1
    cap.release()
    return np.array(frames, dtype=np.float32) if frames else np.zeros((0, FEATURE_DIM), dtype=np.float32)


def main():
    if not RAW_DIR.exists():
        print(f"No raw videos found at {RAW_DIR}. Run download_wlasl_subset.py first, "
              f"or drop your own recorded clips into data/raw/<gloss>/*.mp4")
        return

    gloss_dirs = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    if not gloss_dirs:
        print(f"{RAW_DIR} is empty. Add clips under data/raw/<gloss>/*.mp4 first.")
        return

    for gloss_dir in gloss_dirs:
        gloss = gloss_dir.name
        out_dir = LANDMARKS_DIR / gloss
        out_dir.mkdir(parents=True, exist_ok=True)

        videos = sorted(gloss_dir.glob("*.mp4"))
        for video_path in tqdm(videos, desc=gloss):
            out_path = out_dir / f"{video_path.stem}.npy"
            if out_path.exists():
                continue
            seq = extract_from_video(video_path)
            if seq.shape[0] == 0:
                print(f"  WARNING: no frames read from {video_path}, skipping")
                continue
            np.save(out_path, seq)

    print("Done. Landmark sequences saved under data/landmarks/<gloss>/*.npy")


if __name__ == "__main__":
    main()
