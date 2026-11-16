"""
The moment of truth: point your own webcam at yourself, sign something from
vocab.json, and see whether the trained model recognizes it in real time.

This is a plain OpenCV window (not a browser), which is the fastest way to
sanity-check the model before building the full web app. Press 'q' to quit.

Run this on your own machine (needs a real webcam + mediapipe's models,
which requires normal internet access at least once to fetch/cache).

Usage:
    python scripts/live_demo.py
"""
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras

from landmark_extractor import LandmarkExtractor, draw_landmarks

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

PREDICT_EVERY_N_FRAMES = 8       # how often to run inference (perf/latency tradeoff)

# Confidence bands are for COLOR ONLY now -- the prediction always shows.
# (An earlier version hid anything below 60% behind "...", which made it
# look like nothing was being recognized when really the guess was just
# less confident than an arbitrary cutoff. Always showing the real number
# is more honest and more useful for figuring out what's actually going on.)
HIGH_CONFIDENCE = 0.5
MEDIUM_CONFIDENCE = 0.3


def load_model_and_labels():
    with open(MODELS_DIR / "vocab_used.json") as f:
        seq_len = json.load(f)["sequence_length"]
    with open(MODELS_DIR / "label_map.json") as f:
        label_list = json.load(f)
    model = keras.models.load_model(MODELS_DIR / "sign_classifier.keras")
    return model, label_list, seq_len


def resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    num_frames = seq.shape[0]
    if num_frames == 0:
        return np.zeros((target_len, seq.shape[1] if seq.ndim == 2 else 258), dtype=np.float32)
    if num_frames == target_len:
        return seq
    indices = np.linspace(0, num_frames - 1, target_len).round().astype(int)
    return seq[indices]


def main():
    model, label_list, seq_len = load_model_and_labels()
    print(f"Loaded model for {len(label_list)} signs: {label_list}")

    buffer = deque(maxlen=seq_len * 2)  # keep a bit more than seq_len so resampling has room to work with
    last_prediction = "(warming up -- move your hands into frame)"
    last_color = (255, 255, 255)
    frame_count = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam (index 0). Try a different camera index or check permissions.")
        return

    with LandmarkExtractor() as extractor:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            # IMPORTANT: run detection on the RAW, unflipped frame -- GISLR's
            # training data wasn't captured through a mirrored preview, so
            # feeding the model a mirror-flipped image systematically swaps
            # left/right and was likely why predictions looked random. We
            # still flip a separate copy for on-screen display, since a
            # mirror view feels natural to look at.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vector, points = extractor.process_live_frame(frame_rgb)
            buffer.append(vector)
            frame_count += 1

            if frame_count % PREDICT_EVERY_N_FRAMES == 0 and len(buffer) >= seq_len:
                seq = resample_sequence(np.array(buffer), seq_len)
                probs = model.predict(seq[np.newaxis, ...], verbose=0)[0]
                best_idx = int(np.argmax(probs))
                confidence = float(probs[best_idx])
                last_prediction = f"{label_list[best_idx]} ({confidence:.0%})"
                if confidence >= HIGH_CONFIDENCE:
                    last_color = (0, 255, 0)        # green: confident
                elif confidence >= MEDIUM_CONFIDENCE:
                    last_color = (0, 255, 255)      # yellow: unsure
                else:
                    last_color = (0, 0, 255)        # red: low confidence, probably noise

            display = cv2.flip(frame, 1)
            draw_landmarks(display, points, mirror=True)

            cv2.rectangle(display, (0, 0), (display.shape[1], 60), (0, 0, 0), -1)
            cv2.putText(display, last_prediction, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, last_color, 2)
            cv2.imshow("ASL sign recognition (press q to quit)", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
