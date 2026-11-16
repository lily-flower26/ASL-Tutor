"""
A structured self-test with LIVE feedback: walks through each sign in your
trained vocabulary one at a time and continuously shows the model's current
guess, updating in real time, WHILE you sign -- not a blind record-then-
reveal. You can see it change its mind as you move, just like live_demo.py,
except now each turn has a known target sign so we can score it.

Since you don't know these signs from memory yet, look each one up first
(Lifeprint.com, SigningSavvy, or search "ASL sign for <word>" on YouTube)
so you have something correct to imitate. This measures how well the model
recognizes YOUR attempt -- a different, harder question than the "test
accuracy" train.py printed, which only measured performance against other
(fluent) GISLR signers.

Controls during each sign's turn:
    n / SPACE  - happy with it, move to the next sign early
    q          - quit the quiz (scores whatever's done so far)

Usage:
    python scripts/quiz_mode.py
"""
import json
import random
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras

from landmark_extractor import LandmarkExtractor, draw_landmarks

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

PREP_SECONDS = 2.5
ATTEMPT_SECONDS = 6.0          # max time per sign if you don't advance early
PREDICT_EVERY_N_FRAMES = 8     # same live-update cadence as live_demo.py

# For color/info only now -- NOT for hiding predictions or filtering scoring.
# An earlier version only counted a tick if confidence >= 60%, which meant a
# real (if less-than-60%-confident) correct guess never got shown OR scored.
# Scoring is now a plain majority vote across every tick, full stop.
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


def wait_with_prompt(cap, seconds: float, text: str):
    start = time.time()
    while time.time() - start < seconds:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        remaining = seconds - (time.time() - start)
        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"starting in {remaining:.1f}s", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Quiz mode", frame)
        cv2.waitKey(1)


def run_live_attempt(cap, extractor, model, label_list, seq_len, target_gloss, max_seconds):
    """Continuously predicts and displays the current guess in real time
    while the person signs, exactly like live_demo.py. Returns (final_call,
    all_ticks, quit_requested) where final_call is the majority-vote label
    across every prediction tick during this attempt."""
    buffer = deque(maxlen=seq_len * 2)
    ticks = []  # list of (label, confidence) for every prediction made this attempt
    current_display = "(warming up -- move your hands into frame)"
    current_color = (255, 255, 255)
    frame_count = 0
    start = time.time()
    quit_requested = False

    while time.time() - start < max_seconds:
        ok, frame = cap.read()
        if not ok:
            break
        # Detect on the RAW, unflipped frame -- see the comment in
        # live_demo.py's main loop for why mirroring before detection was
        # likely a significant source of the near-random predictions.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        vector, points = extractor.process_live_frame(frame_rgb)
        buffer.append(vector)
        frame_count += 1

        if frame_count % PREDICT_EVERY_N_FRAMES == 0 and len(buffer) >= seq_len:
            resampled = resample_sequence(np.array(buffer), seq_len)
            probs = model.predict(resampled[np.newaxis, ...], verbose=0)[0]
            best_idx = int(np.argmax(probs))
            confidence = float(probs[best_idx])
            label = label_list[best_idx]
            ticks.append((label, confidence))
            current_display = f"{label} ({confidence:.0%})"
            if confidence >= HIGH_CONFIDENCE:
                current_color = (0, 255, 0)
            elif confidence >= MEDIUM_CONFIDENCE:
                current_color = (0, 255, 255)
            else:
                current_color = (0, 0, 255)

        display = cv2.flip(frame, 1)
        draw_landmarks(display, points, mirror=True)
        remaining = max_seconds - (time.time() - start)
        cv2.rectangle(display, (0, 0), (display.shape[1], 90), (0, 0, 0), -1)
        cv2.putText(display, f"Target: {target_gloss.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, f"Live guess: {current_display}   ({remaining:.1f}s left, n=next, q=quit)",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, current_color, 2)
        cv2.imshow("Quiz mode", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            quit_requested = True
            break
        if key == ord("n") or key == ord(" "):
            break

    # Plain majority vote across every tick, not filtered by confidence --
    # a real (if not super-confident) correct guess should still count.
    all_labels = [label for label, _conf in ticks]
    final_call = Counter(all_labels).most_common(1)[0][0] if all_labels else "(no predictions made)"
    return final_call, ticks, quit_requested


def main():
    model, label_list, seq_len = load_model_and_labels()
    quiz_order = label_list.copy()
    random.shuffle(quiz_order)

    print(f"Quiz on {len(quiz_order)} signs, in this order: {quiz_order}")
    print("Look each one up first (Lifeprint.com, SigningSavvy, or a YouTube search for")
    print("'ASL sign for <word>') so you have something correct to imitate.")
    print("During each turn: press 'n' or SPACE once you're happy with it to move on early, 'q' to quit.\n")
    input("Press Enter when you're ready to start...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    results = []
    with LandmarkExtractor() as extractor:
        for gloss in quiz_order:
            wait_with_prompt(cap, PREP_SECONDS, f"Get ready to sign: {gloss.upper()}")
            final_call, ticks, quit_requested = run_live_attempt(
                cap, extractor, model, label_list, seq_len, gloss, ATTEMPT_SECONDS
            )
            correct = (final_call == gloss)
            results.append((gloss, final_call, correct, ticks))
            print(f"  target: {gloss:10s} final call: {final_call:10s}  "
                  f"{'CORRECT' if correct else 'wrong'}   ({len(ticks)} live predictions made)")
            if quit_requested:
                print("\nQuiz ended early.")
                break

    cap.release()
    cv2.destroyAllWindows()

    if not results:
        print("No attempts recorded.")
        return

    correct_count = sum(1 for r in results if r[2])
    print(f"\n=== Score: {correct_count}/{len(results)} ({correct_count / len(results):.0%}) ===")
    for gloss, final_call, correct, ticks in results:
        mark = "PASS" if correct else "FAIL"
        tick_summary = Counter(label for label, _conf in ticks)
        avg_conf = (sum(conf for _label, conf in ticks) / len(ticks)) if ticks else 0.0
        print(f"  [{mark}] {gloss:10s} -> {final_call:10s}  avg confidence {avg_conf:.0%}  "
              f"(all live guesses seen: {dict(tick_summary)})")


if __name__ == "__main__":
    main()
