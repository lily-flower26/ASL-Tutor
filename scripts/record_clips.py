"""
Record your own short clips of a sign directly from your webcam, saved
straight into data/raw/<gloss>/clip_XXX.mp4 -- ready for
extract_landmarks.py, no manual recording-app trimming/renaming needed.

Usage:
    python scripts/record_clips.py book
    python scripts/record_clips.py book --duration 3 --target 15

Controls (in the video window):
    SPACE  - arm recording (starts after a short countdown, so you have time
             to get your hands back in frame)
    q      - quit and move on to the next sign

Tips for better training data: vary your speed and position a little
between reps (not identical every time), keep your hands and face in frame,
plain background and decent lighting help, and redo a rep if you fumble it
partway through.
"""
import argparse
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

COUNTDOWN_SECONDS = 2.0


def next_clip_path(gloss_dir: Path) -> Path:
    existing = sorted(gloss_dir.glob("clip_*.mp4"))
    next_idx = 0
    for p in existing:
        try:
            idx = int(p.stem.split("_")[1])
            next_idx = max(next_idx, idx + 1)
        except (IndexError, ValueError):
            continue
    return gloss_dir / f"clip_{next_idx:03d}.mp4"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gloss", help="the sign you're recording, e.g. 'book' (should match vocab.json)")
    parser.add_argument("--duration", type=float, default=3.0, help="seconds to record per clip")
    parser.add_argument("--target", type=int, default=15, help="how many clips you're aiming for")
    args = parser.parse_args()

    gloss = args.gloss.lower().strip()
    gloss_dir = RAW_DIR / gloss
    gloss_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam (index 0). Try a different camera index or check permissions.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Recording '{gloss}'. Press SPACE to record a clip ({args.duration:.1f}s each), 'q' when done.")

    state = "idle"  # idle -> countdown -> recording
    state_start = None
    writer = None
    clip_count = len(list(gloss_dir.glob("clip_*.mp4")))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror, feels natural
        display = frame.copy()
        now = time.time()

        if state == "countdown":
            remaining = COUNTDOWN_SECONDS - (now - state_start)
            if remaining <= 0:
                state = "recording"
                state_start = now
                clip_path = next_clip_path(gloss_dir)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))
            else:
                cv2.putText(display, f"Get ready... {remaining:.1f}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        elif state == "recording":
            elapsed = now - state_start
            writer.write(frame)
            cv2.circle(display, (width - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(display, f"REC {elapsed:.1f}/{args.duration:.1f}s", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            if elapsed >= args.duration:
                writer.release()
                writer = None
                clip_count += 1
                state = "idle"
                print(f"  saved clip {clip_count}")

        else:
            cv2.putText(display, f"'{gloss}': {clip_count}/{args.target} clips. SPACE=record, q=quit",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Record ASL clips", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and state == "idle":
            state = "countdown"
            state_start = now

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. {clip_count} clips saved to {gloss_dir}")


if __name__ == "__main__":
    main()
