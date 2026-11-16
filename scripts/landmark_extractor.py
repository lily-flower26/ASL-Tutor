"""
Shared landmark extraction, used by both extract_landmarks.py (offline, on
downloaded video clips) and live_demo.py (live webcam).

Why this exists: the original version of this project used MediaPipe's older
"Solutions" API (`mp.solutions.holistic`), which Google removed from the pip
package starting in mediapipe 0.10.30 (Dec 2025) -- see
https://github.com/google-ai-edge/mediapipe/issues/6192. Current mediapipe
versions only ship the newer "Tasks" API, which doesn't have a single unified
"Holistic" detector -- instead you run a HandLandmarker and a PoseLandmarker
separately and combine their output. This module does that combining once so
the rest of the pipeline doesn't have to care.

Output format is unchanged from before: a 258-dim vector per frame --
pose (33 landmarks x [x,y,z,visibility] = 132) + left hand (21 x [x,y,z] = 63)
+ right hand (21 x [x,y,z] = 63) -- so prepare_dataset.py and train.py don't
need any changes.

Model files (~10-15MB total) are downloaded once on first run and cached
under models/mediapipe_tasks/, so you need internet the first time this runs
but not after.
"""
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision  # noqa: F401 (registers mp.tasks.vision)

ROOT = Path(__file__).resolve().parent.parent
MODEL_CACHE_DIR = ROOT / "models" / "mediapipe_tasks"

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
FEATURE_DIM = POSE_LANDMARKS * 4 + HAND_LANDMARKS * 3 + HAND_LANDMARKS * 3  # 258

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def _ensure_model(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dest.name} (one-time)...")
    urllib.request.urlretrieve(url, dest)
    return dest


class LandmarkExtractor:
    """One HandLandmarker + one PoseLandmarker, both in VIDEO mode.

    VIDEO mode needs strictly increasing millisecond timestamps per call, so
    use process_video_frame() for a video file (pass frame_index/fps) or
    process_live_frame() for a live camera (uses wall-clock time). Create a
    fresh instance per video clip when processing a batch of unrelated clips
    -- the internal tracking state (and timestamp counter) shouldn't carry
    over between clips.
    """

    def __init__(self):
        hand_model_path = _ensure_model(HAND_MODEL_URL, MODEL_CACHE_DIR / "hand_landmarker.task")
        pose_model_path = _ensure_model(POSE_MODEL_URL, MODEL_CACHE_DIR / "pose_landmarker_lite.task")

        hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(hand_model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
        )
        pose_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(pose_model_path)),
            running_mode=VisionRunningMode.VIDEO,
        )
        self._hand_landmarker = HandLandmarker.create_from_options(hand_options)
        self._pose_landmarker = PoseLandmarker.create_from_options(pose_options)
        self._last_ts_ms = -1
        self._live_start = None

    def _next_ts(self, ts_ms: int) -> int:
        # detect_for_video requires strictly increasing timestamps; guard
        # against ties/rounding from either the fps math or wall-clock calls.
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms
        return ts_ms

    def process_video_frame(self, frame_rgb: np.ndarray, frame_index: int, fps: float) -> np.ndarray:
        ts_ms = self._next_ts(int((frame_index / fps) * 1000))
        vector, _points = self._process(frame_rgb, ts_ms)
        return vector

    def process_live_frame(self, frame_rgb: np.ndarray):
        """Returns (vector, points) -- points is a dict of normalized (x, y)
        lists per body part, for drawing an overlay. See draw_landmarks()."""
        if self._live_start is None:
            self._live_start = time.time()
        ts_ms = self._next_ts(int((time.time() - self._live_start) * 1000))
        return self._process(frame_rgb, ts_ms)

    def _process(self, frame_rgb: np.ndarray, ts_ms: int):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        pose_result = self._pose_landmarker.detect_for_video(mp_image, ts_ms)
        hand_result = self._hand_landmarker.detect_for_video(mp_image, ts_ms)

        if pose_result.pose_landmarks:
            pose_lms = pose_result.pose_landmarks[0]
            pose_vec = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility or 0.0] for lm in pose_lms]
            ).flatten()
            pose_points = [(lm.x, lm.y) for lm in pose_lms]
        else:
            pose_vec = np.zeros(POSE_LANDMARKS * 4)
            pose_points = []

        left_vec = np.zeros(HAND_LANDMARKS * 3)
        right_vec = np.zeros(HAND_LANDMARKS * 3)
        left_points, right_points = [], []
        for lm_list, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
            label = handedness[0].category_name  # "Left" or "Right"
            vec = np.array([[lm.x, lm.y, lm.z] for lm in lm_list]).flatten()
            points = [(lm.x, lm.y) for lm in lm_list]
            if label == "Left":
                left_vec, left_points = vec, points
            elif label == "Right":
                right_vec, right_points = vec, points

        vector = np.concatenate([pose_vec, left_vec, right_vec]).astype(np.float32)
        points = {"pose": pose_points, "left_hand": left_points, "right_hand": right_points}
        return vector, points

    def close(self):
        self._hand_landmarker.close()
        self._pose_landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def draw_landmarks(frame_bgr: np.ndarray, points: dict, mirror: bool = False) -> None:
    """Draws simple dot overlays for pose/hand points (in place). Points are
    normalized (x, y) in [0, 1]; frame_bgr is a standard OpenCV BGR frame.
    Set mirror=True when frame_bgr is a horizontally-flipped display copy
    (see the detect-on-unflipped / display-flipped split in live_demo.py and
    quiz_mode.py) so the dots land in the right place on screen.

    (The old mp.solutions.drawing_utils, with its nicer skeleton-line
    rendering, was removed along with the rest of the Solutions API -- this
    is a minimal stand-in, just dots, good enough to confirm tracking is
    working while testing live.)
    """
    h, w = frame_bgr.shape[:2]
    colors = {"pose": (0, 165, 255), "left_hand": (255, 0, 0), "right_hand": (0, 255, 0)}
    for part, color in colors.items():
        for x, y in points.get(part, []):
            px = (1.0 - x) if mirror else x
            center = (int(px * w), int(y * h))
            cv2.circle(frame_bgr, center, 3, color, -1)
