# ASL Tutor — Phase 0 Proof of Concept

Goal of this phase: answer one question — **can a webcam reliably recognize a
small set of ASL signs in real time?** — before building any lesson app
around it. Everything here is scoped to that question.

## Why this runs on your machine, not in the cloud sandbox

The cloud sandbox this was built in has a network policy that blocks PyPI,
npm, and even apt package mirrors, so `pip install` doesn't work there, and it
has no webcam anyway. All of the code below is written and ready to go — you
just need to run it somewhere with normal internet access and a camera. If you
hit an error, paste it back to me and I'll fix the code.

## Known issue you may have already hit: `AttributeError: module 'mediapipe' has no attribute 'solutions'`

Google removed the old "Solutions" API (`mp.solutions.holistic`, the thing
the first version of this code used) from the mediapipe pip package starting
in version 0.10.30 (Dec 2025) — see
[google-ai-edge/mediapipe#6192](https://github.com/google-ai-edge/mediapipe/issues/6192).
Since `requirements.txt` didn't pin a version, `pip install` grabbed the
latest release, which no longer has it. Fixed: the code now uses MediaPipe's
current "Tasks" API instead (`scripts/landmark_extractor.py` — a HandLandmarker
+ PoseLandmarker pair standing in for the old unified Holistic detector). If
you already had the old version of this project, just re-download it; no pip
changes needed, your existing mediapipe install already has the Tasks API.

## What's in here

```
vocab.json                          the ~15 signs targeted for this POC
requirements.txt                    Python dependencies (install locally)
scripts/landmark_extractor.py       shared MediaPipe landmark extraction (hand + pose landmarkers)
scripts/download_wlasl_subset.py    pulls a subset of the public WLASL dataset for our vocab
scripts/extract_landmarks.py        runs MediaPipe Holistic over each video -> landmark sequences
scripts/prepare_dataset.py          pads/resamples sequences, builds train/test split
scripts/train.py                    trains a small LSTM classifier
scripts/live_demo.py                opens your webcam and predicts signs live
```

## Step-by-step

1. **Set up the environment** (Terminal, in this folder):
   ```
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
   On Apple Silicon, if plain `tensorflow` fails to install, swap in
   `tensorflow-macos` (and optionally `tensorflow-metal` for GPU acceleration).
   You'll also need `ffmpeg` on your PATH: `brew install ffmpeg`.

2. **Download the training clips:**
   ```
   python scripts/download_wlasl_subset.py
   ```
   This pulls from WLASL, a public research dataset. Expect some dead links —
   it's normal, the script skips failures and prints a per-sign count at the
   end. If any sign comes back with 0-2 clips, either drop it from
   `vocab.json` or find a replacement gloss (check `data/WLASL_v0.3.json`
   after this runs, since it caches the full metadata locally).

   Alternative: instead of (or in addition to) this dataset, you can drop your
   own recorded clips into `data/raw/<gloss>/*.mp4` — a few short videos of
   you signing each word. This is more effort up front but avoids dataset
   dead-links and tests the model on your own signing style, which is exactly
   what it'll be recognizing later.

3. **Extract landmarks:**
   ```
   python scripts/extract_landmarks.py
   ```
   Runs MediaPipe Holistic over every clip. Takes a few minutes depending on
   how many clips you have. This is also the step most likely to need
   internet the first time it runs, since MediaPipe fetches its model files
   on first use and caches them after that.

4. **Prepare the dataset:**
   ```
   python scripts/prepare_dataset.py
   ```
   Signs with too few clips (below `min_clips_per_sign` in `vocab.json`) get
   dropped automatically, with a printed warning — that's expected on a first
   run, not a bug.

5. **Train:**
   ```
   python scripts/train.py
   ```
   Trains a small LSTM on however many signs made it through step 4. With a
   handful of signs and maybe 10-20 clips each, don't expect great accuracy
   yet — the goal of this run is confirming the *pipeline* works end to end,
   not hitting a target accuracy number.

6. **The real test — live webcam:**
   ```
   python scripts/live_demo.py
   ```
   Sign one of the words from `vocab.json` at your camera and watch what it
   predicts. Press `q` to quit.

## What "success" looks like at this stage

Not high accuracy — just: does a predicted label show up at all when you sign
something, and does it change to a different (ideally correct) label when you
switch signs? That confirms the landmark extraction, the model, and the live
inference loop are all wired together correctly. From there, accuracy
improves by gathering more clips per sign, trimming to a smaller/cleaner
vocabulary, or tuning the model — all straightforward next steps once the
loop itself is proven out.

## Report back

Whatever happens — errors, low accuracy, it working better than expected —
bring the output back to this conversation and we'll iterate from there. If
the webcam loop works reasonably well, the next phase is exporting the model
to run in a browser (TensorFlow.js) so it can become part of the actual web
app instead of a local Python script.
