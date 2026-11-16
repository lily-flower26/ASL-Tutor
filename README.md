# ASL Tutor — Proof of Concept

**can a webcam reliably recognize a
small set of ASL signs in real time?** — before building any lesson app
around it. Everything here is scoped to that question.

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
