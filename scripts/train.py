"""
Train a small sequence classifier on data/prepared/dataset.npz (built by
prepare_dataset.py) and save it to models/sign_classifier.keras plus the
label map it was trained with.

Usage:
    python scripts/train.py
"""
import json
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

ROOT = Path(__file__).resolve().parent.parent
PREPARED_DIR = ROOT / "data" / "prepared"
MODELS_DIR = ROOT / "models"


def build_model(seq_len: int, feature_dim: int, num_classes: int) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(seq_len, feature_dim)),
        keras.layers.Masking(mask_value=0.0),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.3),
        keras.layers.LSTM(32),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    dataset_path = PREPARED_DIR / "dataset.npz"
    label_map_path = PREPARED_DIR / "label_map.json"
    if not dataset_path.exists():
        print(f"{dataset_path} not found. Run prepare_dataset.py first.")
        return

    data = np.load(dataset_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    with open(label_map_path) as f:
        label_list = json.load(f)

    seq_len, feature_dim = X_train.shape[1], X_train.shape[2]
    num_classes = len(label_list)
    print(f"Training on {X_train.shape[0]} clips, testing on {X_test.shape[0]}, "
          f"{num_classes} classes, sequence length {seq_len}, feature dim {feature_dim}")

    model = build_model(seq_len, feature_dim, num_classes)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=150,
        batch_size=16,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.3f} (loss {test_loss:.3f})")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "sign_classifier.keras"
    model.save(model_path)
    shutil.copy(label_map_path, MODELS_DIR / "label_map.json")
    with open(MODELS_DIR / "vocab_used.json", "w") as f:
        json.dump({"sequence_length": seq_len, "feature_dim": feature_dim}, f, indent=2)
    print(f"Saved model to {model_path}")

    if test_acc < 0.5:
        print("\nNOTE: accuracy is low. With this little data (WLASL subset or a handful of "
              "self-recorded clips per sign) that's expected for a first pass -- treat this as "
              "validating the pipeline works end-to-end, then improve by gathering more clips "
              "per sign, trimming to signs with the most data, or simplifying the vocabulary.")


if __name__ == "__main__":
    main()
