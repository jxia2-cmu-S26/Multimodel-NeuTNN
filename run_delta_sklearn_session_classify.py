"""
Quick sanity-check: can delta-modulation IMU spikes classify sessions (ses-02..05)
using a simple sklearn classifier based on spike counts?

This does NOT use NeuTNN, just encodes IMU with DeltaModulationEncoder, extracts
per-window spike-count features, and runs a standard classifier. The goal is to
estimate an approximate upper bound on achievable accuracy with this encoding.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from read_imu_dat import channel_roles

# Use encoders from parent NeuTNN_imu package
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from encoders import DeltaModulationEncoder


IMU_DATASET_ROOT = Path(__file__).resolve().parent / "imu_dataset_clean"
MANIFEST_PATH = IMU_DATASET_ROOT / "manifest.json"


def session_label_from_source(source: str) -> int | None:
    """
    Map a BrainVision source path containing 'ses-YY' to a class index:
      0: ses-02, 1: ses-03, 2: ses-04, 3: ses-05

    ses-01 is treated as separate test-only session in your setup, so it's
    ignored here (returns None).
    """
    p = Path(source)
    ses_id = None
    for part in p.parts:
        if part.startswith("ses-"):
            ses_id = part.split("-", 1)[1]
            break

    if ses_id == "02":
        return 0
    if ses_id == "03":
        return 1
    if ses_id == "04":
        return 2
    if ses_id == "05":
        return 3
    # Ignore ses-01 or anything unexpected
    return None


def load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest


def build_spike_count_dataset(
    manifest: list[dict],
    window_len: int = 64,
    stride: int = 32,
    n_thresholds: int = 5,
    threshold_base: float = 0.0112,
    scale: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    From imu_dataset_clean manifest build (X, y)
      X: (n_samples, n_units) spike-count features
      y: (n_samples,) class labels 0..3 for ses-02..05
    """
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for entry in manifest:
        source = entry.get("source", "")
        if "_task-ERP_imu" not in source:
            continue

        label = session_label_from_source(source)
        if label is None:
            continue

        npy_rel = entry.get("npy")
        ch_names = entry.get("ch_names", [])
        if npy_rel is None:
            continue

        data_path = IMU_DATASET_ROOT / npy_rel
        if not data_path.exists():
            continue

        data = np.load(data_path)  # (n_channels, n_times)
        if data.ndim != 2:
            data = np.asarray(data)
            if data.ndim != 2:
                continue

        roles = channel_roles(ch_names)
        head_acc = [i for i, n in roles["acc"] if n.upper().startswith("H")]
        head_gyro = [i for i, n in roles["gyro"] if n.upper().startswith("H")]
        # Use head accelerometer + head gyroscope channels (up to 6 total)
        head_indices = (head_acc + head_gyro)[:6]
        if not head_indices:
            continue

        data_sel = data[head_indices].astype(np.float64)

        enc = DeltaModulationEncoder(
            n_thresholds=n_thresholds,
            threshold_base=threshold_base,
            scale=scale,
        )
        spikes = enc.encode(data_sel)  # (n_units, n_times-1)
        n_units, n_t = spikes.shape

        if n_t < window_len:
            continue

        start = 0
        while start + window_len <= n_t:
            end = start + window_len
            win = spikes[:, start:end]  # (n_units, window_len)
            # Simple feature: spike count per unit (optionally normalized by window_len)
            feat = win.sum(axis=1).astype(np.float32) / float(window_len)
            X_list.append(feat)
            y_list.append(label)
            start += stride

    if not X_list:
        raise RuntimeError("No IMU windows found for ses-02..05 in manifest.")

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y


def main() -> int:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = load_manifest()
    print(f"Loaded manifest with {len(manifest)} entries.")

    X, y = build_spike_count_dataset(manifest)
    print(f"Spike-count dataset shape: X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Older sklearn versions in this env may not support the multi_class argument,
    # so we rely on the default multi-class handling.
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    acc_train = clf.score(X_train, y_train)
    acc_test = clf.score(X_test, y_test)
    print(f"Train accuracy: {acc_train * 100:.2f}%")
    print(f"Test  accuracy: {acc_test * 100:.2f}%")

    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)

    print("\nClassification report:")
    target_names = ["ses-02", "ses-03", "ses-04", "ses-05"]
    print(classification_report(y_test, y_pred, target_names=target_names))

    # --- Visualization: confusion matrix heatmap and per-class F1 ---
    out_dir = Path(__file__).resolve().parent / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix heatmap
    plt.figure(figsize=(4, 4))
    im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45, ha="right")
    plt.yticks(tick_marks, target_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Delta-encoder (spike counts) – session classification")
    plt.tight_layout()
    cm_path = out_dir / "delta_sklearn_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix figure to {cm_path}")

    # Per-class F1 bar plot
    from sklearn.metrics import precision_recall_fscore_support

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=np.arange(len(target_names))
    )
    x = np.arange(len(target_names))
    plt.figure(figsize=(5, 3))
    plt.bar(x, f1, color="#3498db", alpha=0.8)
    plt.xticks(x, target_names)
    plt.ylim(0.0, 1.0)
    plt.ylabel("F1 score")
    plt.title("Per-class F1 – delta (spike-count) + LogisticRegression")
    plt.tight_layout()
    f1_path = out_dir / "delta_sklearn_per_class_f1.png"
    plt.savefig(f1_path, dpi=150)
    plt.close()
    print(f"Saved per-class F1 figure to {f1_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

