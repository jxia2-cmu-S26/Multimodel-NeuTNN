import os
import sys

# Same as NeuTNN/export_eeg_to_npy.py, but this file lives in eeg_NeuTNN/ — load _load_eeg from NeuTNN/.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "NeuTNN"))

import numpy as np
from sklearn.model_selection import train_test_split

from run_eeg_encodings import _load_eeg


def main(
    out_dir: str = "eeg_preprocessed_npy",
    task: str = "all",
    label_mode: str = "speed",
    n_scalp_channels: int = 32,
    exclude_first_class: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Load real BrainVision EEG using _load_eeg() and export train/test splits as .npy files.

    Only 32 scalp-EEG channels are used; ear-EEG, EOG, and IMU channels are excluded.

    task:
      - "ERP" | "SSVEP" | "all"
    label_mode:
      - "event": classification by events.tsv value codes
      - "speed": classification by session (ses-01..ses-05 -> 0..4)
    n_scalp_channels: number of EEG channels to load (32 for scalp-EEG only).
    exclude_first_class: if True, drop class 0 (e.g. ERP Training 0 m/s) and remap 1..4 -> 0..3 (4 classes).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir_abs = os.path.join(script_dir, out_dir)
    os.makedirs(out_dir_abs, exist_ok=True)

    print("Loading EEG epochs via run_eeg_encodings._load_eeg() (32 scalp-EEG channels only) ...")
    print("task =", task, "| label_mode =", label_mode, "| n_scalp_channels =", n_scalp_channels)
    X_all, y_all = _load_eeg(
        task=task, label_mode=label_mode, n_scalp_channels=n_scalp_channels
    )  # (N, C, T), (N,) with C=32
    if exclude_first_class:
        keep = y_all != 0
        X_all = X_all[keep]
        y_all = y_all[keep] - 1  # remap 1,2,3,4 -> 0,1,2,3
        print("Excluded first class (e.g. Training 0 m/s); remapped to 4 classes 0..3.")
    uniq = np.unique(y_all)
    n_classes = int(uniq.max()) + 1 if uniq.size > 0 else 0
    print("Loaded EEG:", X_all.shape, "| labels:", y_all.shape)
    print("Unique labels:", uniq, "| n_classes:", n_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=test_size,
        random_state=random_state,
        stratify=y_all,
    )

    print("Train:", X_train.shape, "| Test:", X_test.shape)
    print("Saving .npy files into:", out_dir_abs)

    np.save(os.path.join(out_dir_abs, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir_abs, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir_abs, "X_test.npy"), X_test)
    np.save(os.path.join(out_dir_abs, "y_test.npy"), y_test)
    np.save(os.path.join(out_dir_abs, "n_classes.npy"), np.array(n_classes, dtype=np.int64))

    print("Done.")
    print("You can now copy", out_dir_abs, "to Colab and load the arrays there.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Export EEG to .npy for Colab.")
    p.add_argument("--task", type=str, default="all", choices=("all", "ERP", "SSVEP"),
                   help="Which task to load: all, ERP only, or SSVEP only (default: all)")
    p.add_argument("--label_mode", type=str, default="speed", choices=("event", "speed"))
    p.add_argument("--out_dir", type=str, default="eeg_preprocessed_npy")
    p.add_argument("--n_scalp_channels", type=int, default=32)
    p.add_argument("--exclude_first_class", action="store_true",
                   help="Drop first class (e.g. ERP Training 0 m/s); keep 4 classes (Standing, 0.8, 1.6, 2.0 m/s)")
    args = p.parse_args()
    main(
        out_dir=args.out_dir,
        task=args.task,
        label_mode=args.label_mode,
        n_scalp_channels=args.n_scalp_channels,
        exclude_first_class=args.exclude_first_class,
    )
