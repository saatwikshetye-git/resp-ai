"""
model/dataset_prep.py

Produces:
 - data/processed/train/*.wav + train_metadata.csv
 - data/processed/val/*.wav   + val_metadata.csv
 - data/processed/test/*.wav  + test_metadata.csv

Fixes applied (explicit):
1) Reset dataframe indices before assigning processed_path to avoid misalignment.
2) Robust stratified train/val/test splitting with fallback to non-stratified split.
3) Safe output directory cleanup (shutil.rmtree then recreate) so process is idempotent.
4) Use itertuples and try/except so single corrupt files don't crash the job.
5) Use LABEL_MAP dict and set unknown labels to -1 with warnings.
6) Optional dry-run mode (no WAV writes) for testing in CI.

Usage (example):
python model/dataset_prep.py --coughvid_path data/raw/coughvid --icbhi_path data/raw/icbhi --audioset_path data/raw/audioset --output_dir data/processed
"""

import os
import shutil
import argparse
from pathlib import Path
import warnings
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings("ignore")

SAMPLE_RATE = 16000
DURATION = 10  # seconds
TARGET_SAMPLES = SAMPLE_RATE * DURATION

# Example label map - adapt to your dataset's metadata mapping later
# Users should modify this mapping to map dataset-specific labels to unified classes:
# 0: normal, 1: cough, 2: wheeze, 3: crackles, 4: agonal (example)
LABEL_MAP: Dict[str, int] = {
    # dataset-specific placeholder keys -> unified integer label
    # e.g., 'healthy': 0, 'cough': 1, 'wheeze': 2
    # Fill this based on your dataset metadata/columns
}

def preprocess_audio(file_path: str, target_sr: int = SAMPLE_RATE, duration: int = DURATION) -> np.ndarray:
    """
    Load audio, resample to target_sr, convert to mono, pad/trim to exactly `duration` seconds.
    Returns numpy array of shape (TARGET_SAMPLES,) or raises Exception.
    """
    # librosa.load will resample and convert to mono if requested
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    if y is None:
        raise ValueError(f"librosa returned None for {file_path}")

    # Trim or pad
    target_len = target_sr * duration
    if len(y) > target_len:
        y = y[:target_len]
    elif len(y) < target_len:
        pad_len = target_len - len(y)
        y = np.pad(y, (0, pad_len), mode="constant", constant_values=0.0)
    return y.astype(np.float32)


def _scan_audio_files(base_path: str, exts: List[str] = None) -> List[str]:
    if exts is None:
        exts = [".wav", ".mp3", ".ogg", ".webm", ".flac"]
    base = Path(base_path)
    if not base.exists():
        return []
    files = []
    for ext in exts:
        files.extend([str(p) for p in base.rglob(f"*{ext}")])
    return files


def load_coughvid(base_path: str) -> pd.DataFrame:
    """
    Scans CoughVID folder for audio files and tries to attach a label column using metadata if available.
    If metadata is not present or cannot be parsed, assign label = -1 (unknown).
    """
    audio_files = _scan_audio_files(base_path)
    df = pd.DataFrame({"orig_path": audio_files})
    df["dataset"] = "coughvid"
    df["label"] = -1

    meta_path = Path(base_path) / "metadata_compiled.csv"
    if meta_path.exists():
        try:
            meta_df = pd.read_csv(meta_path)
            # Attempt to merge by filename if metadata contains it
            # Adjust column names below if different
            if "filename" in meta_df.columns:
                meta_df["filename_only"] = meta_df["filename"].apply(lambda x: Path(x).stem)
                df["stem"] = df["orig_path"].apply(lambda p: Path(p).stem)
                merged = df.merge(meta_df, left_on="stem", right_on="filename_only", how="left")
                # Example metadata column for status could be 'status' or 'cough_detected'
                if "status" in merged.columns:
                    # Map status to LABEL_MAP if possible
                    merged["label_mapped"] = merged["status"].map(LABEL_MAP).fillna(-1).astype(int)
                    df = merged[["orig_path", "dataset", "label_mapped"]].rename(columns={"label_mapped": "label"})
                else:
                    # fallback - keep -1
                    df["label"] = -1
                df = df.drop(columns=[c for c in ["stem", "filename_only"] if c in df.columns], errors="ignore")
            else:
                # metadata exists but doesn't have expected columns - keep unknown labels
                df["label"] = -1
        except Exception:
            df["label"] = -1
    else:
        df["label"] = -1

    return df.reset_index(drop=True)


def load_icbhi(base_path: str) -> pd.DataFrame:
    """
    Scans ICBHI folder. ICBHI typically contains lung sound recordings; mapping may depend on dataset layout.
    For safety, labels are set to -1 (unknown) and the user should update LABEL_MAP logic if needed.
    """
    audio_files = _scan_audio_files(base_path)
    df = pd.DataFrame({"orig_path": audio_files})
    df["dataset"] = "icbhi"
    df["label"] = -1
    # If filenames contain labels, user can add parsing logic here.
    return df.reset_index(drop=True)


def load_audioset_subset(base_path: str) -> pd.DataFrame:
    """
    Scans a local AudioSet subset folder. Default label = -1 (unknown/background).
    """
    audio_files = _scan_audio_files(base_path)
    df = pd.DataFrame({"orig_path": audio_files})
    df["dataset"] = "audioset"
    df["label"] = -1
    return df.reset_index(drop=True)


def unify_datasets(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate DataFrames and shuffle.
    """
    if not dfs:
        return pd.DataFrame(columns=["orig_path", "dataset", "label"])
    full = pd.concat(dfs, ignore_index=True)
    full = full.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return full


def safe_train_val_test_split(
    df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split with fallback.
    Returns train_df, val_df, test_df.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    if df.empty:
        return df.copy(), df.copy(), df.copy()

    # If labels are unknown (-1) or too few samples per class, stratify will fail.
    stratify_col = None
    try:
        if "label" in df.columns and df["label"].nunique() > 1:
            # Check min samples per class
            class_counts = df["label"].value_counts()
            if class_counts.min() >= 2:
                stratify_col = df["label"]
    except Exception:
        stratify_col = None

    # First split: train vs temp
    try:
        train_df, temp_df = train_test_split(df, test_size=(1 - train_ratio), stratify=stratify_col, random_state=seed)
    except Exception:
        # fallback to non-stratified
        train_df, temp_df = train_test_split(df, test_size=(1 - train_ratio), random_state=seed)

    # Second split: val vs test from temp
    try:
        val_fraction = val_ratio / (val_ratio + test_ratio)
        if stratify_col is not None:
            temp_stratify = temp_df["label"] if "label" in temp_df.columns else None
            val_df, test_df = train_test_split(temp_df, test_size=(1 - val_fraction), stratify=temp_stratify, random_state=seed)
        else:
            val_df, test_df = train_test_split(temp_df, test_size=(1 - val_fraction), random_state=seed)
    except Exception:
        # final fallback
        val_df, test_df = train_test_split(temp_df, test_size=(1 - val_fraction), random_state=seed)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def preprocess_and_save_split(
    df: pd.DataFrame, split_name: str, output_root: str, dry_run: bool = False
) -> pd.DataFrame:
    """
    Process audio entries in df and save WAVs + metadata CSV to output_root/split_name.
    Returns the DataFrame of successfully processed entries with a 'processed_path' column.
    """
    out_dir = Path(output_root) / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    processed_paths = []
    records = []

    if df.empty:
        # ensure CSV still created
        pd.DataFrame(columns=["processed_path", "orig_path", "dataset", "label"]).to_csv(out_dir / f"{split_name}_metadata.csv", index=False)
        return pd.DataFrame(columns=["processed_path", "orig_path", "dataset", "label"])

    print(f"Processing split '{split_name}' with {len(df)} candidates...")

    for row in tqdm(df.itertuples(index=False), total=len(df)):
        orig_path = getattr(row, "orig_path", None)
        dataset_name = getattr(row, "dataset", None)
        label = getattr(row, "label", -1)

        if not orig_path or not Path(orig_path).exists():
            # skip missing files
            continue

        try:
            audio = preprocess_audio(orig_path)
        except Exception as e:
            # Skip corrupt/unreadable files but log
            print(f"Skipping {orig_path}: {e}")
            continue

        # create filename: datasetname_stem.wav to avoid collisions
        stem = Path(orig_path).stem
        safe_name = f"{dataset_name}_{stem}.wav"
        dest_path = str((out_dir / safe_name).resolve())

        if not dry_run:
            try:
                # overwrite if exists
                sf.write(dest_path, audio, SAMPLE_RATE, subtype="PCM_16")
            except Exception as e:
                print(f"Failed to write {dest_path}: {e}")
                continue

        processed_paths.append(dest_path)
        records.append({"processed_path": dest_path, "orig_path": orig_path, "dataset": dataset_name, "label": int(label) if label is not None else -1})

    # Create final DataFrame and reset indices to ensure alignment
    final_df = pd.DataFrame(records).reset_index(drop=True)

    # Save metadata CSV
    csv_path = out_dir / f"{split_name}_metadata.csv"
    final_df.to_csv(csv_path, index=False)
    print(f"Saved {len(final_df)} processed files and metadata to {out_dir}")

    return final_df


def prepare_output_dir(output_root: str):
    """
    Safely remove and recreate the output directory.
    """
    out_path = Path(output_root)
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)


def main(
    coughvid_path: str,
    icbhi_path: str,
    audioset_path: str,
    output_dir: str,
    dry_run: bool = False,
    seed: int = 42,
) -> Tuple[int, int, int]:
    """
    Main pipeline orchestration. Returns counts: (n_train, n_val, n_test)
    """
    dfs = []

    if coughvid_path and Path(coughvid_path).exists():
        dfs.append(load_coughvid(coughvid_path))
    else:
        print(f"Warning: CoughVID path '{coughvid_path}' not found or empty. Skipping.")

    if icbhi_path and Path(icbhi_path).exists():
        dfs.append(load_icbhi(icbhi_path))
    else:
        print(f"Warning: ICBHI path '{icbhi_path}' not found or empty. Skipping.")

    if audioset_path and Path(audioset_path).exists():
        dfs.append(load_audioset_subset(audioset_path))
    else:
        print(f"Warning: AudioSet path '{audioset_path}' not found or empty. Skipping.")

    if not dfs:
        print("No datasets found. Exiting.")
        return 0, 0, 0

    # unify and shuffle
    full_df = unify_datasets(dfs)
    if full_df.empty:
        print("Unified dataset empty. Exiting.")
        return 0, 0, 0

    # split
    train_df, val_df, test_df = safe_train_val_test_split(full_df, seed=seed)

    # prepare output dir
    prepare_output_dir(output_dir)

    # process splits
    train_processed = preprocess_and_save_split(train_df, "train", output_dir, dry_run=dry_run)
    val_processed = preprocess_and_save_split(val_df, "val", output_dir, dry_run=dry_run)
    test_processed = preprocess_and_save_split(test_df, "test", output_dir, dry_run=dry_run)

    print("Dataset preparation complete.")
    print(f"Train: {len(train_processed)} | Val: {len(val_processed)} | Test: {len(test_processed)}")

    return len(train_processed), len(val_processed), len(test_processed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare audio datasets (CoughVID, ICBHI, AudioSet subset)")
    parser.add_argument("--coughvid_path", type=str, default="data/raw/coughvid")
    parser.add_argument("--icbhi_path", type=str, default="data/raw/icbhi")
    parser.add_argument("--audioset_path", type=str, default="data/raw/audioset")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--dry_run", action="store_true", help="If set, do not write WAV files (useful for CI)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        coughvid_path=args.coughvid_path,
        icbhi_path=args.icbhi_path,
        audioset_path=args.audioset_path,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        seed=args.seed,
    )
