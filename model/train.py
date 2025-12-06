"""
model/train.py

Complete training script for Resp-AI.

Features:
- AudioDataset reading `processed_path` from metadata CSVs
- waveform -> log-mel conversion (librosa)
- optional augmentations (noise, gain, time shift, spec augment)
- ResNetSpec model import
- training loop with AdamW, scheduler
- compute class weights from training metadata
- metrics: per-class precision/recall/f1 + distress recall
- checkpoint saving to model/checkpoints/best.pt
- export helpers (TorchScript & ONNX)
- CLI interface with dry_run and smoke-test

Notes:
- Assumes processed WAVs are fixed-length (10s @ 16k => deterministic frames).
- Designed to run in Colab / local / CI. For quick CI/dry runs, use --dry_run.
"""

import os
import sys
import time
import argparse
import csv
import math
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Import the ResNet model we implemented
from model.resnet_spec import ResNetSpec

# ---------------------------
# Hyperparameters & defaults
# ---------------------------
SAMPLE_RATE = 16000
DURATION = 10.0
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
TARGET_SAMPLES = int(SAMPLE_RATE * DURATION)

DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 40
DEFAULT_NUM_CLASSES = 5
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Utilities
# ---------------------------


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_wav_mono(path: str, sr: int = SAMPLE_RATE, duration: float = DURATION) -> Optional[np.ndarray]:
    """
    Load audio with librosa, resample to sr, convert to mono, pad/trim to exact duration.
    Returns numpy float32 array of shape (TARGET_SAMPLES,)
    """
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
    except Exception as e:
        print(f"[load_wav_mono] Error loading {path}: {e}")
        return None
    target_len = int(sr * duration)
    if len(y) > target_len:
        y = y[:target_len]
    elif len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    return y.astype(np.float32)


def waveform_to_logmelspec(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """
    Convert 1-D waveform to log-mel spectrogram (shape: n_mels x time_frames).
    Uses librosa (power spectrogram -> mel -> db). Returns float32.
    """
    # Compute mel spectrogram (power)
    mel_spec = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0
    )
    # Convert to log scale (dB)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    # Normalize per-sample (zero mean, unit std) to stabilize training
    log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-9)
    return log_mel.astype(np.float32)


# ---------------------------
# Augmentations (simple, CPU-friendly)
# ---------------------------


def add_random_noise(waveform: np.ndarray, snr_db_min: float = 5.0, snr_db_max: float = 20.0) -> np.ndarray:
    """
    Additive Gaussian noise scaled to achieve a random SNR in dB between snr_db_min and snr_db_max.
    """
    snr_db = np.random.uniform(snr_db_min, snr_db_max)
    signal_power = np.mean(waveform ** 2)
    if signal_power <= 0:
        return waveform
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), size=waveform.shape).astype(np.float32)
    return waveform + noise


def random_gain(waveform: np.ndarray, min_gain_db: float = -6.0, max_gain_db: float = 6.0) -> np.ndarray:
    gain_db = np.random.uniform(min_gain_db, max_gain_db)
    gain = 10 ** (gain_db / 20.0)
    return (waveform * gain).astype(np.float32)


def time_shift(waveform: np.ndarray, max_shift_seconds: float = 0.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    max_shift = int(max_shift_seconds * sr)
    if max_shift <= 0:
        return waveform
    shift = np.random.randint(-max_shift, max_shift)
    if shift == 0:
        return waveform
    if shift > 0:
        return np.pad(waveform[:-shift], (shift, 0), mode="constant")
    else:
        return np.pad(waveform[-shift:], (0, -shift), mode="constant")


def spec_augment(mel: np.ndarray, time_masking_max: int = 20, freq_masking_max: int = 10) -> np.ndarray:
    """
    Apply simple SpecAugment-like masking on the log-mel spectrogram (in-place on a copy).
    """
    mel = mel.copy()
    num_mels, num_frames = mel.shape
    # time masks
    t = np.random.randint(0, time_masking_max + 1)
    if t > 0:
        t0 = np.random.randint(0, max(1, num_frames - t + 1))
        mel[:, t0 : t0 + t] = 0.0
    # freq masks
    f = np.random.randint(0, freq_masking_max + 1)
    if f > 0:
        f0 = np.random.randint(0, max(1, num_mels - f + 1))
        mel[f0 : f0 + f, :] = 0.0
    return mel


# ---------------------------
# Dataset & DataLoader
# ---------------------------


class AudioDataset(Dataset):
    """
    Dataset reading metadata CSV with at least: processed_path, orig_path, dataset, label
    Returns (tensor: 1 x n_mels x time_frames, label:int)
    """

    def __init__(
        self,
        metadata_csv: str,
        augment: bool = False,
        augment_noise_prob: float = 0.3,
        augment_gain_prob: float = 0.3,
        augment_shift_prob: float = 0.3,
        augment_spec_prob: float = 0.3,
    ):
        self.metadata = pd.read_csv(metadata_csv)
        # basic sanity
        if "processed_path" not in self.metadata.columns:
            raise ValueError("metadata CSV must contain 'processed_path' column")
        self.samples = self.metadata.to_dict(orient="records")
        self.augment = augment
        self.augment_noise_prob = augment_noise_prob
        self.augment_gain_prob = augment_gain_prob
        self.augment_shift_prob = augment_shift_prob
        self.augment_spec_prob = augment_spec_prob

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        path = rec["processed_path"]
        label = int(rec.get("label", -1))

        waveform = load_wav_mono(path)
        if waveform is None:
            # If corrupted, return a zeros sample (should be filtered earlier ideally)
            waveform = np.zeros(TARGET_SAMPLES, dtype=np.float32)

        if self.augment:
            # noise
            if np.random.rand() < self.augment_noise_prob:
                waveform = add_random_noise(waveform)
            if np.random.rand() < self.augment_gain_prob:
                waveform = random_gain(waveform)
            if np.random.rand() < self.augment_shift_prob:
                waveform = time_shift(waveform)
        # waveform -> log-mel
        mel = waveform_to_logmelspec(waveform)
        if self.augment and (np.random.rand() < self.augment_spec_prob):
            mel = spec_augment(mel)
        # Convert to tensor shape (1, n_mels, time)
        mel_tensor = torch.from_numpy(mel).unsqueeze(0)  # float32
        return mel_tensor, label


def collate_fn(batch):
    """
    batch: list of (mel_tensor, label)
    All mel_tensors should have same shape since processed WAVs have same length.
    """
    tensors = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    x = torch.stack(tensors, dim=0)
    return x, labels


# ---------------------------
# Training & Evaluation
# ---------------------------


def compute_class_weights_from_metadata(metadata_csv: str, num_classes: int) -> Optional[torch.Tensor]:
    df = pd.read_csv(metadata_csv)
    if "label" not in df.columns:
        return None
    labels = df["label"].fillna(-1).astype(int).values
    # keep only valid labels 0..num_classes-1
    mask = (labels >= 0) & (labels < num_classes)
    if mask.sum() == 0:
        return None
    classes = np.unique(labels[mask])
    # sklearn compute_class_weight expects all classes present; provide range
    all_classes = np.arange(num_classes)
    try:
        cw = compute_class_weight(class_weight="balanced", classes=all_classes, y=labels[mask])
        cw_tensor = torch.tensor(cw, dtype=torch.float32)
        return cw_tensor
    except Exception:
        return None


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int):
    model.eval()
    preds = []
    trues = []
    losses = []
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            losses.append(loss.item())
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.append(pred)
            trues.append(y.cpu().numpy())
    if len(preds) == 0:
        return {"loss": 0.0, "accuracy": 0.0, "precision": [], "recall": [], "f1": [], "per_class_support": []}
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    avg_loss = float(np.mean(losses)) if losses else 0.0
    acc = float(accuracy_score(trues, preds))
    precision, recall, f1, support = precision_recall_fscore_support(trues, preds, labels=list(range(num_classes)), zero_division=0)
    metrics = {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": support.tolist(),
    }
    return metrics


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    output_dir: str,
    num_classes: int,
    mixed_prec: bool = False,
    grad_clip: float = 1.0,
):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_val_recall = -1.0
    best_checkpoint_path = Path(output_dir) / "model/checkpoints/best.pt"
    best_log = None
    os.makedirs(Path(output_dir) / "model/checkpoints", exist_ok=True)

    # CSV logging
    log_csv = Path(output_dir) / "training_log.csv"
    with open(log_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "val_accuracy",
                "val_recall_distress",  # distress class assumed to be 4 by default (user can adapt)
            ]
        )

    scaler = torch.cuda.amp.GradScaler() if (mixed_prec and device.type == "cuda") else None

    for epoch in range(1, epochs + 1):
        model.train()
        running_losses = []
        t0 = time.time()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = loss_fn(logits, y)
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            running_losses.append(loss.item())

        scheduler.step()
        train_loss = float(np.mean(running_losses)) if running_losses else 0.0

        # Validate
        val_metrics = evaluate_model(model, val_loader, device, num_classes)
        val_loss = val_metrics["loss"]
        val_accuracy = val_metrics["accuracy"]
        # distress class mapping: default to last class (num_classes-1) unless user changes
        distress_idx = num_classes - 1
        val_recall_list = val_metrics.get("recall", [])
        val_recall_distress = float(val_recall_list[distress_idx]) if len(val_recall_list) > distress_idx else 0.0

        epoch_time = time.time() - t0
        print(
            f"[Epoch {epoch}/{epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} val_recall_distress={val_recall_distress:.4f} time={epoch_time:.1f}s"
        )

        # log to CSV
        with open(log_csv, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, train_loss, val_loss, val_accuracy, val_recall_distress])

        # Save best checkpoint by distress recall (sensitivity)
        if val_recall_distress > best_val_recall:
            best_val_recall = val_recall_distress
            best_log = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_recall_distress": val_recall_distress,
            }
            # Save checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_recall": best_val_recall,
                },
                str(best_checkpoint_path),
            )
            print(f"Saved new best checkpoint (val_recall_distress={best_val_recall:.4f}) -> {best_checkpoint_path}")

    print("Training complete. Best:", best_log)
    return best_checkpoint_path


# ---------------------------
# Export helpers
# ---------------------------


def export_model_to_torchscript(model: nn.Module, sample_input: torch.Tensor, out_path: str):
    model.eval()
    try:
        scripted = torch.jit.script(model)
        scripted.save(out_path)
        print(f"TorchScript model saved to {out_path}")
    except Exception as e:
        # fallback to trace if scripting fails
        try:
            traced = torch.jit.trace(model, sample_input)
            traced.save(out_path)
            print(f"TorchScript (traced) model saved to {out_path}")
        except Exception as e2:
            print("TorchScript export failed:", e, e2)


def export_model_to_onnx(model: nn.Module, sample_input: torch.Tensor, out_path: str):
    model.eval()
    try:
        torch.onnx.export(
            model,
            sample_input,
            out_path,
            opset_version=12,
            input_names=["mel_spectrogram"],
            output_names=["logits"],
            dynamic_axes={"mel_spectrogram": {0: "batch", 3: "time"}, "logits": {0: "batch"}},
        )
        print(f"ONNX model saved to {out_path}")
    except Exception as e:
        print("ONNX export failed:", e)


# ---------------------------
# CLI / Orchestration
# ---------------------------


def prepare_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    batch_size: int,
    num_workers: int,
    augment: bool,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = AudioDataset(train_csv, augment=augment)
    val_ds = AudioDataset(val_csv, augment=False)
    test_ds = AudioDataset(test_csv, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


def main_cli(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"Using device: {device}")

    # Validate metadata CSVs exist
    for p in [args.train_csv, args.val_csv, args.test_csv]:
        if not Path(p).exists():
            print(f"ERROR: Required metadata CSV not found: {p}")
            sys.exit(1)

    # Compute class weights (if possible)
    cw = compute_class_weights_from_metadata(args.train_csv, args.num_classes)
    if cw is not None:
        print("Class weights computed:", cw.numpy() if isinstance(cw, torch.Tensor) else cw)
    else:
        print("Class weights not computed (missing or invalid labels). Training will use uniform weights.")

    # Prepare dataloaders
    train_loader, val_loader, test_loader = prepare_dataloaders(
        args.train_csv, args.val_csv, args.test_csv, args.batch_size, args.num_workers, augment=args.augment
    )

    if args.dry_run:
        # Smoke test: run a single batch forward/backward to ensure plumbing works
        print("Running dry-run smoke test...")
        model = ResNetSpec(num_classes=args.num_classes)
        model.to(device)
        model.train()
        try:
            x, y = next(iter(train_loader))
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = nn.CrossEntropyLoss()(logits, y)
            print("Dry-run: forward pass and loss computed:", loss.item())
        except Exception as e:
            print("Dry-run failed:", e)
            sys.exit(1)
        print("Dry-run successful. Exiting (dry_run).")
        return

    # Instantiate model
    model = ResNetSpec(num_classes=args.num_classes)

    # Train
    best_checkpoint = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        mixed_prec=args.mixed_prec,
        grad_clip=args.grad_clip,
    )

    # Load best model for export and test evaluation
    print("Loading best checkpoint for final evaluation & export...")
    ckpt = torch.load(str(best_checkpoint), map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Evaluate on test set
    test_metrics = evaluate_model(model, test_loader, device, args.num_classes)
    print("Test metrics:", test_metrics)

    # Export model (TorchScript & ONNX)
    # Prepare a sample input: compute number of frames for given DURATION
    frames = 1 + int((TARGET_SAMPLES - N_FFT) / HOP_LENGTH)
    sample_input = torch.randn(1, 1, N_MELS, frames).to(device)
    export_dir = Path(args.output_dir) / "model" / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_model_to_torchscript(model, sample_input, str(export_dir / "model_ts.pt"))
    export_model_to_onnx(model, sample_input, str(export_dir / "model.onnx"))

    print("Training + export pipeline finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNetSpec on processed audio data")
    parser.add_argument("--train_csv", type=str, default="data/processed/train/train_metadata.csv", help="Train metadata CSV")
    parser.add_argument("--val_csv", type=str, default="data/processed/val/val_metadata.csv", help="Val metadata CSV")
    parser.add_argument("--test_csv", type=str, default="data/processed/test/test_metadata.csv", help="Test metadata CSV")
    parser.add_argument("--output_dir", type=str, default=".", help="Base output directory for checkpoints and exports")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--augment", action="store_true", help="Enable training augmentations")
    parser.add_argument("--mixed_prec", action="store_true", help="Enable mixed precision training (CUDA only)")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dry_run", action="store_true", help="Run a quick smoke test and exit")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main_cli(args)
