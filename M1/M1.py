# %% [markdown]
# # M1 — shallow CNN baseline
# BraTS 2015 FLAIR → binary whole-tumor segmentation. This file and M1.ipynb
# each contain the complete workflow. Run from the repository root on Windows.

# %%
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


MODEL_ID = "m1"
SEEDS = (42, 123, 2026)
THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
SLICES_PER_CASE = 32
IMAGE_SIZE = 128
CACHE_VERSION = "flair_wt_v1"
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]


def project_root() -> Path:
    """Find the repository whether Jupyter starts in root or M1/."""
    candidates = [Path.cwd(), Path.cwd().parent]
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent.parent)
    for candidate in candidates:
        if (candidate / "data" / "splits_v1.csv").is_file():
            return candidate.resolve()
    raise FileNotFoundError("Cannot find data/splits_v1.csv from the current directory")


ROOT = project_root()


def check_environment() -> torch.device:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
        sys.stderr.reconfigure(errors="backslashreplace")
    expected = ROOT / ".venv" / "Scripts" / "python.exe"
    if sys.platform != "win32" or Path(sys.executable).resolve() != expected.resolve():
        raise RuntimeError(f"Select the Windows .venv kernel/interpreter: {expected}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the Windows .venv; check the PyTorch install")
    device = torch.device("cuda")
    print(f"Python: {sys.executable}")
    print(f"PyTorch {torch.__version__}; GPU: {torch.cuda.get_device_name(0)}")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# %% [markdown]
# ## 1. Fixed data contract
# All three models read the same patient-level split and 201-file checksum list.
# MRI data remain under data/raw/BRATS2015; no patient may cross splits.

# %%
def load_split() -> list[dict[str, str]]:
    with (ROOT / "data" / "splits_v1.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 100 or len({row["case_id"] for row in rows}) != 100:
        raise ValueError("Expected 100 unique cases")
    if Counter((row["grade"], row["split"]) for row in rows) != Counter({
        ("HGG", "train"): 56, ("HGG", "val"): 12, ("HGG", "test"): 12,
        ("LGG", "train"): 14, ("LGG", "val"): 3, ("LGG", "test"): 3,
    }):
        raise ValueError("Patient split differs from benchmark contract")
    for row in rows:
        if row["case_id"] != f"{row['grade']}/{row['patient_id']}":
            raise ValueError(f"Invalid case key: {row['case_id']}")
    return rows


def verify_raw_files() -> None:
    """Check every downloaded file against the committed SHA-256 inventory."""
    inventory = ROOT / "data" / "file_sha256.csv"
    with inventory.open(encoding="utf-8", newline="") as handle:
        entries = list(csv.DictReader(handle))
    if len(entries) != 201:
        raise ValueError("Expected 200 MRI files and one license file")
    raw_root = ROOT / "data" / "raw" / "BRATS2015"
    for entry in entries:
        target = raw_root.joinpath(*entry["path"].split("/"))
        if not target.is_file() or target.stat().st_size != int(entry["bytes"]):
            raise FileNotFoundError(f"Missing or wrong-sized dataset file: {target}")
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"Dataset SHA-256 mismatch: {target}")
    print("BraTS 2015: 201/201 files match the committed SHA-256 inventory")


def preprocess_case(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """Read one 3D pair, then return 32 aligned 2D FLAIR/WT slices."""
    # Some SimpleITK Windows builds cannot open the Vietnamese project path.
    # Stage only this pair under the system's ASCII temp directory, then clean up.
    with tempfile.TemporaryDirectory(prefix="brats_mha_") as stage:
        flair_file, label_file = Path(stage) / "flair.mha", Path(stage) / "ot.mha"
        shutil.copyfile(ROOT / row["flair_relpath"], flair_file)
        shutil.copyfile(ROOT / row["mask_relpath"], label_file)
        flair = sitk.ReadImage(str(flair_file))
        label = sitk.ReadImage(str(label_file))
    image = sitk.GetArrayFromImage(flair).astype(np.float32)
    mask = sitk.GetArrayFromImage(label)
    if image.shape != mask.shape:
        raise ValueError(f"Image/mask shape mismatch: {row['case_id']}")
    if not np.allclose(flair.GetSpacing(), label.GetSpacing()) or not np.allclose(
        flair.GetDirection(), label.GetDirection()
    ):
        raise ValueError(f"Image/mask geometry mismatch: {row['case_id']}")
    if not set(np.unique(mask)).issubset({0, 1, 2, 3, 4}):
        raise ValueError(f"Unexpected OT labels: {row['case_id']}")
    brain = image != 0
    if not brain.any():
        raise ValueError(f"Empty FLAIR volume: {row['case_id']}")
    median = float(np.median(image[brain]))
    q25, q75 = np.percentile(image[brain], [25, 75])
    image = np.clip((image - median) / max(float(q75 - q25), 1e-6), -5, 5)
    image = (image + 5) / 10
    image[~brain] = 0
    depth = image.shape[0]
    indices = np.rint(np.linspace(0.2 * (depth - 1), 0.8 * (depth - 1), SLICES_PER_CASE)).astype(int)
    if len(set(indices)) != SLICES_PER_CASE:
        raise ValueError(f"Volume is too shallow for 32 distinct slices: {row['case_id']}")
    x = torch.from_numpy(image[indices].copy()).unsqueeze(1)
    y = torch.from_numpy(np.isin(mask[indices], (1, 2, 3, 4)).astype(np.float32)).unsqueeze(1)
    x = F.interpolate(x, (IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    y = F.interpolate(y, (IMAGE_SIZE, IMAGE_SIZE), mode="nearest")
    return x[:, 0].numpy().astype(np.float16), y[:, 0].numpy().astype(np.uint8)


def cached_case(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    cache_dir = ROOT / "data" / "processed" / CACHE_VERSION
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / (row["case_id"].replace("/", "__") + ".npz")
    if cache_file.is_file():
        with np.load(cache_file) as saved:
            x, y = saved["image"], saved["mask"]
        if x.shape == y.shape == (SLICES_PER_CASE, IMAGE_SIZE, IMAGE_SIZE):
            return x, y
    x, y = preprocess_case(row)
    temporary = cache_file.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, image=x, mask=y)
    temporary.replace(cache_file)
    return x, y


class SliceDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], augment: bool):
        self.augment = augment
        data = [cached_case(row) for row in rows]
        self.images = np.concatenate([pair[0] for pair in data])
        self.masks = np.concatenate([pair[1] for pair in data])
        self.case_ids = [row["case_id"] for row in rows for _ in range(SLICES_PER_CASE)]

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int):
        image = torch.from_numpy(self.images[index].astype(np.float32)).unsqueeze(0)
        mask = torch.from_numpy(self.masks[index].astype(np.float32)).unsqueeze(0)
        if self.augment and torch.rand(()) < 0.5:
            image, mask = image.flip(-1), mask.flip(-1)
        image = (image.repeat(3, 1, 1) - IMAGENET_MEAN) / IMAGENET_STD
        return image, mask, self.case_ids[index]


# %% [markdown]
# ## 2. M1 model and shared benchmark mathematics
# M1 predicts logits directly; sigmoid appears only inside the loss or at inference.

# %%
class ShallowFCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_model() -> nn.Module:
    return ShallowFCN()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = torch.sigmoid(logits.float())
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    soft_dice_loss = 1 - ((2 * intersection + 1) / (denominator + 1)).mean()
    return 0.5 * bce + 0.5 * soft_dice_loss


def patient_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    tp = int(np.logical_and(prediction, truth).sum())
    fp = int(np.logical_and(prediction, ~truth).sum())
    fn = int(np.logical_and(~prediction, truth).sum())
    if tp + fp + fn == 0:
        return {"dice": 1.0, "iou": 1.0, "precision": 1.0, "recall": 1.0}
    return {
        "dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
    }


@torch.inference_mode()
def predict_cases(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    probability: dict[str, list[np.ndarray]] = defaultdict(list)
    truth: dict[str, list[np.ndarray]] = defaultdict(list)
    for images, masks, case_ids in loader:
        outputs = torch.sigmoid(model(images.to(device))).cpu().numpy()
        for index, case_id in enumerate(case_ids):
            probability[case_id].append(outputs[index, 0])
            truth[case_id].append(masks[index, 0].numpy().astype(bool))
    return {key: np.stack(value) for key, value in probability.items()}, {
        key: np.stack(value) for key, value in truth.items()
    }


def score_cases(probability, truth, threshold: float) -> list[dict]:
    rows = []
    for case_id in sorted(probability):
        scores = patient_metrics(probability[case_id] >= threshold, truth[case_id])
        rows.append({"case_id": case_id, "grade": case_id.split("/")[0], **scores})
    return rows


def best_validation_threshold(probability, truth) -> tuple[float, float]:
    scored = [(float(np.mean([row["dice"] for row in score_cases(probability, truth, t)])), t)
              for t in THRESHOLDS]
    dice, threshold = max(scored, key=lambda item: (item[0], -abs(item[1] - 0.5)))
    return threshold, dice


# %% [markdown]
# ## 3. Train, validation, checkpoint and visual check
# Smoke mode uses 2 train patients and 1 validation patient for one epoch.
# Full mode uses the frozen 70/15 patient split and one of three fixed seeds.

# %%
def make_loaders(rows: list[dict[str, str]], smoke: bool):
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    if smoke:
        train_rows, val_rows = train_rows[:2], val_rows[:1]
    train_data = SliceDataset(train_rows, augment=True)
    val_data = SliceDataset(val_rows, augment=False)
    batch_size = 8 if smoke else 16
    return (DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0),
            DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0), val_data)


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def save_preview(model: nn.Module, data: SliceDataset, threshold: float, device: torch.device,
                 path: Path) -> None:
    index = int(np.argmax(data.masks.reshape(len(data), -1).sum(axis=1)))
    image, mask, case_id = data[index]
    model.eval()
    prediction = torch.sigmoid(model(image[None].to(device)))[0, 0].cpu().numpy() >= threshold
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for axis, overlay, title in zip(axes, (None, mask[0].numpy(), prediction),
                                    ("FLAIR", "Ground truth", "Prediction")):
        axis.imshow(data.images[index], cmap="gray", vmin=0, vmax=1)
        if overlay is not None:
            axis.imshow(np.ma.masked_where(overlay == 0, overlay), cmap="autumn", alpha=0.55)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(f"{MODEL_ID}: {case_id}; threshold={threshold:.2f}")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def train(seed: int, rows: list[dict[str, str]], device: torch.device, smoke: bool) -> Path:
    set_seed(seed)
    train_loader, val_loader, val_data = make_loaders(rows, smoke)
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    run_dir = ROOT / "runs" / ("smoke" if smoke else MODEL_ID)
    if smoke:
        run_dir = run_dir / MODEL_ID
    run_dir = run_dir / str(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_dice = -1.0
    epochs = 1 if smoke else 30
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = segmentation_loss(model(images), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        probability, truth = predict_cases(model, val_loader, device)
        threshold, val_dice = best_validation_threshold(probability, truth)
        history.append({"epoch": epoch + 1, "train_loss": running_loss / len(train_loader),
                        "val_dice": val_dice, "threshold": threshold})
        print(f"{MODEL_ID} seed={seed} epoch={epoch + 1}/{epochs} "
              f"loss={history[-1]['train_loss']:.4f} val_dice={val_dice:.4f}")
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({"model": model.state_dict(), "threshold": threshold,
                        "epoch": epoch + 1, "seed": seed}, run_dir / "best.pt")
            save_csv(run_dir / "metrics_val.csv", score_cases(probability, truth, threshold))
            save_preview(model, val_data, threshold, device, run_dir / "preview.png")
    torch.cuda.synchronize()
    summary = {"model": MODEL_ID, "seed": seed, "smoke": smoke, "epochs": epochs,
               "best_val_dice": best_dice, "train_seconds": time.perf_counter() - started,
               "peak_vram_bytes": torch.cuda.max_memory_allocated(),
               "parameters": sum(parameter.numel() for parameter in model.parameters()),
               "python": sys.version.split()[0], "torch": torch.__version__,
               "gpu": torch.cuda.get_device_name(0)}
    save_csv(run_dir / "history.csv", history)
    (run_dir / "config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Artifacts: {run_dir}")
    return run_dir


# %% [markdown]
# ## 4. Locked test evaluation
# Test is only allowed after all three Mx have trained the same seed.

# %%
def evaluate_test(seed: int, rows: list[dict[str, str]], device: torch.device) -> Path:
    for model_id in ("m1", "m2", "m3"):
        if not (ROOT / "runs" / model_id / str(seed) / "best.pt").is_file():
            raise RuntimeError(f"Train and lock {model_id} seed={seed} before opening test")
    run_dir = ROOT / "runs" / MODEL_ID / str(seed)
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model = build_model().to(device)
    model.load_state_dict(checkpoint["model"])
    test_rows = [row for row in rows if row["split"] == "test"]
    test_data = SliceDataset(test_rows, augment=False)
    loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=0)
    probability, truth = predict_cases(model, loader, device)
    metrics = score_cases(probability, truth, float(checkpoint["threshold"]))
    for row in metrics:
        row.update(model=MODEL_ID, seed=seed, split="test")
    output = run_dir / "metrics_test.csv"
    save_csv(output, metrics)
    print(f"Test patient mean Dice: {np.mean([row['dice'] for row in metrics]):.4f}; {output}")
    return output


def run(action: str = "smoke", seed: int = 42) -> None:
    device = check_environment()
    verify_raw_files()
    rows = load_split()
    if action == "prepare":
        for row in rows:
            cached_case(row)
        print(f"Cached {len(rows)} patients in data/processed/{CACHE_VERSION}")
    elif action == "test":
        evaluate_test(seed, rows, device)
    elif action in ("smoke", "train"):
        train(seed, rows, device, smoke=action == "smoke")
    else:
        raise ValueError(f"Unknown action: {action}")


# %% [markdown]
# ## PowerShell entry point
# From repository root: `& .\.venv\Scripts\python.exe .\M1\M1.py --smoke`
# For final training: replace `--smoke` with `--train --seed 42` (then 123, 2026).

# %% CLI_ONLY
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M1 self-contained BraTS 2015 benchmark")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--smoke", action="store_true", help="One epoch on a tiny patient subset")
    actions.add_argument("--prepare", action="store_true", help="Validate and cache all 100 patients")
    actions.add_argument("--train", action="store_true", help="Train one full seed")
    actions.add_argument("--test", action="store_true", help="Evaluate locked test split")
    parser.add_argument("--seed", type=int, default=42, choices=SEEDS)
    args = parser.parse_args()
    action = "prepare" if args.prepare else "train" if args.train else "test" if args.test else "smoke"
    run(action, args.seed)
