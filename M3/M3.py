# %% [markdown]
# # M3 — ResNet-18 transfer learning
# Chạy các ô theo thứ tự: dữ liệu → model → train → biểu đồ → test.
# Notebook Run All huấn luyện, vẽ hình và test; file .py có thêm --smoke/--prepare/--test.

# %%
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from torchvision.models import ResNet18_Weights, resnet18

# Chế độ chỉ dùng cho file .py; notebook Run All luôn train rồi test.
ACTION = "smoke"  # "smoke", "prepare", "train", "test"
SEED = 42          # 42, 123 hoặc 2026

# File .py nhận cùng lựa chọn qua PowerShell: M3.py --train --seed 42.
IN_NOTEBOOK = "__file__" not in globals()
if not IN_NOTEBOOK:
    parser = argparse.ArgumentParser(description="M3: FLAIR -> whole tumor")
    actions = parser.add_mutually_exclusive_group()
    for name in ("smoke", "prepare", "train", "test"):
        actions.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--seed", type=int, default=42, choices=(42, 123, 2026))
    args = parser.parse_args()
    ACTION = next((name for name in ("prepare", "train", "test") if getattr(args, name)), "smoke")
    SEED = args.seed

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")
    sys.stderr.reconfigure(errors="backslashreplace")

ROOT = Path.cwd()
if not (ROOT / "data" / "splits_v1.csv").exists():
    ROOT = ROOT.parent
CACHE = ROOT / "data" / "processed" / "flair_wt_v1"
CACHE.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if ACTION in ("train", "test") and DEVICE.type != "cuda":
    raise RuntimeError("Cần PyTorch CUDA trong Windows .venv để train/test cuối.")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"Chế độ: {ACTION} | Seed: {SEED} | Thiết bị: {DEVICE}")
print(f"Python: {sys.executable} | PyTorch: {torch.__version__}")

# %% [markdown]
# ## 1. Nạp và chuẩn bị dữ liệu
# Mỗi ca có một ảnh FLAIR 3D và một mask OT 3D. Bảng `splits_v1.csv` chia
# **theo bệnh nhân**, nên 32 lát của cùng một ca luôn nằm trong cùng một tập.
# Hàm `read_case` thực hiện cùng một công thức tiền xử lý cho train/val/test.

# %%
with (ROOT / "data" / "splits_v1.csv").open(encoding="utf-8", newline="") as file:
    cases = list(csv.DictReader(file))
train_cases = [row for row in cases if row["split"] == "train"]
val_cases = [row for row in cases if row["split"] == "val"]
test_cases = [row for row in cases if row["split"] == "test"]
assert (len(train_cases), len(val_cases), len(test_cases)) == (70, 15, 15)
assert len({row["case_id"] for row in cases}) == 100
print(f"BraTS 2015: {len(train_cases)} train | {len(val_cases)} validation | {len(test_cases)} test")
print("HGG/LGG:", dict(Counter(row["grade"] for row in cases)))


def read_case(row):
    """Đọc một ca và trả về 32 lát FLAIR/WT, mỗi lát 128×128."""
    # SimpleITK trên máy này cần đường dẫn tạm không có dấu tiếng Việt.
    with tempfile.TemporaryDirectory(prefix="brats_mha_") as folder:
        flair_file, ot_file = Path(folder) / "flair.mha", Path(folder) / "ot.mha"
        shutil.copyfile(ROOT / row["flair_relpath"], flair_file)
        shutil.copyfile(ROOT / row["mask_relpath"], ot_file)
        flair = sitk.GetArrayFromImage(sitk.ReadImage(str(flair_file))).astype(np.float32)
        ot = sitk.GetArrayFromImage(sitk.ReadImage(str(ot_file)))
    if flair.shape != ot.shape:
        raise ValueError(f"FLAIR/OT khác shape: {row['case_id']}")

    # Chuẩn hóa theo median/IQR của voxel FLAIR khác 0, nền giữ bằng 0.
    brain = flair != 0
    median = np.median(flair[brain])
    q25, q75 = np.percentile(flair[brain], (25, 75))
    flair = np.clip((flair - median) / max(q75 - q25, 1e-6), -5, 5)
    flair = (flair + 5) / 10
    flair[~brain] = 0

    # Vị trí lát chỉ phụ thuộc độ sâu ảnh; không nhìn OT để chọn lát.
    depth = flair.shape[0]
    indices = np.rint(np.linspace(0.2 * (depth - 1), 0.8 * (depth - 1), 32)).astype(int)
    x = torch.from_numpy(flair[indices].copy()).unsqueeze(1)
    y = torch.from_numpy(np.isin(ot[indices], (1, 2, 3, 4)).astype(np.float32)).unsqueeze(1)
    x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
    y = F.interpolate(y, size=(128, 128), mode="nearest")
    return x[:, 0].numpy().astype(np.float16), y[:, 0].numpy().astype(np.uint8)


def load_case(row):
    """Đọc cache chung; nếu chưa có thì tiền xử lý và lưu lại."""
    path = CACHE / (row["case_id"].replace("/", "__") + ".npz")
    if not path.exists():
        image, mask = read_case(row)
        np.savez_compressed(path, image=image, mask=mask)
    with np.load(path) as saved:
        return saved["image"], saved["mask"]


if ACTION == "prepare":
    for row in cases:
        load_case(row)
    print(f"Đã chuẩn bị cache cho {len(cases)} bệnh nhân: {CACHE}")

sample_image, sample_mask = load_case(train_cases[0])
print("Một ca sau xử lý — FLAIR:", sample_image.shape, "| WT:", sample_mask.shape)
if IN_NOTEBOOK:
    slice_index = 16
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    axes[0].imshow(sample_image[slice_index], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("FLAIR")
    axes[1].imshow(sample_mask[slice_index], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("WT mask")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{train_cases[0]['case_id']} — lát {slice_index + 1}/32")
    fig.tight_layout()
    plt.show()
    plt.close(fig)

# %% [markdown]
# ## 2. Đưa các lát ảnh vào DataLoader
# `SliceDataset` trả `(ảnh, mask, case_id)`. Chỉ train mới lật ngang đồng bộ.
# FLAIR được lặp thành 3 kênh và chuẩn hóa như các Mx khác.

# %%
MEAN = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
STD = torch.tensor((0.229, 0.224, 0.225))[:, None, None]


class SliceDataset(Dataset):
    def __init__(self, rows, augment=False):
        pairs = [load_case(row) for row in rows]
        self.images = np.concatenate([x for x, _ in pairs])
        self.masks = np.concatenate([y for _, y in pairs])
        self.case_ids = [row["case_id"] for row in rows for _ in range(32)]
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index].astype(np.float32))[None]
        mask = torch.from_numpy(self.masks[index].astype(np.float32))[None]
        if self.augment and torch.rand(()) < 0.5:
            image, mask = image.flip(-1), mask.flip(-1)
        image = (image.repeat(3, 1, 1) - MEAN) / STD
        return image, mask, self.case_ids[index]


if ACTION in ("smoke", "train"):
    selected_train = train_cases[:2] if ACTION == "smoke" else train_cases
    selected_val = val_cases[:1] if ACTION == "smoke" else val_cases
    train_data = SliceDataset(selected_train, augment=True)
    val_data = SliceDataset(selected_val)
    batch_size = 8 if ACTION == "smoke" else 16
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Số lát dùng: {len(train_data)} train | {len(val_data)} validation")

# %% [markdown]
# ## 3. ResNet-18 transfer learning
# Encoder dùng weights ImageNet; decoder tự viết với bốn skip.
# Lớp cuối trả logits; bảng summary cho thấy thứ tự, shape và số tham số.

# %%
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.layers(torch.cat((x, skip), dim=1))


class TransferUNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = resnet18(weights=weights)
        self.encoder.fc = nn.Identity()
        self.dec4 = UpBlock(512 + 256, 256)
        self.dec3 = UpBlock(256 + 128, 128)
        self.dec2 = UpBlock(128 + 64, 64)
        self.dec1 = UpBlock(64 + 64, 64)
        self.final_block = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e0 = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x)))
        e1 = self.encoder.layer1(self.encoder.maxpool(e0))
        e2 = self.encoder.layer2(e1)
        e3 = self.encoder.layer3(e2)
        e4 = self.encoder.layer4(e3)
        x = self.dec4(e4, e3)
        x = self.dec3(x, e2)
        x = self.dec2(x, e1)
        x = self.dec1(x, e0)
        x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
        return self.head(self.final_block(x))


def loss_fn(logits, mask):
    bce = F.binary_cross_entropy_with_logits(logits, mask)
    probability = torch.sigmoid(logits.float())
    intersection = (probability * mask).sum(dim=(1, 2, 3))
    total = probability.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3))
    soft_dice = 1 - ((2 * intersection + 1) / (total + 1)).mean()
    return 0.5 * bce + 0.5 * soft_dice


# Train/smoke nạp ImageNet; --test nạp toàn bộ trọng số từ best.pt ở cuối file.
model = TransferUNet(pretrained=ACTION in ("smoke", "train")).to(DEVICE)
print(summary(model, input_data=torch.zeros(1, 3, 128, 128, device=DEVICE),
              depth=1, row_settings=("var_names",), mode="eval", verbose=0))

# %% [markdown]
# ## 4. Huấn luyện và chọn checkpoint bằng validation
# Mỗi epoch in loss train và Dice validation **theo bệnh nhân**. Chỉ dùng một
# AdamW; encoder đóng băng 5 epoch rồi mở layer4 trong cùng optimizer.
# Smoke chỉ chạy 1 epoch để kiểm code, không phải kết quả benchmark.

# %%
def patient_scores(prediction, truth):
    tp = int(np.logical_and(prediction, truth).sum())
    fp = int(np.logical_and(prediction, ~truth).sum())
    fn = int(np.logical_and(~prediction, truth).sum())
    tn = int(np.logical_and(~prediction, ~truth).sum())
    if tp + fp + fn == 0:
        return {"dice": 1.0, "iou": 1.0, "precision": 1.0,
                "recall": 1.0, "pixel_accuracy": 1.0}
    return {
        "dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "pixel_accuracy": (tp + tn) / (tp + fp + fn + tn),
    }


@torch.inference_mode()
def predict_by_patient(loader):
    model.eval()
    probabilities, masks = defaultdict(list), defaultdict(list)
    for images, truth, case_ids in loader:
        predicted = torch.sigmoid(model(images.to(DEVICE))).cpu().numpy()
        for i, case_id in enumerate(case_ids):
            probabilities[case_id].append(predicted[i, 0])
            masks[case_id].append(truth[i, 0].numpy().astype(bool))
    return ({key: np.stack(value) for key, value in probabilities.items()},
            {key: np.stack(value) for key, value in masks.items()})


def score_at_threshold(probabilities, masks, threshold):
    return [{"case_id": case_id, "grade": case_id.split("/")[0],
             **patient_scores(probabilities[case_id] >= threshold, masks[case_id])}
            for case_id in sorted(probabilities)]


def save_table(path, records):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)

# %%
if ACTION in ("smoke", "train"):
    run_dir = ROOT / "runs" / ("smoke/m3" if ACTION == "smoke" else "m3") / str(SEED)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Đóng băng encoder 5 epoch; một AdamW duy nhất cho cả hai giai đoạn.
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW([
        {"params": list(model.dec4.parameters()) + list(model.dec3.parameters()) +
                   list(model.dec2.parameters()) + list(model.dec1.parameters()) +
                   list(model.final_block.parameters()) + list(model.head.parameters()), "lr": 3e-4},
        {"params": model.encoder.layer4.parameters(), "lr": 1e-5},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    history = []
    best_dice, best_epoch, best_threshold = -1.0, 0, 0.5
    no_improvement = 0
    max_epochs = 1 if ACTION == "smoke" else 30
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        model.encoder.eval()  # BatchNorm của encoder giữ thống kê ImageNet.
        if epoch == 6:
            for parameter in model.encoder.layer4.parameters():
                parameter.requires_grad = True
            print("Mở layer4 để fine-tune từ epoch 6.")
        total_loss = 0.0
        for images, masks, _ in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=DEVICE.type == "cuda"):
                loss = loss_fn(model(images), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        probabilities, truths = predict_by_patient(val_loader)
        candidates = []
        for threshold in THRESHOLDS:
            records = score_at_threshold(probabilities, truths, threshold)
            candidates.append((np.mean([row["dice"] for row in records]), threshold, records))
        val_dice, threshold, val_records = max(candidates, key=lambda x: (x[0], -abs(x[1] - 0.5)))
        learning_rate = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_loader),
                        "val_dice": float(val_dice), "threshold": threshold,
                        "learning_rate": learning_rate})
        save_table(run_dir / "history.csv", history)
        print(f"Epoch {epoch:02d}/{max_epochs} | loss {history[-1]['train_loss']:.4f} | "
              f"val Dice {val_dice:.4f} | ngưỡng {threshold:.2f} | lr {learning_rate:.2g}")

        if val_dice > best_dice + 1e-4:
            best_dice, best_epoch, best_threshold = float(val_dice), epoch, threshold
            no_improvement = 0
            torch.save({"model": model.state_dict(), "threshold": threshold,
                        "epoch": epoch, "seed": SEED}, run_dir / "best.pt")
            save_table(run_dir / "metrics_val.csv", val_records)
        else:
            no_improvement += 1
        scheduler.step(val_dice)
        if ACTION == "train" and epoch >= 8 and no_improvement >= 6:
            print("Dừng sớm: Dice validation không tăng trong 6 epoch.")
            break

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    save_table(run_dir / "history.csv", history)
    config = {"model": "m3", "architecture": "ResNet18-ImageNet-UNet-decoder",
              "optimizer": "AdamW", "learning_rate_decoder": 3e-4, "learning_rate_layer4": 1e-5,
              "pretrained_weights": "ResNet18_Weights.IMAGENET1K_V1",
              "encoder_frozen_epochs": 5, "weight_decay": 1e-4,
              "seed": SEED, "smoke": ACTION == "smoke", "epochs_run": len(history),
              "best_epoch": best_epoch, "best_threshold": best_threshold,
              "best_val_dice": best_dice, "train_seconds": time.perf_counter() - start,
              "peak_vram_bytes": torch.cuda.max_memory_allocated() if DEVICE.type == "cuda" else 0,
              "parameters": sum(p.numel() for p in model.parameters()),
              "python": sys.version.split()[0], "torch": torch.__version__,
              "torchvision": torchvision.__version__,
              "numpy": np.__version__, "simpleitk": sitk.Version_VersionString(),
              "matplotlib": matplotlib.__version__,
              "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU"}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Checkpoint tốt nhất: epoch {best_epoch}, val Dice {best_dice:.4f}")
    print("Đã lưu:", run_dir)

# %% [markdown]
# ## 5. Biểu đồ học và một dự đoán validation
# Đọc lại **checkpoint tốt nhất** trước khi vẽ; không dùng model ở epoch cuối
# nếu validation của epoch đó kém hơn. Biểu đồ và ảnh được lưu trong `runs/`.

# %%
if ACTION in ("smoke", "train"):
    best = torch.load(run_dir / "best.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(best["model"])
    model.eval()

    with (run_dir / "metrics_val.csv").open(encoding="utf-8", newline="") as file:
        validation_scores = list(csv.DictReader(file))
    print(f"Validation — {len(validation_scores)} bệnh nhân, checkpoint epoch {best_epoch}:")
    for name, label in (("dice", "Dice"), ("iou", "IoU"),
                        ("precision", "Precision"), ("recall", "Recall"),
                        ("pixel_accuracy", "Pixel accuracy")):
        mean_score = np.mean([float(row[name]) for row in validation_scores])
        print(f"  {label}: {mean_score:.4f}")

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(epochs, [row["train_loss"] for row in history], "o-")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss train")
    axes[1].plot(epochs, [row["val_dice"] for row in history], "o-")
    axes[1].set(xlabel="Epoch", ylabel="Dice", title="Dice validation theo bệnh nhân")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "learning_curve.png", dpi=140)
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)

    # Chọn lát có vùng WT rõ để minh họa; chỉ dùng mask để chọn ảnh hiển thị.
    index = int(np.argmax(val_data.masks.reshape(len(val_data), -1).sum(axis=1)))
    image, truth, case_id = val_data[index]
    with torch.inference_mode():
        prediction = torch.sigmoid(model(image[None].to(DEVICE)))[0, 0].cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, overlay, title in zip(axes, (None, truth[0].numpy(), prediction >= best_threshold),
                                  ("FLAIR", "WT thật", "WT dự đoán")):
        ax.imshow(val_data.images[index], cmap="gray", vmin=0, vmax=1)
        if overlay is not None:
            ax.imshow(np.ma.masked_where(overlay == 0, overlay), cmap="autumn", alpha=0.55)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{case_id} | ngưỡng {best_threshold:.2f}")
    fig.tight_layout()
    fig.savefig(run_dir / "preview.png", dpi=140)
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)

# %% [markdown]
# ## 6. Test của riêng M3
# Train xong thì dùng checkpoint/ngưỡng đã chọn từ validation để đánh giá test.
# `--test` chỉ cần checkpoint M3 của seed tương ứng; không chờ M1/M2.

# %%
if ACTION in ("train", "test"):
    run_dir = ROOT / "runs" / "m3" / str(SEED)
    if not (run_dir / "best.pt").exists():
        raise RuntimeError(f"Chưa có checkpoint M3 cho seed {SEED}; hãy chạy --train trước.")
    best = torch.load(run_dir / "best.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(best["model"])
    test_data = SliceDataset(test_cases)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=0)
    probabilities, truths = predict_by_patient(test_loader)
    records = score_at_threshold(probabilities, truths, float(best["threshold"]))
    for row in records:
        row.update(model="m3", seed=SEED, split="test")
    save_table(run_dir / "metrics_test.csv", records)
    print(f"Test chính thức — {len(records)} bệnh nhân:")
    for label, group in (("Tất cả", records),
                         ("HGG", [row for row in records if row["grade"] == "HGG"]),
                         ("LGG", [row for row in records if row["grade"] == "LGG"])):
        print(f"  {label} ({len(group)} ca) | "
              f"Dice {np.mean([row['dice'] for row in group]):.4f} | "
              f"IoU {np.mean([row['iou'] for row in group]):.4f} | "
              f"Precision {np.mean([row['precision'] for row in group]):.4f} | "
              f"Recall {np.mean([row['recall'] for row in group]):.4f} | "
              f"Pixel accuracy {np.mean([row['pixel_accuracy'] for row in group]):.4f}")
    print("Đã lưu:", run_dir / "metrics_test.csv")
