"""Fine-tune the cervical-lip U-Net for cross-scanner (OOD) robustness.

The model already segments in-distribution images excellently (~0.96 Dice),
but degrades on images from other scanners.  Since we have no new labelled
data, the lever is *aggressive photometric augmentation* — brightness,
contrast, gamma, noise, speckle, blur, CLAHE — so the network learns to
tolerate the appearance differences between machines, using the existing
FUGC lip labels unchanged.

Runs on CPU (slow) or GPU.  Designed to be copy-pasteable into Colab.
Never overwrites the original weights: writes to a new --out file.

Example:
    python train_unet.py \
        --data "C:/Users/javie/Documents/unet_dataset-20260705T082940Z-3-001/unet_dataset" \
        --init models/unet_fugc_best.pth --out models/unet_fugc_aug.pth \
        --epochs 60 --batch 4 --lr 1e-4
"""

from __future__ import annotations

import argparse
import glob
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.unet_model import UNet

N_CLASSES = 3


# ---------------------------------------------------------------------------
# Augmentation (manual, cv2/numpy — no extra dependencies)
# ---------------------------------------------------------------------------

def _u(a, b):
    return random.uniform(a, b)


def geometric_aug(gray: np.ndarray, mask: np.ndarray):
    """Geometry applied to BOTH image and mask (keeps them aligned)."""
    if random.random() < 0.5:
        gray, mask = cv2.flip(gray, 1), cv2.flip(mask, 1)
    h, w = gray.shape[:2]
    angle, scale = _u(-12, 12), _u(0.9, 1.1)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += _u(-0.06, 0.06) * w
    M[1, 2] += _u(-0.06, 0.06) * h
    gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_REFLECT_101)
    return gray, mask


def photometric_aug(x: np.ndarray) -> np.ndarray:
    """Appearance-only augmentation on a grayscale float image in [0, 1].

    This is the part that matters for cross-scanner robustness.
    """
    if random.random() < 0.85:                       # brightness
        x = x + _u(-0.22, 0.22)
    if random.random() < 0.85:                       # contrast about the mean
        m = float(x.mean())
        x = (x - m) * _u(0.6, 1.7) + m
    x = np.clip(x, 0, 1)
    if random.random() < 0.7:                        # gamma / dynamic range
        x = np.power(np.clip(x, 0, 1), _u(0.6, 1.7))
    if random.random() < 0.5:                        # additive gaussian noise
        x = x + np.random.normal(0, _u(0.01, 0.06), x.shape)
    if random.random() < 0.5:                        # multiplicative speckle
        x = x * (1 + np.random.normal(0, _u(0.02, 0.10), x.shape))
    x = np.clip(x, 0, 1).astype(np.float32)
    r = random.random()                              # blur or sharpen
    if r < 0.25:
        k = random.choice([3, 5])
        x = cv2.GaussianBlur(x, (k, k), 0)
    elif r < 0.4:
        x = np.clip(x * 1.5 - cv2.GaussianBlur(x, (0, 0), 1.0) * 0.5, 0, 1)
    if random.random() < 0.4:                        # random CLAHE
        u8 = (x * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=_u(1.0, 4.0), tileGridSize=(8, 8))
        x = clahe.apply(u8).astype(np.float32) / 255.0
    return np.clip(x, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CervixDataset(Dataset):
    def __init__(self, base: str, split: str, augment: bool, limit: int = 0):
        self.files = sorted(glob.glob(f"{base}/{split}/images/*.png"))
        if limit:
            self.files = self.files[:limit]
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        p = Path(f)
        lab = p.parent.parent / "labels" / p.name
        gray = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        mask = cv2.imread(str(lab), cv2.IMREAD_UNCHANGED)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if self.augment:
            gray, mask = geometric_aug(gray, mask)
        x = gray.astype(np.float32) / 255.0
        if self.augment:
            x = photometric_aug(x)
        return (
            torch.from_numpy(x).unsqueeze(0),
            torch.from_numpy(mask.astype(np.int64)),
        )


# ---------------------------------------------------------------------------
# Loss & metric
# ---------------------------------------------------------------------------

def dice_loss(logits, target, eps: float = 1.0):
    probs = F.softmax(logits, dim=1)
    tgt = F.one_hot(target, N_CLASSES).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * tgt).sum(dims)
    card = probs.sum(dims) + tgt.sum(dims)
    return 1.0 - ((2 * inter + eps) / (card + eps)).mean()


@torch.no_grad()
def val_dice(model, loader, device):
    model.eval()
    inter = torch.zeros(N_CLASSES)
    card = torch.zeros(N_CLASSES)
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        for c in range(N_CLASSES):
            pc, yc = (pred == c), (y == c)
            inter[c] += (pc & yc).sum()
            card[c] += pc.sum() + yc.sum()
    dice = (2 * inter + 1) / (card + 1)
    return dice  # tensor [bg, anterior, posterior]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def load_init(model, path, device):
    state = torch.load(path, map_location=device)
    state = state.get("model_state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state)
    print(f"[init] loaded weights from {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root (contains train/ val/ test/)")
    ap.add_argument("--init", default="models/unet_fugc_best.pth", help="weights to fine-tune from ('' = from scratch)")
    ap.add_argument("--out", default="models/unet_fugc_aug.pth")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--limit", type=int, default=0, help="cap images per split (smoke test)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(args.device)

    tr = DataLoader(CervixDataset(args.data, "train", True, args.limit),
                    batch_size=args.batch, shuffle=True, num_workers=args.workers)
    va = DataLoader(CervixDataset(args.data, "val", False, args.limit),
                    batch_size=args.batch, shuffle=False, num_workers=args.workers)
    print(f"[data] train={len(tr.dataset)} val={len(va.dataset)} device={device}")

    model = UNet(n_channels=1, n_classes=N_CLASSES).to(device)
    if args.init:
        load_init(model, args.init, device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce = nn.CrossEntropyLoss()

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y) + dice_loss(logits, y)
            loss.backward()
            opt.step()
            running += loss.item()
        sched.step()
        d = val_dice(model, va, device)
        mean_lip = float((d[1] + d[2]) / 2)
        print(f"epoch {epoch:3d}/{args.epochs}  loss={running/max(1,len(tr)):.4f}  "
              f"val Dice ant={d[1]:.3f} post={d[2]:.3f} mean_lip={mean_lip:.3f}"
              + ("  * best" if mean_lip > best else ""))
        if mean_lip > best:
            best = mean_lip
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.out)

    print(f"[done] best mean lip Dice={best:.3f}  saved -> {args.out}")


if __name__ == "__main__":
    main()
