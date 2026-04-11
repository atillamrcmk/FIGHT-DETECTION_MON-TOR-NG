from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet_v3_small
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def list_videos_with_labels(root: str) -> List[Tuple[str, int]]:
    """
    Tries to infer labels from folder names:
      - fight, fights => label 1
      - nonfight, nofight, normal => label 0

    Works with unknown RWF2000 folder layouts (MVP).
    """
    root = os.path.abspath(root)
    out: List[Tuple[str, int]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        lower = dirpath.lower()
        label: Optional[int] = None
        if "fight" in lower and "non" not in lower and "no" not in lower:
            label = 1
        if any(k in lower for k in ["nonfight", "no_fight", "nofight", "non_fight", "normal"]):
            label = 0

        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in VIDEO_EXTS:
                continue
            if label is None:
                # fallback: infer from filename
                f = fn.lower()
                if "fight" in f and "non" not in f and "no" not in f:
                    label = 1
                elif any(k in f for k in ["nonfight", "nofight", "non_fight", "normal"]):
                    label = 0
            if label is None:
                continue
            out.append((os.path.join(dirpath, fn), int(label)))
    return out


def read_clip(path: str, clip_len: int, size: int, stride: int = 1) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return None

    frames = []
    try:
        i = 0
        while len(frames) < clip_len:
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            if (i % stride) == 0:
                fr = cv2.resize(fr, (size, size))
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                frames.append(fr)
            i += 1
        if len(frames) < clip_len:
            return None
        arr = np.stack(frames, axis=0)  # (T,H,W,3)
        return arr
    finally:
        cap.release()


class RwfDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int]], clip_len: int, size: int) -> None:
        self.items = items
        self.clip_len = clip_len
        self.size = size
        self.tf = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, y = self.items[idx]
        clip = read_clip(path, clip_len=self.clip_len, size=self.size, stride=1)
        if clip is None:
            # try next item (MVP robustness)
            j = (idx + 1) % len(self.items)
            path, y = self.items[j]
            clip = read_clip(path, clip_len=self.clip_len, size=self.size, stride=1)
            if clip is None:
                clip = np.zeros((self.clip_len, self.size, self.size, 3), dtype=np.uint8)
                y = 0

        # to tensor: (T,3,H,W)
        xs = [self.tf(clip[t]) for t in range(self.clip_len)]
        x = torch.stack(xs, dim=0)
        y = torch.tensor([float(y)], dtype=torch.float32)
        return x, y


class ClipModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        bb = mobilenet_v3_small(weights=None)
        feat = bb.classifier[-1].in_features
        bb.classifier[-1] = nn.Identity()
        self.bb = bb
        self.head = nn.Sequential(
            nn.Linear(feat, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,3,H,W)
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        f = self.bb(x)
        f = f.reshape(b, t, -1).mean(dim=1)
        return self.head(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="RWF2000 root folder (contains fight/nonfight)")
    ap.add_argument("--out", type=str, default="models/fight_mobilenetv3_small.pth")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip_len", type=int, default=16)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--max_per_class",
        type=int,
        default=0,
        help="Optional cap for faster CPU training (e.g. 200). 0 disables.",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    items = list_videos_with_labels(args.data_root)
    if len(items) < 50:
        raise SystemExit(f"Not enough labeled videos found under {args.data_root}. Found: {len(items)}")

    if int(args.max_per_class or 0) > 0:
        maxn = int(args.max_per_class)
        c0 = [it for it in items if int(it[1]) == 0]
        c1 = [it for it in items if int(it[1]) == 1]
        random.shuffle(c0)
        random.shuffle(c1)
        c0 = c0[:maxn]
        c1 = c1[:maxn]
        items = c0 + c1

    random.shuffle(items)
    n = len(items)
    n_train = int(n * 0.85)
    train_items = items[:n_train]
    val_items = items[n_train:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClipModel().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = optim.AdamW(model.parameters(), lr=args.lr)

    train_ds = RwfDataset(train_items, clip_len=args.clip_len, size=args.size)
    val_ds = RwfDataset(val_items, clip_len=args.clip_len, size=args.size)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    best_val = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logit = model(x)
            loss = loss_fn(logit, y)
            loss.backward()
            opt.step()
            tot += float(loss.item())

        model.eval()
        correct = 0
        count = 0
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device)
                y = y.to(device)
                p = torch.sigmoid(model(x))
                pred = (p >= 0.5).float()
                correct += int((pred == y).sum().item())
                count += int(y.numel())
        acc = (correct / max(1, count))
        print(f"epoch {ep} train_loss={tot/max(1,len(train_dl)):.4f} val_acc={acc:.4f}")

        if acc >= best_val:
            best_val = acc
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            torch.save(model.state_dict(), args.out)
            print(f"saved -> {args.out}")

    print("done. best_val_acc=", best_val)


if __name__ == "__main__":
    main()

