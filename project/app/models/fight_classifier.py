from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small


@dataclass
class FightModelConfig:
    weights_path: str = "models/fight_mobilenetv3_small.pth"
    device: Optional[str] = None  # None => auto
    clip_len: int = 16
    stride: int = 8  # run model every N frames
    input_size: int = 224


class FightClipClassifier:
    """
    Lightweight video classifier:
    - MobileNetV3-Small backbone on individual frames
    - Temporal average pooling over clip_len frames
    - 2-class output: non-fight vs fight (sigmoid on fight logit)

    CPU-friendly baseline; GPU auto if available.
    """

    def __init__(self, cfg: FightModelConfig) -> None:
        self.cfg = cfg
        self._device = self._resolve_device(cfg.device)
        self._frame_count = 0

        self.model = self._build_model()
        self.model.to(self._device)
        self.model.eval()

        self.pre = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((cfg.input_size, cfg.input_size)),
                T.ToTensor(),  # 0..1
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.enabled = self._load_weights_if_present()

    def _resolve_device(self, device: Optional[str]) -> torch.device:
        if device:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self) -> nn.Module:
        backbone = mobilenet_v3_small(weights=None)
        feat_dim = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Identity()

        head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),  # fight logit
        )

        class Model(nn.Module):
            def __init__(self, bb: nn.Module, hd: nn.Module) -> None:
                super().__init__()
                self.bb = bb
                self.head = hd

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: (B,T,3,H,W)
                b, t, c, h, w = x.shape
                x = x.reshape(b * t, c, h, w)
                f = self.bb(x)  # (B*T,feat)
                f = f.reshape(b, t, -1).mean(dim=1)  # temporal average
                return self.head(f)  # (B,1)

        return Model(backbone, head)

    def _load_weights_if_present(self) -> bool:
        path = self.cfg.weights_path
        path = os.path.abspath(path) if not os.path.isabs(path) else path
        if not os.path.exists(path):
            return False
        try:
            state = torch.load(path, map_location="cpu")
            self.model.load_state_dict(state, strict=True)
            return True
        except Exception:
            return False

    @torch.no_grad()
    def predict_score_if_ready(self, clip_bgr: Deque[np.ndarray]) -> Optional[float]:
        """
        Returns fight score 0-100 if (enabled and enough frames and stride hit), else None.
        clip_bgr: deque of BGR frames (uint8).
        """
        if not self.enabled:
            return None
        self._frame_count += 1
        if (self._frame_count % self.cfg.stride) != 0:
            return None
        if len(clip_bgr) < self.cfg.clip_len:
            return None

        frames = list(clip_bgr)[-self.cfg.clip_len :]
        # BGR->RGB for torchvision
        tensors = []
        for fr in frames:
            rgb = fr[:, :, ::-1].copy()
            tensors.append(self.pre(rgb))
        x = torch.stack(tensors, dim=0).unsqueeze(0)  # (1,T,3,H,W)
        x = x.to(self._device)
        logit = self.model(x).float().squeeze(0).squeeze(0)
        p = torch.sigmoid(logit).item()
        return float(max(0.0, min(100.0, p * 100.0)))

