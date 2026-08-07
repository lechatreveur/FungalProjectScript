#!/usr/bin/env python3
"""Review-only full-lineage inference for the tracking correction UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .septum_train_lineage import TimeAwareSeptumModel


class SeptumLineageInference:
    def __init__(self, checkpoint: Path, device: str = "cpu"):
        self.checkpoint = Path(checkpoint)
        self.device = device
        config_path = self.checkpoint.parent / "model_config.json"
        self.config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.is_file()
            else {}
        )
        self.model = TimeAwareSeptumModel(
            embedding_dim=int(self.config.get("embedding_dim", 64)),
            hidden_dim=int(self.config.get("hidden_dim", 96)),
        ).to(device)
        self.model.load_state_dict(
            torch.load(self.checkpoint, map_location=device, weights_only=True)
        )
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        frames: np.ndarray,
        times_min: np.ndarray,
        modalities: np.ndarray,
        boundaries: np.ndarray,
    ) -> dict[str, Any]:
        frames = np.asarray(frames, dtype=np.float32)
        times = np.asarray(times_min, dtype=np.float32)
        modalities = np.asarray(modalities, dtype=np.int64)
        boundaries = np.asarray(boundaries, dtype=np.float32)
        if not (len(frames) == len(times) == len(modalities) == len(boundaries)):
            raise ValueError("Lineage inference arrays must have equal length")
        if len(frames) == 0:
            raise ValueError("Lineage inference received no observed frames")

        if self.config.get("intensity_normalization") == "segment_nonzero_p05_p99":
            # v2 and later are exported already normalized by the service.
            normalized = frames
        else:
            # v1 was trained on the original uint8 / 255 representation.
            normalized = frames / 255.0 if float(frames.max()) > 1.0 else frames

        relative_time = times - times[0]
        delta_t = np.zeros(len(times), dtype=np.float32)
        if len(times) > 1:
            delta_t[1:] = np.diff(times)
        batch = {
            "frames": torch.from_numpy(normalized[:, None]).unsqueeze(0).to(self.device),
            "times": torch.from_numpy(times).unsqueeze(0).to(self.device),
            "relative_time": torch.from_numpy(relative_time).unsqueeze(0).to(self.device),
            "delta_t": torch.from_numpy(delta_t).unsqueeze(0).to(self.device),
            "modality": torch.from_numpy(modalities).unsqueeze(0).to(self.device),
            "boundary": torch.from_numpy(boundaries).unsqueeze(0).to(self.device),
            "lengths": torch.tensor([len(times)], dtype=torch.long),
            "mask": torch.ones(1, len(times), dtype=torch.bool, device=self.device),
        }
        outputs = self.model(batch)
        return {
            key: torch.sigmoid(outputs[key][0]).cpu().numpy()
            for key in ("state", "start", "end")
        } | {"has_event": float(torch.sigmoid(outputs["has"][0]).cpu())}
