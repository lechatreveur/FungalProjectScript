#!/usr/bin/env python3
"""Time-aware, censoring-aware septum training on linked cell lineages.

EXPERIMENTAL VARIANT (censoring soft-label prototype):
Replaces the hard `state_known` mask that previously threw away ALL
per-frame state supervision after a right-censored start (or before a
left-censored end) with a soft target derived from the empirically observed
septum-duration distribution. See `duration_survival_probability` and the
"CENSORING SOFT LABEL" changes in `build_targets` below. Everything else is
unchanged from the original septum_train_lineage.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset


def stable_bucket(value: str, seed: int = 0, buckets: int = 10_000) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % buckets


def load_lineages(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def grouped_split(
    lineages: list[dict[str, Any]],
    test_experiment: str,
    val_fraction: float = 0.2,
    seed: int = 0,
    validation_mode: str = "experiment",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    test = [row for row in lineages if row["experiment"] == test_experiment]
    development = [row for row in lineages if row["experiment"] != test_experiment]
    if not test:
        raise ValueError(f"No lineages found for held-out experiment {test_experiment}")

    if validation_mode == "lineage":
        validation: list[dict[str, Any]] = []
        training: list[dict[str, Any]] = []
        threshold = round(val_fraction * 10_000)
        for row in development:
            target = (
                validation
                if stable_bucket(row["lineage_key"], seed) < threshold
                else training
            )
            target.append(row)
        if not training or not validation:
            raise ValueError("Lineage validation split produced an empty partition")
        return training, validation, test
    if validation_mode != "experiment":
        raise ValueError(f"Unknown validation mode: {validation_mode}")

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in development:
        group = f"{row['experiment']}:{row['sequence']}"
        groups.setdefault(group, []).append(row)
    if len(groups) < 2:
        raise ValueError("At least two acquisition/sequence groups are required")

    target_val = max(1, round(len(development) * val_fraction))
    ordered_groups = sorted(
        groups,
        key=lambda group: (
            abs(len(groups[group]) - target_val),
            stable_bucket(group, seed),
        ),
    )
    val_groups: set[str] = {ordered_groups[0]}
    val_count = len(groups[ordered_groups[0]])
    for group in ordered_groups[1:]:
        candidate_count = val_count + len(groups[group])
        if abs(candidate_count - target_val) < abs(val_count - target_val):
            val_groups.add(group)
            val_count = candidate_count

    train = [
        row
        for group, rows in groups.items()
        if group not in val_groups
        for row in rows
    ]
    val = [
        row
        for group, rows in groups.items()
        if group in val_groups
        for row in rows
    ]
    return train, val, test


def _strip_to_tiles(strip: np.ndarray) -> np.ndarray:
    strip = np.asarray(strip, dtype=np.uint8)
    height = int(strip.shape[0])
    if strip.ndim != 2 or strip.shape[1] % height:
        raise ValueError(f"Invalid strip shape: {strip.shape}")
    length = strip.shape[1] // height
    return strip.reshape(height, length, height).transpose(1, 0, 2)


def _normalize_segment_tiles(tiles: np.ndarray) -> np.ndarray:
    """Robustly align film intensity while preserving within-film dynamics."""
    tiles = tiles.astype(np.float32)
    observed = tiles[tiles > 0]
    if len(observed) < 2:
        return tiles / 255.0
    low, high = np.percentile(observed, [5.0, 99.0])
    scale = max(float(high - low), 1.0)
    return np.clip((tiles - float(low)) / scale, 0.0, 1.0)


def _remap_npz_path(path: str) -> str:
    """Rewrite an absolute npz path recorded on the Mac into whatever this
    process's filesystem actually mounts it at. The dataset stores paths as
    seen on the user's Mac (e.g. "/Volumes/X10 Pro/Movies/..."); when this
    script runs inside the sandbox those files are reachable at a different
    mount point. Set SEPTUM_PATH_REMAP="OLD_PREFIX::NEW_PREFIX" to enable;
    no-op (returns path unchanged) when unset, so this has zero effect when
    run directly on the Mac where the stored paths are already correct."""
    remap = os.environ.get("SEPTUM_PATH_REMAP")
    if not remap or "::" not in remap:
        return path
    old_prefix, new_prefix = remap.split("::", 1)
    if path.startswith(old_prefix):
        return new_prefix + path[len(old_prefix):]
    return path


def load_lineage_observations(
    lineage: dict[str, Any],
    tile_size: int = 96,
) -> dict[str, np.ndarray]:
    frames: list[np.ndarray] = []
    times: list[np.ndarray] = []
    modalities: list[np.ndarray] = []
    boundaries: list[np.ndarray] = []

    for segment in lineage["segments"]:
        with np.load(_remap_npz_path(segment["npz_path"]), allow_pickle=True) as sample:
            tiles = _strip_to_tiles(sample["strip"])
        if tiles.shape[1:] != (tile_size, tile_size):
            raise ValueError(
                f"Expected {tile_size}x{tile_size} tiles, got {tiles.shape[1:]} "
                f"in {segment['npz_path']}"
            )
        length = min(len(tiles), int(segment["length"]))
        frames.append(_normalize_segment_tiles(tiles[:length]))
        start = float(segment["start_time_min"])
        interval = float(segment["interval_min"])
        times.append(start + np.arange(length, dtype=np.float32) * interval)
        modality_id = 1 if segment["modality"] == "gfp" else 0
        modalities.append(np.full(length, modality_id, dtype=np.int64))
        boundary_values = np.zeros(length, dtype=np.float32)
        boundary_values[0] = 1.0
        boundaries.append(boundary_values)

    order = np.argsort(np.concatenate(times), kind="stable")
    frame_array = np.concatenate(frames)[order]
    time_array = np.concatenate(times)[order]
    modality_array = np.concatenate(modalities)[order]
    boundary_array = np.concatenate(boundaries)[order]
    return {
        "frames": frame_array,
        "times": time_array,
        "modalities": modality_array,
        "boundaries": boundary_array,
    }


def choose_physical_window(
    lineage: dict[str, Any],
    observed_times: np.ndarray,
    window_minutes: float,
    rng: np.random.Generator,
    positive_probability: float = 0.75,
) -> tuple[float, float]:
    minimum, maximum = float(observed_times.min()), float(observed_times.max())
    if maximum - minimum <= window_minutes:
        return minimum, maximum + 1e-6

    observed_boundaries = [
        float(value)
        for event in lineage["events"]
        for value in (event.get("start_time_min"), event.get("end_time_min"))
        if value is not None
    ]
    if observed_boundaries and rng.random() < positive_probability:
        anchor = float(rng.choice(observed_boundaries))
    else:
        # Anchor on a real observation so a window cannot fall entirely in an
        # acquisition gap.
        anchor = float(rng.choice(observed_times))
    # Random position within the central 60% of a fixed-duration window.
    relative = float(rng.uniform(0.2, 0.8))
    start = anchor - relative * window_minutes
    start = min(max(start, minimum), maximum - window_minutes)
    return start, start + window_minutes


class DurationDistribution:
    """Log-logistic fit to observed complete-event septum durations.

    Survival function S(t) = P(duration > t) = 1 / (1 + (t / median) ** shape_k).
    Fit from exactly two empirical quantiles (median, p90) of TRAIN-split
    complete events only, so val/test statistics never leak into the prior.
    Chosen over a lognormal fit because it has a closed form with no
    dependency on erf/scipy, and because it matches the observed shape
    (mean slightly above median, i.e. mild right skew) well enough for this
    purpose - this is an engineering prior, not a fitted survival model.
    """

    def __init__(self, median_min: float, p90_min: float):
        self.median_min = float(median_min)
        if p90_min <= median_min:
            raise ValueError("p90 must exceed the median for a valid fit")
        # Solve shape_k from S(p90) = 0.1 given the closed form above.
        self.shape_k = math.log(9.0) / math.log(p90_min / median_min)

    def survival(self, elapsed_min: np.ndarray) -> np.ndarray:
        t = np.clip(np.asarray(elapsed_min, dtype=np.float64), 0.0, None)
        ratio = np.divide(t, self.median_min, out=np.zeros_like(t), where=t > 0)
        return (1.0 / (1.0 + ratio ** self.shape_k)).astype(np.float32)

    def to_dict(self) -> dict[str, float]:
        return {"median_min": self.median_min, "shape_k": self.shape_k}


def fit_duration_distribution(lineages: list[dict[str, Any]]) -> Optional[DurationDistribution]:
    """Fit a DurationDistribution from a set of lineages' complete events only."""
    durations = [
        event["end_time_min"] - event["start_time_min"]
        for lineage in lineages
        for event in lineage["events"]
        if not event["left_censored"] and not event["right_censored"]
    ]
    if len(durations) < 10:
        return None
    durations_sorted = sorted(durations)
    median = float(np.median(durations_sorted))
    p90 = float(np.percentile(durations_sorted, 90))
    return DurationDistribution(median, p90)


def build_targets(
    lineage: dict[str, Any],
    times: np.ndarray,
    endpoint_sigma_min: float,
    duration_dist: Optional["DurationDistribution"] = None,
) -> dict[str, np.ndarray | float]:
    state = np.zeros(len(times), dtype=np.float32)
    state_known = np.ones(len(times), dtype=np.float32)
    start_target = np.zeros(len(times), dtype=np.float32)
    end_target = np.zeros(len(times), dtype=np.float32)

    has_left_censoring = False
    has_right_censoring = False
    for event in lineage["events"]:
        start = event.get("start_time_min")
        end = event.get("end_time_min")
        if start is not None:
            start_target = np.maximum(
                start_target,
                np.exp(-0.5 * ((times - float(start)) / endpoint_sigma_min) ** 2),
            )
        if end is not None:
            end_target = np.maximum(
                end_target,
                np.exp(-0.5 * ((times - float(end)) / endpoint_sigma_min) ** 2),
            )
        if start is not None and end is not None:
            state = np.maximum(
                state,
                ((times >= float(start)) & (times <= float(end))).astype(np.float32),
            )
        elif start is None and end is not None:
            # CENSORING SOFT LABEL (left-censored: end observed, start missing).
            # Instead of masking every frame before `end` out of the loss
            # entirely, use the duration survival function measured
            # *backwards* from the observed end: frames just before `end`
            # are near-certain to have been mid-division; frames long before
            # `end` are increasingly unlikely to have been part of this
            # event. Falls back to the original hard mask if no duration
            # prior is available (e.g. too few complete events to fit one).
            region = times <= float(end)
            if duration_dist is not None:
                elapsed_before_end = float(end) - times[region]
                state = np.maximum(
                    state,
                    _scatter(region, duration_dist.survival(elapsed_before_end), len(times)),
                )
            else:
                state_known[region] = 0.0
            has_left_censoring = True
        elif start is not None and end is None:
            # CENSORING SOFT LABEL (right-censored: start observed, end missing).
            # Symmetric case: frames just after `start` are near-certain to
            # still be mid-division; frames long after `start` increasingly
            # unlikely to still be. This is the common "start observed near
            # the edge of the recording" case - previously fully masked out,
            # now contributes a confident soft label for most of that span.
            region = times >= float(start)
            if duration_dist is not None:
                elapsed_after_start = times[region] - float(start)
                state = np.maximum(
                    state,
                    _scatter(region, duration_dist.survival(elapsed_after_start), len(times)),
                )
            else:
                state_known[region] = 0.0
            has_right_censoring = True

    return {
        "state": state,
        "state_known": state_known,
        "start_target": start_target.astype(np.float32),
        "end_target": end_target.astype(np.float32),
        # Conservatively avoid teaching a missing endpoint as an all-negative trace.
        "start_known": float(not has_left_censoring),
        "end_known": float(not has_right_censoring),
        "has_event": float(
            state.max() > 0
            or start_target.max() >= 0.5
            or end_target.max() >= 0.5
        ),
    }


def _scatter(mask: np.ndarray, values: np.ndarray, length: int) -> np.ndarray:
    out = np.zeros(length, dtype=np.float32)
    out[mask] = values
    return out


def physical_time_weights(
    times: np.ndarray,
    modalities: np.ndarray,
    boundaries: np.ndarray,
) -> np.ndarray:
    """Return per-frame quadrature weights without letting acquisition gaps dominate."""
    weights = np.ones(len(times), dtype=np.float32)
    if len(times) <= 1:
        return weights

    block_starts = [0]
    for index in range(1, len(times)):
        if boundaries[index] > 0 or modalities[index] != modalities[index - 1]:
            block_starts.append(index)
    block_starts.append(len(times))

    for start, end in zip(block_starts[:-1], block_starts[1:]):
        block_times = times[start:end]
        if len(block_times) <= 1:
            weights[start:end] = 1.0
            continue
        intervals = np.diff(block_times)
        positive = intervals[intervals > 0]
        typical = float(np.median(positive)) if len(positive) else 1.0
        # A dropped-frame or inter-film gap should not receive proportional loss mass.
        intervals = np.clip(intervals, 1e-6, max(typical * 5.0, 1e-6))
        block_weights = np.empty(len(block_times), dtype=np.float32)
        block_weights[0] = intervals[0] / 2.0
        block_weights[-1] = intervals[-1] / 2.0
        if len(block_times) > 2:
            block_weights[1:-1] = (intervals[:-1] + intervals[1:]) / 2.0
        weights[start:end] = block_weights

    # Each lineage contributes equal total mass regardless of frame rate or duration.
    return weights / max(float(weights.sum()), 1e-6)


class LineageWindowDataset(Dataset):
    def __init__(
        self,
        lineages: list[dict[str, Any]],
        training: bool,
        window_minutes: float = 60.0,
        endpoint_sigma_min: float = 1.0,
        seed: int = 0,
        duration_dist: Optional["DurationDistribution"] = None,
    ):
        self.lineages = lineages
        self.training = training
        self.window_minutes = float(window_minutes)
        self.endpoint_sigma_min = float(endpoint_sigma_min)
        self.seed = int(seed)
        self.epoch = 0
        # Fit ONLY from the training split (see main()) and reused unchanged
        # for validation/test targets too, so the prior itself never sees
        # val/test durations.
        self.duration_dist = duration_dist

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.lineages)

    def __getitem__(self, index: int) -> dict[str, Any]:
        lineage = self.lineages[index]
        observations = load_lineage_observations(lineage)
        times = observations["times"]
        if self.training:
            item_seed = stable_bucket(
                f"{lineage['lineage_key']}:{self.epoch}", self.seed, 2**32
            )
            rng = np.random.default_rng(item_seed)
            start, end = choose_physical_window(
                lineage, times, self.window_minutes, rng
            )
            keep = (times >= start) & (times <= end)
        else:
            keep = np.ones(len(times), dtype=bool)

        selected_times = times[keep]
        selected_modalities = observations["modalities"][keep]
        selected_boundaries = observations["boundaries"][keep]
        targets = build_targets(
            lineage, selected_times, self.endpoint_sigma_min, self.duration_dist
        )
        delta_t = np.zeros(len(selected_times), dtype=np.float32)
        if len(selected_times) > 1:
            delta_t[1:] = np.diff(selected_times)
        relative_time = selected_times - selected_times[0]

        return {
            "lineage_key": lineage["lineage_key"],
            "frames": torch.from_numpy(
                observations["frames"][keep, None].astype(np.float32)
            ),
            "times": torch.from_numpy(selected_times.astype(np.float32)),
            "relative_time": torch.from_numpy(relative_time.astype(np.float32)),
            "delta_t": torch.from_numpy(delta_t),
            "time_weight": torch.from_numpy(
                physical_time_weights(
                    selected_times,
                    selected_modalities,
                    selected_boundaries,
                )
            ),
            "modality": torch.from_numpy(selected_modalities),
            "boundary": torch.from_numpy(selected_boundaries),
            **{
                key: torch.tensor(value, dtype=torch.float32)
                if np.isscalar(value)
                else torch.from_numpy(value)
                for key, value in targets.items()
            },
        }


def collate_lineages(batch: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = torch.tensor([len(item["times"]) for item in batch], dtype=torch.long)
    max_length = int(lengths.max())
    batch_size = len(batch)
    height, width = batch[0]["frames"].shape[-2:]
    result: dict[str, Any] = {
        "lineage_key": [item["lineage_key"] for item in batch],
        "lengths": lengths,
        "frames": torch.zeros(batch_size, max_length, 1, height, width),
        "mask": torch.zeros(batch_size, max_length, dtype=torch.bool),
    }
    vector_fields = (
        "times",
        "relative_time",
        "delta_t",
        "time_weight",
        "modality",
        "boundary",
        "state",
        "state_known",
        "start_target",
        "end_target",
    )
    for field in vector_fields:
        dtype = torch.long if field == "modality" else torch.float32
        result[field] = torch.zeros(batch_size, max_length, dtype=dtype)
    for field in ("start_known", "end_known", "has_event"):
        result[field] = torch.stack([item[field] for item in batch])

    for batch_index, item in enumerate(batch):
        length = len(item["times"])
        result["frames"][batch_index, :length] = item["frames"]
        result["mask"][batch_index, :length] = True
        for field in vector_fields:
            result[field][batch_index, :length] = item[field]
    return result


class TileEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, embedding_dim, 3, padding=1),
            nn.GroupNorm(8, embedding_dim),
            nn.SiLU(),
        )
        self.pool_average = nn.AdaptiveAvgPool2d(1)
        self.pool_maximum = nn.AdaptiveMaxPool2d(1)
        self.output = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        average = self.pool_average(features).flatten(1)
        maximum = self.pool_maximum(features).flatten(1)
        return self.output(torch.cat([average, maximum], dim=1))


class TimeAwareSeptumModel(nn.Module):
    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 96):
        super().__init__()
        self.encoder = TileEncoder(embedding_dim)
        self.modality_embedding = nn.Embedding(2, 8)
        self.temporal = nn.GRU(
            input_size=embedding_dim + 8 + 4,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        output_dim = hidden_dim * 2
        self.state_head = nn.Linear(output_dim, 1)
        self.start_head = nn.Linear(output_dim, 1)
        self.end_head = nn.Linear(output_dim, 1)
        self.has_head = nn.Linear(output_dim, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        frames = batch["frames"]
        batch_size, length = frames.shape[:2]
        encoded = self.encoder(frames.reshape(-1, *frames.shape[2:]))
        encoded = encoded.reshape(batch_size, length, -1)
        modality = self.modality_embedding(batch["modality"])
        time_features = torch.stack(
            [
                torch.log1p(batch["relative_time"].clamp_min(0)),
                torch.log1p(batch["delta_t"].clamp_min(0)),
                batch["boundary"],
                batch["modality"].float(),
            ],
            dim=-1,
        )
        temporal_input = torch.cat([encoded, modality, time_features], dim=-1)
        packed = pack_padded_sequence(
            temporal_input,
            batch["lengths"].cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.temporal(packed)
        output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=length
        )
        masked_output = output.masked_fill(~batch["mask"].unsqueeze(-1), -1e4)
        pooled = masked_output.max(dim=1).values
        return {
            "state": self.state_head(output).squeeze(-1),
            "start": self.start_head(output).squeeze(-1),
            "end": self.end_head(output).squeeze(-1),
            "has": self.has_head(pooled).squeeze(-1),
        }


def focal_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    time_weight: torch.Tensor,
    balance_classes: bool = False,
    gamma: float = 2.0,
) -> torch.Tensor:
    raw = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    probability = torch.sigmoid(logits)
    target_probability = targets * probability + (1 - targets) * (1 - probability)
    weighted = raw * (1 - target_probability).pow(gamma)
    valid_weight = time_weight * mask.float()
    if balance_classes:
        positive_weight = valid_weight * targets
        negative_weight = valid_weight * (1.0 - targets)
        positive_mass = positive_weight.sum()
        negative_mass = negative_weight.sum()
        if positive_mass > 0 and negative_mass > 0:
            valid_weight = (
                0.5 * positive_weight / positive_mass
                + 0.5 * negative_weight / negative_mass
            )
        else:
            valid_weight = valid_weight / valid_weight.sum().clamp_min(1e-8)
    else:
        valid_weight = valid_weight / valid_weight.sum().clamp_min(1e-8)
    return (weighted * valid_weight).sum()


def endpoint_location_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    time_weight: torch.Tensor,
) -> torch.Tensor:
    """Physical-time KL location loss plus endpoint-presence supervision."""
    losses: list[torch.Tensor] = []
    for index in range(len(logits)):
        valid = mask[index]
        if not bool(valid.any()):
            continue
        valid_logits = logits[index, valid]
        measure = time_weight[index, valid].clamp_min(1e-8)
        measure = measure / measure.sum()
        target_mass = targets[index, valid] * measure
        has_endpoint = target_mass.sum() > 1e-6

        # Treat logits as a density over physical time, not over frame count.
        measured_logits = valid_logits + measure.log()
        presence_score = torch.logsumexp(measured_logits, dim=0)
        presence_target = torch.ones_like(presence_score) if has_endpoint else torch.zeros_like(presence_score)
        loss = 0.25 * nn.functional.binary_cross_entropy_with_logits(
            presence_score,
            presence_target,
        )
        if has_endpoint:
            target_distribution = target_mass / target_mass.sum()
            log_target = target_distribution.clamp_min(1e-8).log()
            log_prediction = nn.functional.log_softmax(measured_logits, dim=0)
            loss = loss + torch.sum(
                target_distribution * (log_target - log_prediction)
            )
        losses.append(loss)
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    valid = batch["mask"]
    state_mask = valid & batch["state_known"].bool()
    start_mask = valid & batch["start_known"].bool().unsqueeze(1)
    end_mask = valid & batch["end_known"].bool().unsqueeze(1)
    time_weight = batch["time_weight"]
    state_loss = focal_bce(
        outputs["state"],
        batch["state"],
        state_mask,
        time_weight,
        balance_classes=True,
    )
    start_loss = endpoint_location_loss(
        outputs["start"],
        batch["start_target"],
        start_mask,
        time_weight,
    )
    end_loss = endpoint_location_loss(
        outputs["end"],
        batch["end_target"],
        end_mask,
        time_weight,
    )
    has_loss = nn.functional.binary_cross_entropy_with_logits(
        outputs["has"], batch["has_event"]
    )
    total = state_loss + start_loss + end_loss + 0.5 * has_loss
    return total, {
        "state": float(state_loss.detach()),
        "start": float(start_loss.detach()),
        "end": float(end_loss.detach()),
        "has": float(has_loss.detach()),
    }


def move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def pick_device(requested: Optional[str]) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def average_precision(
    targets: np.ndarray,
    scores: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    targets = np.asarray(targets, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    weights = (
        np.ones(len(targets), dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    positive_mass = float(weights[targets].sum())
    if positive_mass <= 0:
        return math.nan
    order = np.argsort(-scores, kind="stable")
    ordered_targets = targets[order]
    ordered_weights = weights[order]
    true_positives = np.cumsum(ordered_weights * ordered_targets)
    precision = true_positives / np.cumsum(ordered_weights).clip(min=1e-12)
    return float(
        (precision * ordered_weights * ordered_targets).sum() / positive_mass
    )


def temporal_peaks(
    probabilities: np.ndarray,
    times: np.ndarray,
    threshold: float,
    min_separation_min: float,
) -> list[int]:
    candidates = [
        index
        for index, probability in enumerate(probabilities)
        if probability >= threshold
        and (index == 0 or probability >= probabilities[index - 1])
        and (index == len(probabilities) - 1 or probability >= probabilities[index + 1])
    ]
    selected: list[int] = []
    for index in sorted(candidates, key=lambda item: probabilities[item], reverse=True):
        if all(
            abs(float(times[index] - times[existing])) >= min_separation_min
            for existing in selected
        ):
            selected.append(index)
    return sorted(selected)


def match_endpoint_times(
    predicted_times: list[float],
    target_times: list[float],
    tolerance_min: float,
) -> tuple[int, int, int, list[float]]:
    unused_predictions = set(range(len(predicted_times)))
    errors: list[float] = []
    matched = 0
    for target in target_times:
        if not unused_predictions:
            continue
        best = min(
            unused_predictions,
            key=lambda index: abs(predicted_times[index] - target),
        )
        error = abs(predicted_times[best] - target)
        if error <= tolerance_min:
            unused_predictions.remove(best)
            matched += 1
            errors.append(error)
    false_positives = len(unused_predictions)
    false_negatives = len(target_times) - matched
    return matched, false_positives, false_negatives, errors


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    endpoint_threshold: float = 0.5,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    state_tp = state_tn = state_fp = state_fn = 0.0
    state_targets_all: list[np.ndarray] = []
    state_scores_all: list[np.ndarray] = []
    state_weights_all: list[np.ndarray] = []
    has_targets_all: list[float] = []
    has_scores_all: list[float] = []
    endpoint_errors: list[float] = []
    endpoint_tp = endpoint_fp = endpoint_fn = 0
    top1_endpoint_errors: list[float] = []
    top1_endpoint_tp = top1_endpoint_fp = top1_endpoint_fn = 0

    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        outputs = model(batch)
        loss, _ = compute_loss(outputs, batch)
        losses.append(float(loss))

        state_mask = batch["mask"] & batch["state_known"].bool()
        predictions = torch.sigmoid(outputs["state"]) >= 0.5
        targets = batch["state"].bool()
        metric_weight = batch["time_weight"] * state_mask.float()
        state_tp += float(metric_weight[predictions & targets].sum())
        state_tn += float(metric_weight[~predictions & ~targets].sum())
        state_fp += float(metric_weight[predictions & ~targets].sum())
        state_fn += float(metric_weight[~predictions & targets].sum())
        state_targets_all.append(targets[state_mask].cpu().numpy())
        state_scores_all.append(
            torch.sigmoid(outputs["state"])[state_mask].cpu().numpy()
        )
        state_weights_all.append(metric_weight[state_mask].cpu().numpy())
        has_targets_all.extend(batch["has_event"].cpu().numpy().tolist())
        has_scores_all.extend(torch.sigmoid(outputs["has"]).cpu().numpy().tolist())

        for index in range(len(batch["lengths"])):
            length = int(batch["lengths"][index])
            times = batch["times"][index, :length]
            for head, target_name, known_name in (
                ("start", "start_target", "start_known"),
                ("end", "end_target", "end_known"),
            ):
                if not bool(batch[known_name][index]):
                    continue
                target = batch[target_name][index, :length]
                target_values = target.cpu().numpy()
                time_values = times.cpu().numpy()
                probabilities = torch.sigmoid(
                    outputs[head][index, :length]
                ).cpu().numpy()
                predicted_indices = temporal_peaks(
                    probabilities,
                    time_values,
                    threshold=endpoint_threshold,
                    min_separation_min=2.0,
                )
                target_indices = temporal_peaks(
                    target_values,
                    time_values,
                    threshold=0.5,
                    min_separation_min=2.0,
                )
                matched, false_positive, false_negative, errors = match_endpoint_times(
                    [float(time_values[item]) for item in predicted_indices],
                    [float(time_values[item]) for item in target_indices],
                    tolerance_min=5.0,
                )
                endpoint_tp += matched
                endpoint_fp += false_positive
                endpoint_fn += false_negative
                endpoint_errors.extend(errors)
                top1_index = int(np.argmax(probabilities))
                matched, false_positive, false_negative, errors = match_endpoint_times(
                    [float(time_values[top1_index])],
                    [float(time_values[item]) for item in target_indices],
                    tolerance_min=5.0,
                )
                top1_endpoint_tp += matched
                top1_endpoint_fp += false_positive
                top1_endpoint_fn += false_negative
                top1_endpoint_errors.extend(errors)

    sensitivity = state_tp / max(1, state_tp + state_fn)
    specificity = state_tn / max(1, state_tn + state_fp)
    endpoint_precision = endpoint_tp / max(1, endpoint_tp + endpoint_fp)
    endpoint_recall = endpoint_tp / max(1, endpoint_tp + endpoint_fn)
    endpoint_f1 = (
        2 * endpoint_precision * endpoint_recall
        / max(1e-12, endpoint_precision + endpoint_recall)
    )
    top1_endpoint_precision = top1_endpoint_tp / max(
        1, top1_endpoint_tp + top1_endpoint_fp
    )
    top1_endpoint_recall = top1_endpoint_tp / max(
        1, top1_endpoint_tp + top1_endpoint_fn
    )
    top1_endpoint_f1 = (
        2
        * top1_endpoint_precision
        * top1_endpoint_recall
        / max(1e-12, top1_endpoint_precision + top1_endpoint_recall)
    )
    state_targets_array = np.concatenate(state_targets_all)
    state_scores_array = np.concatenate(state_scores_all)
    state_weights_array = np.concatenate(state_weights_all)
    return {
        "loss": float(np.mean(losses)) if losses else math.nan,
        "state_balanced_accuracy": 0.5 * (sensitivity + specificity),
        "state_sensitivity": sensitivity,
        "state_specificity": specificity,
        "state_average_precision": average_precision(
            state_targets_array,
            state_scores_array,
            state_weights_array,
        ),
        "has_event_average_precision": average_precision(
            np.asarray(has_targets_all), np.asarray(has_scores_all)
        ),
        "endpoint_event_precision_at_5_min": endpoint_precision,
        "endpoint_event_recall_at_5_min": endpoint_recall,
        "endpoint_event_f1_at_5_min": endpoint_f1,
        "endpoint_median_absolute_error_min": (
            float(np.median(endpoint_errors)) if endpoint_errors else math.nan
        ),
        "endpoint_within_2_min": (
            float(np.mean(np.asarray(endpoint_errors) <= 2.0))
            if endpoint_errors
            else math.nan
        ),
        "endpoint_within_5_min": (
            float(np.mean(np.asarray(endpoint_errors) <= 5.0))
            if endpoint_errors
            else math.nan
        ),
        "top1_endpoint_event_precision_at_5_min": top1_endpoint_precision,
        "top1_endpoint_event_recall_at_5_min": top1_endpoint_recall,
        "top1_endpoint_event_f1_at_5_min": top1_endpoint_f1,
        "top1_endpoint_median_absolute_error_min": (
            float(np.median(top1_endpoint_errors))
            if top1_endpoint_errors
            else math.nan
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--test-experiment", default="2026_04_30_M135")
    parser.add_argument(
        "--validation-mode",
        choices=("experiment", "lineage"),
        default="experiment",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--window-minutes", type=float, default=60.0)
    parser.add_argument("--endpoint-sigma-min", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--no-duration-censoring",
        action="store_true",
        help="Disable the duration-survival soft label experiment "
        "(reproduces original hard-mask behavior for A/B comparison).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from model_latest.pt / optimizer_latest.pt / history.json "
        "in --output-dir if present. Lets a long run be split across several "
        "short invocations of this script.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Wall-clock budget for this invocation. When set, the script "
        "checkpoints mid-epoch (model/optimizer/RNG/batch offset) and exits "
        "cleanly once the budget is spent, so a caller with a hard per-call "
        "timeout can make guaranteed forward progress across many --resume "
        "invocations instead of losing an entire in-flight epoch.",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    device = pick_device(args.device)
    lineages = load_lineages(args.dataset_dir / "lineages.jsonl")
    train_rows, val_rows, test_rows = grouped_split(
        lineages,
        args.test_experiment,
        seed=args.seed,
        validation_mode=args.validation_mode,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_path = args.output_dir / "split.json"
    if args.resume and split_path.is_file():
        split_manifest = json.loads(split_path.read_text())
    else:
        split_manifest = {
            "seed": args.seed,
            "test_experiment": args.test_experiment,
            "validation_mode": args.validation_mode,
            "train": [row["lineage_key"] for row in train_rows],
            "validation": [row["lineage_key"] for row in val_rows],
            "test": [row["lineage_key"] for row in test_rows],
        }
        split_path.write_text(
            json.dumps(split_manifest, indent=2, sort_keys=True), encoding="utf-8"
        )

    # Fit the censoring duration prior from TRAIN complete events only, so it
    # never sees validation/test durations. Reused unchanged for val/test
    # target construction (a deployed model only ever has the train-time
    # prior available).
    duration_dist = None
    if not args.no_duration_censoring:
        duration_dist = fit_duration_distribution(train_rows)
        if duration_dist is None:
            print("WARNING: not enough complete train events to fit a duration "
                  "prior; falling back to hard censoring mask.")
        else:
            print(f"Fitted duration prior: median={duration_dist.median_min:.1f} min, "
                  f"shape_k={duration_dist.shape_k:.2f}")
            (args.output_dir / "duration_prior.json").write_text(
                json.dumps(duration_dist.to_dict(), indent=2), encoding="utf-8"
            )

    train_dataset = LineageWindowDataset(
        train_rows,
        training=True,
        window_minutes=args.window_minutes,
        endpoint_sigma_min=args.endpoint_sigma_min,
        seed=args.seed,
        duration_dist=duration_dist,
    )
    val_dataset = LineageWindowDataset(
        val_rows,
        training=False,
        endpoint_sigma_min=args.endpoint_sigma_min,
        seed=args.seed,
        duration_dist=duration_dist,
    )
    test_dataset = LineageWindowDataset(
        test_rows,
        training=False,
        endpoint_sigma_min=args.endpoint_sigma_min,
        seed=args.seed,
        duration_dist=duration_dist,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_lineages,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_lineages,
        num_workers=0,
    )

    model = TimeAwareSeptumModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    history: list[dict[str, Any]] = []
    best_selection_key = (-1.0, -1.0, float("-inf"))
    start_epoch = 1
    batch_offset = 0  # batches already completed within start_epoch
    running_losses: list[float] = []  # partial-epoch losses carried across resumes

    # Resume support: the sandbox this script sometimes runs in kills any
    # process once a single invocation's time budget is up, and does not
    # keep background processes alive across separate invocations. So a
    # single `--epochs 30` run may need to be split into several `--resume`
    # calls. Checkpointing happens mid-epoch (not just epoch-to-epoch) so a
    # call that runs out of time budget partway through an epoch still
    # saves real progress instead of redoing that epoch from scratch.
    latest_model_path = args.output_dir / "model_latest.pt"
    latest_opt_path = args.output_dir / "optimizer_latest.pt"
    history_path = args.output_dir / "history.json"
    progress_path = args.output_dir / "train_progress.json"
    if args.resume and latest_model_path.is_file():
        model.load_state_dict(torch.load(latest_model_path, map_location=device))
        if latest_opt_path.is_file():
            optimizer.load_state_dict(torch.load(latest_opt_path, map_location=device))
        if history_path.is_file():
            history = json.loads(history_path.read_text())
        for row in history:
            selection_key = (
                row["validation_top1_endpoint_event_f1_at_5_min"],
                row["validation_endpoint_event_f1_at_5_min"],
                row["validation_state_balanced_accuracy"],
            )
            if selection_key > best_selection_key:
                best_selection_key = selection_key
        completed_epochs = {int(row["epoch"]) for row in history}
        start_epoch = (max(completed_epochs) + 1) if completed_epochs else 1
        if progress_path.is_file():
            progress = json.loads(progress_path.read_text())
            if progress["epoch"] == start_epoch:
                batch_offset = int(progress["batch_offset"])
                running_losses = list(progress["running_losses"])
        print(f"Resuming at epoch {start_epoch}, batch {batch_offset} "
              f"(best_selection_key={best_selection_key})")

    train_indices_all = list(range(len(train_dataset)))
    deadline = (time.monotonic() + args.max_seconds) if args.max_seconds else None

    def save_checkpoint() -> None:
        torch.save(model.state_dict(), latest_model_path)
        torch.save(optimizer.state_dict(), latest_opt_path)

    def out_of_time() -> bool:
        return deadline is not None and time.monotonic() >= deadline

    ran_out_of_time = False
    for epoch in range(start_epoch, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        rng = random.Random(args.seed + epoch)
        order = train_indices_all[:]
        rng.shuffle(order)
        batches = [
            order[i : i + args.batch_size]
            for i in range(0, len(order), args.batch_size)
        ]

        model.train()
        for b_idx in range(batch_offset, len(batches)):
            raw_batch = collate_lineages([train_dataset[i] for i in batches[b_idx]])
            batch = move_batch(raw_batch, device)
            outputs = model(batch)
            loss, _ = compute_loss(outputs, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_losses.append(float(loss.detach()))

            if out_of_time():
                save_checkpoint()
                progress_path.write_text(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "batch_offset": b_idx + 1,
                            "running_losses": running_losses,
                        }
                    ),
                    encoding="utf-8",
                )
                print(f"Time budget spent mid-epoch {epoch} after batch "
                      f"{b_idx + 1}/{len(batches)}; checkpointed and exiting.")
                ran_out_of_time = True
                break
        if ran_out_of_time:
            break

        validation = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(running_losses)),
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        selection_key = (
            validation["top1_endpoint_event_f1_at_5_min"],
            validation["endpoint_event_f1_at_5_min"],
            validation["state_balanced_accuracy"],
        )
        if selection_key > best_selection_key:
            best_selection_key = selection_key
            torch.save(model.state_dict(), args.output_dir / "model_best.pt")
        save_checkpoint()
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        # Note: this sandbox's mounted filesystem sometimes disallows
        # unlink() (PermissionError) even though overwrite-in-place works,
        # so we mark the in-progress file "done" instead of deleting it.
        # The resume check above only trusts it when
        # progress["epoch"] == start_epoch, so a stale done-marker for a
        # completed epoch is harmless.
        progress_path.write_text(
            json.dumps({"epoch": epoch, "batch_offset": 0, "running_losses": []}),
            encoding="utf-8",
        )
        batch_offset = 0
        running_losses = []

    if ran_out_of_time:
        print("Exiting early to respect --max-seconds; rerun with --resume "
              "to continue training. Skipping final locked-test evaluation.")
        return

    model.load_state_dict(
        torch.load(args.output_dir / "model_best.pt", map_location=device)
    )
    threshold_candidates = np.linspace(0.1, 0.9, 9)
    threshold_results = [
        (
            float(threshold),
            evaluate(model, val_loader, device, endpoint_threshold=float(threshold)),
        )
        for threshold in threshold_candidates
    ]
    endpoint_threshold, calibrated_validation = max(
        threshold_results,
        key=lambda item: (
            item[1]["endpoint_event_f1_at_5_min"],
            item[1]["endpoint_event_precision_at_5_min"],
            -abs(item[0] - 0.5),
        ),
    )
    final_report = {
        "device": device,
        "endpoint_probability_threshold": endpoint_threshold,
        "train_lineages": len(train_rows),
        "validation_lineages": len(val_rows),
        "test_lineages": len(test_rows),
        "validation": calibrated_validation,
        "locked_test": evaluate(
            model,
            test_loader,
            device,
            endpoint_threshold=endpoint_threshold,
        ),
    }
    (args.output_dir / "model_config.json").write_text(
        json.dumps(
            {
                "architecture": "TimeAwareSeptumModel",
                "embedding_dim": 64,
                "hidden_dim": 96,
                "endpoint_probability_threshold": endpoint_threshold,
                "endpoint_sigma_min": args.endpoint_sigma_min,
                "window_minutes": args.window_minutes,
                "intensity_normalization": "segment_nonzero_p05_p99",
                "validation_mode": args.validation_mode,
                "dataset_dir": str(args.dataset_dir),
                "duration_censoring_enabled": duration_dist is not None,
                "duration_prior": duration_dist.to_dict() if duration_dist else None,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(final_report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(final_report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
