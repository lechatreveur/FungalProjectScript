#!/usr/bin/env python3
"""Build an audited, non-destructive lineage index for septum model training.

The output references existing per-film NPZ samples instead of copying image data.
Every observation receives an absolute physical timestamp, modality, acquisition
boundary, and censoring-aware event representation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd


@dataclass(frozen=True)
class AcquisitionMetadata:
    path: str
    acquired_at: datetime
    interval_min: float
    repeat_count: int


def parse_duration_minutes(value: str) -> float:
    match = re.fullmatch(
        r"\s*([0-9]+(?:\.[0-9]+)?)\s*(min|s|ms|µs|us)\s*", value
    )
    if not match:
        raise ValueError(f"Unsupported metadata duration: {value!r}")
    magnitude = float(match.group(1))
    unit = match.group(2)
    return {
        "min": magnitude,
        "s": magnitude / 60.0,
        "ms": magnitude / 60_000.0,
        "µs": magnitude / 60_000_000.0,
        "us": magnitude / 60_000_000.0,
    }[unit]


def parse_acquisition_metadata(path: Path) -> AcquisitionMetadata:
    acquired_at: Optional[datetime] = None
    repeat_count: Optional[int] = None
    interval_min: Optional[float] = None

    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if acquired_at is None and line.startswith("DateAndTime="):
                acquired_at = datetime.strptime(
                    line.split("=", 1)[1], "%Y-%m-%d %H:%M:%S"
                )
            elif repeat_count is None and line.startswith("RepeatCount="):
                repeat_count = int(line.split("=", 1)[1])
            elif interval_min is None and line.startswith("ActualInterval="):
                candidate = line.split("=", 1)[1]
                parsed = parse_duration_minutes(candidate)
                # The first active time-series interval is the useful one.
                if parsed > 0:
                    interval_min = parsed
            if acquired_at is not None and repeat_count is not None and interval_min is not None:
                break

    if acquired_at is None or repeat_count is None or interval_min is None:
        raise ValueError(f"Incomplete acquisition metadata: {path}")
    return AcquisitionMetadata(
        path=str(path),
        acquired_at=acquired_at,
        interval_min=float(interval_min),
        repeat_count=int(repeat_count),
    )


def infer_modality(film: str, film_index: int) -> str:
    upper = film.upper()
    if "BF" in upper:
        return "bf"
    if "GFP" in upper or "FL" in upper:
        return "gfp"
    # M133 alternates fluorescence and brightfield acquisitions.
    return "gfp" if film_index % 2 == 0 else "bf"


def pair_censored_boundaries(boundaries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair chronological boundaries without fabricating unobserved endpoints."""
    ordered = sorted(
        boundaries,
        key=lambda item: (
            float(item["time_min"]),
            0 if item["kind"] == "start" else 1,
        ),
    )
    events: list[dict[str, Any]] = []
    open_start: Optional[dict[str, Any]] = None

    for boundary in ordered:
        if boundary["kind"] == "start":
            if open_start is not None:
                events.append(
                    {
                        "start_time_min": open_start["time_min"],
                        "end_time_min": None,
                        "left_censored": False,
                        "right_censored": True,
                        "start_source": open_start,
                        "end_source": None,
                    }
                )
            open_start = boundary
            continue

        if open_start is None:
            events.append(
                {
                    "start_time_min": None,
                    "end_time_min": boundary["time_min"],
                    "left_censored": True,
                    "right_censored": False,
                    "start_source": None,
                    "end_source": boundary,
                }
            )
        else:
            events.append(
                {
                    "start_time_min": open_start["time_min"],
                    "end_time_min": boundary["time_min"],
                    "left_censored": False,
                    "right_censored": False,
                    "start_source": open_start,
                    "end_source": boundary,
                }
            )
            open_start = None

    if open_start is not None:
        events.append(
            {
                "start_time_min": open_start["time_min"],
                "end_time_min": None,
                "left_censored": False,
                "right_censored": True,
                "start_source": open_start,
                "end_source": None,
            }
        )
    return events


def match_metadata_rule(film: str, rules: list[dict[str, str]], root: Path) -> Path:
    matches = [
        root / rule["metadata"]
        for rule in rules
        if re.search(rule["film_pattern"], film)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one metadata rule for {film}, found {len(matches)}"
        )
    if not matches[0].is_file():
        raise FileNotFoundError(matches[0])
    return matches[0]


def endpoint_value(row: pd.Series, column: str) -> int:
    if column not in row.index or pd.isna(row[column]):
        return -1
    return int(row[column])


def build_experiment(
    spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    experiment_id = str(spec["id"])
    root = Path(spec["root"])
    manifest_path = root / "training_dataset" / "manifest.csv"
    linkage_path = root / "sequence_linkage.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not linkage_path.is_file():
        raise FileNotFoundError(linkage_path)

    manifest = pd.read_csv(manifest_path)
    manifest["cell_id"] = pd.to_numeric(manifest["cell_id"], errors="raise").astype(int)
    manifest_index = {
        (str(row.film_name), int(row.cell_id)): row
        for _, row in manifest.iterrows()
    }
    with linkage_path.open("r", encoding="utf-8") as handle:
        linkage = json.load(handle)

    metadata_by_film: dict[str, AcquisitionMetadata] = {}
    linked_films = {
        film
        for sequence_data in linkage.values()
        if isinstance(sequence_data, dict) and "films" in sequence_data
        for film in sequence_data["films"]
    }
    for film in linked_films:
        metadata_path = match_metadata_rule(
            film, spec["metadata_rules"], root
        )
        metadata_by_film[film] = parse_acquisition_metadata(metadata_path)

    experiment_t0 = min(meta.acquired_at for meta in metadata_by_film.values())
    lineage_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    linked_manifest_keys: set[tuple[str, int]] = set()
    metadata_length_mismatches: list[dict[str, Any]] = []

    for sequence, sequence_data in linkage.items():
        if not isinstance(sequence_data, dict) or "films" not in sequence_data:
            continue
        films = list(sequence_data["films"])
        for global_cell_id, local_ids in sequence_data.get("global_cells", {}).items():
            segments: list[dict[str, Any]] = []
            boundaries: list[dict[str, Any]] = []

            for film_index, film in enumerate(films):
                local_cell_id = int(local_ids[film_index]) if film_index < len(local_ids) else -1
                if local_cell_id < 0:
                    continue
                key = (film, local_cell_id)
                row = manifest_index.get(key)
                if row is None:
                    continue
                linked_manifest_keys.add(key)

                metadata = metadata_by_film[film]
                length = int(row["L"])
                if metadata.repeat_count != length:
                    metadata_length_mismatches.append(
                        {
                            "film": film,
                            "manifest_length": length,
                            "metadata_repeat_count": metadata.repeat_count,
                        }
                    )
                start_time_min = (
                    metadata.acquired_at - experiment_t0
                ).total_seconds() / 60.0
                segment = {
                    "experiment": experiment_id,
                    "sequence": sequence,
                    "global_cell_id": str(global_cell_id),
                    "film_index": film_index,
                    "film": film,
                    "local_cell_id": local_cell_id,
                    "modality": infer_modality(film, film_index),
                    "start_time_min": start_time_min,
                    "interval_min": metadata.interval_min,
                    "length": length,
                    "end_time_min": start_time_min + (length - 1) * metadata.interval_min,
                    "npz_path": str(root / str(row["npz_path"])),
                    "metadata_path": metadata.path,
                    "label_source": str(row.get("label_source", "")),
                    "explicit_negative": bool(int(row.get("has", 0)) == 0),
                }
                segments.append(segment)
                segment_rows.append(segment)

                for event_number, suffix in ((1, ""), (2, "_2")):
                    start_idx = endpoint_value(row, f"start_idx{suffix}")
                    end_idx = endpoint_value(row, f"end_idx{suffix}")
                    for kind, local_frame in (("start", start_idx), ("end", end_idx)):
                        if local_frame < 0:
                            continue
                        boundaries.append(
                            {
                                "kind": kind,
                                "time_min": start_time_min
                                + local_frame * metadata.interval_min,
                                "film": film,
                                "film_index": film_index,
                                "local_cell_id": local_cell_id,
                                "local_frame": local_frame,
                                "modality": segment["modality"],
                                "source_event_number": event_number,
                            }
                        )

            if not segments:
                continue
            events = pair_censored_boundaries(boundaries)
            lineage_key = f"{experiment_id}:{sequence}:{global_cell_id}"
            lineage_rows.append(
                {
                    "lineage_key": lineage_key,
                    "experiment": experiment_id,
                    "sequence": sequence,
                    "global_cell_id": str(global_cell_id),
                    "segment_count": len(segments),
                    "event_count": len(events),
                    "complete_event_count": sum(
                        not event["left_censored"] and not event["right_censored"]
                        for event in events
                    ),
                    "segments": segments,
                    "events": events,
                }
            )
            for event_index, event in enumerate(events):
                event_rows.append(
                    {
                        "lineage_key": lineage_key,
                        "experiment": experiment_id,
                        "sequence": sequence,
                        "global_cell_id": str(global_cell_id),
                        "event_index": event_index,
                        "start_time_min": event["start_time_min"],
                        "end_time_min": event["end_time_min"],
                        "left_censored": event["left_censored"],
                        "right_censored": event["right_censored"],
                        "start_film": (
                            event["start_source"]["film"]
                            if event["start_source"] is not None
                            else None
                        ),
                        "end_film": (
                            event["end_source"]["film"]
                            if event["end_source"] is not None
                            else None
                        ),
                    }
                )

    orphan_keys = sorted(set(manifest_index) - linked_manifest_keys)
    audit = {
        "experiment": experiment_id,
        "manifest_rows": len(manifest),
        "linked_manifest_rows": len(linked_manifest_keys),
        "orphan_manifest_rows": len(orphan_keys),
        "orphan_examples": [
            {"film": film, "cell_id": cell_id} for film, cell_id in orphan_keys[:20]
        ],
        "lineages": len(lineage_rows),
        "segments": len(segment_rows),
        "events": len(event_rows),
        "complete_events": sum(
            not row["left_censored"] and not row["right_censored"]
            for row in event_rows
        ),
        "metadata_length_mismatches": metadata_length_mismatches,
        "metadata": {
            film: {
                **asdict(meta),
                "acquired_at": meta.acquired_at.isoformat(),
            }
            for film, meta in sorted(metadata_by_film.items())
        },
    }
    return lineage_rows, segment_rows, event_rows, audit


def write_dataset(
    output_dir: Path,
    lineages: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    events: list[dict[str, Any]],
    audit: dict[str, Any],
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    lineages_path = output_dir / "lineages.jsonl"
    with lineages_path.open("w", encoding="utf-8") as handle:
        for lineage in sorted(lineages, key=lambda row: row["lineage_key"]):
            handle.write(json.dumps(lineage, sort_keys=True) + "\n")
    pd.DataFrame(segments).to_csv(output_dir / "segments.csv", index=False)
    pd.DataFrame(events).to_csv(output_dir / "events.csv", index=False)
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )

    digest = hashlib.sha256()
    for path in (
        lineages_path,
        output_dir / "segments.csv",
        output_dir / "events.csv",
        output_dir / "audit.json",
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    fingerprint = digest.hexdigest()
    (output_dir / "dataset.sha256").write_text(fingerprint + "\n", encoding="utf-8")
    return fingerprint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("septum_lineage_experiments.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    all_lineages: list[dict[str, Any]] = []
    all_segments: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    experiment_audits: list[dict[str, Any]] = []
    for spec in config["experiments"]:
        lineages, segments, events, audit = build_experiment(spec)
        all_lineages.extend(lineages)
        all_segments.extend(segments)
        all_events.extend(events)
        experiment_audits.append(audit)

    combined_audit = {
        "schema_version": 1,
        "time_unit": "minutes",
        "lineages": len(all_lineages),
        "segments": len(all_segments),
        "events": len(all_events),
        "complete_events": sum(
            not row["left_censored"] and not row["right_censored"]
            for row in all_events
        ),
        "experiments": experiment_audits,
    }
    fingerprint = write_dataset(
        args.output_dir,
        all_lineages,
        all_segments,
        all_events,
        combined_audit,
    )
    print(json.dumps({**combined_audit, "fingerprint": fingerprint}, indent=2))


if __name__ == "__main__":
    main()
