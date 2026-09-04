#!/usr/bin/env python3
"""
migrate_qc_two_level.py

Standalone, idempotent migration script converting existing QC data to the two-level schema:
1. Global Cell Level (identity: <field>_cell_<n>) -> stored in qc_<field>.json
   Allowed statuses: 'good', 'bad', 'pending', 'corrected'
   Legacy global 'mistracked' entries -> mapped to 'bad' (or 'corrected' if in override set).
2. Local Cell Level (identity: <field>_<film>_cell_<n> or raw track ID) -> stored in mistrack_review_state_<field>.json
   Allowed statuses: 'pending', 'mistracked'
   Legacy local 'exhausted' entries -> mapped to status='pending', reviewed=True.

Safeguards:
- Always creates timestamped backups matching: <file>.backup_<YYYYMMDD_HHMM>_qc_two_level_migration
- Runs post-migration validation against formal qc_schema.
- Prints detailed before/after statistics and lists unmapped/edge-case entries.
"""

import sys
import os
import json
import shutil
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Ensure tracking_corrector is importable
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tracking_corrector.qc_schema import (
    GlobalCellQC,
    LocalCellQC,
    validate_global_qc_status,
    validate_local_qc_status,
    InvalidQCStatusError,
)

GLOBAL_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_]+_cell_\d+$")
LOCAL_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_]+_[A-Za-z0-9_]+_cell_\d+$")


def backup_file(filepath: Path, label: str = "qc_two_level_migration") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    backup_name = f"{filepath.name}.backup_{timestamp}_{label}"
    backup_path = filepath.parent / backup_name
    shutil.copy2(filepath, backup_path)
    return backup_path


def classify_key(key: str) -> str:
    """
    Classifies a key into 'global', 'local', or 'unknown'.
    Global format: <field>_cell_<n> (e.g. 3_F0_cell_1)
    Local format: <field>_<film>_cell_<n> (e.g. 3_F0_3_BF1_F0_cell_104)
    """
    if GLOBAL_KEY_PATTERN.match(key):
        # Additional check: local long-form contains film component (e.g. 3_F0_3_BF1_F0_cell_104 has multiple _cell_ or extra parts)
        parts = key.split("_cell_")
        if len(parts) == 2:
            prefix_parts = parts[0].split("_")
            # Field prefix like "3_F0" has 2 parts. Local long form "3_F0_3_BF1_F0" has 5 parts.
            if len(prefix_parts) <= 2 or not any(x in parts[0] for x in ["BF", "FL", "TP"]):
                return "global"
            else:
                return "local"
        return "global"
    elif LOCAL_KEY_PATTERN.match(key):
        return "local"
    elif key.startswith("M156_") or key.startswith("cell_") or key.isdigit():
        return "local_film_track"
    else:
        return "unknown"


def migrate_field_dir(field_dir: Path, execute: bool = False, corrected_overrides: set = None) -> Dict[str, Any]:
    if corrected_overrides is None:
        corrected_overrides = set()

    field_name = field_dir.name
    qc_path = field_dir / f"qc_{field_name}.json"
    review_state_path = field_dir / f"mistrack_review_state_{field_name}.json"

    if not qc_path.exists() and not review_state_path.exists():
        print(f"❌ No QC files found in {field_dir}")
        return {}

    raw_qc = json.loads(qc_path.read_text(encoding="utf-8")) if qc_path.exists() else {}
    raw_review_state = json.loads(review_state_path.read_text(encoding="utf-8")) if review_state_path.exists() else {}

    # Stats tracking
    before_stats = {
        "qc_total_keys": len(raw_qc),
        "qc_global_keys": 0,
        "qc_local_keys": 0,
        "qc_unknown_keys": 0,
        "review_state_total_keys": len(raw_review_state),
        "review_state_statuses": {},
    }

    unmapped_entries = []
    converted_global_mistracked = []

    # Analyze raw_qc before migration
    for k, v in raw_qc.items():
        k_type = classify_key(k)
        if k_type == "global":
            before_stats["qc_global_keys"] += 1
        elif k_type in ("local", "local_film_track"):
            before_stats["qc_local_keys"] += 1
        else:
            before_stats["qc_unknown_keys"] += 1
            unmapped_entries.append((qc_path.name, k, v, "Unrecognized key format"))

    for k, v in raw_review_state.items():
        st = v.get("status") if isinstance(v, dict) else str(v)
        before_stats["review_state_statuses"][st] = before_stats["review_state_statuses"].get(st, 0) + 1

    # Prepare new structures
    new_global_qc: Dict[str, str] = {}
    new_review_state: Dict[str, Dict[str, Any]] = {}

    # 1. Migrate qc_<field>.json -> split into new_global_qc and promote local keys to new_review_state
    for k, v in raw_qc.items():
        k_type = classify_key(k)
        st_raw = v.get("status") if isinstance(v, dict) else str(v)
        st = str(st_raw).strip().lower()

        if k_type == "global":
            if st == "mistracked":
                # Legacy global mistracked conversion (Proposal A)
                target_status = GlobalCellQC.CORRECTED.value if k in corrected_overrides else GlobalCellQC.BAD.value
                new_global_qc[k] = target_status
                converted_global_mistracked.append((k, "mistracked", target_status))
            elif st in GlobalCellQC.valid_statuses():
                new_global_qc[k] = st
            else:
                unmapped_entries.append((qc_path.name, k, v, f"Invalid global status '{st}'"))

        elif k_type in ("local", "local_film_track"):
            # Move local key out of qc_<field>.json into new_review_state
            if st == "mistracked":
                new_review_state[k] = {
                    "status": LocalCellQC.MISTRACKED.value,
                    "reviewed": True,
                    "total_windows": 1,
                    "shown_windows": [],
                    "updated_at": datetime.now().isoformat(),
                }
            elif st in LocalCellQC.valid_statuses():
                new_review_state[k] = {
                    "status": st,
                    "reviewed": False,
                    "total_windows": 1,
                    "shown_windows": [],
                    "updated_at": datetime.now().isoformat(),
                }
            else:
                unmapped_entries.append((qc_path.name, k, v, f"Invalid local status '{st}' moved from qc.json"))

    # 2. Migrate existing mistrack_review_state_<field>.json -> new_review_state
    for k, v in raw_review_state.items():
        if isinstance(v, dict):
            st = str(v.get("status", "pending")).strip().lower()
            total_win = int(v.get("total_windows", 1))
            shown_win = list(v.get("shown_windows", []))

            if st == "exhausted":
                # Proposal B: map exhausted -> pending, reviewed=True
                new_review_state[k] = {
                    "status": LocalCellQC.PENDING.value,
                    "reviewed": True,
                    "total_windows": total_win,
                    "shown_windows": shown_win,
                    "updated_at": datetime.now().isoformat(),
                }
            elif st == LocalCellQC.MISTRACKED.value:
                new_review_state[k] = {
                    "status": LocalCellQC.MISTRACKED.value,
                    "reviewed": True,
                    "total_windows": total_win,
                    "shown_windows": shown_win,
                    "updated_at": datetime.now().isoformat(),
                }
            elif st == LocalCellQC.PENDING.value:
                is_rev = bool(v.get("reviewed", len(shown_win) >= total_win))
                new_review_state[k] = {
                    "status": LocalCellQC.PENDING.value,
                    "reviewed": is_rev,
                    "total_windows": total_win,
                    "shown_windows": shown_win,
                    "updated_at": datetime.now().isoformat(),
                }
            else:
                unmapped_entries.append((review_state_path.name, k, v, f"Unknown status in review state '{st}'"))
        else:
            unmapped_entries.append((review_state_path.name, k, v, "Non-object entry in review state"))

    # After Stats
    after_stats = {
        "global_qc_total": len(new_global_qc),
        "global_qc_good": sum(1 for v in new_global_qc.values() if v == GlobalCellQC.GOOD.value),
        "global_qc_bad": sum(1 for v in new_global_qc.values() if v == GlobalCellQC.BAD.value),
        "global_qc_corrected": sum(1 for v in new_global_qc.values() if v == GlobalCellQC.CORRECTED.value),
        "global_qc_pending": sum(1 for v in new_global_qc.values() if v == GlobalCellQC.PENDING.value),
        "local_review_total": len(new_review_state),
        "local_review_mistracked": sum(1 for v in new_review_state.values() if v.get("status") == LocalCellQC.MISTRACKED.value),
        "local_review_pending_reviewed": sum(1 for v in new_review_state.values() if v.get("status") == LocalCellQC.PENDING.value and v.get("reviewed")),
        "local_review_pending_unreviewed": sum(1 for v in new_review_state.values() if v.get("status") == LocalCellQC.PENDING.value and not v.get("reviewed")),
    }

    # Backup & Write Execution
    backups_created = []
    if execute:
        if qc_path.exists():
            b_qc = backup_file(qc_path)
            backups_created.append(b_qc)
        if review_state_path.exists():
            b_rs = backup_file(review_state_path)
            backups_created.append(b_rs)

        # Write split files
        qc_path.write_text(json.dumps(new_global_qc, indent=2), encoding="utf-8")
        review_state_path.write_text(json.dumps(new_review_state, indent=2), encoding="utf-8")

    # Post-migration Validation Step
    validation_errors = []
    for k, v in new_global_qc.items():
        try:
            validate_global_qc_status(v)
        except InvalidQCStatusError as e:
            validation_errors.append(f"Global QC Key '{k}': {e}")

    for k, v in new_review_state.items():
        st = v.get("status")
        try:
            validate_local_qc_status(st)
        except InvalidQCStatusError as e:
            validation_errors.append(f"Local Review Key '{k}': {e}")

    return {
        "field": field_name,
        "execute": execute,
        "backups_created": backups_created,
        "before_stats": before_stats,
        "after_stats": after_stats,
        "converted_global_mistracked": converted_global_mistracked,
        "unmapped_entries": unmapped_entries,
        "validation_errors": validation_errors,
    }


def print_summary(report: Dict[str, Any]):
    print(f"\n==================================================")
    print(f" QC MIGRATION REPORT FOR FIELD: {report['field']}")
    print(f" Mode: {'EXECUTED (Files Modified)' if report['execute'] else 'DRY RUN (No Files Modified)'}")
    print(f"==================================================")

    if report["backups_created"]:
        print("\n📦 Backups Created:")
        for b in report["backups_created"]:
            print(f"  - {b}")

    before = report["before_stats"]
    print("\n📊 BEFORE MIGRATION:")
    print(f"  qc_{report['field']}.json total keys: {before['qc_total_keys']}")
    print(f"    - Global keys ({GLOBAL_KEY_PATTERN.pattern}): {before['qc_global_keys']}")
    print(f"    - Local keys (per-source): {before['qc_local_keys']}")
    print(f"    - Unknown/Unmapped keys: {before['qc_unknown_keys']}")
    print(f"  mistrack_review_state_{report['field']}.json total keys: {before['review_state_total_keys']}")
    print(f"    - Status distribution: {before['review_state_statuses']}")

    after = report["after_stats"]
    print("\n✨ AFTER MIGRATION (New Two-Level Layout):")
    print(f"  1. Global Cell QC Level (qc_{report['field']}.json): {after['global_qc_total']} total keys")
    print(f"     - good: {after['global_qc_good']}")
    print(f"     - bad: {after['global_qc_bad']}")
    print(f"     - corrected: {after['global_qc_corrected']}")
    print(f"     - pending: {after['global_qc_pending']}")
    print(f"  2. Local Cell QC Level (mistrack_review_state_{report['field']}.json): {after['local_review_total']} total keys")
    print(f"     - mistracked: {after['local_review_mistracked']}")
    print(f"     - pending & reviewed (clean/formerly exhausted): {after['local_review_pending_reviewed']}")
    print(f"     - pending & unreviewed: {after['local_review_pending_unreviewed']}")

    converted = report["converted_global_mistracked"]
    print(f"\n🔄 Global 'mistracked' -> 'bad' Conversions ({len(converted)} total):")
    for k, old_st, new_st in converted[:10]:
        print(f"  - {k}: {old_st} -> {new_st}")
    if len(converted) > 10:
        print(f"  ... and {len(converted) - 10} more")

    unmapped = report["unmapped_entries"]
    print(f"\n⚠️ Unmapped / Edge Case Entries ({len(unmapped)} total):")
    if unmapped:
        for fname, key, val, reason in unmapped:
            print(f"  - [{fname}] Key: '{key}', Val: '{val}' -> Reason: {reason}")
    else:
        print("  None! All keys mapped cleanly.")

    val_errors = report["validation_errors"]
    print(f"\n🔍 Post-Migration Schema Validation:")
    if val_errors:
        print(f"❌ VALIDATION FAILURES DETECTED ({len(val_errors)} errors):")
        for err in val_errors:
            print(f"  - {err}")
    else:
        print("✅ VALIDATION SUCCESSFUL: All migrated records satisfy GlobalCellQC / LocalCellQC schemas!")

    print(f"==================================================\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Migrate QC files to the two-level schema.")
    parser.add_argument("--field-dir", type=str, default="/Volumes/X10 Pro/Movies/2026_07_16_M156/3_F0", help="Path to field directory")
    parser.add_argument("--execute", action="store_true", help="Perform actual migration and file backups")
    args = parser.parse_args()

    field_path = Path(args.field_dir)
    if not field_path.exists():
        print(f"Error: field directory {field_path} does not exist.")
        sys.exit(1)

    report = migrate_field_dir(field_path, execute=args.execute)
    print_summary(report)


if __name__ == "__main__":
    main()
