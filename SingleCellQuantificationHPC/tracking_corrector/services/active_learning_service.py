import json
import os
import shutil
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from SingleCellDataAnalysis.active_learning_label_estimator import (
    RoundMetrics,
    estimate_next_label_target,
)


class ActiveLearningService:
    """Manages active learning coverage tracking, background retraining,

    metrics-gated auto-promotion, and retrain history.
    """

    def __init__(self, base_movie_root: Path):
        self.base_root = base_movie_root
        self._lock = threading.Lock()
        self._state_file = self.base_root / ".tracking_corrector" / "active_learning_state.json"
        self._repo_root = Path("/Users/user/Documents/Python_Scripts/FungalProjectScript")
        self._live_model_dir = self._repo_root / "SingleCellDataAnalysis" / "lineage_model_v1"


    def get_state(self) -> Dict[str, Any]:

        """Load active learning state."""
        with self._lock:
            return self._load_state_unlocked()

    def _load_state_unlocked(self) -> Dict[str, Any]:
        if not self._state_file.exists():
            default_state = {
                "current_round": 1,
                "mobile_labels_saved_count": 0,
                "target_new_labels": 50,  # default initial increment from estimate_next_label_target
                "total_target_lineages": 249,
                "baseline_lineages": 199,
                "training_in_progress": False,
                "training_started_at": None,
                "current_live_checkpoint": str(self._live_model_dir / "model_best.pt"),
                "current_baseline_metrics": {
                    "held_out_experiment": "2026_04_30_M135",
                    "state_balanced_accuracy": 0.7271147406629027,
                    "endpoint_event_f1_at_5_min": 0.2397003745318352,
                    "endpoint_median_absolute_error_min": 2.0,
                },
                "dataset_size_history": [199],
                "metric_history": [
                    {
                        "state_balanced_accuracy": 0.7271147406629027,
                        "endpoint_event_f1_at_5_min": 0.2397003745318352,
                        "endpoint_median_absolute_error_min": 2.0,
                    }
                ],
                "retrain_history": [],
            }
            self._save_state_unlocked(default_state)
            return default_state

        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    data = {}
                return data
        except Exception:
            return {}

    def _save_state_unlocked(self, state: Dict[str, Any]) -> None:
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(state, indent=2)
        tmp_path = self._state_file.parent / f".tmp_{self._state_file.name}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, self._state_file)

    def record_label_saved(self) -> Dict[str, Any]:
        """Increment mobile label counter and check if retraining threshold is reached."""
        with self._lock:
            state = self._load_state_unlocked()
            state["mobile_labels_saved_count"] = int(state.get("mobile_labels_saved_count", 0)) + 1

            count = state["mobile_labels_saved_count"]
            target = state.get("target_new_labels", 50)
            in_progress = state.get("training_in_progress", False)

            trigger = False
            if count >= target and not in_progress:
                state["training_in_progress"] = True
                state["training_started_at"] = datetime.now().isoformat()
                trigger = True

            self._save_state_unlocked(state)

            if trigger:
                round_num = state.get("current_round", 1)
                t = threading.Thread(
                    target=self._run_retrain_job,
                    args=(round_num, False),
                    daemon=True,
                )
                t.start()

            return {
                "mobile_labels_saved_count": count,
                "target_new_labels": target,
                "training_in_progress": state["training_in_progress"],
            }

    def trigger_manual_retrain(self, epochs: int = 30, test_mode: bool = False) -> Dict[str, Any]:
        """Manually trigger active learning retraining."""
        with self._lock:
            state = self._load_state_unlocked()
            if state.get("training_in_progress"):
                return {
                    "status": "warning",
                    "message": "Training is already in progress",
                    "state": state,
                }
            state["training_in_progress"] = True
            state["training_started_at"] = datetime.now().isoformat()
            round_num = state.get("current_round", 1)
            self._save_state_unlocked(state)

        t = threading.Thread(
            target=self._run_retrain_job,
            args=(round_num, test_mode, epochs),
            daemon=True,
        )
        t.start()

        return {
            "status": "success",
            "message": f"Retraining triggered for Round {round_num}",
            "round": round_num,
        }

    def _run_retrain_job(
        self, round_num: int, test_mode: bool = False, epochs: int = 30
    ) -> None:
        """Background worker: rebuilds lineage dataset, trains model, evaluates held-out metrics,

        applies metrics-gated auto-promotion, and re-calculates next round target.
        """
        import subprocess

        print(f"[ActiveLearning] Starting retrain job for Round {round_num} (epochs={epochs})...")

        dataset_out = self._repo_root / "SingleCellDataAnalysis" / f"lineage_dataset_al_r{round_num}"
        model_out = self._repo_root / "SingleCellDataAnalysis" / f"lineage_model_al_r{round_num}"

        env = dict(os.environ)
        env["PYTHONPATH"] = f"{self._repo_root}:{self._repo_root / 'SingleCellQuantificationHPC'}"

        # 1. Rebuild lineage dataset
        try:
            cmd_ds = [
                sys.executable,
                "-m",
                "SingleCellDataAnalysis.septum_lineage_dataset",
                "--output-dir",
                str(dataset_out),
            ]
            res_ds = subprocess.run(cmd_ds, cwd=str(self._repo_root), env=env, capture_output=True, text=True)
            if res_ds.returncode != 0:
                print(f"[ActiveLearning] Dataset build failed stderr:\n{res_ds.stderr}")
                self._fail_retrain(f"Dataset build failed: {res_ds.stderr[-300:] if res_ds.stderr else res_ds.stdout[-300:]}")
                return
        except Exception as exc:
            print(f"[ActiveLearning] Error building dataset for Round {round_num}: {exc}")
            self._fail_retrain(f"Dataset build failed: {exc}")
            return

        # 2. Train septum model from scratch
        # NOTE: septum_train_lineage.py trains from scratch as it does not currently take a fine-tuning checkpoint CLI arg.
        try:
            cmd_train = [
                sys.executable,
                "-m",
                "SingleCellDataAnalysis.septum_train_lineage",
                str(dataset_out),
                "--output-dir",
                str(model_out),
                "--epochs",
                str(epochs if not test_mode else 2),
                "--batch-size",
                "4",
            ]
            res_tr = subprocess.run(cmd_train, cwd=str(self._repo_root), env=env, capture_output=True, text=True)
            if res_tr.returncode != 0:
                print(f"[ActiveLearning] Model training failed stderr:\n{res_tr.stderr}")
                self._fail_retrain(f"Model training failed: {res_tr.stderr[-300:] if res_tr.stderr else res_tr.stdout[-300:]}")
                return
        except Exception as exc:
            print(f"[ActiveLearning] Error training model for Round {round_num}: {exc}")
            self._fail_retrain(f"Model training failed: {exc}")
            return



        # 3. Read evaluation report
        eval_file = model_out / "evaluation.json"
        if not eval_file.is_file():
            self._fail_retrain("evaluation.json missing from output directory")
            return

        try:
            with open(eval_file, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
            locked_test = eval_data.get("locked_test", {})
            new_f1 = float(locked_test.get("endpoint_event_f1_at_5_min", 0.0))
            new_acc = float(locked_test.get("state_balanced_accuracy", 0.0))
            new_mae = float(locked_test.get("endpoint_median_absolute_error_min", 2.0))
            new_lineages = int(eval_data.get("train_lineages", 199))
        except Exception as exc:
            self._fail_retrain(f"Failed to parse evaluation.json: {exc}")
            return

        # 4. Metrics-gated promotion policy check
        with self._lock:
            state = self._load_state_unlocked()
            baseline = state.get("current_baseline_metrics", {})
            baseline_f1 = float(baseline.get("endpoint_event_f1_at_5_min", 0.2397))

            promoted = False
            if new_f1 > baseline_f1:
                promoted = True
                reason = (
                    f"PROMOTED: endpoint_event_f1_at_5_min improved from "
                    f"{baseline_f1:.4f} to {new_f1:.4f}"
                )
                # Overwrite production live checkpoint
                live_pt = self._live_model_dir / "model_best.pt"
                prev_pt = self._live_model_dir / "model_best_prev.pt"
                new_pt = model_out / "model_best.pt"

                if live_pt.exists():
                    shutil.copy2(live_pt, prev_pt)
                if new_pt.exists():
                    shutil.copy2(new_pt, live_pt)

                # Update live evaluation.json
                live_eval_path = self._live_model_dir / "evaluation.json"
                if live_eval_path.exists():
                    shutil.copy2(live_eval_path, self._live_model_dir / "evaluation_prev.json")
                shutil.copy2(eval_file, live_eval_path)

                state["current_baseline_metrics"] = {
                    "held_out_experiment": "2026_04_30_M135",
                    "state_balanced_accuracy": new_acc,
                    "endpoint_event_f1_at_5_min": new_f1,
                    "endpoint_median_absolute_error_min": new_mae,
                }
            else:
                promoted = False
                reason = (
                    f"DID NOT PROMOTE: endpoint_event_f1_at_5_min ({new_f1:.4f}) "
                    f"did not exceed baseline ({baseline_f1:.4f})"
                )

            # Record history
            attempt_record = {
                "round": round_num,
                "timestamp": datetime.now().isoformat(),
                "labels_collected": state.get("mobile_labels_saved_count", 0),
                "total_train_lineages": new_lineages,
                "output_dir": str(model_out),
                "new_metrics": {
                    "endpoint_event_f1_at_5_min": new_f1,
                    "state_balanced_accuracy": new_acc,
                    "endpoint_median_absolute_error_min": new_mae,
                },
                "baseline_metrics": {
                    "endpoint_event_f1_at_5_min": baseline_f1,
                    "state_balanced_accuracy": baseline.get("state_balanced_accuracy"),
                    "endpoint_median_absolute_error_min": baseline.get("endpoint_median_absolute_error_min"),
                },
                "promoted": promoted,
                "reason": reason,
            }
            state.setdefault("retrain_history", []).append(attempt_record)

            # 5. Re-run estimate_next_label_target for next round
            ds_hist = state.get("dataset_size_history", [199]) + [new_lineages]
            m_hist_raw = state.get("metric_history", []) + [{
                "state_balanced_accuracy": new_acc,
                "endpoint_event_f1_at_5_min": new_f1,
                "endpoint_median_absolute_error_min": new_mae,
            }]

            round_metrics_list = [
                RoundMetrics(
                    state_balanced_accuracy=m.get("state_balanced_accuracy", 0.7271),
                    endpoint_event_f1_at_5_min=m.get("endpoint_event_f1_at_5_min", 0.2397),
                    endpoint_median_absolute_error_min=m.get("endpoint_median_absolute_error_min", 2.0),
                )
                for m in m_hist_raw
            ]

            last_metrics = round_metrics_list[-1]
            try:
                next_target = estimate_next_label_target(
                    last_metrics, ds_hist, round_metrics_list
                )
            except Exception as e:
                print(f"[ActiveLearning] Next target estimation warning: {e}")
                next_target = ds_hist[-1] + 50

            new_target_inc = max(10, next_target - ds_hist[-1])

            state["dataset_size_history"] = ds_hist
            state["metric_history"] = m_hist_raw
            state["total_target_lineages"] = next_target
            state["target_new_labels"] = new_target_inc
            state["mobile_labels_saved_count"] = 0
            state["current_round"] = round_num + 1
            state["training_in_progress"] = False
            state["training_started_at"] = None

            self._save_state_unlocked(state)

        print(f"[ActiveLearning] Round {round_num} retrain completed. Outcome: {reason}")

    def _fail_retrain(self, reason: str) -> None:
        with self._lock:
            state = self._load_state_unlocked()
            state["training_in_progress"] = False
            state["training_started_at"] = None
            state.setdefault("retrain_history", []).append({
                "round": state.get("current_round", 1),
                "timestamp": datetime.now().isoformat(),
                "promoted": False,
                "reason": f"FAILED: {reason}",
            })
            self._save_state_unlocked(state)
