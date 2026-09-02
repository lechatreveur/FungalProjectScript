# Training reproducibility checklist

For any trained model kept in this project — the septum classifier
(`SingleCellDataAnalysis/septum_train_binary.py`), the autoencoder and
contrastive families (`FC_AE_*`, `FC_Contrastive_*`, `Video_AE_*`), the lineage
models. `septum_train_binary.py` is the worked reference below.

Referenced by [PROJECT_POLICY.md](PROJECT_POLICY.md) P10. Provenance rules are
P3; checkpoint relocation between SSD / NAS / repo is P5.

---

## 1. Before training — pre-flight

- [ ] **Manifest sanity.** Each `working_dir/training_dataset/manifest.csv` loads,
      is non-empty, and its `has_septum` (positive) count matches what the GUI
      shows for that export — the recurring under-sampling bug
      (`DEVELOPMENT_NOTES.md` §3). Record row count and positive count per
      working dir.
- [ ] **Class balance.** Note the positive fraction; it is what
      `compute_pos_weight_from_manifest()` turns into the BCE `pos_weight`.
      A surprising value means the manifest is wrong, not the model.
- [ ] **Known-trap check.** Confirm in the training code:
      - septum polarity is learned by **50% random inversion**, never a
        label-based flip (`DEVELOPMENT_NOTES.md` §2 — currently
        `septum_train_binary.py` line ~205);
      - `BatchNorm` layers are present in the tile and temporal convolutions
        (their absence was a real "lazy 26% predictor" bug);
      - multi-film manifest rows are keyed by `(film_name, cell_id)`.
- [ ] **Dataset paths resolve.** The NPZ strips each manifest row points at
      exist on this machine (P4 — resolve via config / env, do not assume a
      mount).
- [ ] **Deterministic split.** `split_train_val()` hashes `film_name + cell_id`
      with Python's builtin `hash()`, so the train/val split is only stable
      within a process unless `PYTHONHASHSEED` is fixed. Export
      `PYTHONHASHSEED=0` (or an explicit value) for every run of a given model,
      and record it — otherwise a resume or a second machine gets a different
      split and the val metric is not comparable. See §5.
- [ ] **Seed chosen and written down.** `--seed`. The dataset RNG is
      `np.random.default_rng(seed)` for train, `seed + 1` for val.

## 2. During / at the end — what every kept checkpoint records

The checkpoint payload is minimal (`{"state_dict", "D"}` for the septum model).
Write a sidecar **`<checkpoint>.provenance.json`** next to each kept `.pt`
(P3). Required fields:

```json
{
  "artifact": "model_best.pt",
  "created": "2026-09-02T12:00:00Z",
  "created_by": "SingleCellDataAnalysis/septum_train_binary.py",
  "command": "python septum_train_binary.py \"/Volumes/X10 Pro/Movies/2025_12_31_M92\" \"/Volumes/X10 Pro/Movies/2026_01_08_M93\" --epochs 100 --batch_size 32 --L_max 81 --seed 0",
  "git_commit": "6231da7",
  "git_dirty": false,
  "host": "workstation-ssd",
  "pythonhashseed": "0",
  "training_data": [
    { "working_dir": "/Volumes/X10 Pro/Movies/2025_12_31_M92",
      "manifest_rows": 812, "manifest_positives": 214,
      "manifest_sha256": "…" },
    { "working_dir": "/Volumes/X10 Pro/Movies/2026_01_08_M93",
      "manifest_rows": 640, "manifest_positives": 158,
      "manifest_sha256": "…" }
  ],
  "hyperparams": {
    "D": 64, "L_min": 16, "L_max": 81, "batch_size": 32, "lr": 1e-3,
    "epochs": 100, "include_pos_prob": 0.7, "balanced_sampling": true,
    "seed": 0
  },
  "pos_weight_used": 3.41,
  "augmentation": "4x random 0.5 flips + 0.5 polarity inversion",
  "split": "hash(film_name+cell_id)%100 < 85 train / >= 85 val; PYTHONHASHSEED=0",
  "epoch_saved": 100,
  "best_epoch": 73,
  "val_loss": 0.211,
  "val_metrics": { "acc": 0.94, "f1": 0.88 },
  "resumed_from": null
}
```

The non-negotiable subset (same as P3): **git commit, training data identity
(working dirs + manifest hashes), generating script + command line, seed +
`PYTHONHASHSEED`, hyperparameters, date.**

- [ ] **Per-run log, not the shared append file.** `septum_train_binary.py`
      appends every run to one `training.log` at repo root (gitignored). Copy or
      redirect this run's slice to `checkpoints_binary/<tag>_train.log` next to
      the checkpoint so the history is not lost or interleaved.
- [ ] **Record the exact command line** in the provenance record (verbatim,
      including quoting).
- [ ] **Note any resume.** If `--resume_from` was used, record the parent
      checkpoint and be aware the train/val split may differ from the parent's
      run (§5) — say so in the record.

## 3. Before promoting a checkpoint to production

Promotion (making a checkpoint the one `one_cell_quantification_1CH.py` /
`tracking_corrector` / an inference script loads) is a **Level 4–5** step (P2).

- [ ] **Benchmark against curated ground truth.** For the septum model:
      label-count parity with the GUI and per-interval agreement on a held-out
      film. For trackers: IoU / survival / final-IoU vs the curated set
      (`SingleCellQuantificationHPC/tracker_comparison_summary.md`,
      `benchmark_finetuned.py`). Store the full per-item result.
- [ ] **Compare against the current production checkpoint** on the same
      benchmark (P2 Level 3): state whether it is better, and by how much, on
      which metric.
- [ ] **Level 5 sign-off** for any change to augmentation, label semantics, the
      decision threshold, or the input normalization — these change what the
      output *means*, not just its quality.
- [ ] **Relocation follows P5.** Moving the checkpoint to `/Volumes/.../AI/` or
      the NAS: back up, keep the old one until the new one is benchmarked,
      record the move.

## 4. What to keep

- The promoted checkpoint + its `.provenance.json` + its per-run log.
- The benchmark result table (per-item, not just aggregate).
- Superseded checkpoints stay with their original provenance and their
  benchmark numbers — do not delete them to tidy up (P3).

## 5. Known reproducibility gaps (flag, not yet fixed)

- **`split_train_val()` uses builtin `hash()`.** Deterministic only within one
  process, or with `PYTHONHASHSEED` pinned. Without it, a resume or a run on
  another machine reshuffles train/val, so val metrics across such runs are not
  comparable and there can be train/val leakage across a resume. Mitigation
  today: always pin `PYTHONHASHSEED` and record it. A proper fix (hashlib-based
  stable split) is a Level 3 correctness change, tracked separately.
- **`training.log` is a single shared append file** at repo root and is
  gitignored. Until per-run logging exists, §2's "copy the slice" step is
  manual and must not be skipped.
