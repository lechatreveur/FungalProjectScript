# Development Report: Dense All-Frame Tracking — Pole Stability, Expected-Geometry De-poisoning, and the Contaminated Pass-Through Problem

**Date**: September 11, 2026 (work period 2026-09-10 – 2026-09-11)
**Module**: `SingleCellQuantificationHPC` (scratch: `dense_abbt_interval.py`)
**Dataset**: `2026_08_28_M160` (`5_1_N1_F0`, `5_1_N1_F1`, `5_1_N1_F2`)
**Status**: Validated end-to-end on `hard10` (92/92 segments, 31:05). Eraser detector +
integration complete. One metric-reference defect introduced and documented in §8.4 — the
headline figure is **not** comparable to earlier runs.
**Policy**: introduces **P14** (`PROJECT_POLICY.md` v1.9). All output scratch-only; canonical
masks and keyframes read-only throughout (P5, P14).

---

## 1. Executive Summary

Stage 3 of the pipeline (dense all-frame tracking, per P14) gained **+489 pole-usable frames
against the v4 baseline** across 74 shared cells (52 wins, 5 losses, 17 ties) on the `hard10`
benchmark, with oversize masks cut from 9.9% to 2.8% and division timing unchanged. The headline
rate reads **1,569 / 1,829 (86%)**, which is *lower* than the 91% reported mid-work — because both
the measurement scope and the acceptance reference changed (§2, §8.4). Raw rates across those two
changes are not comparable; the +489 frame delta against a fixed v4 baseline is.

Three defects were diagnosed and fixed, each traced to a concrete mechanism rather than tuned
away:

1. **Expected geometry was poisoned by merge-contaminated frames**, inflating the reference
   length ~2× and causing the pole-recovery machinery to faithfully stretch every real cell to
   the fused scale.
2. **Pole recovery overshot by a constant ~20 px** — exactly the brush radius — tripping its own
   rejection gate, so the repair was computed and then discarded.
3. **`_robust_L` let contaminated interior frames outvote the curated anchors**, violating P14.

A fourth, larger defect was then **identified but not yet fixed**: **536 contaminated frames
(8.9% of FL pass-throughs) never enter the repair set at all** and are invisible to the quality
metric. A helper (`erase_beyond_expected_poles`) and detector (`mark_geometry_bad`) were written
and unit-validated for this; integration was completed and benchmarked (§8.3–8.5).

Two of the author's own earlier conclusions were found to be wrong and are corrected in §6.

---

## 2. Benchmark Results (`hard10`, 92 segments, 1,516 gap frames)

| Metric | v4 baseline | Start of work | + de-poison | + clip fixes | **+ eraser (final)** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Scope (bad frames) | — | 1,516 | 1,516 | 1,516 | **1,829** |
| **Pole-usable** | 67% | 1,079 (71%) | 1,346 (89%) | 1,383 (91%) | **1,569 (86%)** |
| — BF | — | 97% | 99% | 100% | **100%** |
| — FL | — | 69% | 88% | 90% | **85%** |
| **Oversize (>1.35×)** | — | 143 (9.9%) | 66 (4.6%) | 39 (2.7%) | **50 (2.8%)** |
| Recovery discarded | — | 212 | — | 57 | **50** |
| Recovered | — | 1,446 (95%) | 1,442 | 1,443 (95%) | **1,755 (96%)** |
| no_seg | — | 70 (5%) | 74 | 73 (5%) | **74 (4%)** |
| A/B flags vs v4 | — | 38 | 32 | 26 | **23** |

> **The final column is not comparable to the ones before it.** Two things changed at once: the
> scope grew by 313 frames (the detector pulled in previously-invisible contaminated frames), and
> the *acceptance test itself* changed reference (§8.4). The 91% → 86% movement is a change of
> ruler, not a regression in mask quality — but it is also not a clean improvement. See §8.5.

**Division timing unchanged throughout** (P14 §secondary objective): cid514 `t*=36` (matches v4
and curated GT), cid344 `t*=99`, cid22 `t*=75`, cid21 `t*=48`.

Per-cell comparison against v4 on the same cells (v4 pole-usable = `pole_ok + fuse + extrude`):

**Largest gains vs v4 (final run):**

| Cell | dense | v4 | Δ |
| :--- | ---: | ---: | ---: |
| cid425 `FL7_F0` | 97 | 16 | +81 |
| cid404 `FL7_F0` | 85 | 6 | +79 |
| cid513 `FL6_F1` | 66 | 28 | +38 |
| cid489 `FL7_F1` | 53 | 17 | +36 |
| cid18 `FL6_F0` | 59 | 26 | +33 |
| cid19 `FL7_F0` | 48 | 20 | +28 |

**Largest losses vs v4 (final run):**

| Cell | dense | v4 | Δ |
| :--- | ---: | ---: | ---: |
| cid12 `FL5_F0` | 1 | 18 | -17 |
| cid18 `FL4_F0` | 17 | 30 | -13 |
| cid12 `FL2_F0` | 10 | 20 | -10 |
| cid7 `FL4_F0` | 20 | 29 | -9 |
| cid22 `FL4_F0` | 13 | 16 | -3 |

Net across 74 shared cells: **+489** frames (52 wins, 5 losses, 17 ties).
The losses concentrate on cells whose length changes across the interval and are **suspected**
artefacts of the scalar `L_ref` (§8.4) — that is unverified, so they are reported as losses
until confirmed frame-by-frame.

---

## 3. Defect 1 — Merge-Contaminated Expected Geometry

### 3.1 Observation

On `5_1_N1_FL6_F1` cid94 interval [50,100], dense-touched frames rendered at 113–130 px while
the canonical daughter is ~65 px and both interval anchors agree at 71 px.

### 3.2 Mechanism

Probing `interpolate_expected_geometry` directly returned `expected_major_axis ≈ 122–124` for
t54–t69. Three compounding causes:

1. **The interpolation basis was contaminated.** The canonical track has real sister-fusion
   frames at t52, t56–58, t78–79, t86 (95–124 px). These are `bad=False` — unique, non-empty
   RLE — so they sit in `cached_kf_entries` at full weight. `interpolate_expected_geometry`
   (`recover_missegmented_poles.py:288`) filters neighbours that are too **short**
   (`< 0.70 × median`) but never ones that are too **long**.
2. **Positive feedback through write-back.** `_estep_one` writes a frame's geometry back into
   `entries` whenever `pole_usable` is True. Once expected was ~122, a 127 px fused blob passed
   `_pole_usable` at 127/122 = 1.04, was written back, and *raised* the expectation for the next
   frame. The estimate ratcheted and locked at merge scale.
3. **The bisect trigger scaled with the poison.** `bisect_fused_mask` fires at
   `1.35 × exp_maj ≈ 165`, so the one method that could have split the fusion was disarmed
   exactly when needed.

Every pole method then behaved correctly against a reference wrong by 2×: real 65 px daughters
were declared "truncated ~28 px per tip" and extruded toward 120; 120 px blobs were accepted as
pole-usable.

### 3.3 Fix (5 edits)

| Change | Purpose |
| :--- | :--- |
| `_robust_L` | Robust interval length, **anchor-dominant** (see §5) |
| `_clamp_exp_len` | Interp length above `1.25 × anchor span` is replaced by the robust median; tips / `p_max` / `p_min` recomputed consistently |
| `estep_pass` de-poison | Null any non-anchor entry `> 1.4 ×` the robust median before interpolation |
| Fused-strip guard (step f) | Merge-scale length + daughter-scale area → bisect against the robust length; else rebuild a both-poles capsule between nearest clean neighbours (`SYNTH`) |
| Write-back guard + re-centred pole test | A frame re-enters the basis only if also within the robust band; pole inclusion judged against expected geometry **re-centred on the mask itself** — a bisected daughter sits off the lerp centroid, which is a position question, not a pole question |

Result on cid94 [50,100]: touched-frame major axis 113–130 → **51–94 px**, oversize 18 → **0**.

---

## 4. Defect 2 — Pole Recovery Overshoots by Exactly One Brush Radius

### 4.1 Observation

After §3, `pole_short` rose rather than fell. Instrumenting `recover_from_mask` on
`5_1_N1_FL7_F0` cid404: called 85×, **70 reverted as overshoot**.

```
 pre   ->  post    exp   1.35*exp  gain  need(d1+d2)  excess  methods
  74.1 -> 133.6   95.2    128.6   59.5      38.7      20.8   extrude tip2 +39
  75.6 -> 138.7   95.2    128.6   63.0      42.3      20.7   extrude tip2 +42
  70.6 -> 139.4   95.2    128.6   68.8      49.4      19.4   extrude tip2 +49
  ...
median excess (gain − needed) = 19.7 px      median post/exp ratio = 1.50
```

### 4.2 Mechanism

The excess is a constant **19.7 px ≈ `radius = round(W_exp/2)`** with `expected_minor_axis ≈ 40`.
The extrude loop paints `cv2.circle(stroke, pt, radius, ...)` stepping to `delta`, so the **last
circle is centred on the target tip** and its far hemisphere hangs a full radius past it. A mask
needing +38.7 px received +59.5 px, landed at 1.50 × expected, and tripped a 1.35 × gate. The
guard then **reverted to the truncated mask** — all-or-nothing, with no option to clip.

The same unbounded-graft problem existed in strategy 2, where `recovered |= lc` fuses a whole
adjacent label with no length check.

Note the interaction with §3: before de-poisoning, `exp_maj` was inflated, so `1.35 × exp_maj`
was permissive and these overshoots were silently **accepted** — that was the 9.9% oversize.
De-poisoning made the same overshoot correctly **rejected**. The defect moved from *silently
oversized* to *visibly discarded*.

### 4.3 Fix

1. Extrude sweep stops one radius short (`reach = max(0, delta − radius)`).
2. New `_clip_along_axis` — hard bound at `±(L_exp/2 + TRUNC_THR)`, bounding the whole-label fuse
   path too; declines the clip if it would remove >25% of the input.
3. **Clip-then-accept** replaces revert (`RECOVER_CLIPPED`); revert only if the clipped result is
   no closer to expected than the pre-recovery mask.

cid404: discarded 70 → **2**, pole-usable 24 → **29/30**, median post/exp ratio 1.50 → **1.12**.

---

## 5. Defect 3 — Contaminated Interior Outvoting the Curated Anchors

`_robust_L` originally took the median of anchor spans plus all non-bad interior frames. On
cid404 the anchors are **72 / 76 px** but *every* non-bad interior frame is ~**145 px** (median
145.4, min 106) — the cell is chronically fused with a neighbour throughout the canonical track.
43 contaminated frames outvoted 2 curated ones and set `L_ref = 145`, directly contradicting
P14 ("expected length is inferred from the curated keyframes").

**Fix**: `_robust_L` is now **anchor-dominant** — the phase-appropriate anchor seeds the estimate
and interior frames contribute only when they corroborate it (within `[0.75, 1.25] × anchor`).

This mattered independently: without it, cid404's oversize count *rose* to 17 because a
contaminated `L_ref` disagreed with the report's independent Ka→Kb lerp reference. With the fix
the two references agree again and oversize fell to 4.

---

## 6. Corrections to Earlier Conclusions

Two claims made during this work were wrong and are recorded here rather than quietly dropped
(P1, P3).

### 6.1 `pole_short` was a misnamed diagnostic

`pole_short` was computed as `sum(flags["RECOVER_OVERSHOOT"])` — the count of *discarded recovery
attempts*, not frames left pole-short. Because `_estep_one` runs on every interior frame (bad or
not) and flags accumulate there, it could **exceed** `n_bad` (cid404: 70 against 34 bad frames),
and a frame could be flagged yet still be pole-usable (cid489: 49/49 pole-usable with
`pole_short = 12`). The reported "212 honest residue" was wrong; the real recovered-but-not-usable
count was 96.

**Fixed**: `pole_short` now means *emitted frames that are not pole-usable*, with
`recover_discarded` and `recover_clipped` as separate diagnostics.

### 6.2 The proposed Viterbi/Smis gate would not have worked

A per-frame Viterbi decode flagging `argmax γ == Smis` was proposed as the fix for contaminated
pass-through frames. Inspecting the trained emissions shows it cannot work for the dominant case:

```
S0 mother        mean(area, maj) = [0.981, 0.991]   sd [0.065, 0.052]
S1 cytokinesis   mean(area, maj) = [0.526, 0.545]   sd [0.041, 0.032]
S2 daughter      mean(area, maj) = [0.549, 0.564]   sd [0.050, 0.040]
Smis glitch      mean(area, maj) = [0.609, 0.678]   sd [0.205, 0.272]
```

**Every state mean is ≤ 1.0.** `Smis` is an *under*-segmentation state ("transient half area/length
that returns to S0"), which models the t=85 half-cell case (0.55) well but has no vocabulary for a
**fused** frame at 2.0× — that is ~15 sd above S0 and ~5 sd above Smis, in the far tail of all four
states. Viterbi would label it `S0` with terrible likelihood, never `Smis`.

Separately, `TemporalDivisionHMM` exposes only `score_division_posterior()`, which computes the
full forward-backward posterior and then returns **only `γ₅[1]`** — the whole state distribution,
including the `Smis` component, is discarded. There is no `decode()`. As packaged the HMM is a
division-timing scorer, not a per-frame state labeller.

---

## 7. The Remaining Defect — Contaminated Pass-Throughs (open)

### 7.1 Census (75 FL cells in `hard10`)

| | Frames |
| :--- | ---: |
| Dense-rewritten | 1,325 |
| Untouched pass-through | 6,025 |
| — **fused** (> 1.4 × keyframe ref) | **208** |
| — **short** (< 0.75 × keyframe ref) | **328** |
| **Contaminated, never repaired** | **536** (8.9% of untouched) |

The split is informative: **short contamination hits long cells** (kf ref 147–161 — Cellpose
clipping a pole off a long rod); **fusion hits short cells** (kf ref 70–85 — a small cell merged
with its neighbour).

| Cell | kf ref | rewritten | untouched | fused | short |
| :--- | ---: | ---: | ---: | ---: | ---: |
| cid404 `FL7_F0` | 73 | 30 | 68 | **60** | 0 |
| cid57 `FL5_F0` | 85 | 31 | 67 | **35** | 0 |
| cid513 `FL6_F1` | 70 | 63 | 35 | **26** | 0 |
| cid18 `FL6_F0` | 85 | 45 | 53 | **17** | 0 |
| cid12 `FL5_F0` | 161 | 20 | 78 | 0 | **58** |
| cid12 `FL2_F0` | 147 | 22 | 76 | 0 | **45** |

### 7.2 Why they are invisible

`content_bad` only trips on empty or byte-duplicate RLE. A fused or pole-clipped canonical frame
has unique, non-empty content → `bad=False` → never re-linked, emitted verbatim. This is the same
mechanism as the t=85 case (`FL2_F0` cid344, span 92.6 against neighbours at 169/170). The HMM
cannot see them either (§6.2).

**Consequence for the metric**: `pole_usable` is only computed for rewritten frames, so the
reported 91% covered 1,383 frames while **536 sat in the output uninspected**. The true stage-3
quality was worse than that headline — which §8 confirms: bringing those frames into scope and
fixing the reference moved the honest rate to 86%.

### 7.3 Visual confirmation

Vertical pole-QC strips (canonical / v4 / dense, cells rotated horizontal, per-row major axis +
tag + `pole_usable`) show the output alternating between dense-corrected frames at the correct
scale and untouched fused frames beside them:

| Cell | keyframe anchors | dense-rewritten (median, jitter) | untouched (median, jitter) | contaminated, unrepaired |
| :--- | :--- | :--- | :--- | ---: |
| cid18 `FL6_F0` | 85 / 88 / 77 | 89.0, 8.4 px | 92.3, 10.0 px | 17 |
| cid404 `FL7_F0` | 77 / 73 / 73 | 94.2, 6.3 px | **144.9**, 7.6 px | 60 |
| cid12 `FL2_F0` | 147 / 73 / 155 | 86.8, 9.9 px | 96.0, 3.8 px | 0 |

The instability is the **alternation**, not dense wobbling — rewritten frames alone hold 6.3–8.4 px
jitter. Where dense acts on cid404 and cid18 it is now *more correct than the canonical track*.

---

## 8. Eraser Integration — Validated, with a Caveat

### 8.1 `erase_beyond_expected_poles()`

A GTC-eraser equivalent: erases everything beyond the two **expected** pole positions along
`expected_u_long`, so the survivor spans the expected major axis with endpoints at the expected
tips. Cut planes are perpendicular to `u_long`; lateral extent untouched. Keeps only the component
holding the expected centroid; **refuses** the cut if it would leave <10% of the input (meaning
the reference frame is wrong, not the mask).

**Unit validation (complete):**

| Case | n | Target | Span after | Tip offsets |
| :--- | ---: | ---: | :--- | :--- |
| Synthetic fused pair (two 80 px stadiums at 25°) | 1 | 80.0 | **79.7** | +0.14 / +0.14 px |
| cid404 `FL7_F0` real fused frames | 72 | 72.6 | **75.3** (IQR 72.3–76.4) | +1.3 / −2.2 px |
| cid18 `FL6_F0` real fused frames | 43 | 85.2 | **85.7** (IQR 84.8–85.9) | −0.3 / −0.1 px |

**Known risk**: the cut is only as good as `expected_centroid` — it decides *which half is ours*.
cid404's fusions were erased from the `lo` end (~60 px), cid18's from the `hi` end (~70 px); both
correct because the centroid was seeded from the nearest non-fused frame. Seeded wrong, it cleanly
erases the true cell and keeps the sister — the same failure mode as `bisect_fused_mask`.

### 8.2 Integration

- `mark_geometry_bad(DB, meta)` — flags frames outside what **either** curated anchor supports
  (`> 1.4 × max(span_Ka, span_Kb)` → `geom_long`; `< 0.75 × min(...)` → `geom_short`).
  Deliberately conservative and phase-agnostic so a legitimate mother or daughter in a division
  interval is never touched (cid514: anchors 139/72 → band 54–195, both phases safe).
- `geom_long` keeps its canonical mask for the eraser; `geom_short` joins the re-link path.
- Eraser runs **first** in the fused-strip guard, ahead of bisect/synth, with `expected_centroid`
  from `_nearest_clean_geoms`. Accepted only if the result lands in `[0.75, 1.35] × L_ref` and
  keeps IoU ≥ 0.20 with the nearest filled neighbour. Tag `ERASE` / flag `FUSED_ERASED`.

### 8.3 Four integration bugs, each masking the next

The SSD recovered and the integration was tested. It took four fixes, and **none were visible in
the aggregate metric** — every one was found by reading the strips against the keyframe anchors:

| # | Bug | Effect | Found by |
| :-- | :--- | :--- | :--- |
| 1 | Eraser placed after bisect/recover in step (f) | 55 of 59 fused frames on cid404 absorbed by `BISECT`/`RECOVER` before the eraser was reached | `ERASE` tag count (4 of 59) |
| 2 | `slack=TRUNC_THR` (10 px/end) | Every cut landed 20 px long | Strip vs anchors |
| 3 | Eraser cut to `exp_maj`, the window interpolation | `exp_maj` drifts on a contaminated neighbourhood without tripping the 1.25× clamp; cid404 cut to 89.9 px against 73 px anchors | Strip vs anchors |
| 4 | Step (e) re-extruded the deliberate cut | Eraser cut cid18 to ~86 px; `detect_truncated_poles` measured that against the larger `exp_maj`, called it truncated, and grew it back to 100.5 px | Targeted diagnostic |

Fixes: eraser promoted to a new step (c2) ahead of bisect/recover for `geom_long` frames;
`slack=0`; cut target changed to the anchor-derived `min(L_ref, L_hi)` (P14); step (d) guarded by
`not erased` and step (e) skipped entirely for an erased frame — a deliberate cut must not be
re-grown.

**One false alarm, retracted.** An apparent residual "1.15× inflation" was an artifact of comparing
`major_axis_length` (regionprops ellipse axis) against anchors: for a flat-ended cut rod that axis
runs **1.08–1.10×** the physical span, versus ~1.01× for an intact one. Measured correctly, the cut
lands on target — cid404 `L_cut = 73.9` → erased median `physical_span = 76.7`; cid18
`L_cut = 85.8` → `87.1`.

### 8.4 A metric-reference defect, introduced deliberately then found to be half-wrong

`_pole_usable` had been judging masks against `exp_maj` — the same drifted window interpolation the
eraser was cutting to. A metric built on the artefact being repaired endorses whatever the repair
produces, which is exactly why it sat at 91–92% through all four bugs above. It was changed to judge
against the anchor-derived `min(L_ref, L_hi)`.

That is correct for contaminated cells and is what dropped the headline from 92% to 86%. **But it
introduced a new failure**: `_robust_L` returns a *single scalar* for the whole interval, whereas
`exp_maj` varied per frame. On a cell whose length changes across the interval the constant is wrong
at both ends:

| Cell | Keyframe spans | Dense span/anchor (median) | Pole-usable | v4 |
| :--- | :--- | ---: | ---: | ---: |
| cid12 `FL5_F0` | 159 / 156 / **85** (halves) | 0.60 | **1 / 50** | 18 |
| cid18 `FL4_F0` | 138 / 145 / 144 | 0.55 | 17 / 49 | 30 |
| cid404 `FL7_F0` | 76 / 72 / 72 (flat) | **1.07** (p10 1.01, p90 1.11) | **85 / 88** | 6 |

cid404 — a flat-length cell — is exemplary: 69 `ERASE` frames, zero below 0.75× the anchor. cid12
halves across its interval, so a constant reference of ~120 px fails both the 156 px mother end and
the 85 px daughter end. This is the scalar-`L_ref` limitation already noted in §12; it was harmless
while `_pole_usable` used the per-frame interpolation and became load-bearing the moment it did not.

### 8.5 What the 86% actually contains

Of the 186 FL frames now marked not-pole-usable, measured against the keyframe anchors:

| | Frames | Reading |
| :--- | ---: | :--- |
| < 0.75 × anchor | **103 (55%)** | Genuinely too short — real failures, mostly the `geom_short` arm |
| 0.75–1.30 × anchor | 72 (39%) | On target; failed by `detect_truncated_poles`, not the length band |
| > 1.30 × anchor | 11 (6%) | Too long — the eraser's arm, and it is small |

Median ratio 0.68. The **fusion** problem is solved (oversize 2.8%, only 11 long frames). The
dominant remaining defect is the **short** arm — pole-clipped frames that are re-linked but never
restored to full length — plus the metric artefact of §8.4.

---

## 9. Stage-2 Finding — Resegmentation with the Custom Cellpose Model

`cpsam_20260909_neongreen_m160_resumed` (fine-tuned on 129 corrected NeonGreen keyframe pairs)
was applied to two FL cells (ROI-cropped, t=0–100, ~6 min/film) and the trackers re-run against
the new `_seg.tif`.

**Pre-division / division interval — clear win** (`FL6_F1` cid94 [0,50]):

| | Old seg | New seg |
| :--- | ---: | ---: |
| Pole-usable | 8 | **13 / 13** |
| no_seg | 1 | **0** |
| Repair tags | LINK 5, RECOVER 5, BISECT 2 | **LINK 13** (zero synthetic strokes) |
| max_step | 60.3 px | **31.5 px** |
| Division QC | p_comb 0.0, `DIV_QC_LOW` | **p_comb 0.57, unflagged** |

**Post-division daughter — net better, noisier** ([50,100]): pole-usable 2 → **8**, synthetic
strokes 11 → 5, but **oversize 1 → 13** — the new model over-merges the tiny photobleached
daughter. Division timing unchanged in both intervals.

**v4 regressed on the new segmentation** (cid94: recovered 16 → 3, no_seg 16 → 29): its
`robust_span` clamp and `SPAN_CAP = 1.30` reject the new model's larger labels and it has no
bisection to split merges. Further argument for retiring v4.

**Conclusion**: resegmentation materially improves the mother/division phase and eliminates
dropouts, but introduces an over-merge tendency on dim frames — it does **not** on its own remove
the contaminated pass-throughs of §7.

---

## 10. Policy — P14 Introduced

`PROJECT_POLICY.md` v1.9 adds **P14 — Pipeline stage order and per-stage objective priority**:

1. **GTC + ABBT** — keyframe curation and quick QC; authoritative downstream.
2. **Segmentation** — train Cellpose on curated keyframes, segment all timeframes.
3. **Dense tracking** — all-frame linking/gap-fill against stage-2 `_seg.tif`, anchored on stage-1
   keyframes.
4. **Quantification** — polarity-site dynamics.

Stage-3 objective priority, as set by the project owner:

- **Primary — pole inclusion, hence major-axis length stability**, with expected length inferred
  from the curated keyframes. This is what stage 3 is tuned, accepted, and reported on.
- **Secondary — dense division timeframe**, already fixed at keyframe resolution by stage 1
  (98.74% exact, 99.37% within ±1 kf over 316 cells). Frame-level `t*` is a convenience, not a
  deliverable; a low-confidence dense `t*` defers to the stage-1 call.

Supporting rules: do not reorder stages (a tracker can only pick among the labels segmentation
gives it); keyframes read-only from stage 2 onward; never derive expected geometry from
unvalidated interior frames; report pole-usable first; stage boundaries are P2 L4 checkpoints.

---

## 11. Artifacts

| Path | Contents |
| :--- | :--- |
| `scratch/dense_abbt_interval.py` | Stage-3 implementation (all fixes above) |
| `scratch/dense_abbt_out/dense_abbt_interval_report.csv` | `hard10` per-segment report |
| `scratch/render_pole_qc_strips.py` | Pole-QC vertical strip renderer |
| `scratch/pole_qc_strips_20260911T041335Z/` | **Final** QC strips: cid404, cid18, cid513, cid57, cid12, cid514 |
| `scratch/pole_qc_strips_20260910T113543Z/` | Earlier strips (pre-eraser), kept per P3 |
| `scratch/_dabbt_final.log` | Final `hard10` sweep log (31:05) |
| `scratch/reseg_retrack_two_fl.py` + `_20260910T093042Z/` | Reseg A/B (§9), with provenance |
| `scratch/compare_dense_more_cells.py` + `dense_tracking_three_cell_comparison_20260910T090414Z/` | 3-cell v4 vs dense A/B |
| `docs/PROJECT_POLICY.md` | v1.9, P14 |

---

## 12. Next Steps

1. **Interval-varying expected length — now the top priority.** `_robust_L` returns one scalar
   per interval. This was harmless while `_pole_usable` used the per-frame interpolation and became
   load-bearing the moment it did not (§8.4). It must interpolate between the two anchors,
   phase-aware, so a cell that grows or halves across the interval is judged correctly at both ends.
   Until then the 86% is depressed on every length-varying cell (cid12 `FL5_F0` reads 1/50).
2. **The short arm.** 103 of 186 failures are genuinely < 0.75 × the anchor — pole-clipped frames
   re-linked but never restored. The eraser addressed fusion only; there is no equivalent
   "extend back to the expected poles" path that survives the overshoot gate.
3. **Position continuity.** `max_step` still spikes (cid55 103 px, cid392 91 px) — a separate
   failure mode from pole inclusion, untouched by this work.
4. **Re-run the A/B against v4 after (1).** Several current "regressions" (cid12 `FL5_F0`,
   cid18 `FL4_F0`, cid7 `FL4_F0`) are suspected metric artefacts of the scalar reference, not real
   losses, but this is **unverified** — they must be confirmed frame-by-frame before being dismissed.
5. **Stage-2 pre-pass.** Consider resegmenting FL5–FL7 with the custom model before the `full803`
   run, paired with the over-merge handling.
6. **Then** `full803`, P5-gated canonical write, and HPC re-quantification of touched cells.
