# Development Report: Refined Hard-EM Backward Bayesian Tracker & All-Frame Tracking Strategy

**Date**: September 8, 2026  
**Module**: `SingleCellQuantificationHPC`  
**Dataset**: `2026_08_28_M160` (`5_1_N1_F0`, `5_1_N1_F1`, `5_1_N1_F2`)  
**Status**: Validated & Applied to Canonical Linkage / GTC  

---

## 1. Executive Summary & Core Objectives

During the population-scale review of multi-film time-lapse sequences in `2026_08_28_M160`, four primary failure modes were isolated in Cellpose segmentation and inter-film linkage:
1. **Sister Identity Swaps**: Post-division transitions across film boundaries (e.g. `FL1 → BF1`, `FL5 → BF5`) jumping between daughter cells.
2. **Transient Undersegmentation Oscillation**: Cellpose clipping or missing poles across 1–3 frames, creating spurious area drops and rebounds.
3. **Cross-Phase Geometry Contamination**: Premature pole recovery interpolating daughter cells to mother length scales (e.g., Cell 238).
4. **Post-Division Undersegmentation / Fused Daughters**: Cellpose segmenting two touching daughter cells as a single pseudo-mother mask (e.g., Cell 189).

To resolve these challenges without ad-hoc heuristics, we developed and validated the **Hard-EM / Hypothesis-Conditioned Backward Bayesian Tracker (ABBT)**, achieving **98.74% exact division timing accuracy** and **96.46% division classification accuracy** across 316 curated complete cell tracks.

---

## 2. Hard-EM ABBT Architecture & Mathematical Formulation

### 2.1 Backward Formulation (t_end → t_0)
By tracking backward in time:
- The terminal film defines the exact set of distinct daughter cells.
- Forward bifurcation ambiguity is eliminated (every daughter traces to at most one mother).
- Cell length monotonically decreases backward:
  L(t - Δt) ≤ L(t) + ε

### 2.2 Hard-EM / Iterated Conditional Modes (ICM) Optimization
We treat cell identity, division timing k_div, and segmentation status as joint hidden variables:
- **E-Step (Hypothesis-Conditioned Mask Rectification)**:
  Given candidate hypothesis H(k_div):
  - Pre-division keyframes (k < k_div): Maintained as mother cell; undersegmented poles are recovered using expected mother geometry L_mother.
  - Post-division keyframes (k ≥ k_div): Fused daughter masks are cleaved along the minor axis using `bisect_fused_mask(fused_mask, target_centroid, u_long)`. Truncated poles are recovered using expected daughter geometry L_daughter.
- **M-Step (MAP Hypothesis Selection)**:
  Maximizes the joint posterior probability across all candidate drop events:
  P(H(k*) | O) ∝ P(O | H(k*)) · P(H(k*))
  where:
  - P(O | H(k*)) is computed via the 4D/6D Feature HMM + Multivariate Log-Likelihood Ratio (LLR).
  - P(H(k*)) incorporates geometric length priors, area drop constraints (ΔA ≥ 0.25 · A_mother), and boundary drift penalties.

### 2.3 Longitudinal Displacement Sister-Swap Detection
When crossing film boundaries (f → f+1), the displacement vector Δc = c_{f+1} - c_f is projected onto the cell's longitudinal unit axis u_long and transverse axis u_short:
- Δp_long = |Δc · u_long|
- Δp_short = |Δc · u_short|
- Collinearity = Δp_long / ||Δc||

A sister swap is flagged when:
||Δc|| ≥ 35.0 px, Collinearity ≥ 0.70, and Δp_short ≤ 25.0 px

---

## 3. Comprehensive Benchmark Results (316 Curated Cells)

Evaluation was performed across all complete curated cell tracks in `2026_08_28_M160`:

| Benchmark Metric | Evaluation Count / Total | Performance Score |
| :--- | :--- | :--- |
| **Total Curated Cells Evaluated** | **316** | 100.0% |
| **Border-Touching Cells Flagged & Excluded** | **5** | 1.58% |
| **Clean Interior Cells Tested** | **311** | 98.42% |
| **Division Classification Accuracy** | **300 / 311** | **96.46%** |
| **Total True Divisions in Clean Set** | **159** | — |
| **Exact Division Timing Accuracy (k_pred == k_gt)** | **157 / 159** | **98.74%** |
| **Division Timing within ±1 Keyframe Tolerance** | **158 / 159** | **99.37%** |
| **Total Missegmentations Detected & Rectified** | **12** | — |
| **Total Sister Swaps Detected across Film Boundaries** | **14** | — |

---

## 4. Resolution of Specific Target Cases

1. **Cell 246** (`5_1_N1_BF1_F2_cell_246`):
   - True division detected at k = 2 (`FL1 t=100`, P = 1.0000).
   - Sister swap flagged at transition `FL1_F2 → BF1_F2` (Δc = 83.4 px, collinearity = 0.999 along u_long, Δp_short = 2.7 px).
2. **Cell 221** (`5_1_N1_BF5_F2_cell_221`):
   - True division detected at k = 25 (`FL5 t=50`, P = 1.0000).
   - Sister swap flagged at `FL5_F2 → BF5_F2` (Δc = 82.9 px, collinearity = 1.000). Pre-division undersegmentation at `BF4 t=40` rectified (90.5 px → 160.0 px).
3. **Cell 92** (`5_1_N1_FL4_F2_cell_92`):
   - True division detected at k = 20 (`FL4 t=50`, P = 0.9994).
   - Three undersegmented frames (`FL3 t=50`, `FL3 t=100`, `FL4 t=0`) rectified to mother scale, eliminating false division triggers.
4. **Cell 67** (`5_1_N1_BF5_F2_cell_67`):
   - True division detected at k = 25 (`FL5 t=50`, P = 1.0000).
   - Sister swap flagged at `FL5_F2 → BF5_F2` (Δc = 72.9 px, collinearity = 0.997).
5. **Cell 183** (`5_1_N1_FL4_F2_cell_183`):
   - True division detected at k = 18 (`FL4 t=0`, P = 1.0000).
   - Missegmentation at `FL4 t=50` rectified (63.9 px → 87.7 px).
6. **Cell 232** (`5_1_N1_FL7_F2_cell_232`):
   - True division detected at k = 27 (`FL5 t=100`, P = 0.9999).
   - Pre-division missegmentation at `FL5 t=50` rectified (83.0 px → 153.9 px).
7. **Cell 238** (`5_1_N1_BF4_F2_cell_238`):
   - True division detected at k = 18 (`FL4 t=0`, P = 1.0000).
   - Phase-aware geometry interpolation restricted sampling to post-division daughter frames, preventing mother-length elongation.
8. **Cell 189** (`5_1_N1_FL6_F2_cell_189`):
   - True division detected at k = 30 (`FL6 t=0`, P = 1.0000).
   - Hypothesis-conditioned cleavage plane bisection split fused post-division daughter masks, isolating undersegmented frames at `FL4 t=100` and `FL5 t=0`.

---

## 5. Canonical Dataset & GTC Update Summary

- **Sequence Linkage**: Applied `apply_backward_tracking_m160.py` across all 3 sequences in `2026_08_28_M160`.
- **Complete Trajectories**:
  - `5_1_N1_F0`: 403 global cells (348 complete 13/13 tracks, 169/169 curated conserved).
  - `5_1_N1_F1`: 469 global cells (390 complete 13/13 tracks, 100/100 curated conserved).
  - `5_1_N1_F2`: 313 global cells (288 complete 13/13 tracks, 47/47 curated conserved).
  - **Total 13/13 Complete Trajectories**: **1,026 cells** across M160.
- **Data Integrity**: Timestamped backups generated for all linkage JSONs and QC files prior to modification (`*.bak_20260908_172640`).

---

## 6. Strategy for Dense All-Frame Tracking (Beyond 39 Keyframes)

Currently, the ABBT operates on the **39-keyframe backbone** (3 keyframes per film across 13 films). Extending this to all dense frames (~700–1,300 frames per film) requires a 3-tier hierarchical architecture:

```
[Keyframe Backbone (ABBT)]  <-- Global Division Timing (k_div) & Inter-Film Linkage
           │
           ▼
[Dense In-Film Propagation] <-- Dual-Direction Forward/Backward Continuous Tracking
           │
           ▼
[Division Boundary Cleavage] <-- Dense Septum Contraction & Cleavage Propagation
```

### 6.1 Dense In-Film Continuous Tracking
1. **Keyframe Anchor Constraint**:
   - Each film has 3 keyframe anchors: t_0, t_mid, t_end.
   - Within each sub-interval [t_0, t_mid] and [t_mid, t_end], cell masks are tracked frame-by-frame using high-frame-rate Hungarian matching on:
     Cost(i, j) = 0.50 · (1 - IoU(M_i, M_j)) + 0.30 · (||c_i - c_j|| / R_max) + 0.20 · (|A_i - A_j| / A_i)
2. **Dense Mask Interpolation & Pole Verification**:
   - For frames where Cellpose segmentation drops or undersegments, we use **Bidirectional Morphological Skeleton Morphing**:
     - Interpolate centroid c(t) = (1 - α) · c(t_k) + α · c(t_{k+1})
     - Interpolate pole tips along the longitudinal axis u_long(t).
     - Morph contour via level-set distance transform propagation:
       Φ(x, y, t) = (1 - α) · Φ_{t_k}(x, y) + α · Φ_{t_{k+1}}(x, y)

### 6.2 Dense Division Timepoint Localization
1. **Coarse-to-Fine Search**:
   - ABBT establishes the division interval [k_div - 1, k_div] on keyframes (Δt ≈ 50 frames in FL, 20 frames in BF).
   - Within the localized 50-frame window, dense septum intensity contrast S_contrast(t) and minor-axis constriction w_mid(t) are computed on every individual frame t.
2. **Sub-Frame / Exact Frame Division Detection**:
   - Division frame t* is identified at the maximum derivative of constriction contraction:
     t* = argmax_t [ -d/dt (w_mid(t)) + λ · d/dt (S_contrast(t)) ]
   - For t < t*, the single mother mask is tracked.
   - For t ≥ t*, the daughter mask is cleaved along the localized septum coordinates.

### 6.3 Sister Track Continuity across All Frames
- Because the global sister assignment is established at the keyframe boundaries, all intermediate frames t ∈ [t_0, t_end] inherit the unambiguous sister ID from the anchor trajectory, preventing intra-film sister crossing.
