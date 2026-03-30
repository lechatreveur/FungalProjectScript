import os, re, json
import pandas as pd

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ---------------- utilities ----------------

def _list_cell_csvs(dir_path):
    """Return a sorted list of fullpaths for files matching cell_*_masks.csv."""
    out = []
    for fn in os.listdir(dir_path):
        if re.match(r"cell_.*_masks\.csv$", fn):
            out.append(os.path.join(dir_path, fn))
    # stable sort: by integer found in filename if any, then name
    def _key(p):
        m = re.search(r"(\d+)", os.path.basename(p))
        return (int(m.group(1)) if m else -1, os.path.basename(p))
    out.sort(key=_key)
    return out


def _extract_cell_id_from_filename(path):
    """
    Try to extract a numeric cell ID from a filename like 'cell_369_masks.csv';
    if not found, return the basename without extension.
    """
    base = os.path.basename(path)
    m = re.search(r"cell_(\d+)_masks\.csv$", base)
    return int(m.group(1)) if m else os.path.splitext(base)[0]


def _pick_time_row(df, which="first", time_col=None, time_col_candidates=("t","time","frame","tp","Time","Frame")):
    """
    From a single-cell dataframe, pick the row for earliest (first) or latest (last) time.
    If no time column exists, fall back to the first (for 'first') or last (for 'last') row.
    """
    if time_col is None:
        for c in time_col_candidates:
            if c in df.columns:
                time_col = c
                break

    if time_col is not None and time_col in df.columns:
        # coerce to numeric for robust min/max
        tt = pd.to_numeric(df[time_col], errors="coerce")
        if which == "first":
            idx = tt.idxmin()
        else:
            idx = tt.idxmax()
        row = df.loc[idx]
    else:
        row = df.iloc[0 if which == "first" else -1]
    return row


# ---------------- RLE parsing (interval set) ----------------

def _intervals_from_startlen_pairs(rle_str):
    nums = [int(x) for x in str(rle_str).strip().split()]
    if len(nums) % 2 != 0:
        raise ValueError("Start-length RLE has odd number of integers.")
    intervals = []
    for s, l in zip(nums[0::2], nums[1::2]):
        if l <= 0: 
            continue
        intervals.append([s, s + l])
    intervals.sort()
    # merge touching/overlapping
    merged = []
    for s, e in intervals:
        if not merged or s > merged[-1][1]:
            merged.append([s,e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return merged

def _intervals_from_coco_uncompressed(obj):
    counts = obj.get("counts")
    size = obj.get("size")
    if counts is None or size is None:
        raise ValueError("COCO RLE missing 'counts' or 'size'.")
    if isinstance(counts, str):
        raise ValueError("Compressed COCO RLE string not supported without pycocotools.")
    pos = 0
    intervals = []
    for k, c in enumerate(counts):
        c = int(c)
        if c <= 0:
            continue
        start, end = pos, pos + c
        if k % 2 == 1:  # foreground runs
            intervals.append([start, end])
        pos = end
    return intervals

def _to_intervals(rle_value):
    s = str(rle_value).strip()
    if not s or s.lower() in ("nan", "none"):
        return []
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        obj = json.loads(s)
        return _intervals_from_coco_uncompressed(obj)
    return _intervals_from_startlen_pairs(s)

def _interval_len(intervals):
    return sum(e - s for s, e in intervals)

def _rle_intersection_len_from_intervals(a, b):
    i = j = 0
    inter = 0
    while i < len(a) and j < len(b):
        a0, a1 = a[i]; b0, b1 = b[j]
        lo = max(a0, b0); hi = min(a1, b1)
        if hi > lo:
            inter += (hi - lo)
        if a1 <= b1:
            i += 1
        else:
            j += 1
    return inter

def _iou_from_intervals(a, b):
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    inter = _rle_intersection_len_from_intervals(a, b)
    union = _interval_len(a) + _interval_len(b) - inter
    return (inter / union) if union > 0 else 0.0


# ---------------- main: per-cell CSVs with time selection ----------------


def map_gfp_to_bf_ids(
    WORKING_DIR,
    gfp_rel="A14_1TP1_F1/TrackedCells_A14_1TP1_F1/",
    bf_rel="A14_1TP1_BF_F1/TrackedCells_A14_1TP1_BF_F1/",
    gfp_rle_col="rle_gfp",
    bf_rle_col="rle_bf",
    iou_min=0.01,
    gfp_timepoint="last",
    bf_timepoint="first",
    time_col=None,
    time_col_candidates=("t","time","frame","tp","Time","Frame"),
    # NEW:
    assignment="one_to_one",     # "one_to_one" | "gfp_to_best_bf" | "bf_to_topk_gfp"
    k_per_bf=2                   # used when assignment="bf_to_topk_gfp"
):
    """
    Each cell is saved as its own CSV containing multiple rows (timepoints).
    For each GFP cell CSV, we take the row at `gfp_timepoint` (earliest or latest time),
    parse `rle_gfp`; similarly for BF, using `bf_timepoint` and `rle_bf`.
    Then compute pairwise IoUs and produce a one-to-one mapping (GFP → BF).

    Returns:
      mapping: dict {gfp_id: bf_id}
      pairs_df: DataFrame with columns [gfp_id, bf_id, iou]
      used_files: {'gfp_files': [...], 'bf_files': [...]}
    """
    gfp_dir = os.path.join(WORKING_DIR, gfp_rel)
    bf_dir  = os.path.join(WORKING_DIR, bf_rel)

    gfp_files = _list_cell_csvs(gfp_dir)
    bf_files  = _list_cell_csvs(bf_dir)

    if not gfp_files:
        raise FileNotFoundError(f"No GFP cell CSVs found in: {gfp_dir}")
    if not bf_files:
        raise FileNotFoundError(f"No BF cell CSVs found in: {bf_dir}")

    # Build per-cell intervals (one row per cell, chosen timepoint inside the file)
    g_ids, g_ints = [], []
    for fp in gfp_files:
        try:
            df = pd.read_csv(fp)
            row = _pick_time_row(df, which=gfp_timepoint, time_col=time_col, time_col_candidates=time_col_candidates)
            ints = _to_intervals(row[gfp_rle_col])
            if ints:
                g_ids.append(_extract_cell_id_from_filename(fp))
                g_ints.append(ints)
        except Exception:
            # skip malformed / empty
            pass

    b_ids, b_ints = [], []
    for fp in bf_files:
        try:
            df = pd.read_csv(fp)
            row = _pick_time_row(df, which=bf_timepoint, time_col=time_col, time_col_candidates=time_col_candidates)
            ints = _to_intervals(row[bf_rle_col])
            if ints:
                b_ids.append(_extract_cell_id_from_filename(fp))
                b_ints.append(ints)
        except Exception:
            pass

    if not g_ids or not b_ids:
        raise RuntimeError("No decodable RLEs found in one or both sets of per-cell CSVs.")

    # IoU matrix
    nG, nB = len(g_ids), len(b_ids)
    iou_mat = [[_iou_from_intervals(g_ints[i], b_ints[j]) for j in range(nB)] for i in range(nG)]

    # pairs df (for inspection)
    pairs = []
    for i, gid in enumerate(g_ids):
        for j, bid in enumerate(b_ids):
            pairs.append({"gfp_id": gid, "bf_id": bid, "iou": iou_mat[i][j]})
    pairs_df = pd.DataFrame(pairs).sort_values(["gfp_id","iou"], ascending=[True, False]).reset_index(drop=True)
    # Build a convenience structure
    import numpy as np
    iou_np = np.array(iou_mat, dtype=float)  # shape [nG, nB]

    # -------- ASSIGNMENT MODES --------
    mapping = {}                # GFP -> BF (primary return)
    bf_to_gfps = {}             # BF -> [GFP] (handy for divisions)

    if assignment == "one_to_one":
        if _HAS_SCIPY:
            from scipy.optimize import linear_sum_assignment
            cost = 1.0 - iou_np
            r, c = linear_sum_assignment(cost)
            for ri, ci in zip(r, c):
                iou = iou_np[ri, ci]
                if iou >= iou_min:
                    g = g_ids[ri]; b = b_ids[ci]
                    mapping[g] = b
                    bf_to_gfps.setdefault(b, []).append(g)
        else:
            used_g, used_b = set(), set()
            ranked = sorted(((iou_np[i,j], i, j) for i in range(len(g_ids)) for j in range(len(b_ids))),
                            key=lambda x: x[0], reverse=True)
            for iou, i, j in ranked:
                if iou < iou_min: break
                if i in used_g or j in used_b: continue
                g = g_ids[i]; b = b_ids[j]
                mapping[g] = b
                bf_to_gfps.setdefault(b, []).append(g)
                used_g.add(i); used_b.add(j)

    elif assignment == "gfp_to_best_bf":
        # Each GFP picks its single best BF >= threshold (allows many GFP per BF)
        for i, g in enumerate(g_ids):
            j = int(iou_np[i].argmax())
            if iou_np[i, j] >= iou_min:
                b = b_ids[j]
                mapping[g] = b
                bf_to_gfps.setdefault(b, []).append(g)

    elif assignment == "bf_to_topk_gfp":
        # For each BF, take up to k_per_bf best GFP with IoU >= threshold
        for j, b in enumerate(b_ids):
            # sort all GFP by IoU to this BF
            order = np.argsort(iou_np[:, j])[::-1]
            picked = 0
            for i in order:
                iou = iou_np[i, j]
                if iou < iou_min: break
                g = g_ids[i]
                # Map GFP->BF (if a GFP appears in multiple BF lists, keep the BF with higher IoU)
                prev_b = mapping.get(g, None)
                if prev_b is None:
                    mapping[g] = b
                    bf_to_gfps.setdefault(b, []).append(g)
                    picked += 1
                else:
                    # tie-breaker: keep whichever BF gives higher IoU
                    j_prev = b_ids.index(prev_b)
                    if iou_np[i, j] > iou_np[i, j_prev]:
                        # move this g from prev_b to b
                        bf_to_gfps[prev_b].remove(g)
                        mapping[g] = b
                        bf_to_gfps.setdefault(b, []).append(g)
                        # picked still counts for the current BF
                        picked += 1
                if picked >= int(k_per_bf):
                    break
    else:
        raise ValueError("assignment must be 'one_to_one', 'gfp_to_best_bf', or 'bf_to_topk_gfp'")

    # pairs_df construction remains as before...
    pairs = []
    for i, gid in enumerate(g_ids):
        for j, bid in enumerate(b_ids):
            pairs.append({"gfp_id": gid, "bf_id": bid, "iou": float(iou_np[i, j])})
    pairs_df = pd.DataFrame(pairs).sort_values(["gfp_id","iou"], ascending=[True, False]).reset_index(drop=True)

    return mapping, bf_to_gfps, pairs_df, {"gfp_files": gfp_files, "bf_files": bf_files}


import pandas as pd
import numpy as np


def find_gfp_counterparts(
    some_bf_ids,
    map_bf_to_gfp,
    gfp_dfs_dict,
    gfp_id_col="cell_id",
    save_csv_path=None,
):
    import re
    def _to_int_safe(x):
        if pd.isna(x): return pd.NA
        s = str(x).strip()
        s = s.split("_", 1)[0]
        try:
            return int(float(s))
        except Exception:
            m = re.match(r"^\s*(\d+)", s)
            return int(m.group(1)) if m else pd.NA

    some_bf_ids = list(pd.unique(some_bf_ids))
    matched = pd.DataFrame({
        "bf_id": some_bf_ids,
        "gfp_id": [map_bf_to_gfp.get(bf, pd.NA) for bf in some_bf_ids]
    })
    matched["status"] = np.where(matched["gfp_id"].isna(), "unmapped", "mapped")
    matched["gfp_id_norm"] = matched["gfp_id"].map(_to_int_safe)

    gfp_frames = []
    for variant_name, d in gfp_dfs_dict.items():
        if d is None or len(d) == 0: continue
        tmp = d.copy()
        tmp["gfp_variant"] = variant_name
        tmp["cell_id_norm"] = tmp[gfp_id_col].map(_to_int_safe)
        gfp_frames.append(tmp)
    gfp_all = pd.concat(gfp_frames, ignore_index=True) if gfp_frames else pd.DataFrame(columns=[gfp_id_col,"cell_id_norm"])

    # select by normalized ints
    gfp_ids_norm = matched["gfp_id_norm"].dropna().astype("int64").unique().tolist()
    df_gfp_selected = gfp_all[gfp_all["cell_id_norm"].isin(gfp_ids_norm)].copy()

    if save_csv_path:
        matched.to_csv(save_csv_path, index=False)

    # return normalized int list for convenience
    return matched, df_gfp_selected, sorted(list({int(x) for x in gfp_ids_norm}))


#_____
import re
import numpy as np
import pandas as pd

def _norm_id_to_int(x):
    s = str(x).strip().split("_",1)[0]
    m = re.match(r"^\d+", s)
    return int(m.group(0)) if m else None

def _extract_bf_end_timepoint(summary, bf_id_col="cell_id"):
    """
    Treat summary['end'] as a BF TIMEPOINT (frame index).
    Returns: DataFrame [bf_id_norm, anchor_bf_end_tp] (both numeric; NaNs kept).
    """
    s = summary.copy()
    if "end" not in s.columns:
        raise KeyError("summary must contain an 'end' column (BF timepoint / frame index).")
    s["bf_id_norm"] = s[bf_id_col].map(_norm_id_to_int)
    s["anchor_bf_end_tp"] = pd.to_numeric(s["end"], errors="coerce")
    out = (s[["bf_id_norm","anchor_bf_end_tp"]]
           .dropna(subset=["bf_id_norm"])  # keep NaN anchors, drop NaN ids
           .drop_duplicates("bf_id_norm"))
    return out

def align_gfp_to_bf_end_TIMEPOINT(
    df_gfp_selected,
    matched_table,
    summary,
    gfp_id_col="cell_id",
    gfp_time_col="time_point",   # GFP frame index (5 frames/min)
    bf_id_col="cell_id",
    gfp_frames_per_min=5
):
    # 1) BF end timepoint per BF id (as numeric; may contain NaNs)
    bf_anchor_tp = _extract_bf_end_timepoint(summary, bf_id_col=bf_id_col)  # [bf_id_norm, anchor_bf_end_tp]

    # 2) BF → GFP link (normalized ints)
    link = matched_table.dropna(subset=["gfp_id","bf_id"]).copy()
    link["gfp_id_norm"] = link["gfp_id"].map(_norm_id_to_int)
    link["bf_id_norm"]  = link["bf_id"].map(_norm_id_to_int)
    link = link.dropna(subset=["gfp_id_norm","bf_id_norm"]).astype({"gfp_id_norm":"int64","bf_id_norm":"int64"})
    link = link.drop_duplicates(subset=["gfp_id_norm"])

    # 3) merge anchors onto GFP ids (left join to keep all GFPs)
    gfp_anchor_tp = (link.merge(bf_anchor_tp, on="bf_id_norm", how="left")
                          [["gfp_id_norm","anchor_bf_end_tp"]]
                          .drop_duplicates("gfp_id_norm"))

    # 4) attach to GFP rows & compute aligned axes safely (floats; allow NaNs)
    df = df_gfp_selected.copy()
    df["gfp_id_norm"] = df[gfp_id_col].map(_norm_id_to_int)

    df = df.merge(gfp_anchor_tp, on="gfp_id_norm", how="left")

    gfp_tp   = pd.to_numeric(df[gfp_time_col], errors="coerce")
    anchor_tp = pd.to_numeric(df["anchor_bf_end_tp"], errors="coerce")

    # compute in float to tolerate NaNs
    df["aligned_frame"] = gfp_tp - (anchor_tp * float(gfp_frames_per_min))
    df["aligned_min"]   = df["aligned_frame"] / float(gfp_frames_per_min)
    # nullable Int64 can hold <NA>, so this cast is safe
    df["aligned_frame_rounded"] = df["aligned_frame"].round().astype(pd.Int64Dtype())

    # diagnostics
    n_cells = df["gfp_id_norm"].nunique()
    n_with_anchor = df.groupby("gfp_id_norm")["anchor_bf_end_tp"].apply(lambda s: s.notna().any()).sum()
    missing_ids = sorted(set(df["gfp_id_norm"].unique()) -
                         set(df.loc[df["anchor_bf_end_tp"].notna(),"gfp_id_norm"].unique()))
    print(f"[align (TP)] Anchors attached (by timepoint) for {n_with_anchor}/{n_cells} GFP cells.")
    if missing_ids:
        print("[align (TP)] GFP ids missing anchor (first 20):", missing_ids[:20])

    return df
