import os
import json
import base64
import logging
import torch
import numpy as np
import pandas as pd
import umap
from pathlib import Path
from typing import Tuple, Dict

from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data
from SingleCellDataAnalysis.FC_AE_3d_train import MultimodalAutoencoder3D

from .config import load_config, GlobalConfig, ExperimentConfig
from .adapters import get_adapter
from .schemas import validate_stacked_schema, validate_trajectory
from .qc import apply_qc_rules, make_qc_report
from .division_time import load_observed_division_times, estimate_missing_division_times
from .exporter import export_single_html, export_static_site

logger = logging.getLogger(__name__)

def load_cell_areas(gids: list[str]) -> list[float]:
    # Dynamic loader for cell areas mirroring load_cell_areas in old script
    areas = []
    root_sept17 = Path("/Volumes/X10 Pro/Movies/2025_09_17/")
    root_m133 = Path("/Volumes/X10 Pro/Movies/2026_04_29_M133/")
    root_m135 = Path("/Volumes/X10 Pro/Movies/2026_04_30_M135/")
    
    for gid in gids:
        area = np.nan
        try:
            if gid.startswith("Sept17_"):
                suffix = gid[len("Sept17_"):]
                parts = suffix.split('_')
                if len(parts) >= 2 and parts[-1] in ['GFP1', 'GFP2']:
                    source = parts[-1]
                    field = parts[0]
                    orig_id = int(parts[-2])
                    film = f"A14_1TP1_{field}" if source == 'GFP1' else f"A14_1TP2_{field}"
                else:
                    continue
                csv_path = root_sept17 / film / f"TrackedCells_{film}" / f"cell_{orig_id}_data.csv"
            elif gid.startswith("M133_"):
                map_p = root_m133 / "unaligned_pairs_quant" / "id_map_unaligned.csv"
                if map_p.exists():
                    df_map = pd.read_csv(map_p)
                    local_id = int(gid.split("_")[1])
                    sub = df_map[df_map.new_cell_id == local_id]
                    if not sub.empty:
                        row = sub.iloc[0]
                        orig_str = str(row.get('orig_str_id', ''))
                        orig_id = int(orig_str.split(':')[1]) if ':' in orig_str else int(row.get('local_fl_id', local_id))
                        field = row.field
                        source = row.source
                        film = f"A14_TP1_{field}" if source == 'GFP1' else f"A14_TP2_{field}" if source == 'GFP2' else f"A14_TP3_{field}"
                        csv_path = root_m133 / film / f"TrackedCells_{film}" / f"cell_{orig_id}_data.csv"
                    else:
                        continue
                else:
                    continue
            elif gid.startswith("M135_"):
                map_p = root_m135 / "unaligned_pairs_quant" / "id_map_unaligned.csv"
                if map_p.exists():
                    df_map = pd.read_csv(map_p)
                    local_id = int(gid.split("_")[1])
                    sub = df_map[df_map.new_cell_id == local_id]
                    if not sub.empty:
                        row = sub.iloc[0]
                        orig_id = local_id
                        field = row.field
                        source = row.source
                        film = f"A14_10_20min_{field}"
                        csv_path = root_m135 / film / f"TrackedCells_{film}" / f"cell_{orig_id}_data.csv"
                    else:
                        continue
                else:
                    continue
            else:
                areas.append(np.nan)
                continue
                
            if csv_path.exists():
                df_c = pd.read_csv(csv_path)
                if 'cell_area' in df_c.columns:
                    vals = pd.to_numeric(df_c['cell_area'], errors='coerce').dropna().values
                    if len(vals) > 0:
                        area = float(vals.max())
        except Exception:
            pass
        areas.append(area)
    return areas

def run_pipeline(config_path: str, strict: bool = False) -> None:
    logger.info("🚀 Starting Manifold Explorer Pipeline...")
    
    # 1. Load config
    global_cfg = load_config(config_path)
    logger.info(f"Loaded config: {config_path}")
    logger.info(f"Output mode: {global_cfg.output_mode}")
    logger.info(f"Model path: {global_cfg.model_path}")
    
    # Map from ExperimentConfig to root directory
    experiments_dict = {}
    for name, exp_cfg in global_cfg.experiments.items():
        experiments_dict[name] = str(exp_cfg.root)
        
    # 2. Load Trajectories and Features via FC_AE_data_loader
    logger.info("📥 Loading trajectories and engineered features using FC data loader...")
    X_traj, X_feat, gids_loaded, labels_loaded, s_traj, s_feat = load_feature_constrained_data(experiments_dict)
    
    # Un-scale them to build raw canonical datasets
    X_traj_raw = X_traj * (s_traj.std + 1e-8) + s_traj.mean
    X_feat_raw_full = X_feat * (s_feat.std + 1e-8) + s_feat.mean
    
    logger.info(f"Loaded {len(gids_loaded)} cells from dataloader.")
    
    # Load acor results dynamically for present experiments
    acor_dfs = {}
    for name, exp_cfg in global_cfg.experiments.items():
        acor_p = exp_cfg.root / "unaligned_pairs_quant" / "acor_detrended_results.csv"
        if not acor_p.exists():
            acor_p = exp_cfg.root / "acor_detrended_results.csv"
        if acor_p.exists():
            acor_dfs[name] = pd.read_csv(acor_p).set_index('cell_id')
            
    # 3. Build canonical cell table
    records = []
    
    for name, exp_cfg in global_cfg.experiments.items():
        logger.info(f"📥 Resolving metadata and timings for experiment: {name}")
        adapter = get_adapter(exp_cfg.adapter_type)
        
        # Load normalized metadata
        metadata = adapter.load_metadata(exp_cfg)
        qc_map = adapter.load_qc(exp_cfg)
        df_acor = acor_dfs.get(name)
        
        # Calculate division times
        div_times, alignments = load_observed_division_times(exp_cfg)
        no_septum_shifts = estimate_missing_division_times(exp_cfg, div_times)
        
        # Build lookup table of cell metadata by local_cell_id
        meta_by_cid = {row['local_cell_id']: row for _, row in metadata.iterrows()}
        
        # For each loaded cell belonging to this experiment, map metadata
        for i, gid in enumerate(gids_loaded):
            if not gid.startswith(name + "_"):
                continue
                
            # Extract local sequential ID from stable ID or suffix
            suffix = gid[len(name) + 1:]
            parts = suffix.split('_')
            
            # Find local ID mapping
            cid = None
            row_meta = None
            if len(parts) >= 2 and parts[-1] in ['GFP1', 'GFP2']:
                for row in meta_by_cid.values():
                    if row['global_cell_id'] == "_".join(parts[:-1]) and row['source'] == parts[-1]:
                        row_meta = row
                        break
            else:
                try:
                    cid = int(suffix)
                    row_meta = meta_by_cid.get(cid)
                except ValueError:
                    pass
                
            if row_meta is None:
                continue
                
            cid = row_meta['local_cell_id']
            gcid = row_meta['global_cell_id']
            source = row_meta['source']
            field = row_meta['field']
            film = row_meta['film']
            orig_id = row_meta['original_cell_id']
            
            p1 = X_traj_raw[i][:, 0].tolist()
            p2 = X_traj_raw[i][:, 1].tolist()
            
            # QC Status
            qc_status = qc_map.get(gcid, "good")
            
            # Division timing relative calculations
            div_time = div_times.get(gcid)
            alignment_method = "computed"
            
            if div_time is None:
                div_time = no_septum_shifts.get(gcid)
                alignment_method = "aligned" if div_time is not None else None
                
            # Estimated relative times
            mid_T = 0.0
            src_cfg = exp_cfg.sources.get(source)
            if src_cfg:
                mid_T = src_cfg.midpoint_min
            else:
                if source == "GFP1": mid_T = 10.0
                elif source == "GFP2": mid_T = 150.0 if name == "Sept17" else 70.0
                else: mid_T = 130.0
                
            time_to_division = None
            if div_time is not None:
                time_to_division = round(mid_T - div_time, 1)
                
            # Map acor fit params if available
            row_acor = None
            if df_acor is not None and cid in df_acor.index:
                row_acor = df_acor.loc[cid]
                
            if row_acor is not None:
                fit_params = {
                    "p1_A1": float(row_acor.get('pol1_A1', 0.0)),
                    "p1_tau1": float(row_acor.get('pol1_tau1', 0.0)),
                    "p1_tau2": float(row_acor.get('pol1_tau2', 0.0)),
                    "p1_f": float(row_acor.get('pol1_f', 0.0)),
                    "p1_phi": float(row_acor.get('pol1_phi', 0.0)),
                    "p1_C": float(row_acor.get('pol1_C', 0.0)),
                    "p1_acf0": float(row_acor.get('pol1_acf0', 0.0)),
                    
                    "raw_precision_sum": float(row_acor.get('raw_precision_sum', 0.0)),
                    "sse_zero": float(row_acor.get('raw_freq_distance_sum', 0.0)),
                    "log_precision_sum": float(row_acor.get('precision_sum', 0.0)),
                    "log_zero": float(row_acor.get('freq_distance_sum', 0.0)),
                    
                    "p1_y_f": float(row_acor.get('pol1_y_f', 0.0)),
                    "p1_freq_prior": float(row_acor.get('pol1_freq_prior', 0.0)),
                    "p1_penalized": float(row_acor.get('pol1_penalized', 0.0)),

                    "p2_A1": float(row_acor.get('pol2_A1', 0.0)),
                    "p2_tau1": float(row_acor.get('pol2_tau1', 0.0)),
                    "p2_tau2": float(row_acor.get('pol2_tau2', 0.0)),
                    "p2_f": float(row_acor.get('pol2_f', 0.0)),
                    "p2_phi": float(row_acor.get('pol2_phi', 0.0)),
                    "p2_C": float(row_acor.get('pol2_C', 0.0)),
                    "p2_acf0": float(row_acor.get('pol2_acf0', 0.0)),
                    
                    "p2_y_f": float(row_acor.get('pol2_y_f', 0.0)),
                    "p2_freq_prior": float(row_acor.get('pol2_freq_prior', 0.0)),
                    "p2_penalized": float(row_acor.get('pol2_penalized', 0.0))
                }
            else:
                fit_params = {k: 0.0 for k in [
                    "p1_A1", "p1_tau1", "p1_tau2", "p1_f", "p1_phi", "p1_C", "p1_acf0",
                    "raw_precision_sum", "sse_zero", "log_precision_sum", "log_zero",
                    "p1_y_f", "p1_freq_prior", "p1_penalized",
                    "p2_A1", "p2_tau1", "p2_tau2", "p2_f", "p2_phi", "p2_C", "p2_acf0",
                    "p2_y_f", "p2_freq_prior", "p2_penalized"
                ]}
                
            records.append({
                "observation_id": gid,
                "experiment": name,
                "local_cell_id": cid,
                "original_cell_id": orig_id,
                "global_cell_id": gcid,
                "field": field,
                "source": source,
                "film": film,
                "trajectory_p1": p1,
                "trajectory_p2": p2,
                "included": True,
                "exclusion_reason": None,
                "qc_status": qc_status,
                "time_to_division": time_to_division,
                "time_alignment_method": alignment_method,
                "raw_features": X_feat_raw_full[i].tolist(),
                "idx_loaded": i,
                "fit_params": fit_params
            })
            
    # Construct DataFrame
    cells = pd.DataFrame(records)
    cells = cells.set_index("observation_id", verify_integrity=True)
    
    # 4. Apply Quality Control composable rules
    cells = apply_qc_rules(cells)
    
    qc_rep = make_qc_report(cells)
    logger.info("=== QUALITY CONTROL REPORT ===")
    logger.info(f"Loaded observations: {qc_rep['loaded_cells']}")
    logger.info(f"Included observations: {qc_rep['included_cells']}")
    logger.info(f"Excluded observations: {qc_rep['excluded_cells']}")
    
    # Fit scaling parameters on ONLY the curated reference cells (Sept17)
    included_cells = cells[cells["included"]].copy()
    if included_cells.empty:
        raise ValueError("No cells remaining after quality control filtering!")
        
    ref_name = global_cfg.reference_experiment
    ref_cells = included_cells[included_cells["experiment"] == ref_name]
    if ref_cells.empty:
        raise ValueError(f"Reference experiment {ref_name} has 0 cells after QC!")
        
    logger.info(f"Fitting scaler on reference dataset: {ref_name} ({len(ref_cells)} cells)")
    
    ref_indices_loaded = ref_cells["idx_loaded"].values
    X_traj_ref = X_traj_raw[ref_indices_loaded]
    X_feat_ref = X_feat_raw_full[ref_indices_loaded]
    
    mean_traj_ref = np.nanmean(X_traj_ref, axis=0)
    std_traj_ref = np.nanstd(X_traj_ref, axis=0)
    std_traj_ref = np.where(std_traj_ref < 1e-8, 1.0, std_traj_ref)
    
    mean_feat_ref = np.nanmean(X_feat_ref, axis=0)
    std_feat_ref = np.nanstd(X_feat_ref, axis=0)
    std_feat_ref = np.where(std_feat_ref < 1e-8, 1.0, std_feat_ref)
    
    all_indices_loaded = included_cells["idx_loaded"].values
    X_traj_scaled = (X_traj_raw[all_indices_loaded] - mean_traj_ref) / (std_traj_ref + 1e-8)
    X_traj_scaled = np.nan_to_num(X_traj_scaled, nan=0.0)
    
    X_feat_scaled = (X_feat_raw_full[all_indices_loaded] - mean_feat_ref) / (std_feat_ref + 1e-8)
    X_feat_scaled = np.nan_to_num(X_feat_scaled, nan=0.0)
    
    # 5. Load PyTorch autoencoder checkpoint and run inference
    logger.info(f"Loading autoencoder model from: {global_cfg.model_path}")
    model = MultimodalAutoencoder3D().to("cpu")
    model.load_state_dict(torch.load(global_cfg.model_path, map_location="cpu"))
    model.eval()
    
    with torch.no_grad():
        t_tensor = torch.from_numpy(X_traj_scaled).float()
        f_tensor = torch.from_numpy(X_feat_scaled).float()
        _, _, latents = model(t_tensor, f_tensor)
        latents = latents.numpy()
        
    # 6. Fit UMAP on Reference Latents ONLY, then project mutant cells
    logger.info("Fitting UMAP (3D and 2D) on reference latents only...")
    ref_idx_in_included = np.where(included_cells["experiment"] == ref_name)[0]
    latents_ref = latents[ref_idx_in_included]
    
    reducer_3d = umap.UMAP(n_components=3, random_state=global_cfg.umap_seed, n_jobs=1)
    reducer_3d.fit(latents_ref)
    coords_3d = reducer_3d.transform(latents)
    
    reducer_2d = umap.UMAP(n_components=2, random_state=global_cfg.umap_seed, n_jobs=1)
    reducer_2d.fit(latents_ref)
    coords_2d = reducer_2d.transform(latents)
    
    included_cells["coords_3d_x"] = coords_3d[:, 0]
    included_cells["coords_3d_y"] = coords_3d[:, 1]
    included_cells["coords_3d_z"] = coords_3d[:, 2]
    
    included_cells["coords_2d_x"] = coords_2d[:, 0]
    included_cells["coords_2d_y"] = coords_2d[:, 1]
    
    included_cells["latent_3d_x"] = latents[:, 0]
    included_cells["latent_3d_y"] = latents[:, 1]
    included_cells["latent_3d_z"] = latents[:, 2]

    
    gids = list(included_cells.index)
    
    time_rel_display = []
    cycle_display = []
    
    for gid in gids:
        r = included_cells.loc[gid]
        t = r["time_to_division"]
        
        t_val = None
        if t is not None:
            t_val = max(t, -900.0 if gid.startswith("M135_") else -300.0)
            
        time_rel_display.append(t_val)
        
        if t_val is None:
            cycle_display.append(None)
        else:
            cycle_len = 900.0 if gid.startswith("M135_") else 300.0
            score = (t_val + cycle_len) / cycle_len if t_val < 0 else t_val / cycle_len
            score = max(0.0, min(1.0, score))
            cycle_display.append(round(score, 4))
            
    cell_areas = load_cell_areas(gids)
    area_display = [None if np.isnan(v) else round(float(v), 2) for v in cell_areas]
    
    raw_feats_list = list(included_cells["raw_features"])
    
    pol1_mid_display = [round(float(f[1]), 4) for f in raw_feats_list]
    pol2_mid_display = [round(float(f[4]), 4) for f in raw_feats_list]
    periodicity_display = [round(float(f[7]), 4) for f in raw_feats_list]
    nc_score_display = [round(float(f[6]), 4) for f in raw_feats_list]
    
    color_arrays = {
        "Time Relative to Division (min)": time_rel_display,
        "Cell Area (max, px²)":  area_display,
        "Cell Cycle Stage":      cycle_display,
        "Pol1 Mid Intensity":    pol1_mid_display,
        "Pol2 Mid Intensity":    pol2_mid_display,
        "Periodicity":           periodicity_display,
        "NC Score":              nc_score_display,
    }
    
    # 8. Render Plotly structures
    def make_plotly_figure(is_3d: bool, is_latent: bool = False) -> dict:
        data_traces = []
        for exp_name in included_cells["experiment"].unique():
            sub = included_cells[included_cells["experiment"] == exp_name]
            
            customdata = []
            for obs_id, r in sub.iterrows():
                customdata.append([
                    obs_id,
                    r["time_alignment_method"] or "N/A",
                    r["global_cell_id"],
                    r["local_cell_id"],
                    r["film"]
                ])
                
            trace = {
                "type": "scatter3d" if (is_3d or is_latent) else "scatter",
                "mode": "markers",
                "name": exp_name,
                "customdata": customdata,
                "marker": {
                    "size": 5,
                    "opacity": 0.8,
                    "line": {
                        "color": "#475569",
                        "width": 0.5
                    }
                }
            }
            if is_latent:
                trace["x"] = sub["latent_3d_x"].tolist()
                trace["y"] = sub["latent_3d_y"].tolist()
                trace["z"] = sub["latent_3d_z"].tolist()
            elif is_3d:
                trace["x"] = sub["coords_3d_x"].tolist()
                trace["y"] = sub["coords_3d_y"].tolist()
                trace["z"] = sub["coords_3d_z"].tolist()
            else:
                trace["x"] = sub["coords_2d_x"].tolist()
                trace["y"] = sub["coords_2d_y"].tolist()
                
            data_traces.append(trace)
            
        layout = {
            "showlegend": True,
            "legend": {"font": {"color": "#1e293b"}},
            "paper_bgcolor": "#ffffff",
            "plot_bgcolor": "#ffffff",
            "margin": {"l": 0, "r": 0, "b": 0, "t": 40}
        }
        if is_3d or is_latent:
            layout["scene"] = {
                "xaxis": {"gridcolor": "#cbd5e1", "tickcolor": "#475569", "font": {"color": "#475569"}},
                "yaxis": {"gridcolor": "#cbd5e1", "tickcolor": "#475569", "font": {"color": "#475569"}},
                "zaxis": {"gridcolor": "#cbd5e1", "tickcolor": "#475569", "font": {"color": "#475569"}}
            }
        else:
            layout["xaxis"] = {"gridcolor": "#cbd5e1", "tickcolor": "#475569", "font": {"color": "#475569"}}
            layout["yaxis"] = {"gridcolor": "#cbd5e1", "tickcolor": "#475569", "font": {"color": "#475569"}}
            
        return {"data": data_traces, "layout": layout}

    fig_3d = make_plotly_figure(is_3d=True)
    fig_2d = make_plotly_figure(is_3d=False)
    fig_latent = make_plotly_figure(is_3d=True, is_latent=True)
    
    # 9. Build browser data (trajData & colorArrays)
    traj_dict = {}
    strips_dir = Path("/Volumes/X10 Pro/FungalProject_Outputs/video_ae/vertical_strips/")
    
    def correlate_norm(y):
        yd = y - np.mean(y)
        acor = np.correlate(yd, yd, mode='full')
        center = len(acor) // 2
        acor_zero = acor[center]
        if acor_zero > 1e-8:
            acor = acor / acor_zero
        return acor[center:].tolist()

    logger.info("🔬 Compiling final cell list trajectories and autocorrelation arrays...")
    for idx, (obs_id, r) in enumerate(included_cells.iterrows()):
        p1 = r["trajectory_p1"]
        p2 = r["trajectory_p2"]
        
        strip_b64 = ""
        strip_path = strips_dir / f"{obs_id}.png"
        if not strip_path.exists():
            strip_path = strips_dir / f"{r['experiment']}_{r['local_cell_id']}.png"
            
        if strip_path.exists():
            with open(strip_path, "rb") as img_f:
                strip_b64 = "data:image/png;base64," + base64.b64encode(img_f.read()).decode("utf-8")
                
        traj_dict[obs_id] = {
            "p1": p1,
            "p2": p2,
            "acor1": correlate_norm(np.array(p1)),
            "acor2": correlate_norm(np.array(p2)),
            "raw_feats": {
                "pol1_mid": pol1_mid_display[idx],
                "pol2_mid": pol2_mid_display[idx],
                "periodicity": periodicity_display[idx]
            },
            "strip": strip_b64,
            "idx": idx,
            "gcid": r["global_cell_id"],
            "global_cell_id": r["global_cell_id"],
            "local_gfp_id": r["local_cell_id"],
            "gfp_film": r["film"],
            "fit_params": r["fit_params"],
            "f": [
                pol1_mid_display[idx],
                periodicity_display[idx],
                nc_score_display[idx],
                cycle_display[idx],
                time_rel_display[idx],
                r["time_alignment_method"]
            ]
        }
        
    # 10. Call exporter
    template_dir = Path(__file__).parent / "templates"
    
    if global_cfg.output_mode == "static-site":
        export_static_site(
            global_cfg.output_dir,
            fig_2d,
            fig_3d,
            fig_latent,
            traj_dict,
            color_arrays,
            template_dir,
            strips_dir
        )
    else:
        export_single_html(
            global_cfg.output_dir / global_cfg.output_filename,
            fig_2d,
            fig_3d,
            fig_latent,
            traj_dict,
            color_arrays,
            template_dir
        )
        
    nas_html = Path("/Volumes/Ian/UMAP/fc_ae_3d_manifold_explorer_curated_Sept17.html")
    if os.path.exists(nas_html.parent):
        try:
            import shutil
            shutil.copy(global_cfg.output_dir / global_cfg.output_filename, nas_html)
            logger.info(f"✅ Successfully synchronized HTML dashboard to NAS: {nas_html}")
        except Exception as e:
            logger.warning(f"⚠️ Could not sync dashboard to NAS: {e}")
            
    logger.info("🎉 Pipeline execution completed successfully!")
