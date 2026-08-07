
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/list_experiments")
def list_experiments():
    experiments = sorted([d.name for d in BASE_MOVIE_ROOT.iterdir() if d.is_dir() and not d.name.startswith(".") and d.name in RELEVANT_EXPERIMENTS])
    return jsonify({"experiments": experiments})

@app.route("/api/list_films_and_sequences")
def list_films_and_sequences():
    exp = request.args.get("experiment")
    exp_dir = BASE_MOVIE_ROOT / exp
    films = sorted([d.name for d in exp_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    
    seq_data = get_sequence_linkage_data(exp)
    sequences = list(seq_data.keys())
    
    return jsonify({"films": films, "sequences": sequences})

@app.route("/api/list_cells")
def list_cells():
    exp = request.args.get("experiment")
    if "sequence" in request.args:
        sequence = request.args.get("sequence")
        data = get_sequence_linkage_data(exp)
        if sequence in data:
            ensure_pseudo_sequence_cells(exp, sequence, data)
            trigger_pregeneration_for_films(exp, data[sequence]["films"])
            def get_sort_key(k):
                s = str(k)
                m = re.search(r"(\d+)$", s)
                if m:
                    return (0, int(m.group(1)))
                return (1, s)
            cells = sorted(list(data[sequence]["global_cells"].keys()), key=get_sort_key)
            # Filter to keep only cells that have a valid tracking label in the last film
            cells = [c for c in cells if data[sequence]["global_cells"][c][-1] != -1]
            
            def display_name_for(global_id):
                # Extract trailing number from global ID (e.g. "A14_F1_cell_18" → "Cell 18")
                m = re.search(r'(\d+)$', str(global_id))
                return f"Cell {m.group(1)}" if m else str(global_id)
            
            # Build display names, appending film hint on collision
            base_names = {c: display_name_for(c) for c in cells}
            name_count = {}
            for n in base_names.values():
                name_count[n] = name_count.get(n, 0) + 1
            
            def origin_film_hint(global_id, local_ids):
                # Check if global_id contains an inner film name
                # e.g., "A14_F1_A14_BF_2_F1_cell_11" -> "A14_BF_2_F1"
                prefix = f"{sequence}_"
                gid_str = str(global_id)
                if gid_str.startswith(prefix) and "_cell_" in gid_str:
                    inner_part = gid_str[len(prefix):gid_str.rfind("_cell_")]
                    if inner_part:
                        parts = inner_part.split("_")
                        return "_".join(parts[-3:]) if len(parts) >= 3 else inner_part
                
                # Fallback to the first film where the cell is tracked
                films_local = data[sequence]["films"]
                for i, lid in enumerate(local_ids):
                    if lid != -1:
                        # Use a short film suffix, e.g. "FL_1_F1"
                        film = films_local[i] if i < len(films_local) else ""
                        parts = film.split("_")
                        hint = "_".join(parts[-3:]) if len(parts) >= 3 else film
                        return hint
                return "root"
            
            cells_data = []
            for c in cells:
                name = base_names[c]
                if name_count[name] > 1:
                    hint = origin_film_hint(c, data[sequence]["global_cells"][c])
                    name = f"{name} ({hint})"
                cells_data.append({"global_id": c, "display_name": name})
            lineage = data[sequence].get("lineage", {})
            return jsonify({"cells": cells_data, "lineage": lineage})
        return jsonify({"cells": [], "lineage": {}})
        
    film = request.args.get("film")
    trigger_pregeneration_for_films(exp, [film])
    tracked_dir = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}"
    
    cells = []
    if tracked_dir.is_dir():
        for f in tracked_dir.iterdir():
            if f.name.startswith("."):
                continue
            m = re.match(r"^cell_(\d+)_masks\.csv$", f.name)
            if m:
                cells.append(int(m.group(1)))
                
    cells_data = [{"global_id": str(c), "display_name": f"Cell {c}"} for c in sorted(list(set(cells)))]
    return jsonify({"cells": cells_data})

@app.route("/api/cell_masks")
def cell_masks():
    exp = request.args.get("experiment")
    cell_id = request.args.get("cell_id")
    
    if "sequence" in request.args:
        sequence = request.args.get("sequence")
        data = get_sequence_linkage_data(exp)
        if sequence not in data:
            return jsonify({"error": "Sequence not found"}), 404
            
        ensure_pseudo_sequence_cells(exp, sequence, data)
        trigger_pregeneration_for_films(exp, data[sequence]["films"])
        films = data[sequence]["films"]
        local_ids = data[sequence]["global_cells"].get(cell_id, [-1]*len(films))
        
        all_masks = []
        boundaries = []
        current_len = 0
        w, h = 0, 0
        track_channel = 'bf'
        
        for i, film in enumerate(films):
            boundaries.append(current_len)
            
            L, fW, fH = get_film_frame_count_and_size(exp, film)
            if w == 0 and fW > 0:
                w, h = fW, fH
                
            local_id = local_ids[i]
            if local_id == -1:
                all_masks.extend([""] * L)
                current_len += L
                continue
                
            csv_path = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}" / f"cell_{local_id}_masks.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                rle_col = 'rle_bf'
                if 'rle_gfp' in df.columns and any(isinstance(x, str) and x.strip() for x in df['rle_gfp'].dropna()):
                    track_channel = 'gfp'
                    rle_col = 'rle_gfp'
                    
                masks = df[rle_col].fillna("").tolist()
                
                # pad or truncate to L
                if len(masks) < L:
                    masks.extend([""] * (L - len(masks)))
                elif len(masks) > L:
                    masks = masks[:L]
                    
                all_masks.extend(masks)
            else:
                all_masks.extend([""] * L)
                
            current_len += L
            
        return jsonify({
            "masks": all_masks,
            "num_frames": len(all_masks),
            "width": w,
            "height": h,
            "track_channel": track_channel,
            "film_boundaries": boundaries,
            "linkage_details": {"films": films, "local_ids": local_ids},
            "local_film": films[0] if films else None
        })
        
    else:
        film = request.args.get("film")
        trigger_pregeneration_for_films(exp, [film])
        csv_path = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}" / f"cell_{cell_id}_masks.csv"
        df = pd.read_csv(csv_path)
        
        # Determine tracking channel: default to "gfp" if it's a GFP film, else "bf"
        track_channel = "gfp" if "FL" in film else "bf"
        rle_col = 'rle_gfp' if track_channel == 'gfp' else 'rle_bf'
        
        # Override default if either channel has non-empty masks in df
        if 'rle_gfp' in df.columns and any(isinstance(x, str) and x.strip() for x in df['rle_gfp'].dropna()):
            track_channel = 'gfp'
            rle_col = 'rle_gfp'
        elif 'rle_bf' in df.columns and any(isinstance(x, str) and x.strip() for x in df['rle_bf'].dropna()):
            track_channel = 'bf'
            rle_col = 'rle_bf'
                
        masks = df[rle_col].fillna("").tolist()
        return jsonify({
            "masks": masks,
            "num_frames": len(df),
            "width": int(df.iloc[0]['width']),
            "height": int(df.iloc[0]['height']),
            "track_channel": track_channel,
            "local_film": film
        })

@app.route("/api/get_candidates")
def get_candidates():
    exp = request.args.get("experiment")
    film = request.args.get("film")
    tracked_dir = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}"
    
    cells = []
    if tracked_dir.is_dir():
        for f in tracked_dir.iterdir():
            if f.name.startswith("."): continue
            m = re.match(r"^cell_(\d+)_masks\.csv$", f.name)
            if m:
                cells.append(int(m.group(1)))
    return jsonify({"cells": sorted(list(set(cells)))})


@app.route("/api/identify_cell", methods=["POST"])
def identify_cell():
    data = request.json
    exp = data.get("experiment")
    film = data.get("film")
    t = int(data.get("t", 0))
    
    if not film:
        return jsonify({"status": "error", "message": "Film name is required but was empty."})
        
    try:
        x, y = int(data.get("x")), int(data.get("y"))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "Coordinates x and y must be integers."})
    
    tracked_dir = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}"
    if not tracked_dir.is_dir():
        return jsonify({"status": "error", "message": f"TrackedCells directory not found: {tracked_dir}"})
    
    # Fast path: Use the segmentation mask directly to find the label, then see which cell has that label?
    # Wait, cells might not map 1:1 to labels. Just iterate csvs.
    for cf in tracked_dir.iterdir():
        if cf.name.startswith("."): continue
        m = re.match(r"^cell_(\d+)_masks\.csv$", cf.name)
        if m:
            cid = int(m.group(1))
            try:
                df = pd.read_csv(cf)
                if t >= len(df): continue
                
                # Check both masks
                W = int(df.iloc[0]['width'])
                H = int(df.iloc[0]['height'])
                found = False
                for rle_col in ['rle_bf', 'rle_gfp']:
                    if rle_col in df.columns:
                        rle = df.iloc[t][rle_col]
                        if isinstance(rle, str) and rle.strip():
                            mask = rle_decode(rle, (H, W))
                            if y < H and x < W and mask[y, x]:
                                found = True
                                break
                if found:
                    return jsonify({"status": "success", "cell_id": cid})
            except Exception:
                pass
                
    # If no tracked cell is found, check the raw segmentation mask
    try:
        masks_dir = BASE_MOVIE_ROOT / exp / film / f"Masks_{film}"
        files = sorted([f for f in masks_dir.glob(f"{film}_t_{t:03d}_c_*_seg.tif") if not f.name.startswith(".")])
        if not files:
            files = sorted([f for f in masks_dir.glob(f"*_t_{t:03d}_c_*_seg.tif") if not f.name.startswith(".")])
        if files:
            from Cell_tracking_functions import load_segmentation
            from skimage.measure import label
            seg = load_segmentation(str(files[0]))
            seg_lbl = label(seg) if seg.dtype == bool else seg
            if y < seg_lbl.shape[0] and x < seg_lbl.shape[1]:
                val = seg_lbl[y, x]
                if val > 0:
                    return jsonify({"status": "untracked", "label_id": int(val), "message": f"Found raw segment #{val}, but it is not tracked."})
    except Exception as e:
        print(f"Error checking seg mask: {e}")
        
    return jsonify({"status": "error", "message": "No tracked cell found at this location."})

@app.route('/api/get_qc', methods=['GET'])
def get_qc():
    exp = request.args.get('experiment')
    film = request.args.get('film')
    seq = request.args.get('sequence')
    
    target = seq if seq else film
    qc_file = BASE_MOVIE_ROOT / exp / target / f"qc_{target}.json"
    
    if qc_file.exists():
        import json
        with open(qc_file, 'r') as f:
            return jsonify({"status": "success", "qc": json.load(f)})
    return jsonify({"status": "success", "qc": {}})

@app.route("/api/suspicious_cells")
def suspicious_cells():
    exp = request.args.get("experiment")
    sequence = request.args.get("sequence")
    film = request.args.get("film")
    threshold = float(request.args.get("threshold", 15.0))
    
    target = sequence if sequence else film
    cache_key = f"{exp}::{target}::thresh_{threshold}"
    
    if cache_key in SUSPICIOUS_CACHE:
        return jsonify({"suspicious": SUSPICIOUS_CACHE[cache_key]})
        
    # Check disk cache
    target_dir = BASE_MOVIE_ROOT / exp / target
    cache_file = target_dir / f"suspicious_{target}.json"
    if cache_file.exists():
        try:
            import json
            with open(cache_file, "r") as f:
                disk_data = json.load(f)
                SUSPICIOUS_CACHE[cache_key] = disk_data
                return jsonify({"suspicious": disk_data})
        except Exception as e:
            print(f"Error reading disk cache: {e}")
            
    suspicious_data = {}
    
    if sequence:
        seq_data = get_sequence_linkage_data(exp)
        if sequence not in seq_data:
            return jsonify({"suspicious": {}})
        ensure_pseudo_sequence_cells(exp, sequence, seq_data)
        films = seq_data[sequence]["films"]
        cell_mappings = seq_data[sequence]["global_cells"]
    else:
        films = [film]
        tracked_dir = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}"
        cell_mappings = {}
        if tracked_dir.is_dir():
            for f in tracked_dir.iterdir():
                if f.name.startswith("."):
                    continue
                m = re.match(r"^cell_(\d+)_masks\.csv$", f.name)
                if m:
                    cid = m.group(1)
                    cell_mappings[cid] = [int(cid)]
                    
    for cell_id, local_ids in cell_mappings.items():
        film_dfs = []
        
        for i, f_name in enumerate(films):
            local_id = local_ids[i] if i < len(local_ids) else -1
            if local_id == -1:
                film_dfs.append(None)
                continue
            csv_path = BASE_MOVIE_ROOT / exp / f_name / f"TrackedCells_{f_name}" / f"cell_{local_id}_masks.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    film_dfs.append(df)
                except Exception:
                    film_dfs.append(None)
            else:
                film_dfs.append(None)
                
        all_rles = []
        H, W = 0, 0
        for i, df in enumerate(film_dfs):
            f_name = films[i]
            L, fW, fH = get_film_frame_count_and_size(exp, f_name)
            if df is not None and len(df) > 0:
                if H == 0:
                    H, W = int(df.iloc[0]['height']), int(df.iloc[0]['width'])
                
                # Determine rle_col for this dataframe
                rle_col = 'rle_bf'
                if 'rle_gfp' in df.columns and any(isinstance(x, str) and x.strip() for x in df['rle_gfp'].dropna()):
                    rle_col = 'rle_gfp'
                elif 'rle_bf' not in df.columns and 'rle_gfp' in df.columns:
                    rle_col = 'rle_gfp'
                    
                masks = df[rle_col].fillna("").tolist()
                if len(masks) < L:
                    masks.extend([""] * (L - len(masks)))
                elif len(masks) > L:
                    masks = masks[:L]
                all_rles.extend(masks)
            else:
                all_rles.extend([""] * L)
                
        if H == 0 or W == 0:
            continue
            
        centroids = []
        for rle in all_rles:
            if not isinstance(rle, str) or not rle.strip() or rle == "nan":
                centroids.append(None)
                continue
            try:
                mask = rle_decode(rle, (H, W))
                if not mask.any():
                    centroids.append(None)
                else:
                    ys, xs = np.nonzero(mask)
                    centroids.append((float(np.mean(ys)), float(np.mean(xs))))
            except Exception:
                centroids.append(None)
                
        suspicious_frames = []
        for t in range(1, len(centroids)):
            c1 = centroids[t-1]
            c2 = centroids[t]
            if c1 is not None and c2 is not None:
                dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                if dist > threshold:
                    suspicious_frames.append(t)
                    
        if suspicious_frames:
            suspicious_data[str(cell_id)] = suspicious_frames
            
    SUSPICIOUS_CACHE[cache_key] = suspicious_data
    
    # Save to disk cache
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        import json
        with open(cache_file, "w") as f:
            json.dump(suspicious_data, f)
    except Exception as e:
        print(f"Error writing disk cache: {e}")
        
    return jsonify({"suspicious": suspicious_data})

@app.route('/api/save_qc', methods=['POST'])
def save_qc():
    data = request.json
    exp = data.get('experiment')
    film = data.get('film')
    seq = data.get('sequence')
    cell_id = str(data.get('cell_id'))
    status = data.get('status')
    
    target = seq if seq else film
    target_dir = BASE_MOVIE_ROOT / exp / target
    target_dir.mkdir(parents=True, exist_ok=True)
    qc_file = target_dir / f"qc_{target}.json"
    
    import json
    qc_data = {}
    if qc_file.exists():
        with open(qc_file, 'r') as f:
            qc_data = json.load(f)
            
    if status == "pending":
        if cell_id in qc_data:
            del qc_data[cell_id]
    else:
        qc_data[cell_id] = status
    
    with open(qc_file, 'w') as f:
        json.dump(qc_data, f)
        
    return jsonify({"status": "success"})

@app.route("/api/create_new_cell", methods=["POST"])
def create_new_cell():
    data = request.json
    exp = data.get("experiment")
    film = data.get("film")
    
    tracked_dir = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}"
    tracked_dir.mkdir(parents=True, exist_ok=True)
    
    max_id = 9999
    for f in tracked_dir.iterdir():
        if f.name.startswith("."): continue
        m = re.match(r"^cell_(\d+)_masks\.csv$", f.name)
        if m:
            cid = int(m.group(1))
            if cid > max_id:
                max_id = cid
                
    new_id = max_id + 1
    
    L, W, H = get_film_frame_count_and_size(exp, film)
    if L == 0:
        return jsonify({"status": "error", "message": "No frame images or existing cell files found to determine dimensions."})
        
    rows = []
    for t in range(L):
        rows.append({
            "time_point": t,
            "width": W, "height": H,
            "rle_bf": "",
            "touches_border_bf": False,
            "source_bf": "manual" if t == 0 else "",
            "overlap_score_bf": 1.0,
            "smooth_score_bf": 0.0,
            "area_bf": 0,
            "rle_gfp": "",
            "touches_border_gfp": False,
            "source_gfp": "manual" if t == 0 else "",
            "overlap_score_gfp": 1.0,
            "smooth_score_gfp": 0.0,
            "area_gfp": 0
        })
        
    df = pd.DataFrame(rows)
    out_csv = tracked_dir / f"cell_{new_id}_masks.csv"
    df.to_csv(out_csv, index=False)
    
    return jsonify({"status": "success", "cell_id": new_id})

@app.route("/api/quantify_on_hpc", methods=["POST"])
def quantify_on_hpc():
    data = request.json
    exp = data.get("experiment")
    film = data.get("film")
    label_id = data.get("label_id")
    seed_from_csv = data.get("seed_from_csv", False)
    
    track_channel = data.get("track_channel", "gfp" if "FL" in film else "bf")
    
    import subprocess
    
    try:
        exp_dir = BASE_MOVIE_ROOT / exp
        seed_flag = " --seed_from_csv" if seed_from_csv else ""
        
        script_path = Path(__file__).parent / "one_cell_quantification_1CH.py"
        
        import sys
        # Run quantification locally
        local_cmd = f""" "{sys.executable}" "{script_path}" --cell_id {label_id} --experiment_path "{exp_dir}" --file_name "{film}" --track_channel {track_channel} --update_existing{seed_flag} """
        
        result = subprocess.run(local_cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            stderr_out = (result.stderr or "").strip()
            stdout_out = (result.stdout or "").strip()
            full_out = stderr_out if stderr_out else stdout_out
            if not full_out:
                err_msg = f"Process exited with code {result.returncode}"
            else:
                lines = full_out.splitlines()
                if len(lines) > 15:
                    err_msg = "[Truncated progress...]\n" + "\n".join(lines[-15:])
                else:
                    err_msg = full_out
            print(f"Quantification Error: {err_msg}")
            return jsonify({"status": "error", "message": f"Quantification Error: {err_msg}"})
            
        return jsonify({"status": "success", "message": f"Successfully quantified cell #{label_id} locally!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/api/update_linkage", methods=["POST"])
def update_linkage():
    data = request.json
    exp = data.get("experiment")
    sequence = data.get("sequence")
    global_cell = data.get("global_cell")
    film_idx = int(data.get("film_idx"))
    new_local_cell = int(data.get("new_local_cell"))
    
    seq_file = BASE_MOVIE_ROOT / exp / "sequence_linkage.json"
    if not seq_file.exists():
        return jsonify({"status": "error", "message": f"Sequence linkage file not found: {seq_file}"})
        
    with open(seq_file, 'r') as f:
        linkage_data = json.load(f)
        
    if sequence not in linkage_data:
        return jsonify({"status": "error", "message": f"Sequence '{sequence}' not found in linkage data."})
        
    global_cell_str = str(global_cell)
    if global_cell_str not in linkage_data[sequence]["global_cells"]:
        return jsonify({"status": "error", "message": f"Global cell '{global_cell_str}' not found in sequence '{sequence}'."})
        
    linkage_data[sequence]["global_cells"][global_cell_str][film_idx] = new_local_cell
    
    with open(seq_file, 'w') as f:
        json.dump(linkage_data, f, indent=2)
        
    return jsonify({"status": "success"})

# ==============================================================================
# Septum Prediction AI & Labeling Endpoints
# ==============================================================================

def get_inference_runner(exp):
    chkpts = [
        BASE_MOVIE_ROOT / exp / "training_dataset" / "checkpoints_binary" / "model_latest.pt",
        BASE_MOVIE_ROOT / exp / "training_dataset" / "checkpoints" / "model_latest.pt",
        Path("/Volumes/X10 Pro/Movies/2025_12_31_M92/training_dataset/checkpoints_binary/model_latest.pt"),
        Path("/Volumes/X10 Pro/Movies/2025_12_31_M92/training_dataset/checkpoints/model_latest.pt")
    ]
    for cp in chkpts:
        if cp.exists():
            try:
                from SingleCellDataAnalysis.inference_core import FungalInferenceCore
                print(f"Loading FungalInferenceCore with checkpoint {cp}...")
                return FungalInferenceCore(str(cp), device="cpu")
            except Exception as e:
                print(f"Error loading checkpoint {cp}: {e}")
    return None

def get_cell_crop_tile(exp, film, t, rle, pad=10, tile_size=96):
    try:
        frames_dir = BASE_MOVIE_ROOT / exp / film / f"Frames_{film}"
        files = sorted([f for f in frames_dir.glob(f"{film}_t_{t:03d}_c_0.tif") if not f.name.startswith(".")])
        if not files:
            files = sorted([f for f in frames_dir.glob(f"*_t_{t:03d}_c_0.tif") if not f.name.startswith(".")])
        if not files:
            files = sorted([f for f in frames_dir.glob(f"{film}_t_{t:03d}_c_*.tif") if not f.name.startswith(".")])
            if not files:
                files = sorted([f for f in frames_dir.glob(f"*_t_{t:03d}_c_*.tif") if not f.name.startswith(".")])
                
        if not files:
            return None
            
        img = imread(str(files[0]))
        H, W = img.shape[:2]
        mask = rle_decode(rle, (H, W))
        
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            return None
            
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        y0 = max(0, y0 - pad)
        y1 = min(H - 1, y1 + pad)
        x0 = max(0, x0 - pad)
        x1 = min(W - 1, x1 + pad)
        crop = img[y0:y1 + 1, x0:x1 + 1]
        
        Ht, Wt = tile_size, tile_size
        a = np.asarray(crop)
        
        if a.dtype != np.uint8:
            af = a.astype(np.float32)
            lo, hi = np.nanpercentile(af, [1, 99]) if np.isfinite(af).any() else (0.0, 1.0)
            if not np.isfinite(lo): lo = 0.0
            if not np.isfinite(hi) or hi <= lo: hi = lo + 1.0
            af = np.clip((af - lo) / (hi - lo), 0, 1)
            a = (255 * af).astype(np.uint8)
        else:
            a = a.copy()
            
        h, w = a.shape[:2]
        if h > Ht:
            y_start = (h - Ht) // 2
            a = a[y_start:y_start + Ht, :]
            h = Ht
        if w > Wt:
            x_start = (w - Wt) // 2
            a = a[:, x_start:x_start + Wt]
            w = Wt
            
        out = np.zeros((Ht, Wt), dtype=np.uint8)
        y_start = (Ht - h) // 2
        x_start = (Wt - w) // 2
        out[y_start:y_start + h, x_start:x_start + w] = a
        return out
    except Exception as e:
        print(f"Error cropping cell at t={t}: {e}")
        return None

@app.route("/api/get_septum_label", methods=["GET"])
def get_septum_label():
    exp = request.args.get("experiment")
    film = request.args.get("film")
    cell_id = str(request.args.get("cell_id"))
    
    label_dir = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}" / "cell_plots" / "gui_labels"
    json_path = label_dir / "global_septum_alignment.json"
    
    has_septum = False
    local_start = None
    local_end = None
    white_septum = False
    
    has_septum_2 = False
    local_start_2 = None
    local_end_2 = None
    white_septum_2 = False
    
    offset = 0
    
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                js = json.load(f)
            offsets = js.get("offsets", {})
            cell_intervals = js.get("cell_intervals", {})
            
            offset = int(offsets.get(cell_id, 0))
            ci = cell_intervals.get(cell_id, {})
            
            has_septum = bool(ci.get("has_septum", False))
            start_aligned = ci.get("start_aligned")
            end_aligned = ci.get("end_aligned")
            white_septum = bool(ci.get("white_septum", False))
            
            has_septum_2 = bool(ci.get("has_septum_2", False))
            start_aligned_2 = ci.get("start_aligned_2")
            end_aligned_2 = ci.get("end_aligned_2")
            white_septum_2 = bool(ci.get("white_septum_2", False))
            
            if start_aligned is not None:
                local_start = int(start_aligned - offset)
            if end_aligned is not None:
                local_end = int(end_aligned - offset)
                
            if start_aligned_2 is not None:
                local_start_2 = int(start_aligned_2 - offset)
            if end_aligned_2 is not None:
                local_end_2 = int(end_aligned_2 - offset)
        except Exception as e:
            print(f"Error loading global_septum_alignment.json: {e}")
            
    return jsonify({
        "status": "success",
        "has_septum": has_septum,
        "local_start": local_start,
        "local_end": local_end,
        "white_septum": white_septum,
        "has_septum_2": has_septum_2,
        "local_start_2": local_start_2,
        "local_end_2": local_end_2,
        "white_septum_2": white_septum_2,
        "offset": offset
    })

@app.route("/api/save_septum_label", methods=["POST"])
def save_septum_label():
    data = request.json
    exp = data.get("experiment")
    film = data.get("film")
    cell_id = str(data.get("cell_id"))
    
    has_septum = bool(data.get("has_septum", False))
    local_start = data.get("local_start")
    local_end = data.get("local_end")
    white_septum = bool(data.get("white_septum", False))
    
    has_septum_2 = bool(data.get("has_septum_2", False))
    local_start_2 = data.get("local_start_2")
    local_end_2 = data.get("local_end_2")
    white_septum_2 = bool(data.get("white_septum_2", False))
    
    label_dir = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}" / "cell_plots" / "gui_labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    json_path = label_dir / "global_septum_alignment.json"
    
    js = {
        "working_dir": str(BASE_MOVIE_ROOT / exp),
        "film_name": film,
        "cell_order": [],
        "offsets": {},
        "global_interval": {"G0": 0, "G1": 55},
        "cell_intervals": {}
    }
    
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                js = json.load(f)
        except Exception as e:
            print(f"Error loading global_septum_alignment.json: {e}")
            
    offsets = js.setdefault("offsets", {})
    cell_intervals = js.setdefault("cell_intervals", {})
    
    offset = int(offsets.setdefault(cell_id, 0))
    
    start_aligned = (local_start + offset) if (local_start is not None and has_septum) else None
    end_aligned = (local_end + offset) if (local_end is not None and has_septum) else None
    
    start_aligned_2 = (local_start_2 + offset) if (local_start_2 is not None and has_septum_2) else None
    end_aligned_2 = (local_end_2 + offset) if (local_end_2 is not None and has_septum_2) else None
    
    cell_intervals[cell_id] = {
        "has_septum": has_septum,
        "start_aligned": start_aligned,
        "end_aligned": end_aligned,
        "white_septum": white_septum,
        "has_septum_2": has_septum_2,
        "start_aligned_2": start_aligned_2,
        "end_aligned_2": end_aligned_2,
        "white_septum_2": white_septum_2
    }
    
    from datetime import datetime
    js["updated_at"] = datetime.now().isoformat()
    
    try:
        with open(json_path, 'w') as f:
            json.dump(js, f, indent=2)
            
        # Discover all cell IDs in the film to export the CSV
        tracked_dir = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}"
        all_cids = []
        for f in tracked_dir.iterdir():
            if f.name.startswith("."): continue
            m = re.match(r"^cell_(\d+)_masks\.csv$", f.name)
            if m:
                all_cids.append(int(m.group(1)))
        all_cids.sort()
        
        gi = js.get("global_interval", {})
        a_left = int(gi.get("G0", 0))
        
        csv_path = label_dir / "septum_interval_per_cell.csv"
        rows = []
        for cid in all_cids:
            cid_str = str(cid)
            ci = cell_intervals.get(cid_str, {})
            rows.append({
                "cell_id": cid,
                "a_left": a_left,
                "start_aligned": ci.get("start_aligned") if ci.get("start_aligned") is not None else "",
                "end_aligned": ci.get("end_aligned") if ci.get("end_aligned") is not None else "",
                "has": 1 if ci.get("has_septum") else 0,
                "white_septum": 1 if ci.get("white_septum") else 0,
            })
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        
        # Build strip and export training sample in the exact training dataset format
        csv_path_cell = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}" / f"cell_{cell_id}_masks.csv"
        if csv_path_cell.exists():
            try:
                df_cell = pd.read_csv(csv_path_cell)
                rle_col = 'rle_bf'
                if 'rle_gfp' in df_cell.columns and df_cell['rle_gfp'].dropna().any():
                    rle_col = 'rle_gfp'
                    
                tiles = []
                L = len(df_cell)
                for t in range(L):
                    rle = df_cell.iloc[t][rle_col]
                    tile = None
                    if isinstance(rle, str) and rle.strip():
                        tile = get_cell_crop_tile(exp, film, t, rle)
                    if tile is None:
                        tile = np.zeros((96, 96), dtype=np.uint8)
                    tiles.append(tile)
                    
                strip = np.hstack(tiles)
                
                from SingleCellDataAnalysis.septum_training_utils import export_cell_training_sample
                export_cell_training_sample(
                    working_dir=str(BASE_MOVIE_ROOT / exp),
                    film_name=film,
                    cell_id=int(cell_id),
                    strip=strip,
                    tp0=0,
                    offset=offset,
                    start_idx=local_start if (local_start is not None and has_septum) else -1,
                    end_idx=local_end if (local_end is not None and has_septum) else -1,
                    label_source="cell",
                    start_aligned=start_aligned,
                    end_aligned=end_aligned,
                    white_septum=white_septum,
                )
                print(f"Successfully exported training sample for cell {cell_id} in {film} to training_dataset")
            except Exception as e:
                print(f"Error calling export_cell_training_sample for cell {cell_id}: {e}")
        
        return jsonify({"status": "success", "message": "Septum labels saved and CSV/training sample exported successfully!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/api/predict_septum", methods=["POST"])
def predict_septum():
    data = request.json
    exp = data.get("experiment")
    film = data.get("film")
    cell_id = str(data.get("cell_id"))
    
    inference_runner = get_inference_runner(exp)
    if inference_runner is None:
        return jsonify({"status": "error", "message": "Septum AI model checkpoint not found or could not be loaded."})
        
    csv_path = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}" / f"cell_{cell_id}_masks.csv"
    if not csv_path.exists():
        return jsonify({"status": "error", "message": f"Cell masks CSV not found: {csv_path}"})
        
    df = pd.read_csv(csv_path)
    rle_col = 'rle_bf'
    if 'rle_gfp' in df.columns and df['rle_gfp'].dropna().any():
        rle_col = 'rle_gfp'
        
    tiles = []
    L = len(df)
    for t in range(L):
        rle = df.iloc[t][rle_col]
        tile = None
        if isinstance(rle, str) and rle.strip():
            tile = get_cell_crop_tile(exp, film, t, rle)
        if tile is None:
            tile = np.zeros((96, 96), dtype=np.uint8)
        tiles.append(tile)
        
    strip = np.hstack(tiles)
    
    try:
        probs = inference_runner.predict_strip(strip)
        if probs is not None:
            return jsonify({"status": "success", "probs": probs.tolist()})
        else:
            return jsonify({"status": "error", "message": "Model inference failed."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Inference exception: {str(e)}"})



def get_actual_film_and_t(args):
    exp = args.get("experiment")
    t = int(args.get("t"))
    if "sequence" in args:
        seq = args.get("sequence")
        gid = args.get("cell_id")
        return resolve_global_t(exp, seq, gid, t)
    return args.get("film"), int(args.get("cell_id")), t

@app.route("/api/frame_boundaries")
def frame_boundaries():
    film, _, local_t = get_actual_film_and_t(request.args)
    exp = request.args.get("experiment")
    
    masks_dir = BASE_MOVIE_ROOT / exp / film / f"Masks_{film}"
    files = sorted([f for f in masks_dir.glob(f"{film}_t_{local_t:03d}_c_*_seg.tif") if not f.name.startswith(".")])
    if not files:
        files = sorted([f for f in masks_dir.glob(f"*_t_{local_t:03d}_c_*_seg.tif") if not f.name.startswith(".")])
        
    if not files:
        return jsonify({"error": "No segment file found"}), 404
        
    try:
        seg = load_segmentation(str(files[0]))
        from skimage.measure import label
        seg_lbl = (label(seg) if seg.dtype == bool else seg).copy()
        
        # Burn tracked local cells into seg_lbl so their outlines are visible for linking
        tracked_dir = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}"
        if tracked_dir.is_dir():
            max_lbl = int(seg_lbl.max()) if seg_lbl.size > 0 else 0
            next_lbl = max_lbl + 100
            for cf in tracked_dir.iterdir():
                if cf.name.startswith("."): continue
                m = re.match(r"^cell_(\d+)_masks\.csv$", cf.name)
                if m:
                    try:
                        df = pd.read_csv(cf)
                        if local_t < len(df):
                            H, W = seg_lbl.shape
                            for rle_col in ['rle_bf', 'rle_gfp']:
                                if rle_col in df.columns:
                                    rle = df.iloc[local_t][rle_col]
                                    if isinstance(rle, str) and rle.strip():
                                        source_col = 'source_bf' if rle_col == 'rle_bf' else 'source_gfp'
                                        is_manual = False
                                        if source_col in df.columns:
                                            is_manual = (df.iloc[local_t][source_col] == 'manual')
                                        if is_manual:
                                            mask = rle_decode(rle, (H, W))
                                            seg_lbl[mask] = next_lbl
                                            next_lbl += 1
                                            break
                    except Exception:
                        pass
        from skimage.segmentation import find_boundaries
        from scipy.ndimage import binary_dilation
        
        boundaries = find_boundaries(seg_lbl, mode='outer')
        thick_boundaries = binary_dilation(boundaries, structure=np.ones((3, 3)))
        H, W = seg_lbl.shape
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[thick_boundaries] = [234, 179, 8, 140]
        
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(rgba, 'RGBA')
        img_io = BytesIO()
        pil_img.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/frame_image")
def frame_image():
    film, _, local_t = get_actual_film_and_t(request.args)
    exp = request.args.get("experiment")
    
    frames_dir = BASE_MOVIE_ROOT / exp / film / f"Frames_{film}"
    files = sorted([f for f in frames_dir.glob(f"{film}_t_{local_t:03d}_c_*.tif") if not f.name.startswith(".")])
    if not files:
        files = sorted([f for f in frames_dir.glob(f"*_t_{local_t:03d}_c_*.tif") if not f.name.startswith(".")])
        
    if not files:
        return jsonify({"error": f"Frame image not found"}), 404
        
    img = imread(str(files[0]))
    p_lo = np.percentile(img, 1.0)
    p_hi = np.percentile(img, 99.5)
    if p_hi > p_lo:
        img_scaled = np.clip((img - p_lo) / (p_hi - p_lo) * 255.0, 0, 255).astype(np.uint8)
    else:
        img_scaled = (img / img.max() * 255.0).astype(np.uint8) if img.max() > 0 else img.astype(np.uint8)
        
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(img_scaled)
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=85)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

@app.route("/api/frame_crop")
def frame_crop():
    film, local_cid, local_t = get_actual_film_and_t(request.args)
    exp = request.args.get("experiment")
    channel = request.args.get("channel", "bf")
    
    if local_cid == -1:
        # Blank crop
        img_scaled = np.zeros((100, 100), dtype=np.uint8)
    else:
        csv_path = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}" / f"cell_{local_cid}_masks.csv"
        if not csv_path.exists():
            img_scaled = np.zeros((100, 100), dtype=np.uint8)
        else:
            df = pd.read_csv(csv_path)
            H, W = int(df.iloc[0]['height']), int(df.iloc[0]['width'])
            cy, cx = H // 2, W // 2
            
            if local_t < len(df):
                rle_col = 'rle_bf'
                if 'rle_gfp' in df.columns and df['rle_gfp'].dropna().any():
                    rle_col = 'rle_gfp'
                
                if rle_col in df.columns:
                    rle = df.iloc[local_t][rle_col]
                    if isinstance(rle, str) and rle.strip():
                        mask = rle_decode(rle, (H, W))
                        ys, xs = np.where(mask)
                        if len(ys) > 0:
                            cy, cx = int(np.mean(ys)), int(np.mean(xs))
                        
            frames_dir = BASE_MOVIE_ROOT / exp / film / f"Frames_{film}"
            files = sorted([f for f in frames_dir.glob(f"{film}_t_{local_t:03d}_c_*.tif") if not f.name.startswith(".")])
            if not files: files = sorted([f for f in frames_dir.glob(f"*_t_{local_t:03d}_c_*.tif") if not f.name.startswith(".")])
            if files:
                img = imread(str(files[0]))
                crop_size = 100
                y0 = max(0, cy - crop_size // 2); y1 = min(H, cy + crop_size // 2)
                x0 = max(0, cx - crop_size // 2); x1 = min(W, cx + crop_size // 2)
                crop = img[y0:y1, x0:x1]
                
                p_lo = np.percentile(crop, 1.0); p_hi = np.percentile(crop, 99.5)
                if p_hi > p_lo: img_scaled = np.clip((crop - p_lo) / (p_hi - p_lo) * 255.0, 0, 255).astype(np.uint8)
                else: img_scaled = crop.astype(np.uint8)
            else:
                img_scaled = np.zeros((100, 100), dtype=np.uint8)
        
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(img_scaled)
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=80)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

@app.route("/api/population_frame")
def population_frame():
    film, _, local_t = get_actual_film_and_t(request.args)
    exp = request.args.get("experiment")
    
    cache_dir = BASE_MOVIE_ROOT / exp / film / f"PopulationFrames_{film}"
    cache_file = cache_dir / f"frame_{local_t:03d}.jpg"
    
    if cache_file.exists():
        return send_file(str(cache_file), mimetype='image/jpeg')
        
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        img_data = generate_population_frame_image(exp, film, local_t)
        if img_data is not None:
            with open(cache_file, "wb") as f:
                f.write(img_data)
            img_io = BytesIO(img_data)
            return send_file(img_io, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
    return jsonify({"error": "Failed to generate population frame"}), 404

@app.route("/api/click_segment", methods=["POST"])
def click_segment():
    data = request.json
    exp = data.get("experiment")
    t = int(data.get("t"))
    x, y = int(data.get("x")), int(data.get("y"))
    
    if "sequence" in data:
        seq = data.get("sequence")
        gid = data.get("cell_id")
        film, local_cid, local_t = resolve_global_t(exp, seq, gid, t)
    else:
        film, local_cid, local_t = data.get("film"), int(data.get("cell_id")), t
        
    if local_cid == -1:
        return jsonify({"status": "error", "message": "Cannot select segment for an unassigned cell mapping."})
        
    masks_dir = BASE_MOVIE_ROOT / exp / film / f"Masks_{film}"
    files = sorted([f for f in masks_dir.glob(f"{film}_t_{local_t:03d}_c_*_seg.tif") if not f.name.startswith(".")])
    if not files: files = sorted([f for f in masks_dir.glob(f"*_t_{local_t:03d}_c_*_seg.tif") if not f.name.startswith(".")])
    if not files: return jsonify({"status": "error", "message": "Segmentation file not found"}), 404
        
    seg = load_segmentation(str(files[0]))
    seg_lbl = label(seg) if seg.dtype == bool else seg
    H, W = seg_lbl.shape
    if y >= H or x >= W: return jsonify({"status": "error", "message": "Click coordinates out of range"}), 400
        
    lbl = seg_lbl[y, x]
    if lbl == 0: return jsonify({"status": "success", "rle": ""})
        
    segment_mask = (seg_lbl == lbl)
    rle = rle_encode(segment_mask)
    return jsonify({"status": "success", "rle": rle})

@app.route("/api/save_masks", methods=["POST"])
def save_masks():
    data = request.json
    exp = data.get("experiment")
    cell_id = data.get("cell_id")
    channel = data.get("channel", "bf")
    new_masks = data.get("masks")
    
    # Partial update of suspicious cells cache
    seq = data.get("sequence")
    film_param = data.get("film")
    target = seq if seq else film_param
    if exp and target and cell_id:
        try:
            target_dir = BASE_MOVIE_ROOT / exp / target
            cache_file = target_dir / f"suspicious_{target}.json"
            
            disk_data = {}
            if cache_file.exists():
                import json
                with open(cache_file, "r") as f:
                    disk_data = json.load(f)
                    
            if seq:
                seq_data = get_sequence_linkage_data(exp)
                ensure_pseudo_sequence_cells(exp, seq, seq_data)
                films = seq_data[seq]["films"]
                local_ids = seq_data[seq]["global_cells"].get(str(cell_id), [-1]*len(films))
            else:
                films = [film_param]
                local_ids = [int(cell_id)]
                
            film_dfs = []
            for i, f_name in enumerate(films):
                local_id = local_ids[i] if i < len(local_ids) else -1
                if local_id == -1:
                    film_dfs.append(None)
                    continue
                csv_path = BASE_MOVIE_ROOT / exp / f_name / f"TrackedCells_{f_name}" / f"cell_{local_id}_masks.csv"
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        film_dfs.append(df)
                    except Exception:
                        film_dfs.append(None)
                else:
                    film_dfs.append(None)
                    
            all_rles = []
            H, W = 0, 0
            for i, df in enumerate(film_dfs):
                f_name = films[i]
                L, fW, fH = get_film_frame_count_and_size(exp, f_name)
                if df is not None and len(df) > 0:
                    if H == 0:
                        H, W = int(df.iloc[0]['height']), int(df.iloc[0]['width'])
                    
                    rle_col = 'rle_bf'
                    if 'rle_gfp' in df.columns and any(isinstance(x, str) and x.strip() for x in df['rle_gfp'].dropna()):
                        rle_col = 'rle_gfp'
                    elif 'rle_bf' not in df.columns and 'rle_gfp' in df.columns:
                        rle_col = 'rle_gfp'
                        
                    masks = df[rle_col].fillna("").tolist()
                    if len(masks) < L:
                        masks.extend([""] * (L - len(masks)))
                    elif len(masks) > L:
                        masks = masks[:L]
                    all_rles.extend(masks)
                else:
                    all_rles.extend([""] * L)
                    
            if H > 0 and W > 0:
                centroids = []
                for rle in all_rles:
                    if not isinstance(rle, str) or not rle.strip() or rle == "nan":
                        centroids.append(None)
                        continue
                    try:
                        mask = rle_decode(rle, (H, W))
                        if not mask.any():
                            centroids.append(None)
                        else:
                            ys, xs = np.nonzero(mask)
                            centroids.append((float(np.mean(ys)), float(np.mean(xs))))
                    except Exception:
                        centroids.append(None)
                        
                susp_frames = []
                threshold = 15.0
                for t in range(1, len(centroids)):
                    c1 = centroids[t-1]
                    c2 = centroids[t]
                    if c1 is not None and c2 is not None:
                        dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                        if dist > threshold:
                            susp_frames.append(t)
                            
                if susp_frames:
                    disk_data[str(cell_id)] = susp_frames
                else:
                    if str(cell_id) in disk_data:
                        del disk_data[str(cell_id)]
                        
            target_dir.mkdir(parents=True, exist_ok=True)
            import json
            with open(cache_file, "w") as f:
                json.dump(disk_data, f)
                
            for k in list(SUSPICIOUS_CACHE.keys()):
                if k.startswith(f"{exp}::{target}"):
                    SUSPICIOUS_CACHE[k] = disk_data
                    
        except Exception as e:
            print(f"Error updating suspicious cache: {e}")
    
    if "sequence" in data:
        seq = data.get("sequence")
        seq_data = get_sequence_linkage_data(exp)
        ensure_pseudo_sequence_cells(exp, seq, seq_data)
        films = seq_data[seq]["films"]
        local_ids = seq_data[seq]["global_cells"][cell_id]
        
        current_t = 0
        for i, film in enumerate(films):
            L, _, _ = get_film_frame_count_and_size(exp, film)
            film_masks = new_masks[current_t:current_t+L]
            local_id = local_ids[i]
            
            if local_id != -1:
                csv_path = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}" / f"cell_{local_id}_masks.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    if 'rle_gfp' in df.columns and 'rle_bf' not in df.columns:
                        rle_col = 'rle_gfp'
                    elif 'rle_bf' in df.columns and 'rle_gfp' not in df.columns:
                        rle_col = 'rle_bf'
                    else:
                        rle_col = 'rle_bf' if channel == 'bf' else 'rle_gfp'
                        
                    source_col = 'source_bf' if rle_col == 'rle_bf' else 'source_gfp'
                    area_col = 'area_bf' if rle_col == 'rle_bf' else 'area_gfp'
                    
                    if rle_col not in df.columns:
                        df[rle_col] = ""
                    if source_col not in df.columns:
                        df[source_col] = ""
                    
                    if len(df) > len(film_masks):
                        film_masks.extend([""] * (len(df) - len(film_masks)))
                    elif len(df) < len(film_masks):
                        film_masks = film_masks[:len(df)]
                        
                    any_modified = False
                    modified_t_indices = []
                    for t in range(len(df)):
                        old_rle = df.loc[t, rle_col] if pd.notna(df.loc[t, rle_col]) else ""
                        new_rle = film_masks[t] if film_masks[t] is not None else ""
                        if old_rle != new_rle:
                            df.loc[t, rle_col] = new_rle
                            df.loc[t, source_col] = "manual"
                            any_modified = True
                            modified_t_indices.append(t)
                            
                        rle = film_masks[t]
                        H, W = int(df.iloc[t]['height']), int(df.iloc[t]['width'])
                        if isinstance(rle, str) and rle.strip():
                            mask = rle_decode(rle, (H, W))
                            area = int(mask.sum())
                        else:
                            area = 0
                        if area_col in df.columns: df.loc[t, area_col] = area
                    df.to_csv(csv_path, index=False)
                    
                    if any_modified:
                        for t_idx in modified_t_indices:
                            cache_dir = BASE_MOVIE_ROOT / exp / film / f"PopulationFrames_{film}"
                            cache_file = cache_dir / f"frame_{t_idx:03d}.jpg"
                            if cache_file.exists():
                                try:
                                    cache_file.unlink()
                                except Exception:
                                    pass
                            try:
                                img_data = generate_population_frame_image(exp, film, t_idx)
                                if img_data is not None:
                                    with open(cache_file, "wb") as f:
                                        f.write(img_data)
                            except Exception:
                                pass
                            
            current_t += L
            
    else:
        film = data.get("film")
        csv_path = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}" / f"cell_{cell_id}_masks.csv"
        df = pd.read_csv(csv_path)
        
        if 'rle_gfp' in df.columns and 'rle_bf' not in df.columns:
            rle_col = 'rle_gfp'
        elif 'rle_bf' in df.columns and 'rle_gfp' not in df.columns:
            rle_col = 'rle_bf'
        else:
            rle_col = 'rle_bf' if channel == 'bf' else 'rle_gfp'
            
        source_col = 'source_bf' if rle_col == 'rle_bf' else 'source_gfp'
        area_col = 'area_bf' if rle_col == 'rle_bf' else 'area_gfp'
        
        if rle_col not in df.columns:
            df[rle_col] = ""
        if source_col not in df.columns:
            df[source_col] = ""
        
        any_modified = False
        modified_t_indices = []
        for t in range(len(df)):
            if t < len(new_masks):
                old_rle = df.loc[t, rle_col] if pd.notna(df.loc[t, rle_col]) else ""
                new_rle = new_masks[t] if new_masks[t] is not None else ""
                if old_rle != new_rle:
                    df.loc[t, rle_col] = new_rle
                    df.loc[t, source_col] = "manual"
                    any_modified = True
                    modified_t_indices.append(t)
                
                rle = new_masks[t]
                H, W = int(df.iloc[t]['height']), int(df.iloc[t]['width'])
                if isinstance(rle, str) and rle.strip():
                    mask = rle_decode(rle, (H, W))
                    area = int(mask.sum())
                else:
                    area = 0
                if area_col in df.columns: df.loc[t, area_col] = area
        df.to_csv(csv_path, index=False)
        
        if any_modified:
            for t_idx in modified_t_indices:
                cache_dir = BASE_MOVIE_ROOT / exp / film / f"PopulationFrames_{film}"
                cache_file = cache_dir / f"frame_{t_idx:03d}.jpg"
                if cache_file.exists():
                    try:
                        cache_file.unlink()
                    except Exception:
                        pass
                try:
                    img_data = generate_population_frame_image(exp, film, t_idx)
                    if img_data is not None:
                        with open(cache_file, "wb") as f:
                            f.write(img_data)
                except Exception:
                    pass
        
    return jsonify({"status": "success"})
@app.route("/api/auto_fix_segments", methods=["POST"])
def auto_fix_segments():
    data = request.json
    exp = data.get("experiment")
    start_t = int(data.get("start_t"))
    end_t = int(data.get("end_t"))
    
    fixed_count = 0
    modified_dfs = {}
    
    for t in range(start_t, end_t + 1):
        frame_data = dict(data)
        frame_data["t"] = t
        film, local_cid, local_t = get_actual_film_and_t(frame_data)
        
        if local_cid == -1:
            continue
            
        csv_path = BASE_MOVIE_ROOT / exp / film / f"TrackedCells_{film}" / f"cell_{local_cid}_masks.csv"
        
        if csv_path not in modified_dfs:
            if not csv_path.exists():
                continue
            modified_dfs[csv_path] = pd.read_csv(csv_path)
            
        df = modified_dfs[csv_path]
        
        if local_t >= len(df):
            continue
            
        H, W = int(df.iloc[0]['height']), int(df.iloc[0]['width'])
        
        if 'rle_gfp' in df.columns and 'rle_bf' not in df.columns:
            rle_col = 'rle_gfp'
            source_col = 'source_gfp'
        elif 'rle_bf' in df.columns and 'rle_gfp' not in df.columns:
            rle_col = 'rle_bf'
            source_col = 'source_bf'
        else:
            rle_col = 'rle_bf'
            source_col = 'source_bf'
            if 'rle_gfp' in df.columns and df['rle_gfp'].dropna().any():
                rle_col = 'rle_gfp'
                source_col = 'source_gfp'
            
        if source_col not in df.columns:
            df[source_col] = ""
            
        if rle_col not in df.columns:
            continue
            
        existing_rle = df.loc[local_t, rle_col]
        if not isinstance(existing_rle, str) or not str(existing_rle).strip() or str(existing_rle) == "nan":
            continue
            
        existing_mask = rle_decode(str(existing_rle), (H, W))
        if not existing_mask.any():
            continue
            
        masks_dir = BASE_MOVIE_ROOT / exp / film / f"Masks_{film}"
        files = sorted([f for f in masks_dir.glob(f"{film}_t_{local_t:03d}_c_*_seg.tif") if not f.name.startswith(".")])
        if not files: files = sorted([f for f in masks_dir.glob(f"*_t_{local_t:03d}_c_*_seg.tif") if not f.name.startswith(".")])
        if not files:
            continue
            
        raw_seg = imread(str(files[0]))
        
        overlapping_labels, counts = np.unique(raw_seg[existing_mask], return_counts=True)
        
        selected_labels = []
        best_label = 0
        max_iou = 0.0
        existing_area = existing_mask.sum()
        
        for label, count in zip(overlapping_labels, counts):
            if label == 0: continue
            raw_area = np.sum(raw_seg == label)
            coverage = count / raw_area
            iou = count / (existing_area + raw_area - count)
            
            if coverage >= 0.4:
                selected_labels.append(label)
                
            if iou > max_iou:
                max_iou = iou
                best_label = label
                
        if not selected_labels and best_label > 0:
            selected_labels.append(best_label)
            
        if selected_labels:
            new_mask = np.isin(raw_seg, selected_labels)
            new_rle = rle_encode(new_mask)
            df.loc[local_t, rle_col] = new_rle
            df.loc[local_t, source_col] = "manual"
            area_col = 'area_bf' if rle_col == 'rle_bf' else 'area_gfp'
            if area_col in df.columns: df.loc[local_t, area_col] = int(new_mask.sum())
            fixed_count += 1
            
            # Invalidate and regenerate population cache frame
            cache_dir = BASE_MOVIE_ROOT / exp / film / f"PopulationFrames_{film}"
            cache_file = cache_dir / f"frame_{local_t:03d}.jpg"
            if cache_file.exists():
                try:
                    cache_file.unlink()
                except Exception:
                    pass
            try:
                img_data = generate_population_frame_image(exp, film, local_t)
                if img_data is not None:
                    with open(cache_file, "wb") as f:
                        f.write(img_data)
            except Exception:
                pass
            
    for csv_path, df in modified_dfs.items():
        df.to_csv(csv_path, index=False)
        
    return jsonify({"status": "success", "fixed_count": fixed_count})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fungal Cell Tracking Corrector")
    parser.add_argument("--sync-nas", type=str, nargs='?', const='all',
                        help="Sync data from NAS to local SSD. Can specify a specific experiment name (e.g. 2026_01_08_M93) or leave blank/'all' to sync everything.")
    parser.add_argument("--nas-root", type=str, default=str(NAS_MOVIE_ROOT),
                        help="Path to the NAS movie directory")
    parser.add_argument("--local-root", type=str, default=str(BASE_MOVIE_ROOT),
                        help="Path to the local SSD movie directory")
    args = parser.parse_args()
    
    BASE_MOVIE_ROOT = Path(args.local_root)
    NAS_MOVIE_ROOT = Path(args.nas_root)
    
    if args.sync_nas:
        print("🔄 Initiating NAS to Local SSD Sync (Pull)...")
        local_path = Path(args.local_root)
        nas_path = Path(args.nas_root)
        
        local_path.mkdir(parents=True, exist_ok=True)
        
        sync_list = RELEVANT_EXPERIMENTS if args.sync_nas == 'all' else [args.sync_nas]
        pull_errors = False
        
        for exp in sync_list:
            src = str(nas_path / exp) + "/"
            dst = str(local_path / exp) + "/"
            
            print(f"Pulling from NAS: {src} -> Local SSD: {dst}")
            
            if not os.path.exists(src.rstrip("/")):
                print(f"⚠️ Warning: NAS directory '{src}' does not exist. Skipping.")
                continue
                
            local_path_exp = local_path / exp
            local_path_exp.mkdir(parents=True, exist_ok=True)
            
            rsync_cmd = f"rsync -avz --update --exclude='__pycache__' --exclude='*.ims' '{src}' '{dst}'"
            try:
                subprocess.run(rsync_cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ Pull Sync Failed for {exp}: {e}")
                pull_errors = True
                
        if pull_errors:
            print("⚠️ Pull Sync finished with errors.")
            sys.exit(1)
        else:
            print("✅ Pull Sync Completed Successfully!")
            
    port = 5001
    print(f"🚀 Starting Corrector Tool at http://127.0.0.1:{port}")
    
    try:
        app.run(host="0.0.0.0", port=port, debug=False)
    finally:
        if args.sync_nas:
            print("\n🔄 Initiating Local SSD to NAS Sync (Push)...")
            local_path = Path(args.local_root)
            nas_path = Path(args.nas_root)
            
            sync_list = RELEVANT_EXPERIMENTS if args.sync_nas == 'all' else [args.sync_nas]
            
            for exp in sync_list:
                src = str(local_path / exp) + "/"
                dst = str(nas_path / exp) + "/"
                
                if not os.path.exists(src.rstrip("/")):
                    continue
                    
                print(f"Pushing from Local SSD: {src} -> NAS: {dst}")
                
                nas_path_exp = nas_path / exp
                nas_path_exp.mkdir(parents=True, exist_ok=True)
                
                rsync_cmd = f"rsync -avz --update --exclude='__pycache__' --exclude='*.ims' '{src}' '{dst}'"
                try:
                    subprocess.run(rsync_cmd, shell=True, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"❌ Push Sync Failed for {exp}: {e}")
            print("✅ Push Sync Completed Successfully!")
