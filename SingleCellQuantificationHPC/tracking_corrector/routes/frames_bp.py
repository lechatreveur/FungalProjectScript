import io
import re
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Blueprint, jsonify, request, Response, current_app
from skimage.measure import label
from skimage.io import imread
from PIL import Image

logger = logging.getLogger(__name__)

def _parse_cell_id_num(cell_id_str: str) -> int:
    if not cell_id_str:
        return -1
    s = str(cell_id_str).strip()
    if "_cell_" in s:
        s = s.split("_cell_")[-1]
    elif "_" in s:
        s = s.split("_")[-1]
    try:
        return int(s)
    except ValueError:
        return -1

def _create_placeholder_pil(size: int = 100) -> Image.Image:
    """Create a high-contrast placeholder (bright red/magenta pattern) for missing/failed frame decodes."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, :, 0] = 220  # Red
    arr[:, :, 2] = 200  # Magenta
    arr[size//2 - 1:size//2 + 1, :, :] = 255
    arr[:, size//2 - 1:size//2 + 1, :] = 255
    return Image.fromarray(arr)

frames_bp = Blueprint("frames", __name__)

def get_film_frame_count_and_size(base_root: Path, exp: str, film: str):
    tracked_dir = base_root / exp / film / f"TrackedCells_{film}"
    if tracked_dir.exists():
        for cf in tracked_dir.iterdir():
            if cf.name.startswith("."): continue
            if cf.name.endswith("_masks.csv"):
                try:
                    df = pd.read_csv(cf)
                    if not df.empty and "width" in df.columns:
                        return len(df), int(df.iloc[0]["width"]), int(df.iloc[0]["height"])
                except Exception:
                    continue
    frames_dir = base_root / exp / film / f"Frames_{film}"
    if frames_dir.exists():
        files = [f for f in frames_dir.glob("*.tif") if not f.name.startswith(".") and "_seg" not in f.stem]
        if files:
            try:
                img = imread(str(files[0]))
                return len(files), img.shape[1], img.shape[0]
            except Exception:
                return len(files), 512, 512
    return 0, 512, 512

def resolve_global_t(exp: str, sequence: str, global_cell_id: str, global_t: int, base_root: Path):
    from ..services.linkage_service import LinkageService
    from ..repositories.linkage_repository import LinkageRepository
    from ..services.audit_service import AuditService
    
    repo = LinkageRepository(base_root)
    audit = AuditService(base_root)
    linkage_svc = LinkageService(repo, audit)
    seq_res = linkage_svc.get_sequences(exp)
    sequences = seq_res.get("sequences", {})
    
    if sequence not in sequences:
        return sequence, global_cell_id, global_t
        
    seq_info = sequences[sequence]
    films = seq_info.get("films", [sequence])
    global_cells = seq_info.get("global_cells", {})
    local_ids = global_cells.get(str(global_cell_id), [-1] * len(films))
    
    current_t = 0
    for i, film in enumerate(films):
        local_id = local_ids[i] if i < len(local_ids) else -1
        L, W, H = get_film_frame_count_and_size(base_root, exp, film)
        if global_t < current_t + L:
            return film, local_id, global_t - current_t
        current_t += L
        
    return films[-1], local_ids[-1] if local_ids else -1, max(0, global_t - current_t)

def get_actual_film_and_t(args, base_root):
    exp = args.get("experiment")
    t_str = args.get("t", "0")
    try:
        t = int(t_str)
    except ValueError:
        t = 0
        
    cell_id = args.get("cell_id", "")
    
    if "sequence" in args and args.get("sequence"):
        seq = args.get("sequence")
        return resolve_global_t(exp, seq, cell_id, t, base_root)
        
    film = args.get("film") or args.get("sequence")
    try:
        cid = int(cell_id) if cell_id else -1
    except ValueError:
        cid = -1
    return film, cid, t

@frames_bp.route("/api/frame_image", methods=["GET"])
def get_frame_image():
    exp = request.args.get("experiment")
    if not exp:
        return jsonify({"status": "error", "message": "experiment parameter required"}), 400
        
    base_root = current_app.config["APP_CONFIG"].local_movie_root
    film, _, local_t = get_actual_film_and_t(request.args, base_root)
    if not film:
        return jsonify({"status": "error", "message": "film or sequence required"}), 400
        
    frames_service = current_app.config["FRAMES_SERVICE"]
    jpeg_bytes = frames_service.render_frame_jpeg(exp, film, local_t)
    return Response(jpeg_bytes, mimetype="image/jpeg")

@frames_bp.route("/api/population_frame", methods=["GET"])
def get_population_frame():
    exp = request.args.get("experiment")
    if not exp:
        return jsonify({"status": "error", "message": "experiment parameter required"}), 400
        
    base_root = current_app.config["APP_CONFIG"].local_movie_root
    film, _, local_t = get_actual_film_and_t(request.args, base_root)
    if not film:
        return jsonify({"status": "error", "message": "film or sequence required"}), 400
        
    frames_service = current_app.config["FRAMES_SERVICE"]
    jpeg_bytes = frames_service.render_population_frame_jpeg(exp, film, local_t)
    return Response(jpeg_bytes, mimetype="image/jpeg")

@frames_bp.route("/api/frame_crop", methods=["GET"])
def get_frame_crop():
    exp = request.args.get("experiment")
    if not exp:
        return jsonify({"status": "error", "message": "experiment required"}), 400
        
    base_root = current_app.config["APP_CONFIG"].local_movie_root
    film, local_cid, local_t = get_actual_film_and_t(request.args, base_root)
    
    if not film or local_cid == -1:
        img_scaled = np.zeros((100, 100), dtype=np.uint8)
    else:
        csv_path = base_root / exp / film / f"TrackedCells_{film}" / f"cell_{local_cid}_masks.csv"
        if not csv_path.exists():
            img_scaled = np.zeros((100, 100), dtype=np.uint8)
        else:
            try:
                df = pd.read_csv(csv_path)
                H, W = int(df.iloc[0]['height']), int(df.iloc[0]['width'])
                cy, cx = H // 2, W // 2
                
                if local_t < len(df):
                    rle_col = 'rle_bf'
                    if 'rle_gfp' in df.columns and df['rle_gfp'].dropna().any():
                        rle_col = 'rle_gfp'
                    
                    if rle_col in df.columns:
                        rle = df.iloc[local_t][rle_col]
                        if isinstance(rle, str) and rle.strip() and rle.lower() != 'nan':
                            from ..schemas import validate_and_decode_rle
                            mask = validate_and_decode_rle(rle, H, W)
                            ys, xs = np.where(mask > 0)
                            if len(ys) > 0:
                                cy, cx = int(np.mean(ys)), int(np.mean(xs))
                            
                frames_dir = base_root / exp / film / f"Frames_{film}"
                files = sorted([f for f in frames_dir.glob(f"{film}_t_{local_t:03d}_c_*.tif") if not f.name.startswith(".")])
                if not files:
                    files = sorted([f for f in frames_dir.glob(f"*_t_{local_t:03d}_c_*.tif") if not f.name.startswith(".")])
                    
                if files:
                    img = imread(str(files[0]))
                    crop_size = 100
                    y0 = max(0, cy - crop_size // 2); y1 = min(H, cy + crop_size // 2)
                    x0 = max(0, cx - crop_size // 2); x1 = min(W, cx + crop_size // 2)
                    crop = img[y0:y1, x0:x1]
                    
                    p_lo = np.percentile(crop, 1.0) if crop.size > 0 else 0.0
                    p_hi = np.percentile(crop, 99.5) if crop.size > 0 else 255.0
                    if p_hi > p_lo:
                        img_scaled = np.clip((crop - p_lo) / (p_hi - p_lo) * 255.0, 0, 255).astype(np.uint8)
                    else:
                        img_scaled = crop.astype(np.uint8)
                else:
                    img_scaled = np.zeros((100, 100), dtype=np.uint8)
            except Exception:
                img_scaled = np.zeros((100, 100), dtype=np.uint8)
                
    pil_img = Image.fromarray(img_scaled)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return Response(buf.getvalue(), mimetype="image/jpeg")

@frames_bp.route("/api/frame_boundaries", methods=["GET"])
def frame_boundaries():
    exp = request.args.get("experiment")
    if not exp:
        return jsonify({"status": "error", "message": "experiment required"}), 400
        
    base_root = current_app.config["APP_CONFIG"].local_movie_root
    film, _, local_t = get_actual_film_and_t(request.args, base_root)
    if not film:
        return jsonify({"status": "error", "message": "film or sequence required"}), 400
        
    masks_dir = base_root / exp / film / f"Masks_{film}"
    files = sorted([f for f in masks_dir.glob(f"{film}_t_{local_t:03d}_c_*_seg.tif") if not f.name.startswith(".")])
    if not files:
        files = sorted([f for f in masks_dir.glob(f"*_t_{local_t:03d}_c_*_seg.tif") if not f.name.startswith(".")])
        
    if not files:
        return jsonify({"error": "No segment file found"}), 404
        
    try:
        seg = imread(str(files[0]))
        seg_lbl = (label(seg) if seg.dtype == bool else seg).copy()
        
        tracked_dir = base_root / exp / film / f"TrackedCells_{film}"
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
                                    if isinstance(rle, str) and rle.strip() and rle.lower() != 'nan':
                                        from ..schemas import validate_and_decode_rle
                                        mask = validate_and_decode_rle(rle, H, W)
                                        seg_lbl[mask > 0] = next_lbl
                                        next_lbl += 1
                    except Exception:
                        pass
                        
        if request.args.get("format") == "json" or request.args.get("json") == "1":
            import cv2
            contours, _ = cv2.findContours((seg_lbl > 0).astype("uint8"), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            simplified = []
            for c in contours:
                epsilon = 0.005 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, epsilon, True)
                if approx is not None:
                    pts = approx.squeeze().tolist()
                    if isinstance(pts, list) and len(pts) > 0:
                        simplified.append(pts)
            return jsonify({
                "status": "success",
                "width": int(seg_lbl.shape[1]),
                "height": int(seg_lbl.shape[0]),
                "contours": simplified
            })

        from skimage.segmentation import find_boundaries
        from scipy.ndimage import binary_dilation
        import numpy as np
        from PIL import Image as PILImage
        import io

        boundaries = find_boundaries(seg_lbl, mode='outer')
        thick_boundaries = binary_dilation(boundaries, structure=np.ones((3, 3)))
        H, W = seg_lbl.shape
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[thick_boundaries] = [234, 179, 8, 140]
        
        pil_img = PILImage.fromarray(rgba, 'RGBA')
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return Response(buf.getvalue(), mimetype="image/png")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@frames_bp.route("/api/cell_strip_image", methods=["GET"])
def get_cell_strip_image():
    exp = request.args.get("experiment")
    if not exp:
        return jsonify({"status": "error", "message": "experiment required"}), 400
        
    cell_id = request.args.get("cell_id", "")
    channel = request.args.get("channel", "bf")
    
    base_root = current_app.config["APP_CONFIG"].local_movie_root
    frames_service = current_app.config["FRAMES_SERVICE"]
    
    frames_info = []
    
    if "sequence" in request.args and request.args.get("sequence"):
        seq = request.args.get("sequence")
        from ..services.linkage_service import LinkageService
        from ..repositories.linkage_repository import LinkageRepository
        from ..services.audit_service import AuditService
        
        repo = LinkageRepository(base_root)
        audit = AuditService(base_root)
        linkage_svc = LinkageService(repo, audit)
        seq_res = linkage_svc.get_sequences(exp)
        sequences = seq_res.get("sequences", {})
        
        matched_seq = None
        if seq in sequences:
            matched_seq = seq
        else:
            for k in sequences:
                if k == seq or k.endswith(f"_{seq}") or k.endswith(f"_{seq.lstrip('3_')}"):
                    matched_seq = k
                    break

        if matched_seq:
            seq_info = sequences[matched_seq]
            films = seq_info.get("films", [matched_seq])
            global_cells = seq_info.get("global_cells", {})
            
            # Lookup cell_id directly or by numerical suffix
            cell_key = str(cell_id)
            if cell_key not in global_cells:
                num = _parse_cell_id_num(cell_key)
                for gk in global_cells:
                    if _parse_cell_id_num(gk) == num:
                        cell_key = gk
                        break

            local_ids = global_cells.get(cell_key, [_parse_cell_id_num(cell_id)] * len(films))
            
            for i, film in enumerate(films):
                local_id = local_ids[i] if i < len(local_ids) else -1
                L, _, _ = get_film_frame_count_and_size(base_root, exp, film)
                for local_t in range(L):
                    frames_info.append((film, local_id, local_t))
        else:
            film = seq
            cid = _parse_cell_id_num(cell_id)
            L, _, _ = get_film_frame_count_and_size(base_root, exp, film)
            for local_t in range(L):
                frames_info.append((film, cid, local_t))
    else:
        film = request.args.get("film")
        cid = _parse_cell_id_num(cell_id)
        if film:
            L, _, _ = get_film_frame_count_and_size(base_root, exp, film)
            for local_t in range(L):
                frames_info.append((film, cid, local_t))
            
    num_frames = len(frames_info)
    if num_frames == 0:
        logger.warning(f"⚠️ No frames found for exp='{exp}', sequence='{request.args.get('sequence')}', film='{request.args.get('film')}'")
        img = _create_placeholder_pil(100)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return Response(buf.getvalue(), mimetype="image/jpeg")
        
    strip_img = Image.new('RGB', (num_frames * 100, 100), (0, 0, 0))
    for t, (film, local_cid, local_t) in enumerate(frames_info):
        try:
            if local_cid == -1:
                logger.warning(f"⚠️ Unresolved cell_id '{cell_id}' for film '{film}' at frame {local_t}")
                crop_img = _create_placeholder_pil(100)
            else:
                crop_bytes = frames_service.render_frame_crop_jpeg(exp, film, local_cid, local_t, channel)
                if crop_bytes:
                    crop_img = Image.open(io.BytesIO(crop_bytes))
                else:
                    logger.error(f"❌ Empty crop bytes returned for {exp}/{film}/cell_{local_cid}/t_{local_t}")
                    crop_img = _create_placeholder_pil(100)
            strip_img.paste(crop_img, (t * 100, 0))
        except Exception as e:
            logger.error(f"❌ Decode/compositing error at frame {t} (film={film}, cid={local_cid}, t={local_t}): {e}", exc_info=True)
            strip_img.paste(_create_placeholder_pil(100), (t * 100, 0))
            
    width = num_frames * 100
    buf = io.BytesIO()
    if width <= 65500:
        strip_img.save(buf, format="JPEG", quality=85)
        mimetype = "image/jpeg"
    else:
        strip_img.save(buf, format="PNG")
        mimetype = "image/png"
    return Response(buf.getvalue(), mimetype=mimetype)
