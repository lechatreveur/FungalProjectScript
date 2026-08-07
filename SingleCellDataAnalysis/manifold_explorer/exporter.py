import os
import json
import base64
import shutil
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def get_plotly_js() -> str:
    try:
        import plotly
        p = os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return f.read()
    except ImportError:
        pass
    return ""

def export_single_html(
    output_path: Path,
    fig_2d: dict,
    fig_3d: dict,
    fig_latent: dict,
    traj_dict: dict,
    color_arrays: dict,
    template_dir: Path
) -> None:
    html_path = template_dir / "viewer.html"
    css_path = template_dir / "style.css"
    js_path = template_dir / "app.js"
    
    with open(html_path, encoding="utf-8") as f:
        html = f.read()
        
    with open(css_path, encoding="utf-8") as f:
        css = f.read()
        
    with open(js_path, encoding="utf-8") as f:
        js = f.read()
        
    # Replace placeholders
    html = html.replace("<!-- STYLE_PLACEHOLDER -->", f"<style>{css}</style>")
    html = html.replace("PLOTLY_JS_PLACEHOLDER", get_plotly_js())
    html = html.replace("PLOT_JSON_3D_PLACEHOLDER", json.dumps(fig_3d))
    html = html.replace("PLOT_JSON_2D_PLACEHOLDER", json.dumps(fig_2d))
    html = html.replace("PLOT_JSON_LATENT_PLACEHOLDER", json.dumps(fig_latent))
    html = html.replace("TRAJ_JSON_PLACEHOLDER", json.dumps(traj_dict))
    html = html.replace("COLOR_ARRAYS_PLACEHOLDER", json.dumps(color_arrays))
    html = html.replace("<!-- SCRIPT_PLACEHOLDER -->", f"<script>{js}</script>")
    
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
        
    logger.info(f"✅ Standalone HTML Dashboard successfully exported to: {output_path} (Size: {os.path.getsize(output_path)/1024/1024:.2f} MB)")

def export_static_site(
    output_dir: Path,
    fig_2d: dict,
    fig_3d: dict,
    fig_latent: dict,
    traj_dict: dict,
    color_arrays: dict,
    template_dir: Path,
    strips_source_dir: Path
) -> None:
    os.makedirs(output_dir / "assets", exist_ok=True)
    os.makedirs(output_dir / "data", exist_ok=True)
    os.makedirs(output_dir / "strips", exist_ok=True)
    
    # 1. Copy viewer.html to index.html and update paths
    html_path = template_dir / "viewer.html"
    with open(html_path, encoding="utf-8") as f:
        html = f.read()
        
    html = html.replace("<!-- STYLE_PLACEHOLDER -->", '<link rel="stylesheet" href="assets/style.css">')
    html = html.replace("PLOTLY_JS_PLACEHOLDER", "")
    html = html.replace("<script>PLOTLY_JS_PLACEHOLDER</script>", '<script src="assets/plotly.min.js"></script>')
    
    # Update script injection to load JSON files instead of embedding them
    inject_js = """
    <script>
      var plotData3D = {};
      var plotData2D = {};
      var plotDataLatent = {};
      var trajData = {};
      var colorArrays = {};
      
      // Load data files asynchronously
      Promise.all([
        fetch('data/fig_3d.json').then(r => r.json()),
        fetch('data/fig_2d.json').then(r => r.json()),
        fetch('data/fig_latent.json').then(r => r.json()),
        fetch('data/traj_data.json').then(r => r.json()),
        fetch('data/color_arrays.json').then(r => r.json())
      ]).then(values => {
        plotData3D = values[0];
        plotData2D = values[1];
        plotDataLatent = values[2];
        trajData = values[3];
        colorArrays = values[4];
        
        // Dynamic image strip URL override in static-site mode
        for (var gid in trajData) {
          if (trajData[gid].strip) {
             trajData[gid].strip = "strips/" + gid + ".webp";
          }
        }
        
        // Trigger initial load event
        var event = new Event('load');
        window.dispatchEvent(event);
      });
    </script>
    """
    html = html.replace("var plotData3D = PLOT_JSON_3D_PLACEHOLDER;", "")
    html = html.replace("var plotData2D = PLOT_JSON_2D_PLACEHOLDER;", "")
    html = html.replace("var plotDataLatent = PLOT_JSON_LATENT_PLACEHOLDER;", "")
    html = html.replace("var trajData = TRAJ_JSON_PLACEHOLDER;", "")
    html = html.replace("var colorArrays = COLOR_ARRAYS_PLACEHOLDER;", "")
    
    html = html.replace("<!-- SCRIPT_PLACEHOLDER -->", inject_js + '<script src="assets/app.js"></script>')
    
    with open(output_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(html)
        
    # 2. Copy CSS, JS, and Plotly
    shutil.copy(template_dir / "style.css", output_dir / "assets" / "style.css")
    shutil.copy(template_dir / "app.js", output_dir / "assets" / "app.js")
    
    plotly_js = get_plotly_js()
    with open(output_dir / "assets" / "plotly.min.js", "w", encoding="utf-8") as f:
        f.write(plotly_js)
        
    # 3. Write data JSON files
    with open(output_dir / "data" / "fig_3d.json", "w", encoding="utf-8") as f:
        json.dump(fig_3d, f)
    with open(output_dir / "data" / "fig_2d.json", "w", encoding="utf-8") as f:
        json.dump(fig_2d, f)
    with open(output_dir / "data" / "fig_latent.json", "w", encoding="utf-8") as f:
        json.dump(fig_latent, f)
    
    # Strip paths in traj_data JSON (to point to external files)
    traj_stripped = {}
    for gid, cell in traj_dict.items():
        cell_copy = cell.copy()
        # Mark that it has a strip, but strip data will load from webp file
        cell_copy["strip"] = "present" if cell.get("strip") else ""
        traj_stripped[gid] = cell_copy
        
    with open(output_dir / "data" / "traj_data.json", "w", encoding="utf-8") as f:
        json.dump(traj_stripped, f)
    with open(output_dir / "data" / "color_arrays.json", "w", encoding="utf-8") as f:
        json.dump(color_arrays, f)
        
    # 4. Copy image strips to strips directory (and convert to WebP or optimize if possible)
    # For now, copy directly or compress.
    copied_count = 0
    for gid in traj_dict.keys():
        src_png = strips_source_dir / f"{gid}.png"
        dest_webp = output_dir / "strips" / f"{gid}.webp"
        if src_png.exists():
            # If PIL is available, save as optimized webp
            try:
                from PIL import Image
                im = Image.open(src_png)
                im.save(dest_webp, "WEBP", quality=80)
            except ImportError:
                shutil.copy(src_png, output_dir / "strips" / f"{gid}.png")
            copied_count += 1
            
    logger.info(f"✅ Static Site successfully exported to: {output_dir}")
    logger.info(f"   Copied and optimized {copied_count} image strips into strips/")
