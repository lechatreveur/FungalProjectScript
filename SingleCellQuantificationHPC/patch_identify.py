import re

file_path = "/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellQuantificationHPC/manual_correction_tool.py"
with open(file_path, "r") as f:
    content = f.read()

old_code = """                rle_col = 'rle_bf'
                if 'rle_gfp' in df.columns and df['rle_gfp'].dropna().any():
                    rle_col = 'rle_gfp'
                    
                rle = df.iloc[t][rle_col]
                if isinstance(rle, str) and rle.strip():
                    W = int(df.iloc[0]['width'])
                    H = int(df.iloc[0]['height'])
                    mask = rle_decode(rle, (H, W))
                    if y < H and x < W and mask[y, x]:
                        return jsonify({"status": "success", "cell_id": cid})"""

new_code = """                # Check both masks
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
                    return jsonify({"status": "success", "cell_id": cid})"""

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(file_path, "w") as f:
        f.write(content)
    print("Patched identify_cell successfully!")
else:
    print("Could not find old code in identify_cell.")
