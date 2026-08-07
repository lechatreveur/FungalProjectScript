#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure SingleCellQuantificationHPC is on sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from septum_alignment_board.app import create_app

def test_ai_overlay_endpoints():
    print("Initializing septum_alignment_board app...")
    app = create_app()
    client = app.test_client()

    exp = "2026_07_16_M156"
    seq = "3_F0"

    print(f"\nListing sequence cells for {exp}/{seq}...")
    res = client.get(f"/api/list_cells?experiment={exp}&sequence={seq}")
    assert res.status_code == 200
    cells = res.get_json().get("cells", [])
    assert len(cells) > 0, "No cells found"

    cell = cells[0]
    global_id = cell["global_id"]
    print(f"Testing global cell: {global_id}")

    # 1. Test get_septum_ai_cache
    print(f"\nTesting GET /api/get_septum_ai_cache for global_cell_id={global_id}...")
    ai_res = client.get(f"/api/get_septum_ai_cache?experiment={exp}&sequence={seq}&global_cell_id={global_id}")
    assert ai_res.status_code == 200, f"ai_cache endpoint failed: {ai_res.status_code}"
    ai_json = ai_res.get_json()
    print(f"AI Cache Response: status={ai_json.get('status')}, cached={ai_json.get('cached')}")
    assert ai_json.get("status") == "success"

    # 2. Test get_septum_ai_cache with non-existent cell
    fake_res = client.get(f"/api/get_septum_ai_cache?experiment={exp}&sequence={seq}&global_cell_id=nonexistent_cell_xyz")
    assert fake_res.status_code == 200
    assert fake_res.get_json().get("cached") == False, "Nonexistent cell should return cached: false"
    print("Verified non-existent cell returns cached: false cleanly.")

    print("\nSUCCESS! AI Overlay endpoints verified successfully!")

if __name__ == "__main__":
    test_ai_overlay_endpoints()
