#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure SingleCellQuantificationHPC is on sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from tracking_corrector.app import create_app as create_tc_app
from septum_alignment_board.app import create_app as create_board_app

def test_label_persistence():
    print("Initializing tracking_corrector app...")
    tc_app = create_tc_app()
    tc_client = tc_app.test_client()

    print("Initializing septum_alignment_board app...")
    board_app = create_board_app()
    board_client = board_app.test_client()

    exp = "2026_07_16_M156"
    seq = "3_F0"
    
    # 1. Fetch cell list from board_client
    res_cells = board_client.get(f"/api/list_cells?experiment={exp}&sequence={seq}")
    assert res_cells.status_code == 200, "Failed to list cells"
    cells = res_cells.get_json().get("cells", [])
    assert len(cells) > 0, "No cells found in sequence"
    
    cell = cells[0]
    global_cell_id = cell["global_id"]
    film = cell["film"]
    cell_id = cell["cell_id"]
    print(f"Testing global cell: {global_cell_id} (film: {film}, local_cell_id: {cell_id})")

    # 2. Read initial state via tracking_corrector
    tc_read_init = tc_client.get(f"/api/get_septum_label?experiment={exp}&film={film}&cell_id={cell_id}")
    assert tc_read_init.status_code == 200
    init_data = tc_read_init.get_json().get("data", {})
    orig_has1 = init_data.get("has_septum", False)
    orig_start1 = init_data.get("start_aligned")
    orig_end1 = init_data.get("end_aligned")
    orig_has2 = init_data.get("has_septum_2", False)
    orig_start2 = init_data.get("start_aligned_2")
    orig_end2 = init_data.get("end_aligned_2")

    print(f"Initial state in tracking_corrector: Septum1={orig_has1} ({orig_start1}-{orig_end1}), Septum2={orig_has2} ({orig_start2}-{orig_end2})")

    # 3. Save test interval via septum_alignment_board
    test_start1 = 15
    test_end1 = 25
    test_start2 = 40
    test_end2 = 50

    save_payload = {
        "experiment": exp,
        "film": film,
        "cell_id": str(cell_id),
        "sequence": seq,
        "global_cell_id": str(global_cell_id),
        "has_septum": True,
        "start_aligned": test_start1,
        "end_aligned": test_end1,
        "has_septum_2": True,
        "start_aligned_2": test_start2,
        "end_aligned_2": test_end2
    }

    print("Saving new intervals via septum_alignment_board POST /api/save_septum_label...")
    save_res = board_client.post("/api/save_septum_label", json=save_payload)
    assert save_res.status_code == 200, f"Save failed: {save_res.get_json()}"
    assert save_res.get_json().get("status") == "success"

    # 4. Read back label via tracking_corrector and verify exact match
    print("Reading back label via tracking_corrector GET /api/get_septum_label...")
    tc_read_after = tc_client.get(f"/api/get_septum_label?experiment={exp}&film={film}&cell_id={cell_id}")
    assert tc_read_after.status_code == 200
    after_data = tc_read_after.get_json().get("data", {})

    print(f"Read-back data from tracking_corrector: {after_data}")
    assert after_data.get("has_septum") == True
    assert after_data.get("start_aligned") == test_start1
    assert after_data.get("end_aligned") == test_end1
    assert after_data.get("has_septum_2") == True
    assert after_data.get("start_aligned_2") == test_start2
    assert after_data.get("end_aligned_2") == test_end2

    # 5. Restore original state
    print("Restoring original label state...")
    restore_payload = {
        "experiment": exp,
        "film": film,
        "cell_id": str(cell_id),
        "sequence": seq,
        "global_cell_id": str(global_cell_id),
        "has_septum": orig_has1,
        "start_aligned": orig_start1,
        "end_aligned": orig_end1,
        "has_septum_2": orig_has2,
        "start_aligned_2": orig_start2,
        "end_aligned_2": orig_end2
    }
    tc_client.post("/api/save_septum_label", json=restore_payload)

    print("\nSUCCESS! Inter-tool labeling persistence verified 100%!")

if __name__ == "__main__":
    test_label_persistence()
