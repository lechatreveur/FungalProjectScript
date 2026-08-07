#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure SingleCellQuantificationHPC is on sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from tracking_corrector.app import create_app as create_tc_app
from septum_alignment_board.app import create_app as create_board_app

def test_data_reuse():
    print("Initializing tracking_corrector app...")
    tc_app = create_tc_app()
    tc_client = tc_app.test_client()

    print("Initializing septum_alignment_board app...")
    board_app = create_board_app()
    board_client = board_app.test_client()

    exp = "2026_07_16_M156"
    seq = "3_F0"
    cell_id = "3_F0_cell_1"

    print(f"\nTesting sequence cell listing for experiment={exp}, sequence={seq}...")
    res_list = board_client.get(f"/api/list_cells?experiment={exp}&sequence={seq}")
    assert res_list.status_code == 200, f"list_cells failed with status {res_list.status_code}"
    cells_data = res_list.get_json()
    print(f"Total cells returned by septum_alignment_board: {len(cells_data.get('cells', []))}")
    assert len(cells_data.get("cells", [])) > 0, "No cells returned for sequence!"

    print(f"\nRequesting cell strip image for cell_id={cell_id} from tracking_corrector...")
    res_tc = tc_client.get(f"/api/cell_strip_image?experiment={exp}&sequence={seq}&cell_id={cell_id}&channel=bf")
    assert res_tc.status_code == 200, f"tc cell_strip_image failed: {res_tc.status_code}"
    bytes_tc = res_tc.data

    print(f"Requesting cell strip image for cell_id={cell_id} from septum_alignment_board...")
    res_board = board_client.get(f"/api/cell_strip_image?experiment={exp}&sequence={seq}&cell_id={cell_id}&channel=bf")
    assert res_board.status_code == 200, f"board cell_strip_image failed: {res_board.status_code}"
    bytes_board = res_board.data

    print(f"tc strip image byte length: {len(bytes_tc)}")
    print(f"board strip image byte length: {len(bytes_board)}")

    assert bytes_tc == bytes_board, "Error! Strip image outputs are NOT byte-identical!"
    print("\nSUCCESS! The strip images from tracking_corrector and septum_alignment_board are 100% byte-identical!")

if __name__ == "__main__":
    test_data_reuse()
