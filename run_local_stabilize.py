#!/usr/bin/env python3
import os
import subprocess

# Directories to process on the SSD
WORK_DIRS = [
    "/Volumes/X10 Pro/Movies/2026_04_23_M130",
    "/Volumes/X10 Pro/Movies/2026_04_29_M133",
    "/Volumes/X10 Pro/Movies/2026_04_30_M135"
]

# Path to the stabilization script
STABILIZE_SCRIPT = "/Users/user/Documents/Python_Scripts/FungalProjectScript/stabilize_in_place.py"

def run_stabilize(working_dir):
    print(f"\n🌀 Starting stabilization for: {working_dir}")
    # Use the ims_env python for consistency (it has cv2 and numpy)
    python_exe = "/Users/user/Documents/Python_Scripts/FungalProjectScript/ims_env/bin/python3"
    subprocess.run([python_exe, STABILIZE_SCRIPT, working_dir], check=True)
    print(f"✅ Finished stabilization for: {working_dir}")

if __name__ == "__main__":
    for wd in WORK_DIRS:
        if os.path.isdir(wd):
            run_stabilize(wd)
        else:
            print(f"⚠️ Warning: Directory not found: {wd}")
