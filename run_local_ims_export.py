#!/usr/bin/env python3
import os
import subprocess

# Directories to process on the SSD
WORK_DIRS = [
    "/Volumes/X10 Pro/Movies/2026_04_23_M130",
    "/Volumes/X10 Pro/Movies/2026_04_29_M133",
    "/Volumes/X10 Pro/Movies/2026_04_30_M135"
]

# Path to the existing export script
EXPORT_SCRIPT = "/Users/user/Documents/Python_Scripts/FungalProjectScript/batch_ims_export.py"

def run_export(working_dir):
    print(f"\n🚀 Starting export for: {working_dir}")
    # We'll use a temporary environment variable or just modify the script temporarily
    # But a cleaner way is to use a sed command to update the hardcoded path in a temporary file
    temp_script = "/tmp/temp_export.py"
    
    with open(EXPORT_SCRIPT, 'r') as f:
        lines = f.readlines()
    
    with open(temp_script, 'w') as f:
        for line in lines:
            if line.strip().startswith('working_dir ='):
                f.write(f'working_dir = "{working_dir}/"\n')
            else:
                f.write(line)
    
    # Run the modified script using the dedicated ims_env
    python_exe = "/Users/user/Documents/Python_Scripts/FungalProjectScript/ims_env/bin/python3"
    subprocess.run([python_exe, temp_script], check=True)
    print(f"✅ Finished export for: {working_dir}")

if __name__ == "__main__":
    for wd in WORK_DIRS:
        if os.path.isdir(wd):
            run_export(wd)
        else:
            print(f"⚠️ Warning: Directory not found: {wd}")
