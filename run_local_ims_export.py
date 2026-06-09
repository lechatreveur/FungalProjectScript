#!/usr/bin/env python3
import os
import sys
import subprocess

# Path to the existing export script
EXPORT_SCRIPT = "/Users/user/Documents/Python_Scripts/FungalProjectScript/batch_ims_export.py"

def run_export(working_dir):
    print(f"\n🚀 Starting export for: {working_dir}")
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
    if len(sys.argv) < 2:
        print("Usage: python run_local_ims_export.py <movie_folder_1> [<movie_folder_2> ...]")
        print("Example: python run_local_ims_export.py 2026_06_03_M143")
        sys.exit(1)

    for movie in sys.argv[1:]:
        wd = f"/Volumes/X10 Pro/Movies/{movie}"
        if os.path.isdir(wd):
            run_export(wd)
        else:
            print(f"⚠️ Warning: Directory not found: {wd}")
