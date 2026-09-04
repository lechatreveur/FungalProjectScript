import subprocess
import time
import sys
import os

HPC_HOST = "hsushen@172.20.97.21"
REMOTE_OUT_DIR = "/RAID1/working/R402/hsushen/FungalProject/Movies/2026_07_16_M156/cellpose_overexposure_test/"
LOCAL_OUT_DIR = "/Volumes/X10 Pro/Movies/2026_07_16_M156/cellpose_overexposure_test/"

def check_hpc_queue():
    cmd = ["ssh", "-o", "BatchMode=yes", HPC_HOST, "squeue -u hsushen"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.stdout

def count_completed_csvs():
    cmd = ["ssh", "-o", "BatchMode=yes", HPC_HOST, "ls -1 /RAID1/working/R402/hsushen/FungalProject/Movies/2026_07_16_M156/3_FL2_F0/TrackedCells_3_FL2_F0_cpsam_overexp/*.csv 2>/dev/null | wc -l"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return int(res.stdout.strip())
    except ValueError:
        return 0

def sync_results():
    os.makedirs(LOCAL_OUT_DIR, exist_ok=True)
    print(f"[sync] Synchronizing results from HPC to {LOCAL_OUT_DIR}...")
    cmd = ["rsync", "-avz", f"{HPC_HOST}:{REMOTE_OUT_DIR}", LOCAL_OUT_DIR]
    subprocess.run(cmd)
    print("[sync] Sync complete.")

def main():
    print("=== Starting Automated HPC Monitor & Auto-Sync ===")
    poll_interval = 30
    
    while True:
        queue_output = check_hpc_queue()
        val_jobs = [line for line in queue_output.splitlines() if "VAL_M156" in line or "prompt1_" in line]
        completed_csvs = count_completed_csvs()
        
        print(f"[{time.strftime('%H:%M:%S')}] Active SLURM Jobs: {len(val_jobs)} | Completed Cell CSVs: {completed_csvs}/30")
        
        if len(val_jobs) == 0:
            print("\n=== All SLURM Validation Jobs Finished! ===")
            break
            
        time.sleep(poll_interval)
        
    sync_results()
    
    summary_path = os.path.join(LOCAL_OUT_DIR, "prompt1_validation_summary.csv")
    if os.path.exists(summary_path):
        print(f"\n--- Validation Summary ({summary_path}) ---")
        with open(summary_path, 'r') as f:
            print(f.read())
    else:
        print(f"[warning] Summary file {summary_path} not found.")

if __name__ == "__main__":
    main()
