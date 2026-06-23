# Coworker Guide: AI Tracker Checkpoints & Script Update

To keep the git repository lightweight and fast, the AI tracker model checkpoints (`tracker_checkpoints/`, `tracker_checkpoints_m93/`, and `tracker_checkpoints_m93_gfp/`) are now stored externally on the SSD and backed up on the NAS.

The script `one_cell_quantification_1CH.py` has been updated to dynamically search for these checkpoint directories in the following order:
1. **Local SSD**: `/Volumes/X10 Pro/Movies/AI/`
2. **NAS**: `/Volumes/Movies/AI/`
3. **Local Fallback**: Inside the repository under `SingleCellQuantificationHPC/`

Here is how you can update your environment and run the tracking tool.

---

## Step 1: Update the Script

Run the following commands in your terminal to get the latest script updates:

```bash
cd /Users/crissielin/Documents/FungalProjectScript_git/SingleCellQuantificationHPC
git checkout AlignmentPredictionWithSeptum
git pull origin AlignmentPredictionWithSeptum
```

---

## Step 2: Accessing the Checkpoints (Choose Option A or B)

### Option A: Mount the NAS directly (Recommended)
If you are connected to the network, mounting the NAS volume is the easiest approach. The script will automatically discover and load the checkpoints from the NAS mount on the fly without storing anything locally.

1. Open **Finder** on macOS.
2. Select **Go** in the top menu bar, then click **Connect to Server** (or press `Cmd + K`).
3. Enter the server address:
   ```text
   smb://hsushen@R402-NAS._smb._tcp.local/Movies
   ```
4. Click **Connect** and log in with your credentials.
5. Once mounted, the directory `/Volumes/Movies/AI/` will be available, and the quantification script will automatically detect and load the checkpoints from it.

---

### Option B: Download the folders locally (For Offline Use)
If you want to run the tool offline or do not want to mount the NAS constantly:

1. Connect to the NAS as described in Option A.
2. Copy the model folders from the NAS path:
   `/Volumes/Movies/AI/`
3. Paste them directly into your local repository's script folder:
   `/Users/crissielin/Documents/FungalProjectScript_git/SingleCellQuantificationHPC/`
   
   Specifically, your folder structure should look like:
   ```text
   SingleCellQuantificationHPC/
   ├── one_cell_quantification_1CH.py
   ├── tracker_checkpoints/
   │   └── model_latest.pt
   ├── tracker_checkpoints_m93/
   │   └── model_latest.pt
   └── tracker_checkpoints_m93_gfp/
       └── model_latest.pt
   ```

The script's local fallback search will automatically find them there.
