# Coworker Guide: AI Tracker Checkpoints & Script Update

To keep the git repository lightweight and fast, the AI tracker model checkpoints (`tracker_checkpoints/`, `tracker_checkpoints_m93/`, and `tracker_checkpoints_m93_gfp/`) are now stored externally on SSDs and backed up on the NAS.

The script `one_cell_quantification_1CH.py` has been updated to dynamically search for these checkpoint directories in the following order:
1. **Coworker/User SSD**: `/Volumes/X10 Pro/Movies/AI/` or `/X10 Pro/Movies/AI/`
2. **NAS Mount**: `/Volumes/Movies/AI/`
3. **Local Fallback**: Inside the repository under `SingleCellQuantificationHPC/`

Here is how you can update your environment and access the tracking models.

---

## Step 1: Update the Script

Run the following commands in your terminal to pull the latest changes:

```bash
cd /Users/crissielin/Documents/FungalProjectScript_git/SingleCellQuantificationHPC
git checkout AlignmentPredictionWithSeptum
git pull origin AlignmentPredictionWithSeptum
```

---

## Step 2: Accessing the Checkpoints (Choose Option A, B, or C)

### Option A: Mount the NAS directly
If you are connected to the network and want a zero-setup approach, you can mount the NAS. The script will load the checkpoints directly from the NAS on the fly.

1. Open **Finder** on macOS.
2. Select **Go** in the top menu bar, then click **Connect to Server** (or press `Cmd + K`).
3. Enter the server address:
   ```text
   smb://hsushen@R402-NAS._smb._tcp.local/Movies
   ```
4. Click **Connect** and log in with your credentials.
5. Once mounted, the directory `/Volumes/Movies/AI/` is available, and the script will automatically detect it.

---

### Option B: Copy checkpoints to your SSD (X10 Pro)
If you have an external SSD named **X10 Pro**, you can copy the checkpoints onto it to run offline with maximum loading speeds.

1. Connect to the NAS as described in Option A.
2. Plug in your external SSD **X10 Pro**.
3. Create a folder named `Movies/AI` on your SSD: `/Volumes/X10 Pro/Movies/AI/` (Finder shows this as `/X10 Pro/Movies/AI/`).
4. Copy the checkpoint folders from the NAS `/Volumes/Movies/AI/` into `/Volumes/X10 Pro/Movies/AI/`.
   Specifically, you should have:
   - `/Volumes/X10 Pro/Movies/AI/tracker_checkpoints/`
   - `/Volumes/X10 Pro/Movies/AI/tracker_checkpoints_m93/`
   - `/Volumes/X10 Pro/Movies/AI/tracker_checkpoints_m93_gfp/`

---

### Option C: Download the folders locally to the repository (For Offline Use)
If you do not have an external SSD and want to run offline:

1. Connect to the NAS as described in Option A.
2. Copy the checkpoint folders from the NAS path `/Volumes/Movies/AI/` and paste them directly into your local repository's script folder:
   `/Users/crissielin/Documents/FungalProjectScript_git/SingleCellQuantificationHPC/`
   
   Specifically, your repository structure should look like:
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
