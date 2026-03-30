import os
import numpy as np
from tifffile import imread
from skimage.measure import regionprops, label, find_contours
import cv2

# Paths
input_masks_folder = "/Users/user/Documents/FungalProject/TimeLapse/Masks"
output_frames_folder = "/Users/user/Documents/FungalProject/TimeLapse/Frames"
output_tracked_cells_folder = "/Users/user/Documents/FungalProject/TimeLapse/TrackedCells"

# Create necessary output directories
os.makedirs(output_tracked_cells_folder, exist_ok=True)

# Get list of mask files (sorted by timepoint)
mask_files = sorted([f for f in os.listdir(input_masks_folder) if f.endswith('.tif')])
mask_paths = [os.path.join(input_masks_folder, f) for f in mask_files]

# Load the first mask and detect cells
first_mask = imread(mask_paths[0])
labeled_mask = label(first_mask)
regions = regionprops(labeled_mask)

def compute_overlap(mask1, mask2):
    """Computes the percentage of overlap between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    total_area = mask1.sum()
    return intersection / total_area if total_area > 0 else 0


def draw_dashed_contour(image, mask, color, thickness=1, gap=5):
    """Draws a dashed outline around the mask on the given image."""
    contours = find_contours(mask, 0.5)

    for contour in contours:
        contour = np.round(contour).astype(int)
        for i in range(0, len(contour) - 1, gap * 2):  # Draw segments with gaps
            start = tuple(contour[i][::-1])  # Reverse (y, x) to (x, y) for OpenCV
            end = tuple(contour[min(i + gap, len(contour) - 1)][::-1])
            cv2.line(image, start, end, int(color), thickness, cv2.LINE_AA)  # Anti-aliased line
      # if best_match is None:
      #     print(f"Cell {cell_id} lost at frame {t}. Stopping tracking.")
      #     break
# Initialize prev_bbox before entering the loop
prev_bbox = None  

# Process each detected cell
for cell in regions:
    cell_id = cell.label
    centroid = np.round(cell.centroid).astype(int)
    y, x = centroid

    brightfield_timelapse_frames = []
    fluorescent_timelapse_frames_C1 = []
    fluorescent_timelapse_frames_C2 = []

    prev_cell_mask = (labeled_mask == cell_id)

    # Get initial bounding box
    minr, minc, maxr, maxc = cell.bbox
    bbox_size = max(maxr - minr, maxc - minc)
    bbox_size = max(bbox_size, 10)  # Ensure a minimum bbox size

    y_min = max(0, y - bbox_size // 2)
    y_max = min(first_mask.shape[0], y + bbox_size // 2)
    x_min = max(0, x - bbox_size // 2)
    x_max = min(first_mask.shape[1], x + bbox_size // 2)

    prev_bbox = (y_min, y_max, x_min, x_max)  # Initialize bounding box

    for t, mask_path in enumerate(mask_paths):
        brightfield_frame_path = os.path.join(output_frames_folder, f"brightfield_{t:03d}.tif")
        fluorescent_C1_frame_path = os.path.join(output_frames_folder, f"fluorescent_C1_{t:03d}.tif")
        fluorescent_C2_frame_path = os.path.join(output_frames_folder, f"fluorescent_C2_{t:03d}.tif")

        if not os.path.exists(brightfield_frame_path) or not os.path.exists(fluorescent_C1_frame_path) or not os.path.exists(fluorescent_C2_frame_path):
            print(f"Skipping missing frame at time {t}")
            continue

        current_masks = imread(mask_path)
        labeled_current = label(current_masks)
        current_regions = regionprops(labeled_current)

        best_match = None
        max_overlap = 0

        for candidate in current_regions:
            candidate_mask = (labeled_current == candidate.label)
            overlap = compute_overlap(prev_cell_mask, candidate_mask)

            if overlap >= 0.7 and overlap > max_overlap:
                best_match = candidate
                max_overlap = overlap

        if best_match is None:
            print(f"Cell {cell_id} lost at frame {t}. Using previous segmentation.")
            # Keep using the previous mask and bounding box
            current_cell_mask = prev_cell_mask
            y_min, y_max, x_min, x_max = prev_bbox  # Use last known bbox

        else:
            # Update mask tracking when a match is found
            current_cell_mask = (labeled_current == best_match.label)
            centroid = np.round(best_match.centroid).astype(int)
            y, x = centroid
            minr, minc, maxr, maxc = best_match.bbox
            bbox_size = max(maxr - minr, maxc - minc)
            bbox_size = max(bbox_size, 10)

            # Save bbox for reuse if tracking is lost later
            y_min = max(0, y - bbox_size // 2)
            y_max = min(current_cell_mask.shape[0], y + bbox_size // 2)
            x_min = max(0, x - bbox_size // 2)
            x_max = min(current_cell_mask.shape[1], x + bbox_size // 2)
            prev_bbox = (y_min, y_max, x_min, x_max)  # Update bbox


        brightfield_frame = imread(brightfield_frame_path)
        fluorescent_frame_C1 = imread(fluorescent_C1_frame_path)
        fluorescent_frame_C2 = imread(fluorescent_C2_frame_path)
        
        BFmin = brightfield_frame.min()
        BFmax = brightfield_frame.max()
        C1min = 0  # imread(fluorescent_C1_frame_path).min()
        C1max = 8  # imread(fluorescent_C1_frame_path).max()
        C2min = 0  # imread(fluorescent_C2_frame_path).min()
        C2max = 20  # imread(fluorescent_C2_frame_path).max()

        brightfield_frame = np.clip(((brightfield_frame - BFmin) / (BFmax - BFmin) * 255), 0, 255).astype(np.uint8)
        fluorescent_frame_C1 = np.clip(((fluorescent_frame_C1 - C1min) / (C1max - C1min) * 255), 0, 255).astype(np.uint8)
        fluorescent_frame_C2 = np.clip(((fluorescent_frame_C2 - C2min) / (C2max - C2min) * 255), 0, 255).astype(np.uint8)
        
        draw_dashed_contour(brightfield_frame, current_cell_mask, 0)#, best_match.bbox)
        draw_dashed_contour(fluorescent_frame_C1, current_cell_mask, 255)#, best_match.bbox)
        draw_dashed_contour(fluorescent_frame_C2, current_cell_mask, 255)#, best_match.bbox)
        
        brightfield_frame = brightfield_frame[y_min:y_max, x_min:x_max]
        fluorescent_frame_C1 = fluorescent_frame_C1[y_min:y_max, x_min:x_max]
        fluorescent_frame_C2 = fluorescent_frame_C2[y_min:y_max, x_min:x_max]

       
        
        #cv2.putText(image, text, position, font, font_scale, color, thickness, line_type)

        brightfield_frame = cv2.putText(brightfield_frame.copy(), f'{t}', (0, bbox_size), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        fluorescent_frame_C1 = cv2.putText(fluorescent_frame_C1.copy(), f'{t}', (0, bbox_size), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        fluorescent_frame_C2 = cv2.putText(fluorescent_frame_C2.copy(), f'{t}', (0, bbox_size), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        


        
        brightfield_timelapse_frames.append(brightfield_frame)
        fluorescent_timelapse_frames_C1.append(fluorescent_frame_C1)
        fluorescent_timelapse_frames_C2.append(fluorescent_frame_C2)
        
        prev_cell_mask = current_cell_mask
        
    def pad_frame(frame, target_size):
        """Pads a frame to match the target size (height, width) while centering the image."""
        h, w = frame.shape
        target_h, target_w = target_size
        
        pad_top = (target_h - h) // 2
        pad_bottom = target_h - h - pad_top
        pad_left = (target_w - w) // 2
        pad_right = target_w - w - pad_left
        
        return np.pad(frame, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    
    for channel_name, frames in zip(["brightfield", "fluorescent_C1", "fluorescent_C2"],
                                    [brightfield_timelapse_frames, fluorescent_timelapse_frames_C1, fluorescent_timelapse_frames_C2]):
    
        if not frames:
            print(f"No frames found for cell {cell_id} in {channel_name}. Skipping movie creation.")
            continue
    
        # Find the maximum height and width across all frames
        max_height = max(frame.shape[0] for frame in frames)
        max_width = max(frame.shape[1] for frame in frames)
        target_size = (max_height, max_width)
    
        # Define video output path
        video_path = os.path.join(output_tracked_cells_folder, f"cell_{cell_id}_{channel_name}_timelapse.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        video = cv2.VideoWriter(video_path, fourcc, 10, (max_width, max_height), False)
    
        for frame in frames:
            padded_frame = pad_frame(frame, target_size)  # Ensure all frames have the same size
            video.write(padded_frame)
    
        video.release()
        print(f"Saved complete {channel_name} timelapse for Cell {cell_id} at {video_path}")

    # for channel_name, frames in zip(["brightfield", "fluorescent_C1", "fluorescent_C2"],
    #                                 [brightfield_timelapse_frames, fluorescent_timelapse_frames_C1, fluorescent_timelapse_frames_C2]):
        
    #     # Define folder path for the cell
    #     cell_folder = os.path.join(output_tracked_cells_folder, f"cell_{cell_id}")
    #     channel_folder = os.path.join(cell_folder, channel_name)
        
    #     # Create directories if they don't exist
    #     os.makedirs(channel_folder, exist_ok=True)
        
    #     if not frames:
    #         print(f"No frames found for cell {cell_id} in {channel_name}. Skipping frame saving.")
    #         continue
    
    #     # Save each frame as an image
    #     for t, frame in enumerate(frames):
    #         frame_path = os.path.join(channel_folder, f"{channel_name}_frame_{t:03d}.tif")
    #         cv2.imwrite(frame_path, frame)
        
    #     print(f"Saved frames for {channel_name} of Cell {cell_id} in {channel_folder}")
