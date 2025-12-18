import cv2, os, json
import numpy as np

from libraries import image_functions, cv_globals

vid_name = "cat2"
out_imdir = f"output/{vid_name}"

json_fpath = f"output/{vid_name}.json"
with open(json_fpath, "r") as f:
	track_data = json.load(f)

cv_globals.store_debug(True)

# Load one frame to get image size (IMPORTANT)
sample_frame = cv2.imread(f"{out_imdir}/0.png")
H, W = sample_frame.shape[:2]

# { (start_frame, object_id): [ [frame_index, filled_pixels], ... ] }
tracks = {}
active_tracks = {}

frames = sorted(map(int, track_data.keys()))

for frame in frames:
	objects = track_data[str(frame)]
	current_ids = set(map(int, objects.keys()))

	# remove disappeared tracks
	disappeared = set(active_tracks.keys()) - current_ids
	for tid in disappeared:
		del active_tracks[tid]

	for tid_str, contour_pixels in objects.items():
		tid = int(tid_str)

		# Create empty mask
		mask = np.zeros((H, W), dtype=np.uint8)

		# Convert contour pixels to proper contour format
		contour = np.array(contour_pixels, dtype=np.int32)

		# OpenCV wants (N, 1, 2)
		contour = contour.reshape((-1, 1, 2))

		# Fill contour
		cv2.drawContours(mask, [contour], contourIdx=-1, color=255, thickness=-1)

		# Extract filled pixels
		ys, xs = np.where(mask == 255)
		filled_pixels = [[int(x), int(y)] for x, y in zip(xs, ys)]


		# -----------------------------
		# TRACK MANAGEMENT
		# -----------------------------
		if tid not in active_tracks:
			start_frame = frame
			key = (start_frame, tid)
			tracks[key] = []
			active_tracks[tid] = key

		key = active_tracks[tid]
		tracks[key].append([frame, filled_pixels])

# -----------------------------
# VISUALIZE RESULT
# -----------------------------
for key in tracks:
	print("key " + str(key) )
	for frame_index, frame_pixels in tracks[key]:
		print("frame_index " + str(frame_index) )
		image_functions.cr_im_from_pixels( str(frame_index),  out_imdir + '/',	frame_pixels, pixels_rgb=(255, 0, 0) )































