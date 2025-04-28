# import cv2
# import mediapipe as mp
# from ultralytics import YOLO
# import numpy as np
# import math
# from collections import deque
#
# # Initialize YOLO and MediaPipe Pose
# yolo_model = YOLO("yolov8n.pt")
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(
#     static_image_mode=False,
#     model_complexity=2,
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.7
# )
# pose_connections = mp_pose.POSE_CONNECTIONS
#
# # Load video
# video_path = "6.mp4"
# cap = cv2.VideoCapture(video_path)
#
# # Video properties
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# out = cv2.VideoWriter("output_pose_highly_accurate.mp4", fourcc, fps, (width, height))
#
# # Constants
# YOLO_CONF_THRESHOLD = 0.7
# POSE_CONF_THRESHOLD = 0.6
# MAX_TRACKING_HISTORY = 5
# TARGET_SIZE = 512  # Higher resolution for better detail
# BLUR_KERNEL_SIZE = (5, 5)  # For pre-processing
# CONTRAST_ALPHA = 1.3  # Contrast enhancement
#
#
# # Person tracker
# class PersonTracker:
#     def __init__(self, max_history=5):
#         self.next_id = 0
#         self.tracks = {}  # id -> deque of (box, landmarks)
#         self.max_history = max_history
#         self.missing_frames = {}  # id -> number of frames missing
#         self.max_missing = 10  # Maximum frames to track when missing
#
#     def update(self, detections):
#         # Match detections to existing tracks
#         if not self.tracks:
#             # First frame, create new tracks for all detections
#             for box in detections:
#                 self.tracks[self.next_id] = deque(maxlen=self.max_history)
#                 self.tracks[self.next_id].append((box, None))
#                 self.missing_frames[self.next_id] = 0
#                 self.next_id += 1
#             return {id: track[-1][0] for id, track in self.tracks.items()}
#
#         matched_track_ids = set()
#         unmatched_detections = []
#
#         # Match detections to tracks
#         for box in detections:
#             best_iou = 0
#             best_id = None
#             x1, y1, x2, y2 = box
#
#             for track_id, track in self.tracks.items():
#                 if track_id in matched_track_ids:
#                     continue
#
#                 prev_box = track[-1][0]
#                 iou = compute_iou(box, prev_box)
#
#                 if iou > best_iou and iou > 0.3:  # IOU threshold
#                     best_iou = iou
#                     best_id = track_id
#
#             if best_id is not None:
#                 self.tracks[best_id].append((box, None))
#                 self.missing_frames[best_id] = 0
#                 matched_track_ids.add(best_id)
#             else:
#                 unmatched_detections.append(box)
#
#         # Create new tracks for unmatched detections
#         for box in unmatched_detections:
#             self.tracks[self.next_id] = deque(maxlen=self.max_history)
#             self.tracks[self.next_id].append((box, None))
#             self.missing_frames[self.next_id] = 0
#             self.next_id += 1
#
#         # Update missing frames counter for unmatched tracks
#         all_track_ids = list(self.tracks.keys())
#         for track_id in all_track_ids:
#             if track_id not in matched_track_ids:
#                 self.missing_frames[track_id] += 1
#                 if self.missing_frames[track_id] <= self.max_missing:
#                     # Keep predicting the box using velocity
#                     if len(self.tracks[track_id]) >= 2:
#                         prev_box2, _ = self.tracks[track_id][-2]
#                         prev_box, _ = self.tracks[track_id][-1]
#                         # Calculate velocity vector
#                         vx1 = prev_box[0] - prev_box2[0]
#                         vy1 = prev_box[1] - prev_box2[1]
#                         vx2 = prev_box[2] - prev_box2[2]
#                         vy2 = prev_box[3] - prev_box2[3]
#                         # Predict new box
#                         pred_box = (
#                             prev_box[0] + vx1,
#                             prev_box[1] + vy1,
#                             prev_box[2] + vx2,
#                             prev_box[3] + vy2,
#                         )
#                         self.tracks[track_id].append((pred_box, None))
#                 else:
#                     # Remove track if missing for too long
#                     del self.tracks[track_id]
#                     del self.missing_frames[track_id]
#
#         return {id: track[-1][0] for id, track in self.tracks.items()}
#
#     def update_pose(self, track_id, landmarks):
#         if track_id in self.tracks:
#             box, _ = self.tracks[track_id][-1]
#             self.tracks[track_id][-1] = (box, landmarks)
#
#     def get_smoothed_landmarks(self, track_id):
#         if track_id not in self.tracks:
#             return None
#
#         valid_history = []
#         for box, landmarks in self.tracks[track_id]:
#             if landmarks is not None:
#                 valid_history.append(landmarks)
#
#         if not valid_history:
#             return None
#
#         if len(valid_history) == 1:
#             return valid_history[0]
#
#         # Apply exponential smoothing
#         weights = np.exp(np.linspace(-1, 0, len(valid_history)))
#         weights /= weights.sum()
#
#         smoothed = []
#         for i in range(33):  # MediaPipe has 33 landmarks
#             points = []
#             confs = []
#
#             for h_idx, history_item in enumerate(valid_history):
#                 if history_item[i] is not None:
#                     points.append(history_item[i][0])
#                     confs.append(history_item[i][1])
#
#             if points:
#                 points = np.array(points)
#                 avg_x = np.sum(points[:, 0] * weights[-len(points):]) / np.sum(weights[-len(points):])
#                 avg_y = np.sum(points[:, 1] * weights[-len(points):]) / np.sum(weights[-len(points):])
#                 avg_conf = np.mean(confs)
#                 smoothed.append(((int(avg_x), int(avg_y)), avg_conf))
#             else:
#                 smoothed.append(None)
#
#         return smoothed
#
#
# def compute_iou(box1, box2):
#     # Convert predictions to int to ensure consistent calculations
#     box1 = (int(box1[0]), int(box1[1]), int(box1[2]), int(box1[3]))
#     box2 = (int(box2[0]), int(box2[1]), int(box2[2]), int(box2[3]))
#
#     x1 = max(box1[0], box2[0])
#     y1 = max(box1[1], box2[1])
#     x2 = min(box1[2], box2[2])
#     y2 = min(box1[3], box2[3])
#
#     inter_area = max(0, x2 - x1) * max(0, y2 - y1)
#     box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
#     union_area = box1_area + box2_area - inter_area
#
#     return inter_area / union_area if union_area != 0 else 0
#
#
# def nms_boxes(boxes, confidences, iou_threshold=0.6):
#     indices = []
#     sorted_idx = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
#     while sorted_idx:
#         current = sorted_idx.pop(0)
#         indices.append(current)
#         sorted_idx = [i for i in sorted_idx if compute_iou(boxes[current], boxes[i]) < iou_threshold]
#     return indices
#
#
# def enhance_image(image):
#     # Apply series of preprocessing to enhance feature visibility
#     # Step 1: Convert to LAB color space
#     lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#
#     # Step 2: Apply CLAHE to L channel
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     cl = clahe.apply(l)
#
#     # Step 3: Merge channels back
#     enhanced_lab = cv2.merge((cl, a, b))
#     enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
#
#     # Step 4: Apply slight Gaussian blur to reduce noise
#     blurred = cv2.GaussianBlur(enhanced_img, BLUR_KERNEL_SIZE, 0)
#
#     # Step 5: Enhance contrast
#     enhanced = cv2.convertScaleAbs(blurred, alpha=CONTRAST_ALPHA, beta=0)
#
#     return enhanced
#
#
# def smart_crop(frame, box, margin_factor=0.4):
#     x1, y1, x2, y2 = box
#     frame_h, frame_w = frame.shape[:2]
#
#     # Calculate box dimensions
#     box_w, box_h = x2 - x1, y2 - y1
#
#     # Calculate body aspect ratio to determine if person is standing or partially visible
#     aspect_ratio = box_h / max(box_w, 1)
#
#     # Adaptive margin based on detection size and position in frame
#     # Smaller margin for detections near edges, larger for central detections
#     center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
#
#     # Calculate distance from center as percentage (0 = center, 1 = edge)
#     dist_from_center_x = abs((center_x / frame_w) - 0.5) * 2
#     dist_from_center_y = abs((center_y / frame_h) - 0.5) * 2
#     edge_factor = max(dist_from_center_x, dist_from_center_y)
#
#     # Reduce margin when close to edges
#     adaptive_margin = margin_factor * (1 - edge_factor * 0.5)
#
#     # For very tall detections (standing person), add more horizontal margin
#     if aspect_ratio > 2.5:
#         margin_w = int(box_w * (adaptive_margin + 0.2))
#         margin_h = int(box_h * adaptive_margin)
#     # For very wide detections (e.g. lying down), add more vertical margin
#     elif aspect_ratio < 0.7:
#         margin_w = int(box_w * adaptive_margin)
#         margin_h = int(box_h * (adaptive_margin + 0.2))
#     # Regular case
#     else:
#         margin_w = int(box_w * adaptive_margin)
#         margin_h = int(box_h * adaptive_margin)
#
#     # Calculate new coordinates with margins
#     new_x1 = max(0, x1 - margin_w)
#     new_y1 = max(0, y1 - margin_h)
#     new_x2 = min(frame_w, x2 + margin_w)
#     new_y2 = min(frame_h, y2 + margin_h)
#
#     return new_x1, new_y1, new_x2, new_y2
#
#
# # Initialize tracker
# tracker = PersonTracker(max_history=MAX_TRACKING_HISTORY)
#
# frame_count = 0
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     frame_count += 1
#     orig_frame = frame.copy()  # Keep original for drawing
#     h, w, _ = frame.shape
#
#     # Step 1: Enhance the image for better detection
#     enhanced_frame = enhance_image(frame)
#
#     # Step 2: Run YOLO with high confidence threshold
#     results = yolo_model(enhanced_frame, conf=YOLO_CONF_THRESHOLD, iou=0.6)
#
#     person_boxes = []
#     confidences = []
#
#     for result in results:
#         for box in result.boxes:
#             class_id = int(box.cls[0])
#             label = result.names[class_id]
#             if label != "person":
#                 continue
#
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             box_w, box_h = x2 - x1, y2 - y1
#
#             # Skip tiny detections
#             if box_w < 30 or box_h < 60:
#                 continue
#
#             conf = float(box.conf[0])
#             person_boxes.append((x1, y1, x2, y2))
#             confidences.append(conf)
#
#     # Apply NMS with high threshold
#     if person_boxes:
#         selected_indices = nms_boxes(person_boxes, confidences, iou_threshold=0.6)
#         final_boxes = [person_boxes[i] for i in selected_indices]
#     else:
#         final_boxes = []
#
#     # Update tracker with new detections
#     tracked_boxes = tracker.update(final_boxes)
#
#     # Process each tracked person
#     for track_id, box in tracked_boxes.items():
#         # Get an adaptive crop for the person
#         x1_crop, y1_crop, x2_crop, y2_crop = smart_crop(frame, box)
#
#         if x2_crop <= x1_crop or y2_crop <= y1_crop:
#             continue
#
#         person_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
#         if person_crop.size == 0:
#             continue
#
#         # Calculate aspect ratio for proper resizing
#         crop_h, crop_w = person_crop.shape[:2]
#         aspect = crop_w / crop_h
#
#         # Resize to target size preserving aspect ratio
#         if aspect > 1:
#             new_w = TARGET_SIZE
#             new_h = int(TARGET_SIZE / aspect)
#         else:
#             new_h = TARGET_SIZE
#             new_w = int(TARGET_SIZE * aspect)
#
#         person_crop_resized = cv2.resize(person_crop, (new_w, new_h))
#
#         # Add padding to make square
#         if new_w != new_h:
#             padded_img = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
#             if new_w > new_h:
#                 start_y = (TARGET_SIZE - new_h) // 2
#                 padded_img[start_y:start_y + new_h, :, :] = person_crop_resized
#             else:
#                 start_x = (TARGET_SIZE - new_w) // 2
#                 padded_img[:, start_x:start_x + new_w, :] = person_crop_resized
#             person_crop_final = padded_img
#         else:
#             person_crop_final = person_crop_resized
#
#         # Convert to RGB for MediaPipe
#         person_rgb = cv2.cvtColor(person_crop_final, cv2.COLOR_BGR2RGB)
#
#         # Multiple rotated poses for better coverage
#         best_pose = None
#         best_score = -1
#
#         # Try multiple perspectives (original + slightly rotated)
#         angles = [0, -15, 15]
#
#         for angle in angles:
#             if angle == 0:
#                 rotated = person_rgb
#             else:
#                 # Rotate image for different perspective
#                 center = (TARGET_SIZE // 2, TARGET_SIZE // 2)
#                 rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
#                 rotated = cv2.warpAffine(person_rgb, rotation_matrix, (TARGET_SIZE, TARGET_SIZE))
#
#             # Process with MediaPipe
#             pose_results = pose.process(rotated)
#
#             if pose_results.pose_landmarks:
#                 # Calculate pose confidence
#                 landmark_scores = [lm.visibility for lm in pose_results.pose_landmarks.landmark]
#                 avg_score = np.mean(landmark_scores)
#
#                 if avg_score > best_score:
#                     best_score = avg_score
#                     best_pose = (pose_results, angle)
#
#         # Process the best pose if found
#         if best_pose and best_score > POSE_CONF_THRESHOLD:
#             pose_results, angle = best_pose
#
#             # Need to rotate landmarks back if angle wasn't 0
#             landmarks_data = []
#
#             for idx, lm in enumerate(pose_results.pose_landmarks.landmark):
#                 visibility = lm.visibility
#
#                 if visibility < 0.3:  # Skip very low confidence points
#                     landmarks_data.append(None)
#                     continue
#
#                 # Get coordinates in the square padded image
#                 px, py = lm.x * TARGET_SIZE, lm.y * TARGET_SIZE
#
#                 # Rotate back if necessary
#                 if angle != 0:
#                     center = (TARGET_SIZE // 2, TARGET_SIZE // 2)
#                     # Inverse rotation
#                     inv_rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
#                     px_rot = inv_rotation_matrix[0][0] * px + inv_rotation_matrix[0][1] * py + inv_rotation_matrix[0][2]
#                     py_rot = inv_rotation_matrix[1][0] * px + inv_rotation_matrix[1][1] * py + inv_rotation_matrix[1][2]
#                     px, py = px_rot, py_rot
#
#                 # Handle padding if non-square input
#                 if new_w != new_h:
#                     if new_w > new_h:
#                         start_y = (TARGET_SIZE - new_h) // 2
#                         # Adjust y coordinates
#                         py = py - start_y
#                         py = max(0, min(new_h - 1, py))
#                         py = py / new_h
#                     else:
#                         start_x = (TARGET_SIZE - new_w) // 2
#                         # Adjust x coordinates
#                         px = px - start_x
#                         px = max(0, min(new_w - 1, px))
#                         px = px / new_w
#                 else:
#                     px /= TARGET_SIZE
#                     py /= TARGET_SIZE
#
#                 # Map back to original frame
#                 orig_x = int(px * (x2_crop - x1_crop)) + x1_crop
#                 orig_y = int(py * (y2_crop - y1_crop)) + y1_crop
#
#                 # Store coordinates with confidence
#                 landmarks_data.append(((orig_x, orig_y), visibility))
#
#             # Update tracker with landmark data
#             tracker.update_pose(track_id, landmarks_data)
#
#             # Get smoothed landmarks
#             smoothed_landmarks = tracker.get_smoothed_landmarks(track_id)
#
#             if smoothed_landmarks:
#                 x1, y1, x2, y2 = box
#
#                 # Draw the person bounding box
#                 color = (0, 255, 0)
#                 if tracker.missing_frames.get(track_id, 0) > 0:
#                     # Use yellow color for predicted boxes
#                     color = (0, 255, 255)
#
#                 cv2.rectangle(orig_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
#
#                 # Calculate confidence for visualization
#                 valid_landmarks = [lm for lm in smoothed_landmarks if lm is not None]
#                 if valid_landmarks:
#                     avg_confidence = np.mean([lm[1] for lm in valid_landmarks])
#
#                     # Draw ID and confidence
#                     conf_text = f"ID:{track_id} Conf:{avg_confidence:.2f}"
#                     cv2.putText(orig_frame, conf_text, (int(x1), int(y1) - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#
#                     # Draw landmarks with visibility-based coloring
#                     for i, landmark in enumerate(smoothed_landmarks):
#                         if landmark is not None:
#                             point, conf = landmark
#                             # Color gradient based on confidence: green (high) to red (low)
#                             b_val = 0
#                             g_val = int(255 * conf)
#                             r_val = int(255 * (1 - conf))
#                             cv2.circle(orig_frame, point, 4, (b_val, g_val, r_val), -1)
#
#                     # Draw connections
#                     for connection in pose_connections:
#                         start_idx, end_idx = connection
#                         if (start_idx < len(smoothed_landmarks) and end_idx < len(smoothed_landmarks) and
#                                 smoothed_landmarks[start_idx] is not None and smoothed_landmarks[end_idx] is not None):
#
#                             start_point, start_conf = smoothed_landmarks[start_idx]
#                             end_point, end_conf = smoothed_landmarks[end_idx]
#
#                             # Average confidence for this connection
#                             conn_conf = (start_conf + end_conf) / 2
#
#                             # Thicker lines for higher confidence connections
#                             thickness = 1 + int(conn_conf * 3)
#
#                             # Color based on connection type (different colors for limbs)
#                             # Upper body - yellow
#                             if connection in [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]:
#                                 color = (0, 255, 255)
#                             # Lower body - cyan
#                             elif connection in [(23, 24), (11, 23), (12, 24), (23, 25), (24, 26), (25, 27), (26, 28)]:
#                                 color = (255, 255, 0)
#                             # Face - green
#                             elif min(connection) < 11:
#                                 color = (0, 255, 0)
#                             else:
#                                 color = (255, 0, 255)  # magenta for others
#
#                             cv2.line(orig_frame, start_point, end_point, color, thickness)
#
#     out.write(orig_frame)
#     cv2.imshow("Highly Accurate Pose Estimation", orig_frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
#
# cap.release()
# out.release()
# cv2.destroyAllWindows()
















import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np

# Initialize YOLO and MediaPipe Pose
yolo_model = YOLO("yolov8n.pt")  # Detect persons
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)
pose_connections = mp_pose.POSE_CONNECTIONS

# Load video
video_path = "6.mp4"
cap = cv2.VideoCapture(video_path)

# Video properties
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output_pose.mp4", fourcc, fps, (width, height))

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area != 0 else 0

def nms_boxes(boxes, confidences, iou_threshold=0.4):
    indices = []
    sorted_idx = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
    while sorted_idx:
        current = sorted_idx.pop(0)
        indices.append(current)
        sorted_idx = [i for i in sorted_idx if compute_iou(boxes[current], boxes[i]) < iou_threshold]
    return indices

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    results = yolo_model(frame)

    for result in results:
        person_boxes = []
        confidences = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            label = result.names[class_id]
            if label != "person":
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_w, box_h = x2 - x1, y2 - y1
            if box_w < 50 or box_h < 50:
                continue  # Skip small boxes
            conf = float(box.conf[0])
            person_boxes.append((x1, y1, x2, y2))
            confidences.append(conf)

        selected_indices = nms_boxes(person_boxes, confidences)
        final_boxes = [person_boxes[i] for i in selected_indices]

        for (x1, y1, x2, y2) in final_boxes:
            # Expand + square crop
            box_width, box_height = x2 - x1, y2 - y1
            margin = int(max(box_width, box_height) * 0.3)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            side = max(box_width, box_height) + 2 * margin
            x1_new = max(0, cx - side // 2)
            y1_new = max(0, cy - side // 2)
            x2_new = min(w, x1_new + side)
            y2_new = min(h, y1_new + side)

            person_crop = frame[y1_new:y2_new, x1_new:x2_new]
            person_crop = cv2.resize(person_crop, (256, 256))
            person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(person_rgb)

            if results_pose.pose_landmarks:
                landmark_points = []
                for lm in results_pose.pose_landmarks.landmark:
                    if lm.visibility < 0.5:
                        landmark_points.append(None)
                        continue
                    cx_lm = int(lm.x * (x2_new - x1_new)) + x1_new
                    cy_lm = int(lm.y * (y2_new - y1_new)) + y1_new
                    landmark_points.append((cx_lm, cy_lm))
                    cv2.circle(frame, (cx_lm, cy_lm), 5, (0, 0, 255), -1)

                for connection in pose_connections:
                    part1, part2 = connection
                    if (part1 < len(landmark_points) and part2 < len(landmark_points) and
                        landmark_points[part1] is not None and landmark_points[part2] is not None):
                        pt1 = landmark_points[part1]
                        pt2 = landmark_points[part2]
                        cv2.line(frame, pt1, pt2, (255, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Pose Estimation", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()



