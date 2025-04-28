import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import time

# Initialize YOLO and MediaPipe Pose
yolo_model = YOLO("yolov8n.pt")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)
pose_connections = mp_pose.POSE_CONNECTIONS

cap = cv2.VideoCapture(0)

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

def nms_boxes(boxes, confidences, iou_threshold=0.5):
    indices = []
    sorted_idx = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
    while sorted_idx:
        current = sorted_idx.pop(0)
        indices.append(current)
        sorted_idx = [i for i in sorted_idx if compute_iou(boxes[current], boxes[i]) < iou_threshold]
    return indices

prev_time = 0

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

        # Apply NMS
        selected_indices = nms_boxes(person_boxes, confidences, iou_threshold=0.4)
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

    # Show FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) != 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Pose Estimation - Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
