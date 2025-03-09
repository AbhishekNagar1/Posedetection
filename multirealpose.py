import cv2
import mediapipe as mp
from ultralytics import YOLO

# Initialize YOLO and MediaPipe Pose
yolo_model = YOLO("yolov8n.pt")  # YOLO detects persons
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Define pose connections (lines between keypoints)
pose_connections = mp_pose.POSE_CONNECTIONS

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = yolo_model(frame)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # Class ID
            label = result.names[class_id]  # Class label

            if label != "person":
                continue  # Skip if not a person

            x1, y1, x2, y2 = map(int, box.xyxy[0])  # YOLO bounding box coords

            # Crop person & run MediaPipe Pose
            person_crop = frame[y1:y2, x1:x2]
            person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(person_rgb)

            if results_pose.pose_landmarks:
                # Convert landmark coordinates back to original frame
                landmark_points = []
                for lm in results_pose.pose_landmarks.landmark:
                    cx, cy = int(lm.x * (x2 - x1)) + x1, int(lm.y * (y2 - y1)) + y1
                    landmark_points.append((cx, cy))
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # Red dots

                # Draw pose connections (lines)
                for connection in pose_connections:
                    part1, part2 = connection
                    if part1 < len(landmark_points) and part2 < len(landmark_points):
                        pt1 = landmark_points[part1]
                        pt2 = landmark_points[part2]
                        cv2.line(frame, pt1, pt2, (255, 255, 0), 2)  # Cyan lines

    # Show real-time output
    cv2.imshow("Pose Estimation - Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
