import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from tabulate import tabulate  # For table formatting

# Paths
image_path = 'PoseImages/3.png'
output_path = "PoseImages/pose_output.png"

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Load image
image = cv2.imread(image_path)

# Convert to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Define complete list of 33 body parts
body_parts = [
    "Nose", "Left Eye (Inner)", "Left Eye", "Left Eye (Outer)", "Right Eye (Inner)", "Right Eye", "Right Eye (Outer)",
    "Left Ear", "Right Ear", "Mouth (Left)", "Mouth (Right)", "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Pinky", "Right Pinky", "Left Index", "Right Index", "Left Thumb", "Right Thumb",
    "Left Hip", "Right Hip", "Left Knee", "Right Knee", "Left Ankle", "Right Ankle", "Left Heel", "Right Heel",
    "Left Foot Index", "Right Foot Index"
]

# Process image with Pose model
with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # Draw pose landmarks on image
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imwrite(output_path, image)

        # Extract keypoints and confidence scores
        keypoints = [(lm.x, lm.y, lm.visibility) for lm in results.pose_landmarks.landmark]

        # Convert to separate X, Y, and confidence lists
        x_coords, y_coords, confidence_scores = zip(*keypoints)

        # Prepare table data
        table_data = []
        for i, (x, y, conf) in enumerate(zip(x_coords, y_coords, confidence_scores)):
            validity = "✅" if conf > 0.5 else "❌"
            table_data.append([body_parts[i], round(x, 3), round(y, 3), round(conf, 6), validity])

        # Print formatted table
        print("\n### Keypoint Verification Results ###\n")
        print(tabulate(table_data, headers=["Body Part", "X-Coordinate", "Y-Coordinate", "Confidence", "Valid?"],
                       tablefmt="grid"))

        # Define colors based on confidence threshold (0.5)
        colors = ['red' if conf < 0.5 else 'green' for conf in confidence_scores]

        # Create a figure with 2 subplots (Image + Graph)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Show the image with drawn landmarks
        ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax[0].axis('off')
        ax[0].set_title("Pose Detection")

        # Plot the keypoints on a graph with confidence-based coloring
        for x, y, color in zip(x_coords, y_coords, colors):
            ax[1].scatter(x, y, c=color, marker='o', edgecolors='black', s=50)

        ax[1].invert_yaxis()  # Invert Y to match image coordinate system
        ax[1].set_title("Pose Keypoints (Color by Confidence)")
        ax[1].set_xlabel("X Coordinate (Normalized)")
        ax[1].set_ylabel("Y Coordinate (Normalized)")

        # Show the combined figure
        plt.show()

    else:
        print("No pose detected.")
