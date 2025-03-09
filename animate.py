import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Load Image
image_path = "PoseImages/3.png"  # Change to your image path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process Pose
with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
    results = pose.process(image_rgb)

# Extract keypoints
if results.pose_landmarks:
    keypoints = np.array([(lm.x, lm.y) for lm in results.pose_landmarks.landmark])
else:
    print("No pose detected!")
    exit()

# Create figure for animation
fig, ax = plt.subplots(figsize=(5, 6))
ax.set_xlim(0, 1)
ax.set_ylim(1, 0)  # Invert to match image coordinates
scat = ax.scatter([], [], c='red', s=50)

# Draw connections (Skeleton)
connections = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # Head
    (0, 4), (4, 5), (5, 6), (6, 8),  # Head
    (9, 10), (11, 12),  # Shoulders
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # Left Arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),  # Right Arm
    (11, 23), (12, 24), (23, 24),  # Torso
    (23, 25), (25, 27), (27, 29), (29, 31),  # Left Leg
    (24, 26), (26, 28), (28, 30), (30, 32)  # Right Leg
]
lines = [ax.plot([], [], 'bo-', lw=2)[0] for _ in connections]

# Animation update function
def update(frame):
    animated_keypoints = keypoints.copy()

    # Move hands up/down (Bye-Bye)
    animated_keypoints[15][1] -= 0.015 * np.sin(frame * 0.2)  # Left Hand
    animated_keypoints[16][1] -= 0.015 * np.sin(frame * 0.2)  # Right Hand

    # Move elbows up/down slightly
    animated_keypoints[13][1] -= 0.01 * np.sin(frame * 0.2)  # Left Elbow
    animated_keypoints[14][1] -= 0.01 * np.sin(frame * 0.2)  # Right Elbow

    # Update scatter points
    scat.set_offsets(animated_keypoints)

    # Update skeleton connections
    for i, (start, end) in enumerate(connections):
        lines[i].set_data(
            [animated_keypoints[start, 0], animated_keypoints[end, 0]],
            [animated_keypoints[start, 1], animated_keypoints[end, 1]]
        )

    return [scat] + lines

# Create animation
ani = animation.FuncAnimation(fig, update, frames=50, interval=50, blit=True)

# Show animation
plt.show()
