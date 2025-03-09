import cv2
import mediapipe as mp
import time
import math
import numpy as np
import matplotlib.pyplot as plt


class poseDetector():
    def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        """
        Initializes the pose detector using MediaPipe Pose.
        """
        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Initialize MediaPipe Pose
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     model_complexity=1,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        """
        Detects pose landmarks in the given image.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                       self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        """
        Extracts the position of all detected landmarks.
        """
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy, lm.visibility])  # Storing confidence (visibility)

                # Color based on confidence
                color = (0, 255, 0) if lm.visibility > 0.5 else (0, 0, 255)  # Green if high, Red if low

                if draw:
                    cv2.circle(img, (cx, cy), 5, color, cv2.FILLED)
        return self.lmList


def main():
    """Runs a sample video to test the pose detector."""
    cap = cv2.VideoCapture('PoseVideos/2.mp4')
    pTime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()
        if not success:
            break  # Exit if the video ends

        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=True)

        # Extract keypoints for visualization
        if lmList:
            x_coords, y_coords, conf_scores = zip(*[(lm[1], lm[2], lm[3]) for lm in lmList])

            # Set color: red for low confidence, green for high confidence
            colors = ['red' if conf < 0.5 else 'green' for conf in conf_scores]

            # Clear previous figure and plot new keypoints
            plt.clf()
            plt.scatter(x_coords, y_coords, c=colors, edgecolors='black', s=50)
            plt.gca().invert_yaxis()  # Match image coordinate system
            plt.title("Pose Keypoints (Color by Confidence)")
            plt.xlabel("X Coordinate")
            plt.ylabel("Y Coordinate")
            plt.pause(0.001)  # Refresh the plot in real-time

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        cv2.imshow("Video Feed", img)  # Show video with keypoints
        cv2.waitKey(1)


if __name__ == "__main__":
    plt.ion()  # Enable interactive mode for real-time plotting
    main()
