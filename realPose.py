import cv2
import mediapipe as mp
import time
import matplotlib.pyplot as plt
import numpy as np

class PoseDetector:
    def __init__(self, mode=False, model_complexity=1, smooth_landmarks=True,
                 enable_segmentation=False, smooth_segmentation=True,
                 detectionCon=0.5, trackCon=0.5):
        """
        Initializes the pose detector with the given parameters.
        """
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=mode,
                                     model_complexity=model_complexity,
                                     smooth_landmarks=smooth_landmarks,
                                     enable_segmentation=enable_segmentation,
                                     smooth_segmentation=smooth_segmentation,
                                     min_detection_confidence=detectionCon,
                                     min_tracking_confidence=trackCon)

    def findPose(self, img, draw=True):
        """
        Processes the image to detect poses and optionally draws landmarks.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                       self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        """
        Finds the landmarks of the detected pose and returns them as a list.
        """
        lmList = []
        confidence_scores = []
        if self.results.pose_landmarks:
            h, w, c = img.shape
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                confidence = lm.visibility
                lmList.append([id, cx, cy])
                confidence_scores.append(confidence)

                # Color landmark based on confidence
                color = (0, 255, 0) if confidence > 0.5 else (0, 0, 255)
                if draw:
                    cv2.circle(img, (cx, cy), 5, color, cv2.FILLED)
        return lmList, confidence_scores

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = PoseDetector()

    # Initialize Matplotlib
    plt.ion()
    fig, ax = plt.subplots()
    scatter = ax.scatter([], [], c=[], cmap="coolwarm", edgecolors='black', s=50)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # Invert Y to match OpenCV coordinate system
    ax.set_title("Pose Keypoints (Real-Time)")
    ax.set_xlabel("X Coordinate (Normalized)")
    ax.set_ylabel("Y Coordinate (Normalized)")

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img)
        lmList, confidence_scores = detector.findPosition(img, draw=True)

        if lmList:
            # Extract X, Y coordinates and normalize
            x_coords = [lm[1] / img.shape[1] for lm in lmList]
            y_coords = [lm[2] / img.shape[0] for lm in lmList]

            # Color points based on confidence
            colors = ["red" if conf < 0.5 else "green" for conf in confidence_scores]

            # Update the Matplotlib scatter plot
            scatter.set_offsets(np.c_[x_coords, y_coords])
            scatter.set_color(colors)
            plt.draw()
            plt.pause(0.01)  # Small delay for real-time update

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # Show OpenCV video feed
        cv2.imshow("Pose Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
