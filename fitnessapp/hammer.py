# IMPORTS
import cv2
import mediapipe as mp
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

# -------------------- Pose Detection Class --------------------
class PoseDetector:
    def __init__(self):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            h, w, c = img.shape
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        return lmList

# -------------------- Workout Analytics Class --------------------
class WorkoutAnalytics:
    def __init__(self, user_name="user1"):  # <-- FIXED double underscores
        self.user_name = user_name
        self.data_dir = f"workout_data/{user_name}"
        os.makedirs(self.data_dir, exist_ok=True)
        self.today_date = datetime.now().strftime("%Y-%m-%d")
        self.start_time = time.time()
        self.data = {
            "right_curl": 0,
            "left_curl": 0,
            "squat": 0,
            "pushup": 0,
            "lunge": 0,
            "plank": 0
        }

    def update(self, exercise, count):
        if exercise in self.data:
            self.data[exercise] = count

    def end_session(self):
        end_time = time.time()
        total_time = int(end_time - self.start_time)
        calories_burned = (total_time / 60) * 5

        session = {
            "date": self.today_date,
            "total_time_sec": total_time,
            "calories_burned": calories_burned,
            "data": self.data
        }

        file_path = os.path.join(self.data_dir, "workout_history.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                history = json.load(f)
        else:
            history = []

        history.append(session)

        with open(file_path, "w") as f:
            json.dump(history, f, indent=4)

# -------------------- Utility Functions --------------------
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def update_dashboard(dash, reps, timer):
    dash.fill(0)
    cv2.putText(dash, "WORKOUT DASHBOARD", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    y = 70
    for exercise, count in reps.items():
        cv2.putText(dash, f"{exercise}: {int(count)}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 40
    mins, secs = divmod(timer, 60)
    cv2.putText(dash, f"Workout Time: {int(mins)}:{int(secs):02d}", (10, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# -------------------- Main Program --------------------
def main():
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()
    analytics = WorkoutAnalytics()

    dashboard = np.zeros((400, 300, 3), dtype=np.uint8)

    # Counters
    count_right = 0
    count_left = 0
    squat_count = 0
    pushup_count = 0
    lunge_count = 0
    plank_timer = 0

    # Direction flags
    direction_right = 0
    direction_left = 0
    squat_direction = 0
    pushup_direction = 0
    lunge_direction = 0
    plank_direction = 0
    plank_start = None

    pTime = time.time()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)

        if lmList and len(lmList) >= 33:
            # Right Arm
            shoulder_r = lmList[12][1:]
            elbow_r = lmList[14][1:]
            wrist_r = lmList[16][1:]

            # Left Arm
            shoulder_l = lmList[11][1:]
            elbow_l = lmList[13][1:]
            wrist_l = lmList[15][1:]

            # Right Leg
            hip_r = lmList[24][1:]
            knee_r = lmList[26][1:]
            ankle_r = lmList[28][1:]

            # Left Leg
            hip_l = lmList[23][1:]
            knee_l = lmList[25][1:]
            ankle_l = lmList[27][1:]

            angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
            angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
            angle_squat = calculate_angle(hip_r, knee_r, ankle_r)
            angle_lunge = angle_squat
            angle_plank = angle_squat

            # Right Curl
            if angle_r > 160:
                if direction_right == 1:
                    count_right += 1
                    direction_right = 0
            elif angle_r < 50:
                if direction_right == 0:
                    direction_right = 1

            # Left Curl
            if angle_l > 160:
                if direction_left == 1:
                    count_left += 1
                    direction_left = 0
            elif angle_l < 50:
                if direction_left == 0:
                    direction_left = 1

            # Squat
            if angle_squat > 160:
                if squat_direction == 1:
                    squat_count += 1
                    squat_direction = 0
            elif angle_squat < 110:
                if squat_direction == 0:
                    squat_direction = 1

            # Push-up
            if angle_r < 80 and angle_l < 80:
                if pushup_direction == 0:
                    pushup_direction = 1
            elif angle_r > 160 and angle_l > 160:
                if pushup_direction == 1:
                    pushup_count += 1
                    pushup_direction = 0

            # Lunge
            if 60 < angle_lunge < 80:
                if lunge_direction == 1:
                    lunge_count += 1
                    lunge_direction = 0
            elif angle_lunge > 150:
                if lunge_direction == 0:
                    lunge_direction = 1

            # Plank
            body_angle = calculate_angle(shoulder_l, hip_l, knee_l)
            if angle_plank > 170 and body_angle < 20:
                if plank_direction == 0:
                    plank_direction = 1
                    plank_start = time.time()
            elif angle_plank < 160:
                if plank_direction == 1:
                    plank_direction = 0
                    if plank_start:
                        plank_timer += time.time() - plank_start
                        plank_start = None

            if plank_direction == 1 and plank_start:
                plank_timer = time.time() - plank_start

        # Update analytics
        analytics.update("right_curl", count_right)
        analytics.update("left_curl", count_left)
        analytics.update("squat", squat_count)
        analytics.update("pushup", pushup_count)
        analytics.update("lunge", lunge_count)
        analytics.update("plank", int(plank_timer))

        # Dashboard
        elapsed = int(time.time() - analytics.start_time)
        reps = {
            "Right Curls": count_right,
            "Left Curls": count_left,
            "Squats": squat_count,
            "Push-ups": pushup_count,
            "Lunges": lunge_count,
            "Plank (s)": int(plank_timer)
        }
        update_dashboard(dashboard, reps, elapsed)

        cv2.imshow("Fitness Tracker", img)
        cv2.imshow("Workout Dashboard", dashboard)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    analytics.end_session()

    # Plot today's workout
    today_counts = reps
    exercises = list(today_counts.keys())
    values = list(today_counts.values())

    figures_dir = os.path.join(analytics.data_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(exercises, values, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6'])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, int(yval), ha='center', va='bottom')
    plt.title(f"Workout Summary ({analytics.today_date})")
    plt.ylabel('Repetitions / Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plot_path = os.path.join(figures_dir, f"workout_{analytics.today_date}.png")
    plt.savefig(plot_path)
    plt.show()

if __name__ == "__main__":
    main()
