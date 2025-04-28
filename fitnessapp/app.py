# # IMPORTS
# from flask import Flask, jsonify, request, Response
# import cv2
# import mediapipe as mp
# import time
# import numpy as np
# import os
# import json
# from datetime import datetime
# import threading
# import base64
#
# app = Flask(__name__)
#
#
# # -------------------- Pose Detection Class --------------------
# class PoseDetector:
#     def __init__(self):
#         self.mpDraw = mp.solutions.drawing_utils
#         self.mpPose = mp.solutions.pose
#         self.pose = self.mpPose.Pose()
#         self.results = None
#
#     def findPose(self, img, draw=True):
#         imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         self.results = self.pose.process(imgRGB)
#         if self.results.pose_landmarks and draw:
#             self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
#         return img
#
#     def findPosition(self, img, draw=True):
#         lmList = []
#         if self.results.pose_landmarks:
#             h, w, c = img.shape
#             for id, lm in enumerate(self.results.pose_landmarks.landmark):
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 lmList.append([id, cx, cy])
#                 if draw:
#                     cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
#         return lmList
#
#
# # -------------------- Workout Analytics Class --------------------
# class WorkoutAnalytics:
#     def __init__(self, user_name="user1"):
#         self.user_name = user_name
#         self.data_dir = f"workout_data/{user_name}"
#         os.makedirs(self.data_dir, exist_ok=True)
#         self.today_date = datetime.now().strftime("%Y-%m-%d")
#         self.start_time = time.time()
#         self.data = {
#             "right_curl": 0,
#             "left_curl": 0,
#             "squat": 0,
#             "pushup": 0,
#             "lunge": 0,
#             "plank": 0
#         }
#
#     def update(self, exercise, count):
#         if exercise in self.data:
#             self.data[exercise] = count
#
#     def end_session(self):
#         end_time = time.time()
#         total_time = int(end_time - self.start_time)
#         calories_burned = (total_time / 60) * 5
#
#         session = {
#             "date": self.today_date,
#             "total_time_sec": total_time,
#             "calories_burned": calories_burned,
#             "data": self.data
#         }
#
#         file_path = os.path.join(self.data_dir, "workout_history.json")
#         if os.path.exists(file_path):
#             with open(file_path, "r") as f:
#                 history = json.load(f)
#         else:
#             history = []
#
#         history.append(session)
#
#         with open(file_path, "w") as f:
#             json.dump(history, f, indent=4)
#
#         return session
#
#
# # -------------------- Utility Functions --------------------
# def calculate_angle(a, b, c):
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)
#     radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
#     angle = np.abs(radians * 180.0 / np.pi)
#     if angle > 180.0:
#         angle = 360 - angle
#     return angle
#
#
# # Global workout session tracking
# class WorkoutSession:
#     def __init__(self):
#         self.active = False
#         self.detector = None
#         self.analytics = None
#         self.count_right = 0
#         self.count_left = 0
#         self.squat_count = 0
#         self.pushup_count = 0
#         self.lunge_count = 0
#         self.plank_timer = 0
#         self.direction_right = 0
#         self.direction_left = 0
#         self.squat_direction = 0
#         self.pushup_direction = 0
#         self.lunge_direction = 0
#         self.plank_direction = 0
#         self.plank_start = None
#         self.last_processed_frame = None
#
#
# # Initialize the global session
# workout_session = WorkoutSession()
#
#
# # -------------------- API Endpoints --------------------
# @app.route('/api/start_session', methods=['POST'])
# def start_session():
#     global workout_session
#
#     # Get user from request
#     data = request.get_json()
#     user_name = data.get('user_name', 'user1')
#
#     # Initialize session
#     workout_session = WorkoutSession()
#     workout_session.active = True
#     workout_session.detector = PoseDetector()
#     workout_session.analytics = WorkoutAnalytics(user_name)
#
#     return jsonify({
#         'status': 'success',
#         'message': f'Workout session started for {user_name}',
#         'start_time': workout_session.analytics.start_time
#     })
#
#
# @app.route('/api/end_session', methods=['POST'])
# def end_session():
#     global workout_session
#
#     if not workout_session.active:
#         return jsonify({
#             'status': 'error',
#             'message': 'No active workout session'
#         }), 400
#
#     # End the session and get summary
#     session_data = workout_session.analytics.end_session()
#     workout_session.active = False
#
#     # Return summary
#     return jsonify({
#         'status': 'success',
#         'message': 'Workout session ended',
#         'summary': session_data
#     })
#
#
# @app.route('/api/process_frame', methods=['POST'])
# def process_frame():
#     global workout_session
#
#     if not workout_session.active:
#         return jsonify({
#             'status': 'error',
#             'message': 'No active workout session'
#         }), 400
#
#     # Get the image from the request
#     if 'image' not in request.files:
#         return jsonify({
#             'status': 'error',
#             'message': 'No image in request'
#         }), 400
#
#     # Decode the image
#     file = request.files['image']
#     img_bytes = file.read()
#     nparr = np.frombuffer(img_bytes, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#
#     if img is None:
#         return jsonify({
#             'status': 'error',
#             'message': 'Invalid image'
#         }), 400
#
#     # Process the image
#     img = workout_session.detector.findPose(img)
#     lmList = workout_session.detector.findPosition(img, draw=False)
#
#     # Save processed frame
#     workout_session.last_processed_frame = img
#
#     # Process exercise counts
#     if lmList and len(lmList) >= 33:
#         # Right Arm
#         shoulder_r = lmList[12][1:]
#         elbow_r = lmList[14][1:]
#         wrist_r = lmList[16][1:]
#
#         # Left Arm
#         shoulder_l = lmList[11][1:]
#         elbow_l = lmList[13][1:]
#         wrist_l = lmList[15][1:]
#
#         # Right Leg
#         hip_r = lmList[24][1:]
#         knee_r = lmList[26][1:]
#         ankle_r = lmList[28][1:]
#
#         # Left Leg
#         hip_l = lmList[23][1:]
#         knee_l = lmList[25][1:]
#         ankle_l = lmList[27][1:]
#
#         angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
#         angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
#         angle_squat = calculate_angle(hip_r, knee_r, ankle_r)
#         angle_lunge = angle_squat
#
#         # Right Curl
#         if angle_r > 160:
#             if workout_session.direction_right == 1:
#                 workout_session.count_right += 1
#                 workout_session.direction_right = 0
#         elif angle_r < 50:
#             if workout_session.direction_right == 0:
#                 workout_session.direction_right = 1
#
#         # Left Curl
#         if angle_l > 160:
#             if workout_session.direction_left == 1:
#                 workout_session.count_left += 1
#                 workout_session.direction_left = 0
#         elif angle_l < 50:
#             if workout_session.direction_left == 0:
#                 workout_session.direction_left = 1
#
#         # Squat
#         if angle_squat > 160:
#             if workout_session.squat_direction == 1:
#                 workout_session.squat_count += 1
#                 workout_session.squat_direction = 0
#         elif angle_squat < 110:
#             if workout_session.squat_direction == 0:
#                 workout_session.squat_direction = 1
#
#         # Push-up
#         if angle_r < 80 and angle_l < 80:
#             if workout_session.pushup_direction == 0:
#                 workout_session.pushup_direction = 1
#         elif angle_r > 160 and angle_l > 160:
#             if workout_session.pushup_direction == 1:
#                 workout_session.pushup_count += 1
#                 workout_session.pushup_direction = 0
#
#         # Lunge
#         if 60 < angle_lunge < 80:
#             if workout_session.lunge_direction == 1:
#                 workout_session.lunge_count += 1
#                 workout_session.lunge_direction = 0
#         elif angle_lunge > 150:
#             if workout_session.lunge_direction == 0:
#                 workout_session.lunge_direction = 1
#
#         # Plank
#         body_angle = calculate_angle(shoulder_l, hip_l, knee_l)
#         if angle_squat > 170 and body_angle < 20:
#             if workout_session.plank_direction == 0:
#                 workout_session.plank_direction = 1
#                 workout_session.plank_start = time.time()
#         elif angle_squat < 160:
#             if workout_session.plank_direction == 1:
#                 workout_session.plank_direction = 0
#                 if workout_session.plank_start:
#                     workout_session.plank_timer += time.time() - workout_session.plank_start
#                     workout_session.plank_start = None
#
#         if workout_session.plank_direction == 1 and workout_session.plank_start:
#             workout_session.plank_timer = time.time() - workout_session.plank_start
#
#     # Update analytics
#     workout_session.analytics.update("right_curl", workout_session.count_right)
#     workout_session.analytics.update("left_curl", workout_session.count_left)
#     workout_session.analytics.update("squat", workout_session.squat_count)
#     workout_session.analytics.update("pushup", workout_session.pushup_count)
#     workout_session.analytics.update("lunge", workout_session.lunge_count)
#     workout_session.analytics.update("plank", int(workout_session.plank_timer))
#
#     # Get current stats
#     elapsed = int(time.time() - workout_session.analytics.start_time)
#     reps = {
#         "right_curls": workout_session.count_right,
#         "left_curls": workout_session.count_left,
#         "squats": workout_session.squat_count,
#         "pushups": workout_session.pushup_count,
#         "lunges": workout_session.lunge_count,
#         "plank_seconds": int(workout_session.plank_timer)
#     }
#
#     # Convert processed image to base64 for response
#     _, buffer = cv2.imencode('.jpg', workout_session.last_processed_frame)
#     img_str = base64.b64encode(buffer).decode('utf-8')
#
#     return jsonify({
#         'status': 'success',
#         'workout_time': elapsed,
#         'exercises': reps,
#         'processed_image': img_str
#     })
#
#
# @app.route('/api/get_workout_stats', methods=['GET'])
# def get_workout_stats():
#     global workout_session
#
#     if not workout_session.active:
#         return jsonify({
#             'status': 'error',
#             'message': 'No active workout session'
#         }), 400
#
#     # Get current stats
#     elapsed = int(time.time() - workout_session.analytics.start_time)
#     reps = {
#         "right_curls": workout_session.count_right,
#         "left_curls": workout_session.count_left,
#         "squats": workout_session.squat_count,
#         "pushups": workout_session.pushup_count,
#         "lunges": workout_session.lunge_count,
#         "plank_seconds": int(workout_session.plank_timer)
#     }
#
#     return jsonify({
#         'status': 'success',
#         'workout_time': elapsed,
#         'exercises': reps
#     })
#
#
# @app.route('/api/get_workout_history', methods=['GET'])
# def get_workout_history():
#     user_name = request.args.get('user_name', 'user1')
#     data_dir = f"workout_data/{user_name}"
#     file_path = os.path.join(data_dir, "workout_history.json")
#
#     if not os.path.exists(file_path):
#         return jsonify({
#             'status': 'success',
#             'history': []
#         })
#
#     with open(file_path, "r") as f:
#         history = json.load(f)
#
#     return jsonify({
#         'status': 'success',
#         'history': history
#     })
#
#
# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=5000, debug=True)


# IMPORTS
from flask import Flask, jsonify, request, Response
from flask_cors import CORS  # Add CORS support
import cv2
import mediapipe as mp
import time
import numpy as np
import os
import json
from datetime import datetime
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# -------------------- Pose Detection Class --------------------
class PoseDetector:
    def __init__(self):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.results = None

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
    def __init__(self, user_name="user1"):
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

        return session


# -------------------- Utility Functions --------------------
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


# Global workout session tracking
class WorkoutSession:
    def __init__(self):
        self.active = False
        self.detector = None
        self.analytics = None
        self.count_right = 0
        self.count_left = 0
        self.squat_count = 0
        self.pushup_count = 0
        self.lunge_count = 0
        self.plank_timer = 0
        self.direction_right = 0
        self.direction_left = 0
        self.squat_direction = 0
        self.pushup_direction = 0
        self.lunge_direction = 0
        self.plank_direction = 0
        self.plank_start = None
        self.last_processed_frame = None


# Initialize the global session
workout_session = WorkoutSession()


# Simple test endpoint to verify API is working
@app.route('/api/test', methods=['GET'])
def test_api():
    return jsonify({
        'status': 'success',
        'message': 'API is working correctly!'
    })


# -------------------- API Endpoints --------------------
@app.route('/api/start_session', methods=['POST'])
def start_session():
    global workout_session

    # Get user from request
    data = request.get_json()
    user_name = data.get('user_name', 'user1')

    # Initialize session
    workout_session = WorkoutSession()
    workout_session.active = True
    workout_session.detector = PoseDetector()
    workout_session.analytics = WorkoutAnalytics(user_name)

    return jsonify({
        'status': 'success',
        'message': f'Workout session started for {user_name}',
        'start_time': workout_session.analytics.start_time
    })


@app.route('/api/end_session', methods=['POST'])
def end_session():
    global workout_session

    if not workout_session.active:
        return jsonify({
            'status': 'error',
            'message': 'No active workout session'
        }), 400

    # End the session and get summary
    session_data = workout_session.analytics.end_session()
    workout_session.active = False

    # Return summary
    return jsonify({
        'status': 'success',
        'message': 'Workout session ended',
        'summary': session_data
    })


@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    global workout_session

    if not workout_session.active:
        return jsonify({
            'status': 'error',
            'message': 'No active workout session'
        }), 400

    # Get the image from the request
    if 'image' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No image in request'
        }), 400

    # Decode the image
    file = request.files['image']
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({
            'status': 'error',
            'message': 'Invalid image'
        }), 400

    # Process the image
    img = workout_session.detector.findPose(img)
    lmList = workout_session.detector.findPosition(img, draw=False)

    # Save processed frame
    workout_session.last_processed_frame = img

    # Process exercise counts
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

        # Right Curl
        if angle_r > 160:
            if workout_session.direction_right == 1:
                workout_session.count_right += 1
                workout_session.direction_right = 0
        elif angle_r < 50:
            if workout_session.direction_right == 0:
                workout_session.direction_right = 1

        # Left Curl
        if angle_l > 160:
            if workout_session.direction_left == 1:
                workout_session.count_left += 1
                workout_session.direction_left = 0
        elif angle_l < 50:
            if workout_session.direction_left == 0:
                workout_session.direction_left = 1

        # Squat
        if angle_squat > 160:
            if workout_session.squat_direction == 1:
                workout_session.squat_count += 1
                workout_session.squat_direction = 0
        elif angle_squat < 110:
            if workout_session.squat_direction == 0:
                workout_session.squat_direction = 1

        # Push-up
        if angle_r < 80 and angle_l < 80:
            if workout_session.pushup_direction == 0:
                workout_session.pushup_direction = 1
        elif angle_r > 160 and angle_l > 160:
            if workout_session.pushup_direction == 1:
                workout_session.pushup_count += 1
                workout_session.pushup_direction = 0

        # Lunge
        if 60 < angle_lunge < 80:
            if workout_session.lunge_direction == 1:
                workout_session.lunge_count += 1
                workout_session.lunge_direction = 0
        elif angle_lunge > 150:
            if workout_session.lunge_direction == 0:
                workout_session.lunge_direction = 1

        # Plank
        body_angle = calculate_angle(shoulder_l, hip_l, knee_l)
        if angle_squat > 170 and body_angle < 20:
            if workout_session.plank_direction == 0:
                workout_session.plank_direction = 1
                workout_session.plank_start = time.time()
        elif angle_squat < 160:
            if workout_session.plank_direction == 1:
                workout_session.plank_direction = 0
                if workout_session.plank_start:
                    workout_session.plank_timer += time.time() - workout_session.plank_start
                    workout_session.plank_start = None

        if workout_session.plank_direction == 1 and workout_session.plank_start:
            workout_session.plank_timer = time.time() - workout_session.plank_start

    # Update analytics
    workout_session.analytics.update("right_curl", workout_session.count_right)
    workout_session.analytics.update("left_curl", workout_session.count_left)
    workout_session.analytics.update("squat", workout_session.squat_count)
    workout_session.analytics.update("pushup", workout_session.pushup_count)
    workout_session.analytics.update("lunge", workout_session.lunge_count)
    workout_session.analytics.update("plank", int(workout_session.plank_timer))

    # Get current stats
    elapsed = int(time.time() - workout_session.analytics.start_time)
    reps = {
        "right_curls": workout_session.count_right,
        "left_curls": workout_session.count_left,
        "squats": workout_session.squat_count,
        "pushups": workout_session.pushup_count,
        "lunges": workout_session.lunge_count,
        "plank_seconds": int(workout_session.plank_timer)
    }

    # Convert processed image to base64 for response
    _, buffer = cv2.imencode('.jpg', workout_session.last_processed_frame)
    img_str = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'status': 'success',
        'workout_time': elapsed,
        'exercises': reps,
        'processed_image': img_str
    })


@app.route('/api/get_workout_stats', methods=['GET'])
def get_workout_stats():
    global workout_session

    if not workout_session.active:
        return jsonify({
            'status': 'error',
            'message': 'No active workout session'
        }), 400

    # Get current stats
    elapsed = int(time.time() - workout_session.analytics.start_time)
    reps = {
        "right_curls": workout_session.count_right,
        "left_curls": workout_session.count_left,
        "squats": workout_session.squat_count,
        "pushups": workout_session.pushup_count,
        "lunges": workout_session.lunge_count,
        "plank_seconds": int(workout_session.plank_timer)
    }

    return jsonify({
        'status': 'success',
        'workout_time': elapsed,
        'exercises': reps
    })


@app.route('/api/get_workout_history', methods=['GET'])
def get_workout_history():
    user_name = request.args.get('user_name', 'user1')
    data_dir = f"workout_data/{user_name}"
    file_path = os.path.join(data_dir, "workout_history.json")

    if not os.path.exists(file_path):
        return jsonify({
            'status': 'success',
            'history': []
        })

    with open(file_path, "r") as f:
        history = json.load(f)

    return jsonify({
        'status': 'success',
        'history': history
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)