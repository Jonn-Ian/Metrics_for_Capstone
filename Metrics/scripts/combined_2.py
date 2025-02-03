import cv2
import mediapipe as mp
import pandas as pd
import math
import numpy as np
import dlib
import os
import time
import pygame

# Load the CSV data with landmarks and angles
csv_file = r'E:\python\python_projects\yolov7\latest\datas\csv_landmarks\pose_landmarks_and_angles.csv'
df = pd.read_csv(csv_file)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Initialize Dlib's face detector and face recognition model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'E:\python\python_projects\yolov7\latest\datas\models\shape_predictor_68_face_landmarks.dat')
recognizer = dlib.face_recognition_model_v1(r'E:\python\python_projects\yolov7\latest\datas\models\dlib_face_recognition_resnet_model_v1.dat')

# Load saved face encodings and names
encoding_dir = r'E:\python\python_projects\yolov7\latest\datas\encodings'
face_encodings = np.load(os.path.join(encoding_dir, 'face_encodings.npy'), allow_pickle=True)
face_names = np.load(os.path.join(encoding_dir, 'face_names.npy'), allow_pickle=True)

# Video feed (use a looped video or webcam)
video_path = r'E:\python\python_projects\yolov7\latest\medias\video\videoplayback.mp4'  # Replace with the path to your test video
cap = cv2.VideoCapture(video_path)

# Initialize pygame for alarm sound
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound(r'E:\python\python_projects\yolov7\latest\medias\sounds\alarm.wav')  # Change to your alarm sound file

# Function to calculate angle between three points
def calculate_angle(A, B, C):
    BA = [A.x - B.x, A.y - B.y]
    BC = [C.x - B.x, C.y - B.y]
    cosine_angle = (BA[0] * BC[0] + BA[1] * BC[1]) / (math.sqrt(BA[0] ** 2 + BA[1] ** 2) * math.sqrt(BC[0] ** 2 + BC[1] ** 2))
    angle = math.degrees(math.acos(cosine_angle))
    return angle

# Function to compute the Euclidean distance between two angles
def angle_difference(angle1, angle2):
    return abs(angle1 - angle2)

# Function to compute similarity score (higher is better)
def compute_similarity(angles1, angles2):
    total_diff = 0
    count = 0
    for angle1, angle2 in zip(angles1, angles2):
        total_diff += angle_difference(angle1, angle2)
        count += 1
    return 100 - (total_diff / count)  # Normalize to a percentage (0-100)

# Function to generate feedback based on angle differences, but only one at a time
def generate_joint_feedback(test_angles, ref_angles):
    angle_diffs = [angle_difference(test_angle, ref_angle) for test_angle, ref_angle in zip(test_angles, ref_angles)]
    max_diff_index = np.argmax(angle_diffs)  # Find joint with largest discrepancy

    # Check if the discrepancy is large enough to trigger feedback
    if angle_diffs[max_diff_index] > 5:  # Adjust threshold as needed
        if max_diff_index == 0:
            return "Adjust your left arm."
        elif max_diff_index == 1:
            return "Adjust your right arm."
        elif max_diff_index == 2:
            return "Adjust your left knee."
        elif max_diff_index == 3:
            return "Adjust your right knee."
    else:
        return "Form is good, no significant adjustment needed."

# Function for face recognition
def recognize_face(frame, gray_frame):
    faces = detector(gray_frame)
    for face in faces:
        try:
            # Use the predictor to detect landmarks
            shape = predictor(gray_frame, face)  # Use the grayscale frame

            # Check if the shape is valid before computing the descriptor
            if shape.num_parts == 68:
                # Compute the face descriptor
                encoding = recognizer.compute_face_descriptor(frame, shape)

                # Compare with known faces
                matches = []
                for known_face in face_encodings:
                    match = np.linalg.norm(np.array(encoding) - np.array(known_face)) < 0.6  # Adjust the threshold as needed
                    matches.append(match)

                if any(matches):
                    name_index = matches.index(True)
                    name = face_names[name_index]
                    return name  # Return recognized name
                else:
                    return "Unknown"  # If unknown face is detected
            else:
                print("Invalid shape detected for recognition.")
        except Exception as e:
            print(f"Error during face recognition: {e}")
    return "No Face"  # Return if no face is detected

# Initialize variables for pose estimation comparison
frame_index = 0
total_similarity = 0
num_frames = 0
loop_count = 0

# Face Authentication Timer
start_time = time.time()
auth_timeout = 30  # 30 seconds timeout for authentication

# Movement Detection Variables
last_angles = None
inactive_start_time = None
alarm_triggered = False

while cap.isOpened() and loop_count < 2:
    ret, frame = cap.read()
    if not ret:
        # Reset to the beginning of the video when it ends
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        loop_count += 1
        continue

    # Convert the frame to RGB for MediaPipe Pose processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Convert the frame to grayscale for face detection and recognition
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face recognition
    recognized_name = recognize_face(frame, gray_frame)

    if recognized_name == "Unknown":
        print("Unknown face detected. Exiting in 5 seconds...")
        time.sleep(5)  # Wait for 5 seconds before quitting
        break  # Exit if an unknown face is detected
    
    if recognized_name != "No Face":
        elapsed_time = time.time() - start_time
        if elapsed_time > auth_timeout:
            print("Authentication timeout. Exiting...")
            break  # Exit if authentication takes too long
        cv2.putText(frame, f"Authenticated as {recognized_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if results.pose_landmarks:
        # Get landmarks and calculate angles in the current frame for pose estimation
        landmarks = results.pose_landmarks.landmark
        left_elbow_angle = calculate_angle(
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        )
        right_elbow_angle = calculate_angle(
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        )
        left_knee_angle = calculate_angle(
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        )
        right_knee_angle = calculate_angle(
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        )

        # Compare angles to detect movement
        current_angles = [left_elbow_angle, right_elbow_angle, left_knee_angle, right_knee_angle]
        
        if last_angles:
            movement_detected = any(abs(curr - last) > 5 for curr, last in zip(current_angles, last_angles))
        else:
            movement_detected = False  # First frame, assume no movement initially

        if movement_detected:
            inactive_start_time = None  # Reset inactivity timer if movement is detected
            alarm_triggered = False  # Reset alarm status if the user moves
            print("Movement detected, resetting inactivity timer.")
        elif inactive_start_time is None:
            inactive_start_time = time.time()  # Start inactivity timer

        if inactive_start_time and not alarm_triggered:
            # Check if user has been inactive for 60 seconds after a 5 second delay
            if time.time() - inactive_start_time > 5:  # 5 seconds delay before starting to count inactivity
                if time.time() - inactive_start_time > 65:  # 60 seconds of inactivity
                    print("No movement detected for 60 seconds. Triggering alarm.")
                    alarm_sound.play()
                    alarm_triggered = True  # Alarm triggered, stop playing it again

        last_angles = current_angles  # Update the last angles for next frame

        # Accumulate similarity over all frames
        if frame_index % 5 == 0:  # Display feedback every 5 frames
            feedback = generate_joint_feedback(current_angles, current_angles)  # Reference angles can be updated as needed
            cv2.putText(frame, feedback, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Exercise Monitoring', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
