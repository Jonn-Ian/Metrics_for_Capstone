import cv2
import dlib
import numpy as np
import time
import signal
import sys
import mediapipe as mp
import pygame
import math
import pandas as pd
import os
# Initialize Pygame mixer for sound
pygame.mixer.init()

# Load the sound file
try:
    warning_sound = pygame.mixer.Sound(r'E:\python\python_projects\yolov7\latest\medias\sounds\alarm.wav')
except FileNotFoundError:
    print("Warning sound file not found. Please check the path.")
    exit()

# Load Dlib's face detector and face recognition model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'E:\python\python_projects\yolov7\latest\datas\models\shape_predictor_68_face_landmarks.dat')
recognizer = dlib.face_recognition_model_v1(r'E:\python\python_projects\yolov7\latest\datas\models\dlib_face_recognition_resnet_model_v1.dat')

# Load the CSV data with landmarks and angles
csv_file = r'E:\python\python_projects\yolov7\latest\datas\csv_landmarks\pose_landmarks_and_angles.csv'
df = pd.read_csv(csv_file)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Directory to save encodings
encoding_dir = r'E:\python\python_projects\yolov7\latest\datas\encodings'

# Function to encode faces from a directory
def encode_faces_from_directory(image_folder):
    encodings = []
    names = []
    
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.jpg', '.png')):  # Handle case sensitivity
            image_path = os.path.join(image_folder, filename)
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            if faces:
                face = faces[0]  # Take the first detected face
                shape = predictor(gray, face)  # Get the shape of the detected face
                
                if shape.num_parts == 68:  # Check if the expected number of landmarks is detected
                    encoding = recognizer.compute_face_descriptor(img, shape)
                    encodings.append(np.array(encoding))
                    name = os.path.basename(image_path).split('.')[0]  # Use the filename as the name
                    names.append(name)
                else:
                    print(f"Unexpected number of landmarks detected in {filename}. Expected 68, got {shape.num_parts}.")
            else:
                print(f"No face detected in {filename}.")

    return encodings, names

# Example usage for encoding
image_folder = r'E:\python\python_projects\yolov7\latest\medias\images'  # Update this path to your folder
face_encodings, face_names = encode_faces_from_directory(image_folder)

# Save encodings for later use
if face_encodings:
    np.save(os.path.join(encoding_dir, 'face_encodings.npy'), face_encodings)
    np.save(os.path.join(encoding_dir, 'face_names.npy'), face_names)

# Load saved encodings
face_encodings = np.load(os.path.join(encoding_dir, 'face_encodings.npy'), allow_pickle=True)
face_names = np.load(os.path.join(encoding_dir, 'face_names.npy'), allow_pickle=True)

# Function to calculate angle between three points
def calculate_angle(A, B, C):
    """Calculate the angle ABC (in degrees) between points A, B, and C."""
    BA = [A.x - B.x, A.y - B.y]
    BC = [C.x - B.x, C.y - B.y]
    cosine_angle = (BA[0] * BC[0] + BA[1] * BC[1]) / (math.sqrt(BA[0] ** 2 + BA[1] ** 2) * math.sqrt(BC[0] ** 2 + BC[1] ** 2))
    angle = math.degrees(math.acos(cosine_angle))
    return angle

# Function to calculate distance between landmarks
def calculate_distance(landmarks):
    if landmarks:
        distances = []
        for i in range(len(landmarks)):
            for j in range(i + 1, len(landmarks)):
                point1 = np.array([landmarks[i].x, landmarks[i].y])
                point2 = np.array([landmarks[j].x, landmarks[j].y])
                distances.append(np.linalg.norm(point1 - point2))
        return np.mean(distances)
    return 0

# Function to handle cleanup on exit
def signal_handler(sig, frame):
    print("Exiting gracefully...")
    cap.release()  # Release the video
    cv2.destroyAllWindows()  # Close all OpenCV windows
    sys.exit(0)  # Exit the program

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# Main function
def main():
    global cap
    cap = cv2.VideoCapture(0)  # Use webcam as the input source
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    last_motion_time = time.time()  # Track the last time motion was detected
    previous_dist = 0  # Track previous distance for movement comparison
    frame_index = 0  # For pose feedback reference
    completed_reps = 0  # Initialize completed repetitions counter
    rep_threshold = 30  # Threshold angle for a completed push-up (example)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video.")
            break

        # Convert the frame to RGB for MediaPipe and face detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face detection
        faces = detector(gray_frame)
        face_recognized = None
        
        for face in faces:
            try:
                shape = predictor(gray_frame, face)  # Use the grayscale frame
                
                if shape.num_parts == 68:
                    encoding = recognizer.compute_face_descriptor(frame, shape)

                    # Compare with known faces
                    matches = [np.linalg.norm(np.array(encoding) - np.array(known_face)) < 0.6 for known_face in face_encodings]

                    if any(matches):
                        name_index = matches.index(True)
                        face_recognized = face_names[name_index]
                    else:
                        face_recognized = "Unknown"

            except Exception as e:
                print(f"Error during face recognition: {e}")

        # Process the image for pose detection
        pose_results = pose.process(rgb_frame)

        if pose_results.pose_landmarks:
            current_dist = calculate_distance(pose_results.pose_landmarks.landmark)

            if abs(current_dist - previous_dist) > 0.01:  # Adjust threshold as needed
                last_motion_time = time.time()

            previous_dist = current_dist

            # Calculate angles and provide feedback
            left_elbow_angle = calculate_angle(
                pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value]
            )
            right_elbow_angle = calculate_angle(
                pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            )
            left_knee_angle = calculate_angle(
                pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value],
                pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value],
                pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            )
            right_knee_angle = calculate_angle(
                pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value],
                pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            )

            try:
                ref_row = df.iloc[frame_index]
                feedback = []

                # Check angles and provide feedback
                if abs(left_elbow_angle - ref_row['Left_Elbow_Angle']) > 10:
                    feedback.append("Adjust left elbow position.")
                if abs(right_elbow_angle - ref_row['Right_Elbow_Angle']) > 10:
                    feedback.append("Adjust right elbow position.")
                if abs(left_knee_angle - ref_row['Left_Knee_Angle']) > 10:
                    feedback.append("Adjust left knee position.")
                if abs(right_knee_angle - ref_row['Right_Knee_Angle']) > 10:
                    feedback.append("Adjust right knee position.")

                # Display feedback on video frame
                for i, message in enumerate(feedback):
                    cv2.putText(frame, message, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # If feedback is empty, user is doing well
                if not feedback:
                    cv2.putText(frame, "Good form!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Count repetitions based on angles
                if left_elbow_angle < rep_threshold and right_elbow_angle < rep_threshold:
                    completed_reps += 1
                    print(f"Completed Repetitions: {completed_reps}")

            except IndexError:
                # If frame index exceeds the reference data, reset or loop
                frame_index = 0

        # Display the frame
        cv2.imshow('Face Detection and Pose Estimation', frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Increment frame index for CSV reference
        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
