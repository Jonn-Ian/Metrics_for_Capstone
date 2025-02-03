import cv2
import dlib
import numpy as np
import os
import time
import signal
import sys
import mediapipe as mp
import pygame
import math
import pandas as pd

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

# Load the Caffe model
face_net = cv2.dnn.readNetFromCaffe(
    r'E:\python\python_projects\yolov7\latest\datas\for_cuda\deploy.prototxt',
    r'E:\python\python_projects\yolov7\latest\datas\for_cuda\res10_300x300_ssd_iter_140000.caffemodel'
)

# Set the preferable backend and target
face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Function to encode faces from a directory
def encode_faces_from_directory(image_folder):
    encodings = []
    names = []
    
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.jpg', '.png')):
            image_path = os.path.join(image_folder, filename)
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            if faces:
                face = faces[0]  # Take the first detected face
                shape = predictor(gray, face)
                
                if shape.num_parts == 68:
                    encoding = recognizer.compute_face_descriptor(img, shape)
                    encodings.append(np.array(encoding))
                    name = os.path.basename(image_path).split('.')[0]
                    names.append(name)
                else:
                    print(f"Unexpected number of landmarks detected in {filename}. Expected 68, got {shape.num_parts}.")
            else:
                print(f"No face detected in {filename}.")

    return encodings, names

# Example usage for encoding
image_folder = r'E:\python\python_projects\yolov7\latest\medias\images'
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
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# Main function
def main():
    global cap
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    last_motion_time = time.time()
    previous_dist = 0
    frame_index = 0
    start_time = time.time()
    completed_exercises = 0
    recognized_user = None
    exercise_done = False
    frame_skip = 2

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video")
            break

        if frame_index % frame_skip != 0:
            frame_index += 1
            continue

        # Prepare the frame for detection
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()

        # Face detection
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Adjust confidence threshold as needed
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")
                faces.append((startX, startY, endX, endY))

        if faces:
            (startX, startY, endX, endY) = faces[0]
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            try:
                shape = predictor(gray_frame, dlib.rectangle(startX, startY, endX, endY))
                if shape.num_parts == 68:
                    encoding = recognizer.compute_face_descriptor(frame, shape)
                    matches = [np.linalg.norm(np.array(encoding) - np.array(known_face)) < 0.6 for known_face in face_encodings]

                    if any(matches):
                        name_index = matches.index(True)
                        recognized_user = face_names[name_index]
                    else:
                        recognized_user = "Unknown"
            except Exception as e:
                print(f"Error during face recognition: {e}")

        # Process the image for pose detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_frame)

        # Motion detection
        motion_detected = False

        if pose_results.pose_landmarks:
            current_dist = calculate_distance(pose_results.pose_landmarks.landmark)

            if abs(current_dist - previous_dist) > 0.01:  # Adjust threshold as needed
                last_motion_time = time.time()
                motion_detected = True
            
            previous_dist = current_dist

        # Check elapsed time without significant motion
        elapsed_time = time.time() - last_motion_time

        total_elapsed_time = time.time() - start_time
        print(f"Elapsed Time: {total_elapsed_time:.2f} seconds")

        if elapsed_time >= 5:
            print("No significant movement detected for 5 seconds! Playing warning sound...")
            warning_sound.play()
            time.sleep(2)
            warning_sound.stop()
            last_motion_time = time.time()

        # Feedback for pose estimation
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            
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

            try:
                ref_row = df.iloc[frame_index]
                feedback = []

                if abs(left_elbow_angle - ref_row['Left_Elbow_Angle']) > 10:
                    feedback.append("Adjust left elbow position.")
                if abs(right_elbow_angle - ref_row['Right_Elbow_Angle']) > 10:
                    feedback.append("Adjust right elbow position.")
                if abs(left_knee_angle - ref_row['Left_Knee_Angle']) > 10:
                    feedback.append("Adjust left knee position.")
                if abs(right_knee_angle - ref_row['Right_Knee_Angle']) > 10:
                    feedback.append("Adjust right knee position.")

                for i, message in enumerate(feedback):
                    cv2.putText(frame, message, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if recognized_user and recognized_user != "Unknown" and not exercise_done:
                    completed_exercises += 1
                    exercise_done = True
                    print(f"Exercises Completed: {completed_exercises}")
                elif recognized_user == "Unknown":
                    exercise_done = False

                if not feedback:
                    cv2.putText(frame, "Good form!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            except IndexError:
                frame_index = 0

        # Display the frame
        cv2.imshow('Face Detection, Motion Detection, and Pose Estimation', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
