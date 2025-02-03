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

# Load the face detection model
face_net = cv2.dnn.readNetFromCaffe(
    r'E:\python\python_projects\yolov7\latest\datas\for_cuda\deploy.prototxt',
    r'E:\python\python_projects\yolov7\latest\datas\for_cuda\res10_300x300_ssd_iter_140000.caffemodel'
)

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

# Load saved encodings
encoding_dir = r'E:\python\python_projects\yolov7\latest\datas\encodings'
image_folder = r'E:\python\python_projects\yolov7\latest\medias\images'  # Update this path to your folder
face_encodings, face_names = encode_faces_from_directory(image_folder)

if face_encodings:
    np.save(os.path.join(encoding_dir, 'face_encodings.npy'), face_encodings)
    np.save(os.path.join(encoding_dir, 'face_names.npy'), face_names)

# Load encodings if already saved
face_encodings = np.load(os.path.join(encoding_dir, 'face_encodings.npy'), allow_pickle=True)
face_names = np.load(os.path.join(encoding_dir, 'face_names.npy'), allow_pickle=True)

# Function to calculate angle between three points
def calculate_angle(A, B, C):
    BA = [A.x - B.x, A.y - B.y]
    BC = [C.x - B.x, C.y - B.y]
    cosine_angle = (BA[0] * BC[0] + BA[1] * BC[1]) / (math.sqrt(BA[0] ** 2 + BA[1] ** 2) * math.sqrt(BC[0] ** 2 + BC[1] ** 2))
    angle = math.degrees(math.acos(cosine_angle))
    return angle

# Function to handle cleanup on exit
def signal_handler(sig, frame):
    print("Exiting gracefully...")
    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all OpenCV windows
    sys.exit(0)  # Exit the program

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# Main function
def main():
    global cap
    cap = cv2.VideoCapture(0)  # Open the default camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Set lower resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Set lower resolution
    last_motion_time = time.time()  # Track the last time motion was detected
    previous_dist = 0  # Track previous distance for movement comparison
    frame_index = 0  # For pose feedback reference
    start_time = time.time()  # Record the start time
    recognized_user = None  # Store recognized user name
    exercise_done = False  # Flag to track if the exercise was already counted
    frame_skip = 2  # Process every 2nd frame

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video")
            break

        # Skip frames for processing
        if frame_index % frame_skip != 0:
            frame_index += 1
            continue

        # Convert the frame to RGB for DNN model
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        blob = cv2.dnn.blobFromImage(rgb_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()

        # Face detection
        faces = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        if faces:
            face = faces[0]  # Process only the first detected face
            try:
                shape = predictor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), face)
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
        pose_results = pose.process(rgb_frame)

        # Feedback for pose estimation
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            
            # Calculate angles and provide feedback
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

            print(f"Left Elbow Angle: {left_elbow_angle:.2f} degrees, Right Elbow Angle: {right_elbow_angle:.2f} degrees")

        # Display recognized user
        if recognized_user:
            cv2.putText(frame, f"Recognized User: {recognized_user}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the output frame
        cv2.imshow("Video Feed", frame)

        frame_index += 1  # Increment frame index

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
