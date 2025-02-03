import time
import mediapipe as mp
import pygame
import numpy as np
import cv2

# Initialize Pygame mixer for sound
pygame.mixer.init()

# Load the sound file
try:
    warning_sound = pygame.mixer.Sound(r'E:\python\python_projects\yolov7\latest\medias\sounds\alarm.wav')
    print("Warning sound file loaded successfully.")
except FileNotFoundError:
    print("Warning sound file not found. Please check the path.")
    exit()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7)

# Initialize previous_dist as a global variable
previous_dist = 0

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

def motion_detection(frame, last_motion_time):
    global previous_dist  # Access the global variable
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb_frame)

    motion_detected = False

    if pose_results.pose_landmarks:
        current_dist = calculate_distance(pose_results.pose_landmarks.landmark)
        if abs(current_dist - previous_dist) > 0.01:
            last_motion_time = time.time()
            motion_detected = True
            print("Significant motion detected!")
        
        previous_dist = current_dist

    elapsed_time = time.time() - last_motion_time
    print(f"Elapsed time since last motion: {elapsed_time:.2f} seconds")  # Debugging line

    # Only play sound if a certain period has passed without motion
    if elapsed_time >= 10 and not motion_detected:  # Change this value as needed
        print(f"No significant movement detected for {int(elapsed_time)} seconds! Playing warning sound...")
        warning_sound.play()
        time.sleep(2)  # Be cautious with sleep in real-time applications
        warning_sound.stop()
        last_motion_time = time.time()  # Reset time after sound plays

    if motion_detected:
        cv2.rectangle(frame, (10, 10), (250, 50), (0, 255, 0), -1)
        cv2.putText(frame, "Motion Detected!", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return motion_detected, last_motion_time
