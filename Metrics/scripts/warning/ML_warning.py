import cv2
import time
import mediapipe as mp
import pygame
import signal
import sys
import numpy as np

# Initialize Pygame mixer for sound
pygame.mixer.init()

# Load the sound file
try:
    warning_sound = pygame.mixer.Sound(r'E:\python\python_projects\yolov7\latest\medias\sounds\alarm.wav')
except FileNotFoundError:
    print("Warning sound file not found. Please check the path.")
    exit()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7)

# Function to handle cleanup on exit
def signal_handler(sig, frame):
    print("Exiting gracefully...")
    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all OpenCV windows
    sys.exit(0)  # Exit the program

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

def calculate_distance(landmarks):
    if landmarks:
        # Calculate the average distance between key points
        distances = []
        for i in range(len(landmarks)):
            for j in range(i + 1, len(landmarks)):
                point1 = np.array([landmarks[i].x, landmarks[i].y])
                point2 = np.array([landmarks[j].x, landmarks[j].y])
                distances.append(np.linalg.norm(point1 - point2))
        return np.mean(distances)
    return 0

def main():
    global cap
    cap = cv2.VideoCapture(0)  # Open the default camera
    last_motion_time = time.time()  # Track the last time motion was detected
    previous_dist = 0  # Track previous distance for movement comparison

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video")
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image for pose detection
        pose_results = pose.process(rgb_frame)

        # Flag to check if any motion is detected
        motion_detected = False

        # Calculate movement based on detected pose landmarks
        if pose_results.pose_landmarks:
            # Calculate the average distance between landmarks
            current_dist = calculate_distance(pose_results.pose_landmarks.landmark)

            # Reset last_motion_time only if significant movement is detected
            if abs(current_dist - previous_dist) > 0.01:  # Adjust threshold as needed
                last_motion_time = time.time()
                motion_detected = True
                print("Significant motion detected!")  # Debugging line
            
            previous_dist = current_dist

        # Check elapsed time without movement
        elapsed_time = time.time() - last_motion_time
        print(f"Elapsed time without significant motion: {elapsed_time:.2f} seconds")  # Debugging line

        if elapsed_time >= 5:  # If no significant movement for 5 seconds
            print("No significant movement detected for 5 seconds! Playing warning sound...")
            warning_sound.play()  # Play the warning sound
            time.sleep(2)  # Wait for a couple of seconds before stopping the sound
            warning_sound.stop()
            last_motion_time = time.time()  # Reset last motion time after playing the sound

        # Visual feedback for motion detection
        if motion_detected:
            cv2.rectangle(frame, (10, 10), (250, 50), (0, 255, 0), -1)  # Green box
            cv2.putText(frame, "Motion Detected!", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the image without landmarks
        cv2.imshow('Motion Detection', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
