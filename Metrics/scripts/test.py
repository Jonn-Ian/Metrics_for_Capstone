import cv2
import numpy as np
import time
import mediapipe as mp
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Function to track repetitions using line crossing
def count_reps_using_line(exercise_type='push_up', video_path=None, line_position=300):
    """Count repetitions based on line crossing for general exercises."""
    
    # Open video file or webcam feed
    if video_path:
        cap = cv2.VideoCapture(video_path)  # Open the video file
    else:
        cap = cv2.VideoCapture(0)  # Open the default camera
    
    # Repetition count variables
    rep_count = 0
    crossed_line = False  # To track if the body part has crossed the line
    direction = 0  # 1 means crossing downward, -1 means crossing upward, 0 means idle

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video")
            break

        # Convert the frame to RGB for MediaPipe pose detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image for pose detection
        pose_results = pose.process(rgb_frame)

        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark

            # Draw the pose landmarks on the frame
            for landmark in landmarks:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw landmarks as green circles

            # Define the body part to track (e.g., wrist or elbow for push-ups)
            # For push-ups, you can use the wrist or elbow, depending on the exercise mechanics
            wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]  # Tracking left wrist for example
            
            # Get the Y-coordinate of the wrist
            wrist_y = wrist.y * frame.shape[0]
            
            # Check if the wrist crosses the line (threshold)
            if wrist_y > line_position and not crossed_line:  # Moving down and crosses the line
                crossed_line = True
                direction = 1  # Crossing downwards
            elif wrist_y < line_position and crossed_line:  # Moving up and crosses the line
                if direction == 1:  # Ensure that the direction is correct (from bottom to top)
                    rep_count += 1  # Increment repetition count
                    crossed_line = False  # Reset the crossing flag
                    direction = -1  # Now the direction is up

            # Draw the line in a similar way to the skeleton
            line_color = (255, 0, 0)  # Blue color for the line
            line_thickness = 2
            cv2.line(frame, (0, line_position), (frame.shape[1], line_position), line_color, line_thickness)

            # Display the repetition count on the video frame
            cv2.putText(frame, f"Reps: {rep_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame with the landmarks and line
        cv2.imshow(f'{exercise_type.capitalize()} Repetition Counter', frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage for push-ups
video_path = r'E:\python\python_projects\yolov7\latest\medias\video\crunches.mp4'  # Replace with your video file path
count_reps_using_line(exercise_type='push_up', video_path=video_path, line_position=300)
