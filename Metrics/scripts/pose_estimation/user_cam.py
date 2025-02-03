import cv2
import mediapipe as mp
import pandas as pd
import math
import numpy as np

# Load the CSV data with landmarks and angles
csv_file = r'E:\python\python_projects\yolov7\latest\datas\csv_landmarks\pose_landmarks_and_angles.csv'
df = pd.read_csv(csv_file)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Video feed (use the webcam instead of a video file)
cap = cv2.VideoCapture(0)  # 0 corresponds to the default webcam

# Function to calculate angle between three points
def calculate_angle(A, B, C):
    """Calculate the angle ABC (in degrees) between points A, B, and C."""
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

# Function to provide feedback based on angle differences
def give_feedback(test_angles, ref_angles):
    feedback = []
    for i, (test_angle, ref_angle) in enumerate(zip(test_angles, ref_angles)):
        diff = abs(test_angle - ref_angle)
        if diff > 10:  # Threshold for providing feedback, adjust as needed
            joint = ["Left Elbow", "Right Elbow", "Left Knee", "Right Knee"]
            feedback.append(f"Adjust {joint[i]}: Difference of {diff:.2f} degrees.")
    
    # Return the first feedback message if available, otherwise None
    return feedback[0] if feedback else None

try:
    frame_index = 0
    total_similarity = 0
    num_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # Get landmarks and calculate angles in the current frame
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

            # Get reference angles from CSV for the current frame index
            try:
                ref_row = df.iloc[frame_index]
                
                # Extract angles from the reference CSV
                ref_angles = [
                    ref_row['Left_Elbow_Angle'],
                    ref_row['Right_Elbow_Angle'],
                    ref_row['Left_Knee_Angle'],
                    ref_row['Right_Knee_Angle']
                ]
                
                # Compare angles between the test video and the reference CSV
                test_angles = [left_elbow_angle, right_elbow_angle, left_knee_angle, right_knee_angle]
                
                # Compute similarity for this frame
                frame_similarity = compute_similarity(test_angles, ref_angles)
                total_similarity += frame_similarity
                num_frames += 1

                # Provide feedback if the similarity is low
                feedback_message = give_feedback(test_angles, ref_angles)
                if feedback_message:
                    cv2.putText(frame, feedback_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Good form!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            except IndexError:
                # If frame index exceeds the reference data, reset or loop
                frame_index = 0

        # Increment frame index for CSV reference
        frame_index += 1

        # Show the current frame (without skeleton)
        cv2.imshow('Pose Feedback', frame)

        # Wait for user to press 'q' to quit or continue with the video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calculate average similarity across all frames
    average_similarity = total_similarity / num_frames if num_frames > 0 else 0
    print(f"Accuracy: {average_similarity:.2f}%")

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
