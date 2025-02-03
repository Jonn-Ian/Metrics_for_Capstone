import cv2
import mediapipe as mp
import pandas as pd
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Load Video
video_path = r'E:\python\python_projects\yolov7\latest\medias\video\Shoulder_abduction.mp4'  # Replace with the path to your video
cap = cv2.VideoCapture(video_path)

# Data storage for CSV
data = []

# Function to calculate angle between three points
def calculate_angle(A, B, C):
    """Calculate the angle ABC (in degrees) between points A, B, and C."""
    BA = [A.x - B.x, A.y - B.y]
    BC = [C.x - B.x, C.y - B.y]
    cosine_angle = (BA[0] * BC[0] + BA[1] * BC[1]) / (math.sqrt(BA[0] ** 2 + BA[1] ** 2) * math.sqrt(BC[0] ** 2 + BC[1] ** 2))
    angle = math.degrees(math.acos(cosine_angle))
    return angle

try:
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # Get landmarks
            landmarks = results.pose_landmarks.landmark
            row = []

            # Append frame index
            row.append(frame_index)

            # Calculate specific joint angles
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

            # Append angles to row
            row.extend([left_elbow_angle, right_elbow_angle, left_knee_angle, right_knee_angle])

            # Append x, y, z coordinates for each landmark
            for landmark in landmarks:
                row.extend([landmark.x, landmark.y, landmark.z])

            # Add the row to data list
            data.append(row)

        # Increment frame index
        frame_index += 1

finally:
    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

    # Define column names
    columns = (
        ['Frame_Index', 'Left_Elbow_Angle', 'Right_Elbow_Angle', 'Left_Knee_Angle', 'Right_Knee_Angle'] +
        [f'Landmark_{i}_x' for i in range(33)] +
        [f'Landmark_{i}_y' for i in range(33)] +
        [f'Landmark_{i}_z' for i in range(33)]
    )

    # Save data to CSV
    output_path = r'E:\python\python_projects\yolov7\latest\datas\csv_landmarks\pose_landmarks_and_angles.csv'
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
