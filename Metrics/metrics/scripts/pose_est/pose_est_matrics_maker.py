import cv2
import mediapipe as mp
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the CSV data with landmarks and angles
csv_file = r'E:\python\python_projects\yolov7\latest\datas\csv_landmarks\pose_landmarks_and_angles.csv'
df = pd.read_csv(csv_file)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Video feed (use a looped video or webcam)
video_path = r'E:\python\python_projects\yolov7\latest\medias\video\videoplayback.mp4'  # Replace with the path to your test video
cap = cv2.VideoCapture(video_path)

# Function to calculate angle between three points
def calculate_angle(A, B, C):
    BA = [A.x - B.x, A.y - B.y]
    BC = [C.x - B.x, C.y - B.y]
    cosine_angle = (BA[0] * BC[0] + BA[1] * BC[1]) / (math.sqrt(BA[0] ** 2 + BA[1] ** 2) * math.sqrt(BC[0] ** 2 + BC[1] ** 2))
    angle = math.degrees(math.acos(cosine_angle))
    return angle

# Function to compute the Euclidean distance between two 3D points (landmarks)
def euclidean_distance(coords1, coords2):
    return np.sqrt((coords1[0] - coords2[0])**2 + (coords1[1] - coords2[1])**2 + (coords1[2] - coords2[2])**2)

# Function to compute MAE (Mean Absolute Error) for angles
def mean_absolute_error(predicted, reference):
    return np.mean(np.abs(np.array(predicted) - np.array(reference)))

# Function to compute RMSE (Root Mean Squared Error) for angles
def root_mean_squared_error(predicted, reference):
    return np.sqrt(np.mean((np.array(predicted) - np.array(reference))**2))

# Function to compute similarity score (higher is better)
def compute_similarity(angles1, angles2):
    total_diff = 0
    count = 0
    for angle1, angle2 in zip(angles1, angles2):
        total_diff += abs(angle1 - angle2)
        count += 1
    return 100 - (total_diff / count)  # Normalize to a percentage (0-100)

# Min-Max Normalization function
def min_max_normalize(values, feature_range=(0, 1)):
    min_val = min(values)
    max_val = max(values)
    normalized_values = [(value - min_val) / (max_val - min_val) * (feature_range[1] - feature_range[0]) + feature_range[0] for value in values]
    return normalized_values

# Path for saving the graphs
output_graph_path = r'E:\python\python_projects\yolov7\latest\otherpaths\pose_est'
if not os.path.exists(output_graph_path):
    os.makedirs(output_graph_path)

# Video Writer to save output video with overlays
output_video_path = r'E:\python\python_projects\yolov7\latest\otherpaths\pose_est'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

try:
    frame_index = 0
    total_similarity = 0
    total_mae = 0
    total_rmse = 0
    total_landmark_distance = 0
    num_frames = 0
    loop_count = 0

    similarity_scores = []
    mae_scores = []
    rmse_scores = []
    landmark_distances = []

    while cap.isOpened() and loop_count < 0:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            loop_count += 1
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
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

            try:
                ref_row = df.iloc[frame_index]
                
                ref_angles = [
                    ref_row['Left_Elbow_Angle'],
                    ref_row['Right_Elbow_Angle'],
                    ref_row['Left_Knee_Angle'],
                    ref_row['Right_Knee_Angle']
                ]
                
                test_angles = [left_elbow_angle, right_elbow_angle, left_knee_angle, right_knee_angle]
                
                normalized_ref_angles = min_max_normalize(ref_angles, feature_range=(0, 1))
                normalized_test_angles = min_max_normalize(test_angles, feature_range=(0, 1))
                
                frame_similarity = compute_similarity(normalized_test_angles, normalized_ref_angles)
                frame_mae = mean_absolute_error(test_angles, ref_angles)
                frame_rmse = root_mean_squared_error(test_angles, ref_angles)

                frame_landmark_distance = 0
                for i in range(33):
                    ref_x = ref_row[f'Landmark_{i}_x']
                    ref_y = ref_row[f'Landmark_{i}_y']
                    ref_z = ref_row[f'Landmark_{i}_z']
                    ref_coords = [ref_x, ref_y, ref_z]
                    current_coords = [landmarks[i].x, landmarks[i].y, landmarks[i].z]
                    frame_landmark_distance += euclidean_distance(current_coords, ref_coords)

                similarity_scores.append(frame_similarity)
                mae_scores.append(frame_mae)
                rmse_scores.append(frame_rmse)
                landmark_distances.append(frame_landmark_distance)

                num_frames += 1

                # Overlay metrics on the frame with a background rectangle
                overlay_text = (
                    f"Sim: {frame_similarity:.2f}% | MAE: {frame_mae:.2f} | RMSE: {frame_rmse:.2f} | "
                    f"Landmark Dist: {frame_landmark_distance:.2f}"
                )
                
                # Draw a black rectangle for better text visibility
                cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, 70), (0, 0, 0), -1)  # Black rectangle
                
                # Add white text on top of the black rectangle
                cv2.putText(frame, overlay_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            except IndexError:
                frame_index = 0

        frame_index += 1

        # Write the frame with overlays to the output video
        out.write(frame)

    average_similarity = np.mean(similarity_scores) if num_frames > 0 else 0
    average_mae = np.mean(mae_scores) if num_frames > 0 else 0
    average_rmse = np.mean(rmse_scores) if num_frames > 0 else 0
    average_landmark_distance = np.mean(landmark_distances) if num_frames > 0 else 0

    print(f"Average Similarity: {average_similarity:.2f}%")
    print(f"Average MAE: {average_mae:.2f}")
    print(f"Average RMSE: {average_rmse:.2f}")
    print(f"Average Landmark Distance: {average_landmark_distance:.2f}")

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
