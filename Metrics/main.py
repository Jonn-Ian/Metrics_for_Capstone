import cv2
import mediapipe as mp
import pandas as pd
import math
import numpy as np
import os

# Load the CSV data with landmarks and angles
csv_file = r'path here'
df = pd.read_csv(csv_file)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.85, min_tracking_confidence=0.85)

# for input as an mp4
video_path = r'path here'  #Replace it with an mp4 path
cap = cv2.VideoCapture(video_path)

#to calculate angle between three points
def calculate_angle(A, B, C):
    BA = [A.x - B.x, A.y - B.y]
    BC = [C.x - B.x, C.y - B.y]
    cosine_angle = (BA[0] * BC[0] + BA[1] * BC[1]) / (math.sqrt(BA[0] ** 2 + BA[1] ** 2) * math.sqrt(BC[0] ** 2 + BC[1] ** 2))
    angle = math.degrees(math.acos(cosine_angle))
    return angle

#to compute the Euclidean distance between two 3D points
def euclidean_distance(coords1, coords2):
    return np.sqrt((coords1[0] - coords2[0])**2 + (coords1[1] - coords2[1])**2 + (coords1[2] - coords2[2])**2)

#to compute (Mean Absolute Error) for angles
def mean_absolute_error(predicted, reference):
    return np.mean(np.abs(np.array(predicted) - np.array(reference)))

#to compute RMSE (Root Mean Squared Error) for angles
def root_mean_squared_error(predicted, reference):
    return np.sqrt(np.mean((np.array(predicted) - np.array(reference))**2))

# to compute similarity score (higher is better)
def compute_similarity(angles1, angles2, threshold=0.1):
    diff = np.abs(np.array(angles1) - np.array(angles2))
    similarity = np.mean(diff <= threshold) * 100 #100 is as 100%
    return similarity

#to validate angles within tolerance range (e.g., ±5 degrees)
def validate_angles_with_tolerance(angles, tolerance=5):
    for angle in angles:
        if not (0 <= angle <= 180) or angle < tolerance:
            return False
    return True

def min_max_normalize(values, feature_range=(0, 1)):
    min_val = min(values)
    max_val = max(values)
    normalized_values = [(value - min_val) / (max_val - min_val) * (feature_range[1] - feature_range[0]) + feature_range[0] for value in values]
    return normalized_values

# Path for saving the graphs
output_graph_path = r'path here'#change the path to your path
if not os.path.exists(output_graph_path):
    os.makedirs(output_graph_path)

# Path for saving the chunk CSVs
output_video_path = r'path here' #change the path to your own path
if not os.path.exists(output_video_path):
    os.makedirs(output_video_path)

#this helps to reduce the time complecxity, instead of frame by frame, it takes the 25 frames and take the average of them all
# to make it as 1 chunk
try:
    frame_index = 0
    total_similarity = 0
    total_mae = 0
    total_rmse = 0
    total_landmark_distance = 0
    num_frames = 0
    total_processed_frames = 0  # Add a counter for total processed frames
    chunk_counter = 1  # Counter for chunks of frames
    chunk_size = 25  # Every 25 frames will be considered as one chunk

    similarity_scores = []
    mae_scores = []
    rmse_scores = []
    landmark_distances = []

    all_chunk_similarities = []
    all_chunk_maes = []
    all_chunk_rmses = []
    all_chunk_landmark_distances = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            break  # If video is finished, exit the loop

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
                
                # Normalize angles
                normalized_ref_angles = min_max_normalize(ref_angles, feature_range=(0, 1))
                normalized_test_angles = min_max_normalize(test_angles, feature_range=(0, 1))
                
                # Compute similarity score
                frame_similarity = compute_similarity(normalized_test_angles, normalized_ref_angles, threshold=0.1)
                
                # Calculate MAE and RMSE
                frame_mae = mean_absolute_error(test_angles, ref_angles)
                frame_rmse = root_mean_squared_error(test_angles, ref_angles)

                # Calculate landmark distances
                frame_landmark_distance = 0
                for i in range(33):
                    ref_x = ref_row[f'Landmark_{i}_x']
                    ref_y = ref_row[f'Landmark_{i}_y']
                    ref_z = ref_row[f'Landmark_{i}_z']
                    ref_coords = [ref_x, ref_y, ref_z]
                    current_coords = [landmarks[i].x, landmarks[i].y, landmarks[i].z]
                    frame_landmark_distance += euclidean_distance(current_coords, ref_coords)

                # Store results
                similarity_scores.append(frame_similarity)
                mae_scores.append(frame_mae)
                rmse_scores.append(frame_rmse)
                landmark_distances.append(frame_landmark_distance)

                num_frames += 1
                total_processed_frames += 1

            except IndexError:
                frame_index = 0

        # After processing all frames in the chunk, save results and reset accumulators
        if num_frames % chunk_size == 0:
            average_similarity = np.mean(similarity_scores) if len(similarity_scores) > 0 else 0
            average_mae = np.mean(mae_scores) if len(mae_scores) > 0 else 0
            average_rmse = np.mean(rmse_scores) if len(rmse_scores) > 0 else 0
            average_landmark_distance = np.mean(landmark_distances) if len(landmark_distances) > 0 else 0

            # Create results table for this chunk
            results_table = pd.DataFrame({
                'Metric': ['Average Similarity (%)', 'Average MAE (°)', 'Average RMSE (°)', 'Average Landmark Distance (mm)', 'Number of Frames Processed'],
                'Value': [round(average_similarity, 4), round(average_mae, 4), round(average_rmse, 4), round(average_landmark_distance, 4), float(num_frames)]
            })

            # Save results table to CSV for this chunk
            chunk_filename = os.path.join(output_video_path, f'chunk_{chunk_counter}_results.csv')
            results_table.to_csv(chunk_filename, index=False)

            # Print results for the chunk to the terminal
            print(f"\nResults for Chunk {chunk_counter}:")
            print(results_table)

            # Save chunk results to list for the final output
            all_chunk_similarities.append(average_similarity)
            all_chunk_maes.append(average_mae)
            all_chunk_rmses.append(average_rmse)
            all_chunk_landmark_distances.append(average_landmark_distance)

            # Reset accumulators for the next chunk
            similarity_scores = []
            mae_scores = []
            rmse_scores = []
            landmark_distances = []
            num_frames = 0
            chunk_counter += 1

        frame_index += 1

    # After processing all chunks, calculate and save final averages
    final_avg_similarity = np.mean(all_chunk_similarities) if len(all_chunk_similarities) > 0 else 0
    final_avg_mae = np.mean(all_chunk_maes) if len(all_chunk_maes) > 0 else 0
    final_avg_rmse = np.mean(all_chunk_rmses) if len(all_chunk_rmses) > 0 else 0
    final_avg_landmark_distance = np.mean(all_chunk_landmark_distances) if len(all_chunk_landmark_distances) > 0 else 0

    final_results = pd.DataFrame({
        'Metric': ['Final Average Similarity (%)', 'Final Average MAE (°)', 'Final Average RMSE (°)', 'Final Average Landmark Distance (mm)', 'Total Frames Processed'],
        'Value': [round(final_avg_similarity, 4), round(final_avg_mae, 4), round(final_avg_rmse, 4), round(final_avg_landmark_distance, 4), float(total_processed_frames)]
    })

    # Save final results to a CSV
    final_results_filename = os.path.join(output_video_path, 'final_results.csv')
    final_results.to_csv(final_results_filename, index=False)

    # Print final results to the terminal
    print("\nFinal Results:")
    print(final_results)

finally:
    cap.release()
    cv2.destroyAllWindows()
