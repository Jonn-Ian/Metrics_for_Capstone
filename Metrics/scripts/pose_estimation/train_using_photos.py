import cv2
import mediapipe as mp
import pandas as pd
import math

# Load the CSV data with landmarks and angles (trained reference angles)
csv_file = r'E:\python\python_projects\yolov7\latest\datas\csv_landmarks\pose_landmarks_and_angles.csv'
df = pd.read_csv(csv_file)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Video feed (use a looped video or webcam)
video_path = r'E:\python\python_projects\yolov7\latest\medias\video\push_up.mp4'  # Replace with the path to your video
cap = cv2.VideoCapture(video_path)

# Function to calculate angle between three points
def calculate_angle(A, B, C):
    """Calculate the angle ABC (in degrees) between points A, B, and C."""
    BA = [A.x - B.x, A.y - B.y]
    BC = [C.x - B.x, C.y - B.y]
    cosine_angle = (BA[0] * BC[0] + BA[1] * BC[1]) / (math.sqrt(BA[0] ** 2 + BA[1] ** 2) * math.sqrt(BC[0] ** 2 + BC[1] ** 2))
    angle = math.degrees(math.acos(cosine_angle))
    return angle

# Function to draw the skeleton on the frame
def draw_skeleton(frame, landmarks):
    # Define the connections between landmarks
    connections = [
        (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_SHOULDER),
        (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.RIGHT_SHOULDER),
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
        (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
        (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
        (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
        (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
        (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
        (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
    ]

    for connection in connections:
        start_idx, end_idx = connection
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        cv2.line(frame, (int(start.x * frame.shape[1]), int(start.y * frame.shape[0])),
                 (int(end.x * frame.shape[1]), int(end.y * frame.shape[0])), (0, 255, 0), 2)

# Variables for repetition tracking
repetition_counter = 0
is_exercise_in_progress = False
previous_state = "up"  # Assuming starting in an "up" position (e.g., standing or straight arms in a push-up)

try:
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop the video
            continue

        # Convert the frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # Get landmarks and calculate angles in the current frame
            landmarks = results.pose_landmarks.landmark
            draw_skeleton(frame, landmarks)  # Draw the skeleton

            # Calculate relevant joint angles (e.g., elbow and knee angles)
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

                # Compare real-time angles to reference angles from CSV for feedback
                feedback = []
                if abs(left_elbow_angle - ref_row['Left_Elbow_Angle']) > 10:
                    feedback.append("Adjust left elbow position.")
                if abs(right_elbow_angle - ref_row['Right_Elbow_Angle']) > 10:
                    feedback.append("Adjust right elbow position.")
                if abs(left_knee_angle - ref_row['Left_Knee_Angle']) > 10:
                    feedback.append("Adjust left knee position.")
                if abs(right_knee_angle - ref_row['Right_Knee_Angle']) > 10:
                    feedback.append("Adjust right knee position.")

                # Check if repetition is complete based on angle transitions
                if abs(left_knee_angle - ref_row['Left_Knee_Angle']) < 10 and abs(right_knee_angle - ref_row['Right_Knee_Angle']) < 10:
                    # Transition from down to up (up position)
                    if previous_state == "down" and abs(left_knee_angle - ref_row['Left_Knee_Angle']) < 10:
                        repetition_counter += 1
                        previous_state = "up"
                    # Transition from up to down (down position)
                    elif previous_state == "up" and abs(left_knee_angle - ref_row['Left_Knee_Angle']) > 70:  # Example threshold
                        previous_state = "down"

                # Display feedback on video frame
                for i, message in enumerate(feedback):
                    cv2.putText(frame, message, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # If feedback is empty, user is doing well
                if not feedback:
                    cv2.putText(frame, "Good form!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Display repetition count
                cv2.putText(frame, f"Reps: {repetition_counter}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            except IndexError:
                # If frame index exceeds the reference data, reset or loop
                frame_index = 0

        # Display the frame
        cv2.imshow('Pose Estimation Feedback', frame)

        # Break on 'q' key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        # Increment frame index for CSV reference
        frame_index += 1

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
