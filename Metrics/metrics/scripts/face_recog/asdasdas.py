import os
import cv2
import dlib
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score

# Initialize Dlib's face detector and face recognition model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'E:\python\python_projects\yolov7\latest\assets\datas\models\shape_predictor_68_face_landmarks.dat')
recognizer = dlib.face_recognition_model_v1(r'E:\python\python_projects\yolov7\latest\assets\datas\models\dlib_face_recognition_resnet_model_v1.dat')

# Load precomputed encodings
encoding_dir = r'E:\python\python_projects\yolov7\latest\assets\datas\encodings\James_Serafin'
known_encodings = np.load(os.path.join(encoding_dir, 'face_encodings.npy'), allow_pickle=True)
known_names = np.load(os.path.join(encoding_dir, 'face_names.npy'), allow_pickle=True)

# Test dataset
test_image_folder = r'E:\python\python_projects\yolov7\latest\assets\medias\images\validation\James_Serafin\10_faces'

# Directories for outputs
csv_output_path = r'E:\python\python_projects\yolov7\latest\metrics\otherpaths\face_recog\csv\metrics.csv'
graph_output_path = r'E:\python\python_projects\yolov7\latest\metrics\otherpaths\face_recog\imgs\metrics_graph.png'

# Ensure output directories exist
os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
os.makedirs(os.path.dirname(graph_output_path), exist_ok=True)

# Metrics initialization
y_true, y_pred = [], []
false_positives, false_negatives, true_positives, true_negatives = 0, 0, 0, 0

# Function to compare face encodings
def compare_faces(known_encodings, face_encoding, threshold=0.6):
    distances = np.linalg.norm(known_encodings - face_encoding, axis=1)
    min_distance = np.min(distances)
    return (distances <= threshold).any(), np.argmin(distances) if min_distance <= threshold else -1

# Function to display the image with scaling if necessary
def display_image(title, image, max_width=800, max_height=800):
    """
    Display an image scaled to fit within max dimensions while maintaining aspect ratio.
    """
    h, w = image.shape[:2]
    if w > max_width or h > max_height:
        scaling_factor = min(max_width / w, max_height / h)
        new_w = int(w * scaling_factor)
        new_h = int(h * scaling_factor)
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized_image = image  # No scaling needed
    cv2.imshow(title, resized_image)

# Process test images
for filename in os.listdir(test_image_folder):
    if filename.lower().endswith(('.jpg', '.png')):
        image_path = os.path.join(test_image_folder, filename)
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Error loading {filename}. Skipping...")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if not faces:
            # No face detected, assume "Unknown"
            cv2.putText(frame, "No face detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            display_image('Manual Classification', frame)
            print("Press 1 for False Negative, 3 for True Negative.")
            key = cv2.waitKey(0) & 0xFF

            if key == ord('1'):
                false_negatives += 1
                y_true.append(1)  # Actual known
                y_pred.append(0)  # Predicted unknown
            elif key == ord('3'):
                true_negatives += 1
                y_true.append(0)  # Actual unknown
                y_pred.append(0)  # Predicted unknown
            cv2.destroyAllWindows()
            continue

        for face in faces:
            shape = predictor(gray, face)
            face_encoding = np.array(recognizer.compute_face_descriptor(frame, shape))
            match, index = compare_faces(known_encodings, face_encoding)

            if match:
                predicted_name = known_names[index]
                cv2.putText(frame, f"Matched: {predicted_name}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                predicted_name = "Unknown"
                cv2.putText(frame, "No Match", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            display_image('Manual Classification', frame)
            print("Press 1 for False Negative, 2 for False Positive, 3 for True Negative, 4 for True Positive.")
            key = cv2.waitKey(0) & 0xFF

            if key == ord('1'):
                false_negatives += 1
                y_true.append(1)  # Actual known
                y_pred.append(0)  # Predicted unknown
            elif key == ord('2'):
                false_positives += 1
                y_true.append(0)  # Actual unknown
                y_pred.append(1)  # Predicted known
            elif key == ord('3'):
                true_negatives += 1
                y_true.append(0)  # Actual unknown
                y_pred.append(0)  # Predicted unknown
            elif key == ord('4'):
                true_positives += 1
                y_true.append(1)  # Actual known
                y_pred.append(1)  # Predicted known
            cv2.destroyAllWindows()

# Compute metrics
precision = precision_score(y_true, y_pred, zero_division=0)
accuracy = accuracy_score(y_true, y_pred)
far = false_positives / (false_positives + true_negatives) if false_positives + true_negatives > 0 else 0
frr = false_negatives / (false_negatives + true_positives) if true_positives > 0 else 0

# Save metrics to CSV with percentages
with open(csv_output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Metric', 'Value'])
    writer.writerow(['Precision', f"{precision * 100:.2f}%"])
    writer.writerow(['Accuracy', f"{accuracy * 100:.2f}%"])
    writer.writerow(['False Acceptance Rate (FAR)', f"{far * 100:.2f}%"])
    writer.writerow(['False Rejection Rate (FRR)', f"{frr * 100:.2f}%"])

# Save metrics graph
plt.bar(['Precision', 'Accuracy', 'FAR', 'FRR'], [precision, accuracy, far, frr], color=['blue', 'green', 'red', 'orange'])
plt.title('Face Recognition Metrics')
plt.ylabel('Value')
plt.savefig(graph_output_path)
plt.close()

print(f"Metrics saved to {csv_output_path}")
print(f"Graph saved to {graph_output_path}")
