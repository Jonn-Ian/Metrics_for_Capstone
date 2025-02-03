import dlib
import cv2
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Load Dlib's face detector and face recognition model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'E:\python\python_projects\yolov7\latest\datas\models\shape_predictor_68_face_landmarks.dat')
recognizer = dlib.face_recognition_model_v1(r'E:\python\python_projects\yolov7\latest\datas\models\dlib_face_recognition_resnet_model_v1.dat')

# Load saved encodings
encoding_dir = r'E:\python\python_projects\yolov7\latest\datas\encodings\James_Serafin'
face_encodings = np.load(os.path.join(encoding_dir, 'face_encodings.npy'), allow_pickle=True)
face_names = np.load(os.path.join(encoding_dir, 'face_names.npy'), allow_pickle=True)

# Directory of images to test
test_image_folder = r'E:\python\python_projects\yolov7\latest\medias\images\validation\James_Serafin'  # Update to your folder

# Initialize counters for evaluation
total_faces = 0
known_faces = 0
unknown_faces = 0
y_true = []  # Actual labels (0: unknown, 1: known)
y_pred = []  # Predicted labels (0: unknown, 1: known)

# Process all images in the test folder
for filename in os.listdir(test_image_folder):
    if filename.lower().endswith(('.jpg', '.png')):  # Handle case sensitivity
        image_path = os.path.join(test_image_folder, filename)
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Error loading image: {filename}.")
            continue
        else:
            print(f"Image loaded successfully: {filename}.")

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_frame)

        print(f"Detected {len(faces)} face(s) in {filename}.")  # Debugging output

        for face in faces:
            try:
                shape = predictor(gray_frame, face)  # Use the grayscale frame
                
                if shape.num_parts == 68:
                    # Compute the face descriptor
                    encoding = recognizer.compute_face_descriptor(frame, shape)

                    # Compare with known faces
                    matches = []
                    for known_face in face_encodings:
                        match = np.linalg.norm(np.array(encoding) - np.array(known_face)) < 0.6  # Adjust the threshold as needed
                        matches.append(match)

                    if any(matches):
                        name_index = matches.index(True)
                        name = face_names[name_index]
                        known_faces += 1  # Increment known faces counter
                        y_true.append(1)  # Actual label is known (1)
                        y_pred.append(1)  # Predicted label is known (1)
                    else:
                        name = "Unknown"
                        unknown_faces += 1  # Increment unknown faces counter
                        y_true.append(0)  # Actual label is unknown (0)
                        y_pred.append(0)  # Predicted label is unknown (0)

                    total_faces += 1  # Increment total faces counter
                else:
                    print(f"Invalid shape detected for recognition in {filename}.")
            except Exception as e:
                print(f"Error during face recognition in {filename}: {e}")

# After processing all images, calculate the evaluation metrics
if total_faces > 0:
    accuracy = (known_faces / total_faces) * 100
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    false_positives = sum((np.array(y_true) == 0) & (np.array(y_pred) == 1))
    false_negatives = sum((np.array(y_true) == 1) & (np.array(y_pred) == 0))
else:
    accuracy = 0
    precision = 0
    recall = 0
    f1 = 0
    false_positives = 0
    false_negatives = 0

# Display the evaluation metrics
print(f"\n--- Evaluation ---")
print(f"Total faces detected: {total_faces}")
print(f"Known faces: {known_faces}")
print(f"Unknown faces: {unknown_faces}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"False Positives: {false_positives}")
print(f"False Negatives: {false_negatives}")

# Path to save the graph
output_path = r'E:\python\python_projects\yolov7\latest\otherpaths\face_recog'

# Plot the metrics
metrics = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'False Positives': false_positives,
    'False Negatives': false_negatives
}

# Create a bar graph
plt.figure(figsize=(10, 6))
plt.bar(metrics.keys(), metrics.values(), color=['#4CAF50', '#FF9800', '#2196F3', '#F44336', '#9C27B0', '#00BCD4'])
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Face Recognition Performance Metrics')
plt.ylim(0, max(metrics.values()) + 10)  # Set y-axis limit

# Save the graph as an image
plt.savefig(output_path)
plt.close()

# Print where the graph is saved
print(f"Graph saved at {output_path}")
