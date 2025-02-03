import dlib
import cv2
import numpy as np
import os

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
                    else:
                        name = "Unknown"
                        unknown_faces += 1  # Increment unknown faces counter

                    total_faces += 1  # Increment total faces counter
                else:
                    print(f"Invalid shape detected for recognition in {filename}.")
            except Exception as e:
                print(f"Error during face recognition in {filename}: {e}")

# After processing all images, calculate the evaluation metrics
if total_faces > 0:
    accuracy = (known_faces / total_faces) * 100
else:
    accuracy = 0

print(f"\n--- Evaluation ---")
print(f"Total faces detected: {total_faces}")
print(f"Known faces: {known_faces}")
print(f"Unknown faces: {unknown_faces}")
print(f"Accuracy: {accuracy:.2f}%")
