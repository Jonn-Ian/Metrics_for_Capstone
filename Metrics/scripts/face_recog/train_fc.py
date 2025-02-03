import dlib
import cv2
import numpy as np
import os

# Initialize Dlib's face detector and face recognition model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'E:\python\python_projects\yolov7\latest\assets\datas\models\shape_predictor_68_face_landmarks.dat')
recognizer = dlib.face_recognition_model_v1(r'E:\python\python_projects\yolov7\latest\assets\datas\models\dlib_face_recognition_resnet_model_v1.dat')

# Directory to save encodings
encoding_dir = r'E:\python\python_projects\yolov7\latest\assets\datas\encodings\James_Serafin'

# Function to encode faces from a directory
def encode_faces_from_directory(image_folder):
    encodings = []
    names = []
    folder_name = os.path.basename(image_folder)  # Get the folder name (e.g., "James_Serafin")
    save_folder = os.path.join(encoding_dir, folder_name)

    # Create the folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.jpg', '.png')):  # Handle case sensitivity
            image_path = os.path.join(image_folder, filename)
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            if faces:
                face = faces[0]  # Take the first detected face
                shape = predictor(gray, face)  # Get the shape of the detected face
                
                # Ensure the shape is valid before computing the descriptor
                if shape.num_parts == 68:  # Check if the expected number of landmarks is detected
                    encoding = recognizer.compute_face_descriptor(img, shape)
                    encodings.append(np.array(encoding))
                    # Save the **folder name** (instead of the image filename) as the name
                    names.append(folder_name)  # All images in this folder share the same name (folder name)
                else:
                    print(f"Unexpected number of landmarks detected in {filename}. Expected 68, got {shape.num_parts}.")
            else:
                print(f"No face detected in {filename}.")  # Log when no face is detected

    # Save encodings and names to the folder
    if encodings:
        np.save(os.path.join(save_folder, 'face_encodings.npy'), encodings)
        np.save(os.path.join(save_folder, 'face_names.npy'), names)
        print(f"Encodings saved to {save_folder}")

    return encodings, names

# Example usage for training phase
image_folder = r'E:\python\python_projects\yolov7\latest\assets\medias\images\training\James_Serafin'  # Update this path to your folder
face_encodings, face_names = encode_faces_from_directory(image_folder)
