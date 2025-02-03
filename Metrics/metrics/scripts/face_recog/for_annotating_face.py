import dlib
import cv2
import numpy as np
import os

# Initialize Dlib's face detector and face recognition model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'E:\python\python_projects\yolov7\latest\datas\models\shape_predictor_68_face_landmarks.dat')
recognizer = dlib.face_recognition_model_v1(r'E:\python\python_projects\yolov7\latest\datas\models\dlib_face_recognition_resnet_model_v1.dat')

# Directory to save encodings
encoding_dir = r'E:\python\python_projects\yolov7\latest\datas\encodings\James_Serafin'

# Load pre-existing face encodings
def load_encodings(encoding_dir):
    encoding_path = os.path.join(encoding_dir, 'face_encodings.npy')
    names_path = os.path.join(encoding_dir, 'face_names.npy')

    if os.path.exists(encoding_path) and os.path.exists(names_path):
        encodings = np.load(encoding_path, allow_pickle=True)
        names = np.load(names_path, allow_pickle=True)
        return encodings, names
    else:
        return [], []

# Function to recognize faces from a directory and save annotated images
def recognize_faces_from_directory(image_folder, encoding_dir, save_folder):
    known_encodings, known_names = load_encodings(encoding_dir)

    # Ensure the directory to save annotated images exists
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
                
                # Draw a rectangle around the face
                cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                
                # Annotate the landmarks with much bigger dots (increase the radius)
                for i in range(0, 68):  # Loop over all 68 landmarks
                    x, y = shape.part(i).x, shape.part(i).y
                    cv2.circle(img, (x, y), 10, (0, 0, 255), -1)  # Draw a much bigger red circle at each landmark (radius = 50)
                
                # Compute the face descriptor
                if shape.num_parts == 68:  # Check if the expected number of landmarks is detected
                    encoding = recognizer.compute_face_descriptor(img, shape)
                    encoding = np.array(encoding)
                    
                    # Compare the encoding with known encodings
                    distances = np.linalg.norm(known_encodings - encoding, axis=1)
                    threshold = 0.6  # Distance threshold for recognition
                    if len(distances) > 0 and min(distances) < threshold:
                        recognized_name = known_names[np.argmin(distances)]
                    else:
                        recognized_name = "Unrecognized"

                    # Add the label on the image
                    cv2.putText(img, recognized_name, (face.left(), face.top() - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    # Save the annotated image to the specified directory
                    annotated_image_path = os.path.join(save_folder, f"annotated_{filename}")
                    cv2.imwrite(annotated_image_path, img)
                else:
                    print(f"Unexpected number of landmarks detected in {filename}. Expected 68, got {shape.num_parts}.")
            else:
                print(f"No face detected in {filename}.")  # Log when no face is detected

# Example usage for face recognition
image_folder = r'E:\python\python_projects\yolov7\latest\medias\images\validation\James_Serafin'  # Update this path to your folder
save_folder = r'E:\python\python_projects\yolov7\latest\medias\annotated_faces\James_Serafin'  # Folder where annotated images will be saved

recognize_faces_from_directory(image_folder, encoding_dir, save_folder)

print(f"Annotated images saved to {save_folder}")
