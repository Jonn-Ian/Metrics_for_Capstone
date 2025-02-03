import dlib
import cv2
import numpy as np
import os

# Load Dlib's face detector and face recognition model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'E:\python\python_projects\yolov7\latest\datas\models\shape_predictor_68_face_landmarks.dat')
recognizer = dlib.face_recognition_model_v1(r'E:\python\python_projects\yolov7\latest\datas\models\dlib_face_recognition_resnet_model_v1.dat')

# Directory to save encodings
encoding_dir = r'E:\python\python_projects\yolov7\latest\datas\encodings'

# Function to encode faces from a directory
def encode_faces_from_directory(image_folder):
    encodings = []
    names = []
    
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.jpg', '.png')):  # Handle case sensitivity
            image_path = os.path.join(image_folder, filename)
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            if faces:
                face = faces[0]  # Take the first detected face
                shape = predictor(gray, face)  # Get the shape of the detected face
                
                if shape.num_parts == 68:  # Check if the expected number of landmarks is detected
                    encoding = recognizer.compute_face_descriptor(img, shape)
                    encodings.append(np.array(encoding))
                    name = os.path.basename(image_path).split('.')[0]  # Use the filename as the name
                    names.append(name)
                else:
                    print(f"Unexpected number of landmarks detected in {filename}. Expected 68, got {shape.num_parts}.")
            else:
                print(f"No face detected in {filename}.")  # Log when no face is detected

    return encodings, names

# Example usage
image_folder = r'E:\python\python_projects\yolov7\latest\medias\my_photos'  # Update this path to your folder
face_encodings, face_names = encode_faces_from_directory(image_folder)

# Save encodings for later use
if face_encodings:
    np.save(os.path.join(encoding_dir, 'face_encodings.npy'), face_encodings)
    np.save(os.path.join(encoding_dir, 'face_names.npy'), face_names)

# Load saved encodings
face_encodings = np.load(os.path.join(encoding_dir, 'face_encodings.npy'), allow_pickle=True)
face_names = np.load(os.path.join(encoding_dir, 'face_names.npy'), allow_pickle=True)

# Process all images in a specified folder
test_image_folder = r'E:\python\python_projects\yolov7\latest\medias\my_photos'  # Update this to your test images folder

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
                    # Draw landmarks
                    for i in range(68):
                        x = shape.part(i).x
                        y = shape.part(i).y
                        cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)  # Draw a small circle at each landmark

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
                    else:
                        name = "Unknown"

                    # Draw rectangles and labels
                    cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                    cv2.putText(frame, name, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    print("Invalid shape detected for recognition.")
            except Exception as e:
                print(f"Error during face recognition in {filename}: {e}")

        # Resize the image for display
        max_width = 800
        max_height = 600

        height, width = frame.shape[:2]
        scale = min(max_width / width, max_height / height)
        new_dimensions = (int(width * scale), int(height * scale))
        resized_frame = cv2.resize(frame, new_dimensions)

        # Display the resulting image
        cv2.imshow('Face Detection and Recognition with Landmarks', resized_frame)
        cv2.waitKey(0)  # Wait until a key is pressed

cv2.destroyAllWindows()
