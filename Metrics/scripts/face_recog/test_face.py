import dlib
import cv2
import numpy as np
import os

# Initialize Dlib's face detector and face recognition model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'E:\python\python_projects\yolov7\latest\datas\models\shape_predictor_68_face_landmarks.dat')
recognizer = dlib.face_recognition_model_v1(r'E:\python\python_projects\yolov7\latest\datas\models\dlib_face_recognition_resnet_model_v1.dat')

# Load saved face encodings and names
encoding_dir = r'E:\python\python_projects\yolov7\latest\datas\encodings'
face_encodings = np.load(os.path.join(encoding_dir, 'face_encodings.npy'), allow_pickle=True)
face_names = np.load(os.path.join(encoding_dir, 'face_names.npy'), allow_pickle=True)

# Capture video from the webcam
cap = cv2.VideoCapture(0)  # Use 0 for webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_frame)

    for face in faces:
        try:
            # Use the predictor to detect landmarks
            shape = predictor(gray_frame, face)  # Use the grayscale frame
            
            # Check if the shape is valid before computing the descriptor
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
                else:
                    name = "Unknown"

                # Draw rectangles and labels
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                cv2.putText(frame, name, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                print("Invalid shape detected for recognition.")
        except Exception as e:
            print(f"Error during face recognition: {e}")

    # Display the resulting frame
    cv2.imshow('Face Detection and Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
