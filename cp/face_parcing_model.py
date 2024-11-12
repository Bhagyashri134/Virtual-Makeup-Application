import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe face mesh model
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Initialize the face mesh model
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open a video capture or use an image
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with image path for static image

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for better user experience
    frame = cv2.flip(frame, 1)

    # Convert the image to RGB (MediaPipe works with RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and find landmarks
    results = face_mesh.process(rgb_frame)

    # If face landmarks are found
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw landmarks on the face
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
            
            # Get the points for lips (for applying lipstick)
            # Lips are typically between points 61 to 81 in the face mesh
            lip_points = []
            for i in range(61, 81):
                x = int(face_landmarks.landmark[i].x * frame.shape[1])
                y = int(face_landmarks.landmark[i].y * frame.shape[0])
                lip_points.append((x, y))

            # Create a mask for lipstick (for simplicity, we'll fill the lips area with red)
            lip_points = np.array(lip_points, dtype=np.int32)
            cv2.fillConvexPoly(frame, lip_points, (0, 0, 255))  # Red color for lipstick

    # Display the resulting frame
    cv2.imshow('Virtual Makeup - Face Parsing', frame)

    # Exit loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
