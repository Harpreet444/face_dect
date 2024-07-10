import cv2
import streamlit as st
import numpy as np

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.title("Live Camera Face Detection")

run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

camera = None

def get_camera_index():
    # Try to access the first few possible camera indices
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return -1

camera_index = get_camera_index()
if camera_index == -1:
    st.error("Error: Could not find a working camera. Please check your camera connection.")
else:
    camera = cv2.VideoCapture(camera_index)

if camera is not None and not camera.isOpened():
    st.error(f"Error: Could not open camera with index {camera_index}.")
    camera.release()
else:
    while run:
        if camera is not None:
            ret, frame = camera.read()
            if not ret:
                st.error("Error: Failed to capture image. Please ensure the camera is not being used by another application.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Draw rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            FRAME_WINDOW.image(frame)
        else:
            st.error("Error: Camera not initialized.")

    if camera is not None:
        camera.release()
