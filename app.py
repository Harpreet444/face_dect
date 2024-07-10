import streamlit as st
import numpy as np
import cv2

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.title("Live Camera Face Detection")

run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

if run:
    camera_input = st.camera_input()

    if camera_input is not None:
        # Convert the image to OpenCV format
        frame = np.array(camera_input)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Convert image back to RGB and display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame, channels="RGB")

