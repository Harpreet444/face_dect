import cv2
import streamlit as st
import numpy as np
from PIL import Image

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.title("Live Camera Face Detection")

camera_input = st.camera_input("Take a picture")

if camera_input:
    # To read image file buffer with OpenCV
    image = Image.open(camera_input)
    img_array = np.array(image)

    # Convert the image from BGR to RGB
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Detect faces
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Convert image back to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Display the image
    st.image(img, caption='Captured Image with Detected Faces', use_column_width=True)
    st.write(f"Found {len(faces)} face(s)")
