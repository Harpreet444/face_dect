# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
# import cv2
# import logging

# # Setup logging
# logging.basicConfig(level=logging.DEBUG)

# st.title("Live Face Detection")

# class FaceDetectionProcessor(VideoProcessorBase):
#     def __init__(self):
#         logging.debug("Initializing FaceDetectionProcessor")
#         self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#         if self.face_cascade.empty():
#             logging.error("Failed to load face cascade")

#     def recv(self, frame):
#         img = frame.to_ndarray(format="bgr24")

#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

#         for (x, y, w, h) in faces:
#             cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

#         return frame.from_ndarray(img, format="bgr24")

# def app():
#     webrtc_ctx = webrtc_streamer(
#         key="example",
#         mode=WebRtcMode.SENDRECV,
#         video_processor_factory=FaceDetectionProcessor,
#         media_stream_constraints={
#             "video": True,
#             "audio": False
#         },
#         async_processing=True
#     )

#     st.write("Please ensure that your browser has permission to access the camera. If you see a black screen, try refreshing the page or checking your camera settings.")

# if __name__ == "__main__":
#     app()


import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)

st.title("Basic Webcam Stream")

def app():
    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

    st.write("Please ensure that your browser has permission to access the camera. If you see a black screen, try refreshing the page or checking your camera settings.")

if __name__ == "__main__":
    app()
