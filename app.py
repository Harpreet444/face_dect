import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase
import cv2
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)

st.title("Live Face Detection")

class FaceDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        logging.debug("Initializing FaceDetectionTransformer")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.face_cascade.empty():
            logging.error("Failed to load face cascade")

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        return img

def app():
    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=FaceDetectionTransformer,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

    if webrtc_ctx.video_transformer:
        webrtc_ctx.video_transformer.result_queue.clear()
    
    st.write("Please ensure that your browser has permission to access the camera. If you see a black screen, try refreshing the page or checking your camera settings.")

if __name__ == "__main__":
    app()
