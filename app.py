import streamlit as st
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Set page config
st.set_page_config(page_title="YOLOv8 Live Object Detection", page_icon="📷", layout="centered")

st.title("📷 Live Real-Time Object Detection")
st.write("Turn on your webcam to detect objects live using YOLOv8 Nano.")

# Load the YOLO model (cached so it doesn't reload on every interaction)
@st.cache_resource
def load_model():
    # Load a pre-trained YOLOv8 nano model
    model = YOLO('yolov8n.pt')
    return model

model = load_model()

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame):
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")

        # Run inference
        results = self.model(img)
        
        # Plot the results on the image (returns a BGR numpy array)
        res_plotted = results[0].plot()

        return res_plotted

webrtc_ctx = webrtc_streamer(
    key="example",
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
)

if webrtc_ctx.state.playing:
    st.write("Webcam is active. Look out for detections!")
