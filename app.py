import streamlit as st
import cv2
import av
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer

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

# The callback function for streamlit-webrtc to process video frames
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # Convert frame to numpy array
    img = frame.to_ndarray(format="bgr24")

    # Run inference
    results = model(img)
    
    # Plot the results on the image (returns a BGR numpy array)
    res_plotted = results[0].plot()

    # Return the processed frame
    return av.VideoFrame.from_ndarray(res_plotted, format="bgr24")

webrtc_streamer(
    key="example",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
)
