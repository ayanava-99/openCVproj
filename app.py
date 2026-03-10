import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Set page config
st.set_page_config(page_title="YOLOv8 Object Detection", page_icon="📷", layout="centered")

st.title("📷 Real-Time Object Detection")
st.write("Take a picture to detect objects using YOLOv8 Nano.")

# Load the YOLO model (cached so it doesn't reload on every interaction)
@st.cache_resource
def load_model():
    # Load a pre-trained YOLOv8 nano model
    model = YOLO('yolov8n.pt')
    return model

model = load_model()

# Camera input
picture = st.camera_input("Take a picture!")

if picture is not None:
    # Convert the file to a PIL image
    image = Image.open(picture)
    
    with st.spinner('Detecting objects...'):
        # Run inference
        results = model(image)
        
        # Plot the results on the image (returns a BGR numpy array)
        res_plotted = results[0].plot()
        
        # Convert BGR to RGB for Streamlit
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        # Display the output image
        st.image(res_rgb, caption='Detected Objects', use_container_width=True)
        
        # Display detection text
        st.subheader("Detections:")
        
        if len(results[0].boxes) == 0:
            st.write("No objects detected.")
        else:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                conf = float(box.conf[0])
                st.write(f"- **{class_name}** (Confidence: {conf:.2f})")
