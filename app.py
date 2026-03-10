import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline

# Set page config
st.set_page_config(page_title="YOLOv8 Object Detection", page_icon="📷", layout="centered")

st.title("📷 Real-Time Object Detection & Analysis")
st.write("Take a picture to detect objects. If a person is detected, the app will predict their age and emotion!")

# Load the YOLO model (cached so it doesn't reload on every interaction)
@st.cache_resource
def load_yolo_model():
    # Load a pre-trained YOLOv8 nano model
    model = YOLO('yolov8n.pt')
    return model

# Load the Age Classifier (cached)
@st.cache_resource
def load_age_model():
    # Lightweight Vision Transformer for age classification
    age_pipeline = pipeline("image-classification", model="nateraw/vit-age-classifier")
    return age_pipeline

# Load the Emotion Classifier (cached)
@st.cache_resource
def load_emotion_model():
    # Lightweight Vision Transformer for emotion detection
    emotion_pipeline = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
    return emotion_pipeline

model = load_yolo_model()
age_predictor = load_age_model()
emotion_predictor = load_emotion_model()

# Camera input
picture = st.camera_input("Take a picture!")

if picture is not None:
    # Convert the file to a PIL image
    image = Image.open(picture)
    
    with st.spinner('Detecting objects and analyzing persons...'):
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
                
                # If a person is detected, predict their age and emotion
                if class_name == "person":
                    # Get bounding box coordinates for cropping
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Crop the person from the original PIL image
                    person_crop = image.crop((x1, y1, x2, y2))
                    
                    # Predict age and emotion
                    try:
                        age_preds = age_predictor(person_crop)
                        top_age = age_preds[0]['label']
                        
                        emotion_preds = emotion_predictor(person_crop)
                        # Emotion models sometimes return labels in lowercase or with underscores
                        top_emotion = emotion_preds[0]['label'].replace("_", " ").title()
                        
                        st.write(f"- 🧔 **person** (Confidence: {conf:.2f}) | **Predicted Age:** {top_age} | **Emotion:** {top_emotion}")
                    except Exception as e:
                        st.write(f"- 🧔 **person** (Confidence: {conf:.2f}) | Age/Emotion Prediction Failed")
                else:
                    st.write(f"- **{class_name}** (Confidence: {conf:.2f})")
