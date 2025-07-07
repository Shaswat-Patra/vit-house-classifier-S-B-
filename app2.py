import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import timm
import requests

# ------------------- Configuration -------------------
MODEL_PATH = "best_vit_model.pth"
CLASS_NAMES = ['Kutcha House', 'Pucca House']
CONFIDENCE_THRESHOLD = 0.80

# ------------------- Ensemble Model Class -------------------
class EnsembleModel(torch.nn.Module):
    def __init__(self, model_a, model_b):
        super(EnsembleModel, self).__init__()
        self.model_a = model_a
        self.model_b = model_b

    def forward(self, x):
        out1 = self.model_a(x)
        out2 = self.model_b(x)
        return (out1 + out2) / 2  # Averaged output

# ------------------- Download and Load Ensemble -------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model..."):
            model_url = st.secrets["MODEL_URL"]
            response = requests.get(model_url)
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.content)

    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

    # Initialize models
    swin = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=len(CLASS_NAMES))
    convnext = timm.create_model('convnext_base', pretrained=False, num_classes=len(CLASS_NAMES))

    # Load weights
    swin.load_state_dict(checkpoint['swin_state_dict'])
    convnext.load_state_dict(checkpoint['convnext_state_dict'])

    # Create ensemble
    model = EnsembleModel(swin, convnext)
    model.eval()
    return model

# ------------------- Preprocessing -------------------
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Batch dimension

# ------------------- Prediction -------------------
def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.nn.functional.softmax(output, dim=1).numpy()[0]
        max_prob = np.max(probs)
        predicted_idx = np.argmax(probs)
        if max_prob < CONFIDENCE_THRESHOLD:
            return "❌ Cannot detect class. Please upload a valid full house photo.", probs
        return CLASS_NAMES[predicted_idx], probs

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="🏠 House Type Classifier", layout="centered")

st.title("🏠 House Type Classifier")
st.caption("Identify whether a house is **Kutcha** or **Pucca** from uploaded images using an ensemble deep learning model.")

# Sidebar
with st.sidebar:
    st.header("📘 About")
    st.markdown("""
    This AI model classifies house images into two categories:
    - Kutcha House
    - Pucca House

    It uses an **ensemble** of Swin Transformer and ConvNeXt models for robust prediction.
    """)
    st.header("👤 Developer")
    st.markdown("""
    **Name:** Shaswat Patra  
    **Email:** patrarishu@gmail.com
    """)

# Load model
model = load_model()

# Upload instruction
st.warning("⚠️ This model is trained only on house images. Irrelevant or out-of-context images may still be classified as 'Kutcha' or 'Pucca' incorrectly. Please upload only valid clear full house images for accurate results.")

# Upload
uploaded_files = st.file_uploader("Upload house image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.divider()
        st.info(f"📷 Processing: `{uploaded_file.name}`")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("🔍 Predicting..."):
            input_tensor = preprocess_image(image)
            label, probs = predict(model, input_tensor)

        if "Cannot detect" in label:
            st.warning(label)
        else:
            confidence = np.max(probs) * 100
            st.success(f"🏷️ Predicted Class: **{label}**  \n📊 Confidence: **{confidence:.1f}%**")
