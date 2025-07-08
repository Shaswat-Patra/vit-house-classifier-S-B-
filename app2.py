import gradio as gr
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
MODEL_PATH = "best_vit_model(swin+convnext).pth"
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
def load_model():
    if not os.path.exists(MODEL_PATH):
        model_url = os.environ.get("MODEL_URL")
        response = requests.get(model_url)
        if response.status_code != 200:
            raise RuntimeError("❌ Failed to download model. Check permissions or URL.")
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)

    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

    # Initialize models
    swin = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=len(CLASS_NAMES))
    convnext = timm.create_model('convnext_base', pretrained=False, num_classes=len(CLASS_NAMES))

    # Load weights
    swin.load_state_dict(checkpoint['swin_state_dict'])
    convnext.load_state_dict(checkpoint['convnext_state_dict'])

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

# -------------- Gradio Interface Function --------------
model = load_model()

def classify_house(image):
    image = image.convert("RGB")
    input_tensor = preprocess_image(image)
    label, probs = predict(model, input_tensor)
    if "Cannot detect" in label:
        return label
    confidence = np.max(probs) * 100
    return f"🏷️ Predicted Class: {label} | 📊 Confidence: {confidence:.1f}%"


# ------------------- Gradio UI -------------------
import gradio as gr

model = load_model()

def classify_images(images):
    results = []
    for img in images:
        img = img.convert("RGB")
        tensor = preprocess_image(img)
        label, probs = predict(model, tensor)

        if "Cannot detect" in label:
            results.append((img, label))
        else:
            confidence = np.max(probs) * 100
            results.append((img, f"🏷️ {label} | 📊 Confidence: {confidence:.1f}%"))
    return results

with gr.Blocks() as demo:
    gr.Markdown("# 🏠 House Type Classifier")
    gr.Markdown("Identify whether a house is **Kutcha** or **Pucca** using an **ensemble of Swin Transformer and ConvNeXt** models.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload House Image(s)", tool=None, image_mode='RGB', sources="upload", multiple=True)
            submit_btn = gr.Button("🔍 Predict")
        with gr.Column():
            output_gallery = gr.Gallery(label="📸 Results").style(grid=[1], height="auto")

    submit_btn.click(fn=classify_images, inputs=image_input, outputs=output_gallery)

    gr.Markdown("""
    ### 📘 About  
    This model classifies house images into two categories:  
    - **Kutcha House**  
    - **Pucca House**  
    Built with a deep learning ensemble for more robust predictions.

    ### 👤 Developer  
    **Name:** Shaswat Patra  
    **Email:** patrarishu@gmail.com  
    """)

if __name__ == "__main__":
    demo.launch()
