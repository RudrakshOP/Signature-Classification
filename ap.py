import os
import torch
import streamlit as st
from PIL import Image
from transformers import ViTForImageClassification
from torchvision import transforms
import requests

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Signature Classifier 🎯",
    page_icon="🖋️",
    layout="centered"
)

# ---------- BEAUTIFUL CUSTOM CSS ----------
st.markdown("""
    <style>
    html, body {
        background: linear-gradient(to right, #ffecd2 0%, #fcb69f 100%);
    }
    .stApp {
        background: transparent;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        text-align: center;
        font-size: 50px;
        font-weight: 800;
        color: #2f3640;
        margin-top: 10px;
        margin-bottom: 5px;
        text-shadow: 2px 2px #f5f6fa;
    }
    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #353b48;
        margin-bottom: 30px;
    }
    .uploadbox > label {
        background-color: #ff6b81;
        color: white;
        padding: 10px 20px;
        border-radius: 12px;
        cursor: pointer;
        font-weight: bold;
        font-size: 16px;
    }
    .uploadbox > label:hover {
        background-color: #ff4757;
    }
    .prediction-box {
        margin-top: 25px;
        background: linear-gradient(135deg, #81ecec, #74b9ff);
        color: #2d3436;
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.1);
    }
    .footer {
        text-align: center;
        color: #636e72;
        font-size: 14px;
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- IMAGE TRANSFORMATION ----------
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image):
    image = image.convert('RGB')
    image = data_transform(image)
    return image.unsqueeze(0)

# ---------- MODEL LOADING FROM HUGGING FACE ----------
@st.cache_resource
def load_model():
    model_url = "https://huggingface.co/RudrakshOP/SignatureClassificationn/resolve/main/modeler.pth"
    model_path = "modeler.pth"

    # Download model if not exists
    if not os.path.exists(model_path):
        with st.spinner("Downloading model..."):
            response = requests.get(model_url)
            with open(model_path, "wb") as f:
                f.write(response.content)

    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k', num_labels=16)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# ---------- USER CLASSES ----------
user_names = ['Aditya', 'Ayushi', 'deepak', 'deepam', 'devesh', 'harsh', 'himarshini', 'Moksh',
              'Nikita', 'ninja', 'nishant', 'priydarshini', 'rajsabi', 'rishab', 'rudraksh', 'shomesh']

# ---------- HEADER ----------
st.markdown('<div class="title">🔍 Signature Verification</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a signature image to identify the person behind it</div>', unsafe_allow_html=True)

# ---------- FILE UPLOAD ----------
with st.container():
    st.markdown('<div class="uploadbox">📂</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg", "bmp"])

# ---------- IMAGE PROCESSING ----------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Signature", use_column_width=True)

    input_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.logits.argmax(-1).item()
        predicted_user = user_names[predicted_class]

    st.markdown(f'<div class="prediction-box">✅ <strong>Predicted User:</strong> {predicted_user}</div>', unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown('<div class="footer">Made with ❤️ using Streamlit & Vision Transformer</div>', unsafe_allow_html=True)
