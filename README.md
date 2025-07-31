# 🖋️ Signature Recognition using Vision Transformer

This is a deep learning-based signature recognition system that identifies the person behind a handwritten signature image. The project is built using the **Vision Transformer (ViT)** architecture, trained on a custom dataset of 16 users and deployed using **Streamlit Cloud**.

---

## 🚀 Live Demo

🎯 Try the live app here:  
👉 [https://your-streamlit-link.streamlit.app  ](https://signature-classification-hbkjpqtefgn9tqkwgspqhf.streamlit.app/)


Upload a signature image and instantly get the predicted user's name!

---

## 📖 Project Description

The goal of this project is to classify signature images by the user who wrote them. We collected signature samples from **16 users**, with **16 samples each**, totaling **256 images**. Using data augmentation techniques, we expanded and diversified the dataset for better training.

We fine-tuned a **pretrained Vision Transformer** model (`google/vit-base-patch16-224-in21k`) using PyTorch and Hugging Face Transformers. The model is then integrated into a user-friendly Streamlit app.

---

## 📁 Dataset

- Total Users: 16
- Signature Samples per User: 16
- Dataset Directory: `new_test/`
- Each user has a dedicated subfolder with their signature images

Example:
