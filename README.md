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

## 🔄 Preprocessing

- Images resized to 224x224
- Normalized using ImageNet mean and std
- Converted to PyTorch tensors
- Applied data augmentation:
  - Random rotation
  - Horizontal/Vertical flips
  - Color jittering
  - Scaling

---

## 🧠 Model Details

- Base model: Vision Transformer (ViT)
- Pretrained on ImageNet
- Fine-tuned for 16-class classification (one for each user)
- Loss Function: CrossEntropyLoss
- Optimizer: AdamW
- Framework: PyTorch + Hugging Face Transformers

---

## 📊 Performance

- Achieved high accuracy on the test set
- Verified using real signature images in the `new_test` folder
- Most users are correctly identified; very few misclassifications due to similar writing patterns

---

## 🧪 How to Use

1. Open the Streamlit app from the link above
2. Upload a signature image (`.jpg`, `.jpeg`, `.png`,`.bmp`)
3. The app will display the predicted user name

You can also test with the sample signatures in the `new_test/` folder.

🙋‍♂️ Author
Rudraksh Kaushik
Data Science and AI Enthusiast

GitHub: RudrakshOP

LinkedIn: (https://www.linkedin.com/in/rudraksh-kaushik-17a2831a3/)


