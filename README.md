# 🤖 AI-Generated Image Detection using Deep Learning

## 📌 Project Overview

With the rapid advancement of generative AI models such as GANs and diffusion models, AI-generated images have become highly realistic. This project aims to build a deep learning system that classifies whether an image is:

- ✅ Real (camera-captured)
- ❌ AI-generated

The model learns hidden visual patterns and artifacts that distinguish synthetic images from real photographs.

---

## 🎯 Objectives

- Build a CNN-based binary image classifier
- Train on a large balanced dataset (60,000 images)
- Achieve strong generalization performance
- Compare custom CNN with Transfer Learning
- Deploy using Streamlit for real-time detection

---

## 📊 Dataset

- 30,000 Real images (Training)
- 30,000 AI-generated images (Training)
- 10,000 Real images (Testing)
- 10,000 AI-generated images (Testing)

Dataset was balanced to prevent model bias.

Images were resized to 224x224 and normalized before training.

---

## 🧠 Model Architecture (Custom CNN)

The architecture includes:

- 2 Convolutional Layers
- MaxPooling Layers
- Flatten Layer
- Dense Layers
- Dropout (Regularization)
- Sigmoid Output Layer

### Loss Function:
Binary Crossentropy

### Optimizer:
Adam

### Regularization:
Dropout + EarlyStopping

---

## 📈 Results

Test Set Performance (20,000 images):

Accuracy: **91%**

| Metric      | Real | Fake |
|------------|------|------|
| Precision  | 0.92 | 0.91 |
| Recall     | 0.91 | 0.92 |
| F1-Score   | 0.91 | 0.91 |

The model shows balanced performance across both classes.

---

## 🔬 Transfer Learning Experiment

MobileNetV2 was tested using transfer learning.

Result:
- Feature extraction only: 81% accuracy
- Custom CNN outperformed for this dataset.

This highlights the importance of dataset-specific training.

---

## 🚀 Deployment

The model is deployed using Streamlit.

Features:
- Image upload
- Real/Fake prediction
- Confidence score display
- Probability visualization

Run locally:

```bash
streamlit run app.py