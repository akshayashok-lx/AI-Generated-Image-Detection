# 🤖 AI-Generated Image Detection using Deep Learning

## 📌 Project Overview

With the rapid advancement of GANs and diffusion models, AI-generated images have become highly realistic.  
This project builds a Deep Learning-based binary classifier to detect:

✅ Real (Camera-Captured Image)  
❌ AI-Generated (Synthetic Image)

The model learns hidden artifacts, texture inconsistencies, and structural patterns that differentiate synthetic images from real photographs.

---

# 🎯 Objectives

- Build a Custom CNN binary classifier
- Train on a large balanced dataset (60,000+ images)
- Compare with Transfer Learning (MobileNetV2)
- Deploy using Streamlit for real-time detection
- Achieve strong generalization on unseen data

---

# 📊 Dataset

## Dataset Distribution

Training:
- 30,000 Real
- 30,000 Fake

Testing:
- 10,000 Real
- 10,000 Fake

✔ Balanced dataset  
✔ Resized to 224x224  
✔ Normalized (pixel values scaled 0–1)

---

# 📂 Dataset Folder Structure (Important)

Your dataset must be arranged like this:

dataset/
│
├── train/
│   ├── real/
│   └── fake/
│
├── test/
│   ├── real/
│   └── fake/

The folder names (`real`, `fake`) act as class labels.

---

# 🧠 Model Architecture (Custom CNN)

- Conv2D (32 filters)
- MaxPooling
- Conv2D (64 filters)
- MaxPooling
- Flatten
- Dense (128)
- Dropout
- Dense (1, Sigmoid)

### Training Configuration

- Loss: Binary Crossentropy
- Optimizer: Adam
- Batch Size: 32
- Image Size: 224x224
- EarlyStopping
- ModelCheckpoint (best_model.h5 saved)

---

# 🏗 How to Train the Model (Step-by-Step)

## 1️⃣ Clone the Repository

git clone https://github.com/akshayashok-lx/Ai-Generated-Image-Detection.git
cd ai-image-detection

---

## 2️⃣ Create Virtual Environment (Recommended)

Windows:
python -m venv venv
venv\Scripts\activate

Mac/Linux:
python3 -m venv venv
source venv/bin/activate

---

## 3️⃣ Install Dependencies

pip install -r requirements.txt

---

## 4️⃣ Place Dataset

Download your dataset and place it inside:

dataset/train/
dataset/test/

Make sure the structure matches the format shown above.

---

## 5️⃣ Train the Model

Run:

python model_training.py

During training:
- Model will train for defined epochs
- EarlyStopping will monitor validation loss
- Best model will be saved automatically

Saved model location:
models/best_model.h5

---

# 📈 Model Performance

Test Accuracy: 91%

| Metric    | Real | Fake |
|-----------|------|------|
| Precision | 0.92 | 0.91 |
| Recall    | 0.91 | 0.92 |
| F1 Score  | 0.91 | 0.91 |

Balanced performance across both classes.

---

# 🔬 Transfer Learning Experiment

MobileNetV2 (Feature Extraction Only)

Accuracy: 81%

Observation:
Custom CNN outperformed pretrained feature extraction for this dataset.

---

# 🚀 How to Run the Streamlit App

Make sure model file exists inside:

models/best_model.h5

Then run:

streamlit run app.py

The app will open in your browser at:

http://localhost:8501

---

# 🖥 Application Features

- Upload Image
- Real/Fake Prediction
- Confidence Score
- Probability Visualization

---

# 📦 Requirements

tensorflow
keras
numpy
opencv-python
matplotlib
streamlit
scikit-learn
pillow

---

# 🗂 Project Structure

AI-Image-Detection/
│
├── app.py
├── model_training.py
├── requirements.txt
├── models/
│   └── best_model.h5
├── dataset/
│   ├── train/
│   └── test/
└── README.md

---

# 🔮 Future Improvements

- Add Grad-CAM visualization
- Add Frequency Domain Analysis
- Deploy on cloud (AWS / Render / HuggingFace)
- Use EfficientNet for higher accuracy
- Convert to ONNX for optimization

---

# 👨‍💻 Author

Akshay A  
Machine Learning | Deep Learning | GenAI Enthusiast

---

⭐ If you found this project useful, give it a star!

streamlit run app.py


