# A-EYE
# 👁️ Eye.AI – Smart Eye Disease Detection with Deep Learning

**Eye.AI** is an AI-powered web application that detects and classifies common eye diseases from **Fundus** and **OCT** images. Designed for early screening and medical support, Eye.AI combines deep learning models with an interactive frontend to deliver fast, accurate predictions — complete with basic medical advice.

---

## 🚀 Demo

Upload a Fundus or OCT image, and Eye.AI will:
- Auto-detect the image type
- Run the appropriate deep learning model
- Return the predicted disease
- Display tailored medical advice

---

## 🧠 Features

- **Dual-Model System**:
  - `Fundus Model` (EfficientNetB0): Detects Cataract, Glaucoma, Diabetic Retinopathy, Normal
  - `OCT Model` (ResNet50): Detects CNV, DME, Drusen, Normal
- **TFLite Models** for lightweight and fast inference
- **Auto Image Type Detection** from uploaded images
- **Medical Feedback Generator** with recommendations for each condition
- **Responsive Frontend** using HTML + Tailwind CSS
- **Flask API Backend** to serve predictions

---

## 🖼️ Sample Prediction

| Input | Model | Output |
|------|--------|--------|
| ![Fundus Sample](images/sample_fundus.jpg) | Fundus Model | Cataract |
| ![OCT Sample](images/sample_oct.jpg) | OCT Model | CNV |

---

## 🛠️ Tech Stack

- **Frontend**: HTML, Tailwind CSS, JavaScript
- **Backend**: Python, Flask
- **ML Models**: TensorFlow Lite (EfficientNetB0 + ResNet50)
- **Deployment Ready**

---

## 📁 Project Structure

