---
title: License Plate Digits Classification
emoji: 🚗
colorFrom: indigo
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# 🚗 License Plate Digits Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)](https://fastapi.tiangolo.com/)

A premium, end-to-end classification system for license plate digits and letters (0-9 and A-Z) using deep learning and a sleek modern web interface.

---

## ✨ Features

- **High Accuracy Model**: Custom 3-layer CNN achieving **96.16% validation accuracy**.
- **Modern Web App**: Premium dark-mode interface with glassmorphism, drag-and-drop, and real-time predictions.
- **Interactive CLI**: Easy-to-use command-line tool for quick inference and testing.
- **Robust Pipeline**: Clean modules for dataset handling, model definition, and training.

---

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch, Torchvision
- **Backend**: FastAPI, Uvicorn
- **Frontend**: Vanilla JS, Modern CSS (Glassmorphism), HTML5
- **Data Handling**: PIL, NumPy

---

## 📂 Project Structure

- `app.py`: FastAPI server for the web application and model serving.
- `model.py`: CNN architecture definition (optimized 32x32 input).
- `dataset.py`: Data loading, augmentation, and preprocessing.
- `predict.py`: Interactive CLI script for classification.
- `train.py`: Training script with automated model checkpointing.
- `static/`: Modern web frontend assets (HTML, CSS, JS).
- `best_model.pth`: Pre-trained model weights.

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Web App (Recommended)
Start the modern web interface:
```bash
python app.py
```
Visit **[http://localhost:8000](http://localhost:8000)** in your browser to start classifying!

### 3. Use the CLI
Run the interactive prediction tool:
```bash
python predict.py
```
Or provide an image path directly:
```bash
python predict.py "path/to/image.jpg"
```

### 4. Training
If you wish to retrain the model from scratch:
```bash
python train.py
```

---

## 📊 Dataset
The project uses the **CNN letter Dataset**, containing categorized images of digits (0-9) and uppercase letters (A-Z). The pipeline automatically handles resizing, grayscale conversion, and normalization.

---

## 📝 GitHub Deployment Instructions
1. Create a new repository on GitHub.
2. Link your local repo and push:
   ```bash
   git remote add origin https://github.com/snehadm-25/License-Plate-Digits-Classification.git
   git branch -M main
   git push -u origin main
   ```

---

## 👤 Author
Developed by **Sneha D M**