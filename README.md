# 🥔 Potato Disease Classification - Deep Learning Project

A deep learning web application built with **TensorFlow**, **FastAPI**, and **LangChain** that classifies potato leaf diseases and provides an **AI-powered agricultural chatbot** for farmers and researchers.  

The app predicts whether a potato leaf is:

- **Early Blight** 🍂  
- **Late Blight** 🧫  
- **Healthy** ✅

---

## 🚀 Features

- 📷 **Image Upload & Real-Time Disease Prediction**  
- 💬 **RAG-based Agricultural Chatbot** integrated using LangChain  
- 🎨 Clean and responsive **frontend** (HTML, CSS, JS)  
- 🌆 Background image with **blur effect**  
- 🧠 **Convolutional Neural Network (CNN)** for disease classification  
- ⚡ **FastAPI Backend** for prediction and chatbot APIs  
- 🔒 `.env` file support for secret keys (e.g., Hugging Face token)

---

## 🧠 Model Overview

- **Input Shape:** `256x256x3`  
- **Architecture:**  
  - 6 × Conv2D layers (ReLU activation)  
  - MaxPooling for downsampling  
  - Dense layers for classification  
  - Softmax output layer  
- **Model File:** `potatoes.h5`  
- **Framework:** TensorFlow / Keras

---

## 🤖 RAG-Based Chatbot Integration

Alongside disease prediction, this project includes a **Retrieval-Augmented Generation (RAG)** chatbot that answers agricultural queries related to **potato diseases**, **treatment methods**, and **farming practices**.

### 🧩 Chatbot Pipeline

1. **Knowledge Base Creation**  
   - Collected domain-specific text documents about potato diseases and farming techniques.  
   - Loaded using **`TextLoader`** from LangChain.

2. **Text Processing**
   - Split into smaller chunks using **`RecursiveCharacterTextSplitter`** for efficient retrieval.  

3. **Vector Store**
   - Created embeddings using:  
     ```python
     model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
     ```
   - Stored embeddings in **FAISS (Facebook AI Similarity Search)** for high-speed retrieval.  

4. **RAG Chain**
   - Combined **retriever** + **Hugging Face language model** for contextual question answering.  
   - Deployed via FastAPI endpoints alongside the disease classification API.

---
## 🎥 Demo Video

Watch the full project demo on YouTube:
👉 https://youtu.be/ZknDwnZHyRk

## ⚙️ Installation & Setup

### ✅ 1. Clone the repo

bash
git clone https://github.com/aliahmad552/potato_disease_recognition.git
cd Potato-Disease-Classification

### ✅ 2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
### ✅ 3. Run the Flask app
bash
Copy
Edit
python app.py
Now visit: http://127.0.0.1:5000

### 📂 Dataset
Custom dataset with three classes (Early Blight, Late Blight, Healthy)

Train/Validation/Test split handled using ImageDataGenerator

Image size: 256x256, normalized between 0-1

### 💻 Author
Ali Ahmad
BS Software Engineering – The Islamia University of Bahawalpur
GitHub: aliahmad552

