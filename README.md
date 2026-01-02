
# 🛡️ ChildGuard AI – Hybrid Harmful Content Classifier

### BERT + TF‑IDF Logistic Regression Hybrid Classifier

ChildGuard AI is a **hybrid text classification system** developed to detect **harmful content targeting children** in online texts. The project combines **Deep Learning (BERT)** with **Classical Machine Learning (TF‑IDF + Logistic Regression)** approaches to produce more balanced, explainable, and high‑accuracy results.

The application provides **real‑time analysis** through a **Gradio‑based web interface** by loading pre‑trained and serialized models (`.pkl`, `save_pretrained`).

---

## 🚀 Hybrid Architecture (Next‑Generation Approach)

The system combines the outputs of two different models using a **weighted decision mechanism**:

* **BERT (Transformers)**
  Analyzes the contextual and semantic structure of the text.  
  **Weight:** 60%

* **Logistic Regression (Feature‑Engineered)**
  Performs statistical analysis using TF‑IDF vectors along with textual and demographic features.  
  **Weight:** 40%

This approach provides **generalizability** and **stability** without relying solely on deep learning.

---

## 📌 Technical Details

### 1️⃣ Model Serialization

* **Joblib / Pickle**  
  The Logistic Regression model and TF‑IDF vectorizer are saved in `.pkl` format, eliminating the need for retraining on each run.

* **HuggingFace – save_pretrained**  
  BERT models and tokenizers are optimized for fast loading from local directories.

---

### 2️⃣ Feature Engineering

To improve the performance of the classical model, the following features were used in accordance with data mining principles:

* **TF‑IDF Vectors**
  * Maximum dimension: 5000  
  * Text‑based weighted feature extraction

* **Numerical Metadata**
  * Text length (`text_len`)  
  * Word count (`word_cnt`)

* **One‑Hot Encoding**
  * Target age group information

This structure significantly improves the contribution of the classical model, especially for short and ambiguous texts.

---

### 3️⃣ Hybrid Decision Mechanism

The probability scores obtained from both models are combined using the following formula:

Final Score = (BERT_prob × 0.6) + (LR_prob × 0.4)

This score forms the basis of the final classification decision.

---

## 📁 Project File Structure

```text
├── final_models/
│   ├── bert_pre-teen/
│   ├── bert_teen/
│   ├── bert_younger/
│   ├── logistic_regression_model.pkl
│   └── tfidf_vectorizer.pkl
├── app.py
├── childguardhybrid.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```
---

## ⚙️ Installation and Execution

### 1️⃣ Preparing the Models

Place the trained `.pkl` files and BERT model directories into the following path:

final_models/

---

### 2️⃣ Installing Dependencies

pip install -r requirements.txt

---

### 3️⃣ Running the Application

python app.py

After the application starts, you can access the interface via the link shown in the terminal:

http://127.0.0.1:7860

---

## 🐳 Running with Docker (Optional)

The project is optimized with **Docker layer caching** support.

docker-compose up -d --build

This method enables fast deployment by preventing repeated dependency downloads.

---

## 🎯 Use Cases

* Social media content moderation
* Content filtering in educational platforms
* Child‑focused digital safety systems
* Academic data mining and NLP research

---

## 👨‍💻 Developer

**Ömer Özcan**  
Afyon Kocatepe University – Computer Engineering

📌 This project was developed as part of the **Data Mining course final assignment**.

---

> 🛡️ *ChildGuard AI is designed to help children stay safer in the digital world.*
