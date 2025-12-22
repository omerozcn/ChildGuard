import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gradio as gr
import tensorflow as tf
import joblib
from transformers import BertTokenizer, TFBertForSequenceClassification
import re
import numpy as np
from scipy.sparse import hstack, csr_matrix

# ==========================================
# 1. MODELLERİ YÜKLE
# ==========================================
MODEL_PATH = "./final_models/bert_younger"

# BERT Yükle
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
bert_model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH)

# Lojistik Regresyon & TF-IDF Yükle
lr_model = None
tfidf = None
lr_available = False
status_msg = ""

try:
    lr_model = joblib.load("./final_models/logistic_regression_model.pkl")
    tfidf = joblib.load("./final_models/tfidf_vectorizer.pkl")
    lr_available = True
    status_msg = "✅ Hibrit Sistem Aktif (BERT + Logistic Regression)"
except Exception as e:
    status_msg = f"⚠️ LR Yükleme Hatası: {e}"
    lr_available = False

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s!?.,]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ==========================================
# 2. ANALİZ LOGIC (HİBRİT + FEATURE ENGINEERING)
# ==========================================
def analyze_text(text):
    if not text:
        # Arayüzdeki 5 çıktı alanı için boş dönüş (Etiket, Hibrit, BERT, LR, Durum)
        return "⚠️ Lütfen bir metin girin.", 0, 0, 0, "N/A"

    cleaned = clean_text(text)
    
    # --- 1. BERT Tahmini ---
    inputs = tokenizer(cleaned, return_tensors="tf", truncation=True, padding=True, max_length=128)
    bert_logits = bert_model(inputs).logits
    bert_probs = tf.nn.softmax(bert_logits, axis=1).numpy()[0]
    bert_score = float(bert_probs[1])

    # --- 2. Lojistik Regresyon Tahmini ---
    lr_score = 0.0
    current_status = status_msg
    
    if lr_available:
        try:
            tfidf_vec = tfidf.transform([cleaned])
            text_len = len(cleaned)
            word_cnt = len(cleaned.split())
            age_group_features = np.array([[0, 0, 1, 0]]) 
            
            X_final = hstack([tfidf_vec, csr_matrix([[text_len, word_cnt]]), csr_matrix(age_group_features)])
            
            lr_probs = lr_model.predict_proba(X_final)[0]
            lr_score = float(lr_probs[1])
            
        except ValueError as ve:
            current_status = f"❌ Boyut Hatası: {ve}"
            lr_score = 0.0
        except Exception as e:
            current_status = f"❌ LR Tahmin Hatası: {e}"
            lr_score = 0.0

    # --- 3. Hibrit Karar ---
    if lr_available and lr_score >= 0:
        final_score = (bert_score * 0.6) + (lr_score * 0.4)
    else:
        final_score = bert_score

    label = "🚨 RİSKLİ / SİBER ZORBALIK" if final_score > 0.45 else "✅ GÜVENLİ / TEMİZ"
    
    # Sırasıyla: Etiket, Hibrit Skor, BERT Skor, LR Skor, Durum Mesajı
    return label, round(final_score, 4), round(bert_score, 4), round(lr_score, 4), current_status

# ==========================================
# 3. MODERN ARAYÜZ (GÜNCELLENMİŞ)
# ==========================================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate")) as demo:
    gr.Markdown("# 🛡️ ChildGuard AI - Hibrit Analiz Paneli")
    gr.Markdown("*BERT ve Lojistik Regresyon modellerinin tahminlerini aşağıda ayrı ayrı görebilirsiniz.*")

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(label="Analiz Edilecek Mesaj", placeholder="Buraya şüpheli bir metin yazın...", lines=5)
            with gr.Row():
                clear_btn = gr.Button("Temizle")
                submit_btn = gr.Button("Analiz Et", variant="primary")

        with gr.Column(scale=1):
            output_label = gr.Label(label="Tespit Durumu")
            confidence_bar = gr.Slider(label="Final Hibrit Skor (%60 BERT + %40 LR)", minimum=0, maximum=1, interactive=False)
            
            # Skorları yan yana göstermek için yeni bir satır
            with gr.Row():
                bert_val = gr.Number(label="BERT Tekil Skor", interactive=False)
                lr_val = gr.Number(label="LR Tekil Skor", interactive=False)
                
            model_info = gr.Textbox(label="Sistem Durumu", interactive=False)

    gr.Examples(
        examples=[
            ["You are a loser, nobody likes you in this school."],
            ["You are a disgusting piece of trash and the world would be better without you."],
            ["I love playing football with my friends."],
            ["Look at yourself in the mirror, you are absolutely hideous."],
            ["You're so incredibly stupid, stop talking forever."]
        ],
        inputs=text_input
    )

    submit_btn.click(
        fn=analyze_text,
        inputs=text_input,
        outputs=[output_label, confidence_bar, bert_val, lr_val, model_info]
    )
    # Temizle butonu için 5 alanı sıfırla
    clear_btn.click(lambda: [None, 0, 0, 0, ""], outputs=[text_input, confidence_bar, bert_val, lr_val, model_info])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)