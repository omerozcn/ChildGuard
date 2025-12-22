# 🛡️ ChildGuard AI – Hibrit Zararlı İçerik Sınıflandırıcı
### BERT + TF‑IDF Logistic Regression Hybrid Classifier

Bu proje, çevrim içi metinlerde çocukları hedef alan zararlı içerikleri tespit etmek için **Derin Öğrenme (BERT)** ve **Klasik Makine Öğrenmesi (Logistic Regression)** yaklaşımlarını birleştiren hibrit bir sistemdir. Sistem, eğitilmiş modelleri serileştirilmiş (`.pkl` ve `save_pretrained`) formatta yükleyerek **Gradio** arayüzü üzerinden anlık analiz sunar.

---

## 🚀 Yeni Nesil Hibrit Yapı

Eski versiyonlardan farklı olarak sistem artık iki farklı mimariyi eş zamanlı çalıştırarak karar verir:

* **BERT (Transformers):** Metnin anlamsal (contextual) yapısını analiz eder ve %60 ağırlığa sahiptir.
* **Logistic Regression (Feature Engineered):** TF-IDF vektörlerine ek olarak metin uzunluğu, kelime sayısı ve yaş grubu verilerini harmanlayarak istatistiksel analiz yapar ve %40 ağırlığa sahiptir.

---

## 📌 Teknik Detaylar ve Veri Madenciliği Referansları

### 1) Model Serileştirme (Serialization)
* **Joblib & Pickle:** Eğitilen klasik model ve TF-IDF vektörleştirici `.pkl` formatında kaydedilerek, her seferinde tekrar eğitim yapmadan anında yüklenmesi sağlanmıştır.
* **HuggingFace Save/Load:** BERT modelleri ve tokenizer'ları `save_pretrained` metodu ile yerel dizinden yüklenecek şekilde optimize edilmiştir.

### 2) Öznitelik Mühendisliği (Feature Engineering)
Veri madenciliği prensiplerine uygun olarak klasik modelin başarısını artırmak için şu nitelikler kullanılmıştır:
* **TF-IDF Vektörleri:** 5000 boyutlu metin temsil matrisi.
* **Sayısal Meta Veriler:** Metin karakter uzunluğu (`text_len`) ve kelime sayısı (`word_cnt`).
* **One-Hot Encoding:** Hedef yaş grubunun sayısal matrise dönüştürülmesi.

### 3) Hibrit Karar Mekanizması
Modellerden gelen olasılık skorları şu formül ile birleştirilir:

$$Final Score = (BERT_{prob} \times 0.6) + (LR_{prob} \times 0.4)$$

---

## 📁 Dosya Yapısı

```text
├── final_models/
│   ├── bert_pre-teen/            # BERT Model (11-13 Yaş Grubu)
│   ├── bert_teen/                # BERT Model (13-17 Yaş Grubu)
│   ├── bert_younger/             # BERT Model (<11 Yaş Grubu)
│   ├── logistic_regression_model.pkl  # Eğitilmiş LR modeli
│   └── tfidf_vectorizer.pkl      # Eğitilmiş TF-IDF nesnesi
├── app.py                        # Gradio Web Interface (Ana Uygulama)
├── childguardhybrid.py           # Model Eğitim ve Test Kodları
├── docker-compose.yml            # Docker Servis Konfigürasyonu
├── Dockerfile                    # Konteynır İmaj Dosyası
├── requirements.txt              # Gerekli Kütüphaneler
└── README.md                     # Proje Dökümantasyonu

---

## 🚀 Kurulum ve Çalıştırma

### 1) Modelleri Hazırlama
Eğittiğiniz `.pkl` ve BERT klasörünü `final_models` dizini altına yerleştirin.

### 2) Bağımlılıkları Yükleme
```bash
pip install -r requirements.txt

### 3) Uygulamayı Başlatma
```bash
python app.py

Uygulama başladığında terminalde oluşan linke (örn: http://127.0.0.1:7860) tıklayarak arayüze erişebilirsiniz.

🐳 Docker ile Canlıya Alma (Opsiyonel)
Proje, internet tasarrufu sağlayan Layer Caching teknolojisiyle Dockerize edilmiştir:
```bash
docker-compose up -d --build

👨‍💻 Geliştirici
Ömer Özcan – AKÜ Bilgisayar Mühendisliği Bu proje Veri Madenciliği dersi final ödevi kapsamında hazırlanmıştır.