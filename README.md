# 🛡️ ChildGuard AI – Hibrit Zararlı İçerik Sınıflandırıcı

### BERT + TF‑IDF Logistic Regression Hybrid Classifier

ChildGuard AI, çevrim içi metinlerde **çocukları hedef alan zararlı içeriklerin** tespiti için geliştirilmiş **hibrit bir metin sınıflandırma sistemidir**. Proje; **Derin Öğrenme (BERT)** ile **Klasik Makine Öğrenmesi (TF‑IDF + Logistic Regression)** yaklaşımlarını birleştirerek daha dengeli, açıklanabilir ve yüksek doğruluklu sonuçlar üretir.

Uygulama, önceden eğitilmiş ve serileştirilmiş modelleri (`.pkl`, `save_pretrained`) yükleyerek **Gradio** tabanlı bir web arayüzü üzerinden **anlık analiz** sunar.

---

## 🚀 Hibrit Mimari (Yeni Nesil Yaklaşım)

Sistem, iki farklı modelin çıktısını **ağırlıklı karar mekanizması** ile birleştirir:

* **BERT (Transformers)**
  Metnin bağlamsal (contextual) ve anlamsal yapısını analiz eder.
  **Ağırlık:** %60

* **Logistic Regression (Feature‑Engineered)**
  TF‑IDF vektörlerine ek olarak metinsel ve demografik öznitelikler ile istatistiksel analiz yapar.
  **Ağırlık:** %40

Bu yaklaşım, yalnızca derin öğrenmeye bağımlı kalmadan **genellenebilirlik** ve **kararlılık** sağlar.

---

## 📌 Teknik Detaylar

### 1️⃣ Model Serileştirme (Serialization)

* **Joblib / Pickle**
  Logistic Regression modeli ve TF‑IDF vektörleştirici `.pkl` formatında kaydedilmiştir. Böylece her çalıştırmada yeniden eğitim gerekmez.

* **HuggingFace – save_pretrained**
  BERT modelleri ve tokenizer’lar yerel dizinden hızlı yükleme için optimize edilmiştir.

---

### 2️⃣ Öznitelik Mühendisliği (Feature Engineering)

Klasik modelin performansını artırmak amacıyla veri madenciliği prensiplerine uygun şekilde aşağıdaki öznitelikler kullanılmıştır:

* **TF‑IDF Vektörleri**

  * Maksimum 5000 boyut
  * Metin temelli ağırlıklı özellik çıkarımı

* **Sayısal Meta Veriler**

  * Metin uzunluğu (`text_len`)
  * Kelime sayısı (`word_cnt`)

* **One‑Hot Encoding**

  * Hedef yaş grubu bilgisi

Bu yapı, özellikle kısa ve belirsiz metinlerde klasik modelin katkısını artırır.

---

### 3️⃣ Hibrit Karar Mekanizması

Her iki modelden elde edilen olasılık skorları aşağıdaki formül ile birleştirilir:

```text
Final Score = (BERT_prob × 0.6) + (LR_prob × 0.4)
```

Bu skor, nihai sınıflandırma kararının temelini oluşturur.

---

## 📁 Proje Dosya Yapısı

```text
├── final_models/
│   ├── bert_pre-teen/                 # BERT Model (11–13 yaş)
│   ├── bert_teen/                     # BERT Model (13–17 yaş)
│   ├── bert_younger/                  # BERT Model (<11 yaş)
│   ├── logistic_regression_model.pkl  # Eğitilmiş LR modeli
│   └── tfidf_vectorizer.pkl           # Eğitilmiş TF‑IDF nesnesi
│
├── app.py                             # Gradio Web Arayüzü
├── childguardhybrid.py                # Eğitim ve test kodları
├── docker-compose.yml                 # Docker servis konfigürasyonu
├── Dockerfile                         # Docker imaj tanımı
├── requirements.txt                   # Python bağımlılıkları
└── README.md                          # Proje dokümantasyonu
```

---

## ⚙️ Kurulum ve Çalıştırma

### 1️⃣ Modelleri Hazırlama

Eğitilmiş `.pkl` dosyalarını ve BERT model klasörlerini aşağıdaki dizine yerleştirin:

```text
final_models/
```

---

### 2️⃣ Bağımlılıkları Yükleme

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Uygulamayı Başlatma

```bash
python app.py
```

Uygulama çalıştıktan sonra terminalde verilen bağlantı üzerinden arayüze erişebilirsiniz:

```text
http://127.0.0.1:7860
```

---

## 🐳 Docker ile Çalıştırma (Opsiyonel)

Proje, **Docker layer caching** desteğiyle optimize edilmiştir.

```bash
docker-compose up -d --build
```

Bu yöntem, bağımlılıkların tekrar indirilmesini önleyerek hızlı dağıtım sağlar.

---

## 🎯 Kullanım Senaryoları

* Sosyal medya içerik denetimi
* Eğitim platformlarında içerik filtreleme
* Çocuklara yönelik dijital güvenlik sistemleri
* Akademik veri madenciliği ve NLP çalışmaları

---

## 👨‍💻 Geliştirici

**Ömer Özcan**
Afyon Kocatepe Üniversitesi – Bilgisayar Mühendisliği

📌 Bu proje, **Veri Madenciliği dersi final ödevi** kapsamında geliştirilmiştir.

---

> 🛡️ *ChildGuard AI, çocukların dijital dünyada daha güvenli bir ortamda bulunabilmesi için tasarlanmıştır.*