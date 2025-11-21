# ChildGuard – Age-Aware Harmful Content Classifier  
### BERT + TF‑IDF Logistic Regression Projesi

Bu proje, çevrim içi metinlerde **çocukları hedef alan zararlı içerikleri tespit etmek** için yaş gruplarına duyarlı (Age‑Aware) bir sınıflandırma modeli geliştirir.  
Model iki ana yaklaşımı birleştirir:

- **BERT (Age‑Aware Fine-Tuning / Her yaş grubuna özel model)**
- **Klasik Model: TF‑IDF + One‑Hot + Logistic Regression**

---

## 📌 Proje İçeriği

### ✔ 1) Dataset  
- Kullanılan veri seti: **ChildGuard.csv**  
- Kolonlar:
  - `text`: Ham metin
  - `actual_class`: Gerçek sınıf (0 = Zararsız, 1 = Çocuk hedefli zararlı)
  - `Age_Group`: İçeriğin yöneldiği yaş grubu

### ✔ 2) Veri Temizleme  
Kod içerisinde şu işlemler yapılmaktadır:

- URL silme  
- Mention temizleme (@user)  
- Emoji/sembol temizleme  
- Küçük harfe dönüştürme  
- Fazla boşlukları silme  

Bu süreç veri madenciliği dersinin **ön işleme (preprocessing)** bölümüne referans olarak yorum satırlarında belirtilmiştir.

---

## 📌 Model Yapısı

### 🟦 **A) Age-Aware BERT Modelleri**
Her yaş grubu için **ayrı bir BERT modeli** eğitilir.

Yaş grupları:
- *Teen (13–17)*
- *Pre‑Teen (11–13)*
- *Younger (Under 11)*

Her modelde:
- BERT tokenizer  
- Fine‑tuning  
- Class Weight dengelemesi  
- Train / Validation / Test bölme  
- Threshold Tuning yapılır  

Bu bölüm veri madenciliği dersinin:
- **Nitelik seçimi**
- **Sınıflandırma**
- **Derin öğrenme temelli modeller**
- **Model değerlendirme (precision, recall, F1)**  
konularını uygulamalı olarak içerir.

---

### 🟨 **B) Klasik Model: TF‑IDF + One‑Hot + Logistic Regression**

Ekstra özellikler:
- Metin uzunluğu
- Kelime sayısı
- AgeGroup One‑Hot Encoding

TF‑IDF → Sparse Matrix  
One‑Hot → Sparse Matrix  
Hepsi birleştirilerek Logistic Regression’a verilir.

Bu bölüm veri madenciliği dersindeki:
- **TF‑IDF**
- **Öznitelik mühendisliği**
- **One‑Hot Encoding**
- **Lojistik regresyon**
- **Dengesiz sınıf problemlerinde class_weight kullanımı**  
konularını referans alır.

---

## 📌 Sonuçlar

| Model | Accuracy | F1 | Açıklama |
|------|---------|------|----------|
| **BERT (Age‑Aware)** | Çok yüksek | Çoğu >0.80 | Yaş grubuna duyarlı en iyi model |
| **TF‑IDF Logistic Regression** | 0.78 | Dengeli | Ekstra klasik yöntem karşılaştırması |

---

## 🚀 Projenin Çalıştırılması

### 1) Gereksinimler
```bash
pip install transformers tensorflow datasets sklearn scipy
```

### 2) Dosyayı çalıştır
```bash
python childguard_main.py
```

Modeller eğitim sonunda sonuçları terminale yazacaktır.

---

## 📌 Kod Dosyası İçindeki Yorumlar

Kod içinde tüm adımlar **hocanın veri madenciliği dersinde işlediği konulara uygun şekilde açıklamalı yorum satırlarıyla** anlatılmıştır:

- Preprocessing açıklamaları  
- TF‑IDF sürecinin eğitimdeki karşılığı  
- Logistic Regression’ın avantajları  
- BERT fine‑tuning mantığı  
- Threshold tuning  
- Veri madenciliği 2–3–4 ipynb dosyalarındaki içeriklere referanslar  

---

## 📁 Dosya Yapısı

```
├── ChildGuard.csv
├── childguard_main.py
└── README.md
```

---

## 👨‍💻 Geliştirici
**Ömer Özcan – AKÜ Bilgisayar Mühendisliği**  
Bu proje Veri Madenciliği dersi kapsamında hazırlanmıştır.

---

## 📜 Lisans
Bu proje akademik kullanım içindir.
