FROM python:3.12-slim

WORKDIR /app

# --- KRİTİK KATMAN ---
# Önce sadece requirements kopyalanır
COPY requirements.txt .

# requirements.txt değişmedikçe bu katman cached (önbellekte) kalır, internet harcamaz!
RUN pip install --no-cache-dir -r requirements.txt
# ---------------------

# En son kodları kopyalıyoruz
COPY . .

# Modelleri volume ile bağlayacağımız için Dockerfile içinde kopyalamaya gerek yok
# Ama yerel testlerde hata almamak için klasörün var olduğundan emin olalım
RUN mkdir -p /app/final_models

CMD ["python", "app.py"]