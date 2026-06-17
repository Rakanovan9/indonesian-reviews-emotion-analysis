# 📌 Analisis Emosi Ulasan Twitter/X di Play Store
> **Transisi dari Klasifikasi Sentimen ke Klasifikasi Emosi 5 Kelas**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Tradisional-orange.svg)](https://scikit-learn.org/)
[![NLP](https://img.shields.io/badge/NLP-Bahasa%20Indonesia-green.svg)](https://github.com/)

Proyek ini adalah sistem klasifikasi teks (NLP) untuk mengelompokkan ulasan pengguna aplikasi Twitter/X di Google Play Store ke dalam 5 emosi: **Joy (Senang), Anger (Marah), Sadness (Sedih), Fear (Takut), dan Disgust (Muak)**.

Proyek ini merupakan **pengembangan (evolusi)** dari analisis sentimen sederhana (Positif/Negatif/Netral) ke deteksi emosi yang lebih detail dan spesifik.

---

## 🔄 Evolusi Proyek

Perubahan dari analisis sentimen ke analisis emosi meningkatkan pemahaman detail keluhan pengguna.

### Perbandingan Sentimen vs. Emosi

| Aspek | Analisis Sentimen (Lama) | Analisis Emosi (Baru) |
| :--- | :--- | :--- |
| **Kelas Label** | 3 Kelas (Positif, Negatif, Netral) | 5 Kelas (Joy, Anger, Sadness, Fear, Disgust) |
| **Tingkat Kesulitan** | Rendah (Hanya polaritas baik/buruk) | Tinggi (Banyak kata emosi yang mirip/tumpang tindih) |
| **Hasil Output** | Arah sentimen umum | Nuansa emosi yang spesifik |
| **Model** | Klasifikasi sederhana | Optimasi parameter dengan GridSearch dan Word2Vec |

### Perubahan Struktur Sistem
* **Yang Digunakan Kembali:** Pembersihan teks dasar, kamus slang bahasa Indonesia, dan data mentah.
* **Yang Diubah:** Jalur preprocessing dipisah (TF-IDF menggunakan stemming Sastrawi, Word2Vec tanpa stemming). Pembagian data diperketat menggunakan *Stratified Train/Val/Test (70:15:15)*. Metrik evaluasi diganti ke *Macro F1-score* karena data tidak seimbang.
* **Yang Ditambahkan:** Pelabelan otomatis menggunakan leksikon emosi dengan aturan prioritas (`Anger > Sadness > Disgust > Fear > Joy`). Pelatihan Word2Vec mandiri, visualisasi t-SNE, grafik performa model, matriks konfusi, dan fitur prediksi kalimat baru secara langsung.

---

## 📊 Data Proyek

Ulasan Play Store Indonesia untuk aplikasi Twitter/X:
* **Total Data Mentah:** 49.070 ulasan.
* **Total Data Setelah Saring:** 44.217 ulasan (ulasan netral/faktual dibuang).
* **5 Kelas Emosi:** Joy (Senang), Anger (Marah), Sadness (Sedih), Fear (Takut), dan Disgust (Muak).

---

## ⚙️ Alur Pipeline

Setiap tahapan dikerjakan berurutan dalam 5 Jupyter Notebook:

1. **01_eda_and_filtering.ipynb (Pemuatan & Penyaringan):**
   * Menyaring ulasan netral agar model fokus pada ulasan yang mengandung emosi.
2. **02_preprocessing.ipynb (Labeling & Pra-pengolahan):**
   * Melabeli data otomatis secara objektif dengan leksikon, membagi data (70:15:15), dan membersihkan kata slang.
3. **03_feature_extraction.ipynb (Ekstraksi Fitur):**
   * Mengubah teks menjadi angka menggunakan TF-IDF dan representasi Word2Vec kustom (Skip-Gram).
4. **04_model_training.ipynb (Pelatihan Model):**
   * Mencari model dan parameter terbaik menggunakan GridSearchCV 5-Fold.
5. **05_error_analysis.ipynb (Analisis Eror & Uji Coba):**
   * Mengukur akurasi pada data uji, menggambar matriks konfusi, menganalisis ulasan yang salah prediksi, serta menyediakan fitur tes kalimat baru.

---

## 🤖 Model yang Digunakan

* **Machine Learning Tradisional:** Logistic Regression, LinearSVC, Complement Naive Bayes (CNB), dan Random Forest.
* **Kenapa Tidak Pakai Deep Learning (BERT/LSTM)?**
  1. Dataset (44 ribu baris) dinilai terlalu kecil untuk melatih Deep Learning tanpa risiko *overfitting*.
  2. Label data berasal dari aturan kamus (leksikon), sehingga Deep Learning hanya akan menghafal kamus tersebut tanpa belajar makna kalimat sebenarnya.
  3. Model tradisional lebih mudah dipahami dan dijelaskan bobot katanya (*interpretable*).

---

## 📈 Hasil Performa Model

Berikut adalah tabel performa model pada data uji (diurutkan berdasarkan nilai Macro F1):

| Model | Fitur | Val Macro F1 | Test Macro F1 | Akurasi Uji |
| :--- | :--- | :---: | :---: | :---: |
| **LinearSVC** | **TF-IDF** | **0.950** | **0.940** | **97.00%** |
| **Logistic Regression** | TF-IDF | 0.938 | 0.925 | 96.20% |
| **Complement Naive Bayes** | TF-IDF | 0.932 | 0.918 | 95.80% |
| **Random Forest** | TF-IDF | 0.895 | 0.880 | 93.40% |
| **LinearSVC** | Word2Vec (Weighted) | 0.865 | 0.852 | 91.20% |
| **LinearSVC** | Word2Vec (Average) | 0.840 | 0.825 | 89.80% |

* **Kesimpulan:** Model **LinearSVC dengan fitur TF-IDF** adalah yang terbaik dengan F1-Score **0.940** dan Akurasi **97.00%**. TF-IDF lebih unggul karena ulasan Play Store cenderung pendek dan langsung menggunakan kata emosi kunci (seperti *"kecewa"*, *"bagus"*) yang dibobotkan secara kuat oleh TF-IDF.

---

## 🗂️ Struktur Folder

```bash
emotion-analysis/
│
├── data/                    # Data mentah (raw) dan data siap pakai (processed)
├── notebooks/               # 5 berkas notebook analisis bertahap
├── src/                     # Kode python modular (pembantu)
├── models/                  # Berkas model terlatih (.joblib dan .model)
├── reports/figures/         # Hasil gambar visualisasi (t-SNE, confusion matrix, dll)
└── emotion_analysis_plan/   # Dokumen rancangan tugas
```

---

## ▶️ Cara Menjalankan

Jalankan perintah ini di terminal:

```bash
# 1. Install library yang dibutuhkan
pip install -r requirements.txt

# 2. Jalankan notebook secara berurutan (01 s.d 05) di Jupyter Anda
```

---

## 💡 Temuan Penting

* **Ambiguitas Emosi:** Emosi lebih sulit diprediksi daripada sentimen karena batas antar emosi tipis (misal keluhan *"uninstall"* bisa masuk kategori Sadness atau Disgust).
* **Ketidakseimbangan Data:** Emosi *Fear* dan *Joy* mendominasi ulasan, sedangkan *Anger* sangat sedikit. Penggunaan metrik Macro F1 memastikan model tetap adil dan akurat pada kelas minoritas.

---

## 🔮 Rencana Lanjutan

1. **Anotasi Manual:** Memperbaiki label otomatis dengan penilaian manusia agar lebih akurat.
2. **Fine-Tuning IndoBERT:** Mencoba pemodelan Transformer bahasa Indonesia di masa mendatang.
3. **Interpretabilitas (SHAP):** Menambahkan visualisasi penjelasan keputusan model.

---

## 🧾 Catatan
* Proyek ini dibuat sebagai tugas mata kuliah **Text Mining / Natural Language Processing** Semester 6.
