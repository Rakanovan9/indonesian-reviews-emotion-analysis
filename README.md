# Perbandingan Algoritma Machine Learning untuk Klasifikasi Emosi pada Ulasan Aplikasi X di Google Play Store
> **Sistem Klasifikasi Teks Multi-Kelas untuk Mendeteksi 5 Kategori Emosi pada Ulasan Pengguna Aplikasi X**

---

## 1. Overview

### Apa itu Analisis Emosi?
Berbeda dari analisis sentimen biasa yang hanya membagi teks menjadi Positif/Negatif, **Analisis Emosi** memetakan teks ke kategori emosi yang lebih spesifik. Proyek ini mengelompokkan ulasan pengguna Twitter/X di Play Store ke dalam 5 kelas emosi dasar: **Joy (Senang), Anger (Marah), Sadness (Sedih), Fear (Takut), dan Disgust (Muak)**.

### Masalah yang Diselesaikan
Membantu pengembang menyortir ulasan secara otomatis untuk merespon keluhan secara tepat sasaran. Ulasan berkategori *Anger* (kemarahan tentang bug) akan diarahkan langsung ke tim teknis, sedangkan ulasan berkategori *Disgust* (kemaklan tentang iklan/spam) akan diarahkan ke tim produk/kebijakan.

---

## 2. Objectives

* Membangun pipeline NLP modular dari pembersihan data mentah hingga evaluasi akhir model.
* Membandingkan performa representasi fitur berbasis frekuensi kata (**TF-IDF unigram & bigram**) dengan model semantik kontinu (**Word2Vec Skip-Gram**).
* Melatih dan mengoptimasi berbagai algoritma Machine Learning tradisional untuk menemukan model klasifikasi terbaik.
* Membuat fitur inferensi interaktif untuk menguji prediksi kalimat ulasan baru secara langsung.

---

## 3. Features

* **Algoritma Klasifikasi:** Logistic Regression, LinearSVC (SVM), Complement Naive Bayes (CNB), dan Random Forest.
* **Ekstraksi Fitur:** TF-IDF (Unigram + Bigram) dan Word2Vec Skip-Gram (Simple Average & TF-IDF Weighted Average).
* **Pra-pengolahan Dual-Jalur:** Pembersihan slang teks secara menyeluruh, serta pembagian jalur khusus (menggunakan stemming untuk TF-IDF dan tanpa stemming untuk Word2Vec agar context semantik terjaga).
* **Uji Signifikansi Statistik:** Menggunakan Uji McNemar untuk memvalidasi perbedaan performa model terbaik secara ilmiah.
* **Desain Eksperimen Bersih:** Pembagian data Stratified Train/Val/Test (70:15:15) dan pelatihan Word2Vec terisolasi guna menghindari kebocoran data (*data leakage*).

---

## 4. Project Structure

```bash
emotion-analysis/
│
├── data/
│   ├── raw/                  # Dataset ulasan mentah hasil scraping (.csv)
│   └── processed/            # File hasil filter emosi, pelabelan, dan pembagian split data (.csv)
│
├── notebooks/                # Jupyter Notebook alur kerja berurutan dari 01 s.d 05
│   ├── 01_eda_and_filtering.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_extraction.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_error_analysis.ipynb
│
├── src/                      # Berkas python modular pembantu
│   ├── data_utils.py         # Logika penyaringan ulasan non-emosional
│   ├── preprocessing.py      # Dual preprocessing pipeline (stemming vs no-stemming)
│   ├── modeling.py           # Pembuatan model GridSearchCV & fungsi evaluasi
│   └── evaluation.py         # Kode visualisasi confusion matrix & matriks eror
│
├── models/                   # Berkas model dan representasi vektor yang terlatih (.joblib & .model)
└── reports/figures/          # Seluruh gambar grafik ekspor visualisasi performa (.png)
```

---

## 5. Installation

Pastikan Anda menggunakan Python versi **3.12** atau di atasnya.

```bash
# 1. Clone repositori proyek
git clone https://github.com/Rakanovan9/indonesian-reviews-emotion-analysis.git
cd indonesian-reviews-emotion-analysis

# 2. Buat virtual environment baru
python -m venv venv

# 3. Aktifkan virtual environment
# Untuk Windows:
.\venv\Scripts\activate
# Untuk macOS/Linux:
source venv/bin/activate

# 4. Install dependensi
pip install -r requirements.txt
```

---

## 6. How to Run

Jalankan berkas Jupyter Notebook di folder `notebooks/` secara berurutan:
1. **`01_eda_and_filtering.ipynb`**: Menyaring ulasan netral/faktual yang tidak beremosi.
2. **`02_preprocessing.ipynb`**: Melabeli ulasan otomatis menggunakan leksikon dan membagi data secara stratified.
3. **`03_feature_extraction.ipynb`**: Mengekstrak matriks fitur TF-IDF dan melatih Word2Vec Skip-Gram pada data latih.
4. **`04_model_training.ipynb`**: Mencari hyperparameter terbaik via GridSearchCV 5-Fold.
5. **`05_error_analysis.ipynb`**: Mengevaluasi model akhir pada set uji, menggambar matriks konfusi, dan mencoba prediksi kalimat baru.

*Opsi eksekusi otomatis lewat terminal tanpa GUI:*
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

---

## 7. Methodology

1. **Data Collection & Filtering:** Menyaring 49.070 ulasan mentah menjadi 44.217 ulasan bernilai emosi tinggi dengan membuang ulasan netral.
2. **Preprocessing:** Membersihkan noise, menormalkan kata slang, dan menerapkan stemming khusus untuk representasi TF-IDF guna menekan dimensi fitur yang sparse.
3. **Feature Extraction:** Merepresentasikan dokumen ulasan ke bentuk angka menggunakan TF-IDF unigram+bigram dan rata-rata vektor kata Word2Vec Skip-Gram.
4. **Modeling:** Melatih 4 algoritma ML tradisional dengan optimasi parameter GridSearch pada set data latih.
5. **Evaluation:** Mengukur performa pada set validasi untuk memilih model terbaik, lalu mengujinya secara final pada data uji menggunakan Macro F1-Score.

---

## 8. Models Used

### A. Machine Learning Tradisional
* **LinearSVC (SVM):** Model linier tangguh yang unggul dalam menentukan hyperplane pemisah pada dimensi fitur sparse yang sangat tinggi (TF-IDF).
* **Logistic Regression:** Classifier probabilistik yang cepat, stabil, dan menjadi pembanding utama.
* **Complement Naive Bayes:** Varian Naive Bayes yang dioptimalkan untuk meminimalkan bias prediksi pada data kelas tidak seimbang.
* **Random Forest:** Ensemble pohon keputusan untuk menangkap pola non-linier, meskipun cenderung lambat pada matriks TF-IDF.

### B. Deep Learning (Desain Arsitektur)
* **IndoBERT (Fine-Tuning):** Model berbasis arsitektur Transformer yang memanfaatkan mekanisme *self-attention* dua arah untuk memahami konteks semantik kalimat yang lebih kompleks (seperti sarkasme atau negasi jarak jauh). Pemasangan model ditangguhkan pada eksperimen ini untuk menghindari risiko overfitting akibat sirkularitas pelabelan kamus leksikon otomatis.

---

## 9. Results & Evaluation

Berikut adalah tabel performa ke-10 model yang diuji (diurutkan dari skor Macro F1 tertinggi pada set data uji):

| Model | Fitur | Val Macro F1 | Test Macro F1 | Akurasi Uji |
| :--- | :--- | :---: | :---: | :---: |
| **LinearSVC** | **TF-IDF** | **0.941** | **0.941** | **96.75%** |
| Logistic Regression | TF-IDF | 0.928 | 0.933 | 96.09% |
| Random Forest | TF-IDF | 0.851 | 0.847 | 89.37% |
| Complement Naive Bayes | TF-IDF | 0.822 | 0.834 | 89.59% |
| LinearSVC | Word2Vec (Average) | 0.790 | 0.799 | 87.07% |
| Logistic Regression | Word2Vec (Average) | 0.775 | 0.786 | 85.79% |
| LinearSVC | Word2Vec (Weighted) | 0.715 | 0.720 | 80.90% |
| Logistic Regression | Word2Vec (Weighted) | 0.700 | 0.706 | 78.93% |
| Random Forest | Word2Vec (Average) | 0.695 | 0.698 | 79.22% |
| Random Forest | Word2Vec (Weighted) | 0.649 | 0.652 | 75.13% |

* **Model Terbaik:** **LinearSVC dengan fitur TF-IDF** memenangkan eksperimen secara mutlak dengan skor F1 uji **0.941** dan Akurasi Uji **96.75%**.

---

## 10. Visualizations

Semua visualisasi tersimpan di folder `reports/figures/`:

### Sebaran Rating & Panjang Karakter
<img src="reports/figures/raw_rating_distribution.png" width="350" alt="Sebaran Rating Mentah" />
<img src="reports/figures/char_length_comparison.png" width="400" alt="Perbandingan Panjang Karakter" />
* Menunjukkan bahwa ulasan dengan rating ekstrem (1 dan 5) mendominasi, dan panjang ulasan berkurang drastis setelah disaring untuk meminimalisasi noise.

### Distribusi Kelas & Word Clouds
<img src="reports/figures/class_distribution.png" width="350" alt="Distribusi Kelas Emosi" />
<img src="reports/figures/wordclouds.png" width="550" alt="Word Clouds Emosi" />
* Menunjukkan ketidakseimbangan jumlah sampel emosi (kelas *Fear* mendominasi, *Anger* paling sedikit), serta memvisualisasikan kata kunci unik per kelas.

### Proyeksi Semantik t-SNE & Perbandingan Model
<img src="reports/figures/word2vec_tsne.png" width="380" alt="Visualisasi t-SNE" />
<img src="reports/figures/model_comparison_chart.png" width="480" alt="Grafik Perbandingan Performa Model" />
* Memvisualisasikan kluster kata-kata beremosi sejenis di ruang Word2Vec, serta membandingkan performa Macro F1-score seluruh model pada set validasi secara grafis.

### Confusion Matrix & Error Transition Matrix
<img src="reports/figures/normalized_confusion_matrix.png" width="380" alt="Normalized Confusion Matrix" />
<img src="reports/figures/error_transition_matrix.png" width="380" alt="Error Transition Matrix" />
* Menampilkan recall per-kelas model terbaik yang tinggi (89%-99%), serta memetakan arah salah prediksi terbesar yang dominan terjadi di antara kelas *Anger* dan *Disgust*.

---

## 11. Example Predictions

Berikut adalah hasil uji coba inferensi langsung model terbaik (**LinearSVC + TF-IDF**) terhadap beberapa kalimat ulasan kustom baru:

| No. | Ulasan Masukan (Input) | Hasil Preprocessing | Prediksi Emosi | Distribusi Probabilitas |
| :---: | :--- | :--- | :---: | :--- |
| 1 | "aplikasinya jelek banget, tiap buka loading terus sering force close bikin emosi!" | "aplikasi jelek banget tiap buka loading terus sering force close bikin emosi" | ![Anger](https://img.shields.io/badge/ANGER-red?style=flat-square) | **Anger: 98.43%**<br>Disgust: 0.40%<br>Fear: 0.21%<br>Joy: 0.56%<br>Sadness: 0.40% |
| 2 | "wah gokil sih fitur barunya keren banget memudahkan saya bersosialisasi makasih ya dev!" | "wah gokil fitur baru keren banget mudah sosial makasih ya dev" | ![Joy](https://img.shields.io/badge/JOY-brightgreen?style=flat-square) | Anger: 6.67%<br>Disgust: 8.47%<br>Fear: 13.24%<br>**Joy: 65.65%**<br>Sadness: 5.98% |
| 3 | "khawatir banget sama kebocoran data privasi apalagi ada berita akun di hack orang lain" | "khawatir banget sama bocor data privasi apalagi ada berita akun hack orang lain" | ![Fear](https://img.shields.io/badge/FEAR-orange?style=flat-square) | Anger: 4.81%<br>Disgust: 5.34%<br>**Fear: 80.40%**<br>Joy: 5.46%<br>Sadness: 4.00% |
| 4 | "sedih banget melihat akun saya tidak bisa dipulihkan, padahal banyak data penting di sana." | "sedih banget lihat akun tidak bisa pulih padahal banyak data penting sana" | ![Sadness](https://img.shields.io/badge/SADNESS-blue?style=flat-square) | Anger: 7.77%<br>Disgust: 6.06%<br>Fear: 9.13%<br>Joy: 6.06%<br>**Sadness: 70.99%** |
| 5 | "aplikasinya ampas, isinya cuma iklan, bot, sama spam yang mengganggu banget. uninstall aja lah." | "aplikasi ampas isi cuma iklan bot sama spam ganggu banget copot aja" | ![Disgust](https://img.shields.io/badge/DISGUST-yellowgreen?style=flat-square) | Anger: 21.22%<br>**Disgust: 76.11%**<br>Fear: 0.23%<br>Joy: 1.24%<br>Sadness: 1.20% |
| 6 | "woy respon dong admin! ini login gagal terus padahal koneksi internet lancar jaya!" | "woy respon dong admin login gagal terus padahal koneksi internet lancar jaya" | ![Anger](https://img.shields.io/badge/ANGER-red?style=flat-square) | **Anger: 89.57%**<br>Disgust: 2.39%<br>Fear: 4.24%<br>Joy: 2.65%<br>Sadness: 1.14% |
| 7 | "terima kasih banyak dev, update kali ini keren abis, aplikasinya jadi lancar dan responsif sekali!" | "terima kasih banyak dev update kali keren abis aplikasi jadi lancar responsif sekali" | ![Joy](https://img.shields.io/badge/JOY-brightgreen?style=flat-square) | Anger: 20.36%<br>Disgust: 9.67%<br>Fear: 11.61%<br>**Joy: 32.08%**<br>Sadness: 26.28% |
| 8 | "was-was banget kalau mau masukin nomor hp, takut disalahgunakan buat penipuan." | "was-was banget kalau mau masukin nomor hp takut disalahgunakan buat tipu" | ![Fear](https://img.shields.io/badge/FEAR-orange?style=flat-square) | Anger: 24.84%<br>Disgust: 14.61%<br>**Fear: 37.58%**<br>Joy: 12.87%<br>Sadness: 10.11% |

---

## 12. Key Findings

* **TF-IDF Mengalahkan Word2Vec:** Ulasan Play Store yang sangat pendek (langsung menyebut kata kunci emosi) membuat pembobotan IDF sangat sensitif dan diskriminatif bagi model linier.
* **Kebocoran Data Word2Vec Dibersihkan:** Menghilangkan bias dengan melatih Skip-Gram eksklusif pada data latih menurunkan F1-score Word2Vec ke performa riilnya (79.90%), membuktikan pentingnya integritas data split.
* **Uji Signifikansi Kuat:** Uji McNemar membuktikan keunggulan LinearSVC dibanding Logistic Regression valid secara ilmiah (p-value 0.000000).
* **Kelas Tersulit (Anger vs. Disgust):** Tumpang tindih leksikon kata keluhan teknis (seperti *"aplikasi sampah"*) membuat kedua emosi negatif ini sering kali membingungkan model.

---

## 13. Limitations

* **Sirkularitas Pelabelan Kamus:** Karena dataset dilabeli otomatis oleh kamus leksikon, akurasi tinggi (97%) mencerminkan performa model meniru aturan leksikon, bukan kepekaan emosi kognitif manusia yang alami.
* **Bahasa Slang Dinamis:** Kamus normalisasi slang bersifat statis, berisiko tidak mampu menangani istilah slang internet terbaru di masa mendatang.

---

## 14. Future Work

* **Anotasi Data oleh Manusia:** Mengganti pelabelan kamus otomatis dengan anotasi manual panel ahli bahasa untuk menghilangkan bias leksikon.
* **Eksperimen IndoBERT:** Menguji model Transformer bahasa Indonesia guna menangkap konteks negasi jarak jauh dan mendeteksi sarkasme teks ulasan secara implisit.
* **Augmentasi Data:** Menambahkan variasi sampel teks pada kelas minoritas (*Anger*) menggunakan teknik sinonim berbasis ruang Word2Vec.

---

## 15. Conclusion

Sistem klasifikasi teks multi-kelas analisis emosi berhasil diimplementasikan secara modular. Model **LinearSVC dengan fitur TF-IDF unigram+bigram** terbukti memberikan performa terbaik dengan skor Macro F1 **0.941** dan akurasi uji **96.75%**. Pengurangan dimensionalitas Word2Vec terbukti kurang optimal karena mereduksi intensitas kata emosi unik pada teks ulasan pendek.
