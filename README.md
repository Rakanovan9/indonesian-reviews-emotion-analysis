# Analisis Emosi Multi-Kelas pada Ulasan Aplikasi Twitter/X di Google Play Store
> **Studi Komparatif Ekstraksi Fitur TF-IDF vs. Word2Vec Menggunakan Algoritma Machine Learning Tradisional dan Desain Arsitektur Fine-Tuning Transformer**

---

## 1. Project Overview (Deskripsi Proyek)

### Pendahuluan
Ulasan pengguna di platform digital seperti Google Play Store merupakan cerminan langsung dari kepuasan, keluhan, dan harapan mereka terhadap sebuah aplikasi. Twitter/X, sebagai salah satu platform sosial media terbesar, sering kali menerima ulasan dengan polaritas emosi yang sangat kontras dan intens. Proyek ini memodelkan klasifikasi emosi multi-kelas (5 kelas) menggunakan pendekatan pemrosesan bahasa alami (NLP) berbasis Machine Learning tradisional serta membandingkannya dengan analisis teoritis arsitektur Deep Learning (Transformer).

### Definisi Analisis Emosi
Analisis emosi (*Emotion Analysis*) berbeda dengan analisis sentimen konvensional. Sentimen memetakan teks ke dalam polaritas linear sederhana (**Positif, Negatif, Netral**). Sementara itu, Analisis Emosi mendeteksi keadaan kognitif spesifik pengguna yang diwakili oleh kategori emosi dasar psikologis: **Joy (Senang), Anger (Marah), Sadness (Sedih), Fear (Takut), dan Disgust (Muak)**.

### Sasaran & Objektif
1. Membangun pipeline NLP modular dari scraping, eksplorasi data (EDA), pra-pengolahan bercabang, ekstraksi fitur, pemodelan, hingga analisis kesalahan linguistik.
2. Membandingkan secara empiris efektivitas fitur leksikal berbasis frekuensi (**TF-IDF unigram & bigram**) terhadap representasi semantik kontinu terdistribusi (**Word2Vec Skip-Gram kustom**).
3. Melakukan pengujian statistik untuk membuktikan validitas keunggulan model terbaik secara ilmiah.
4. Menyediakan antarmuka inferensi interaktif untuk menguji performa model secara langsung pada ulasan kustom baru.

---

## 2. Background & Problem Statement (Latar Belakang & Masalah)

### Pentingnya Analisis Emosi vs. Analisis Sentimen
Analisis sentimen tradisional terlalu menyederhanakan feedback pengguna. Sebagai contoh, dua ulasan negatif berikut memiliki muatan tindakan yang berbeda bagi pengembang aplikasi:
* *Ulasan A:* "Aplikasinya jelek banget, tiap buka loading terus sering force close bikin emosi!" -> **Anger (Marah)** (membutuhkan perbaikan bug performa segera).
* *Ulasan B:* "Isinya cuma iklan, bot, sama spam yang mengganggu banget. Uninstall aja lah." -> **Disgust (Muak)** (membutuhkan kebijakan moderasi konten dan peninjauan iklan).

Dengan mendeteksi emosi spesifik, tim pengembang dapat melakukan triase keluhan secara cerdas berdasarkan kategori urgensi masalah kognitif pengguna.

### Tantangan Klasifikasi Emosi Multi-Kelas
1. **Ambiguitas Semantik:** Batas linguistik antar emosi negatif sangat tipis. Kata-kata kasar atau informal sering kali merepresentasikan kemarahan (*Anger*) sekaligus kemuakan (*Disgust*) secara bersamaan.
2. **Keterbatasan Konteks:** Ulasan aplikasi Play Store cenderung sangat pendek dan tidak menggunakan tata bahasa formal. Kata-kata slang, ejaan tidak baku, penulisan sarkasme, dan negasi implisit meningkatkan kesulitan klasifikasi.
3. **Ketidakseimbangan Kelas (Class Imbalance):** Distribusi emosi di dunia nyata sangat tidak seimbang; ulasan emosional didominasi oleh kekhawatiran akun ditangguhkan (*Fear*) atau kepuasan umum (*Joy*), sementara kelas kemarahan (*Anger*) memiliki volume sampel yang jauh lebih sedikit.

---

## 3. Objectives (Tujuan Penelitian)

1. **Penyusunan Pipeline Modular:** Membangun struktur kode python modular di folder `src/` yang memisahkan fungsi pembersihan data, pra-pengolahan dual-jalur, pemodelan, dan evaluasi.
2. **Hardening Kebocoran Data (Data Leakage):** Memastikan proses pelatihan Word2Vec Skip-Gram terisolasi secara induktif hanya pada data latih (*train split*) guna menjaga integritas pengujian ilmiah.
3. **Studi Komparatif Ekstraksi Fitur:** Menganalisis kelebihan dan kelemahan pembobotan TF-IDF unigram+bigram terhadap model representasi vektor kata Word2Vec (Simple Average vs. TF-IDF Weighted Average).
4. **Verifikasi Signifikansi Statistik:** Melakukan pengujian hipotesis McNemar untuk membuktikan superioritas model terbaik secara matematis di atas variasi acak pembagian data.

---

## 4. Dataset Description (Deskripsi Dataset)

* **Sumber Data:** Ulasan aplikasi Twitter/X (`com.twitter.android`) di Google Play Store Indonesia.
* **Volume Data Mentah:** 49.070 ulasan.
* **Volume Data Setelah Saring:** 44.217 ulasan (ulasan faktual/netral dibuang melalui filter rating ekstrim dan pencocokan kata emosi leksikon).
* **Bahasa:** Bahasa Indonesia (informal, kaya kata slang, singkatan, dan ejaan tidak baku).
* **Kategori Kelas Emosi:**
  * **Joy:** Ulasan kepuasan pengguna, pujian fitur baru, atau ucapan terima kasih.
  * **Anger:** Keluhan kegagalan sistem, aplikasi lambat, atau respon lambat dari layanan pelanggan.
  * **Sadness:** Kekecewaan atas hilangnya fitur lama, akun yang diblokir, atau hilangnya data historis.
  * **Fear:** Kekhawatiran mengenai kebocoran data privasi, peretasan akun, atau keamanan data pribadi.
  * **Disgust:** Kemuakan terhadap kehadiran iklan yang masif, spamming bot akun, atau antarmuka aplikasi yang buruk.

---

## 5. End-to-End Pipeline Methodology (Metodologi Sistem)

```
[Raw Data] ──> [01. EDA & Saring] ──> [02. Labeling & Split] ──┬──> [Slang + Stemming] ──> [TF-IDF] ────┬──> [GridSearch] ──> [Best Model]
                                                               └──> [Slang Only] ───────> [Word2Vec] ──┘
```

1. **Scraping & Pengumpulan Data:** Menggunakan ulasan Google Play Store.
   * *Alternatif:* Menggunakan dataset publik Indonesia seperti EmoT.
   * *Justifikasi Pilihan:* Menggunakan data ulasan store riil untuk menangkap bahasa slang spesifik ekosistem aplikasi mobile yang tidak terdapat di dataset Twitter umum.
2. **Exploratory Data Analysis (EDA):** Menganalisis sebaran rating asli, panjang karakter ulasan, dan visualisasi distribusi emosi awal.
3. **Pelabelan Otomatis (Lexicon Labeling):** Melabeli data latih secara objektif menggunakan kamus emosi hibrida bahasa Indonesia.
   * *Alternatif:* Anotasi manual oleh manusia.
   * *Justifikasi Penundaan:* Kecepatan pemrosesan untuk melabeli 44 ribu baris ulasan secara konsisten dalam waktu singkat. Dampak keterbatasan ini dibahas pada bagian Keterbatasan.
4. **Pra-pengolahan Dual-Jalur:** Memisahkan pra-pengolahan untuk TF-IDF (menggunakan stemming) dan Word2Vec (tanpa stemming). Detail di Bagian 6.
5. **Ekstraksi Fitur:** Membangun fitur TF-IDF unigram+bigram dan Word2Vec Skip-Gram.
6. **Pemodelan & GridSearchCV:** Menjalankan GridSearch 5-Fold Stratified CV pada Logistic Regression, LinearSVC, Complement Naive Bayes, dan Random Forest.
7. **Analisis Kesalahan & Signifikansi:** Menggunakan Uji McNemar dan Matriks Transisi Kesalahan untuk membedah perilaku klasifikasi model terbaik secara mendalam.

---

## 6. Preprocessing (Pra-pengolahan Teks)

Proyek ini menerapkan pra-pengolahan bercabang (*branched preprocessing*) berdasarkan jenis representasi fitur yang digunakan:

### Langkah Pembersihan Umum
* **Case Folding:** Mengubah seluruh karakter menjadi huruf kecil untuk menghilangkan sensitivitas kapitalisasi (misal: *"Kecewa"* dan *"kecewa"* dihitung sebagai token yang sama).
* **Cleaning Noise:** Menghapus tautan URL, username (@user), angka, tanda baca, simbol emoji, dan spasi ganda menggunakan ekspresi reguler.
* **Slang Normalization:** Memetakan kata slang/informal ke bentuk bakunya (misal: *"ga"* -> *"tidak"*, *"bgt"* -> *"banget"*, *"lemot"* -> *"lambat"*) menggunakan kamus slang bahasa Indonesia untuk menyatukan variasi kata sebelum perhitungan frekuensi.

### Justifikasi Jalur Khusus Fitur

#### Jalur A: TF-IDF (Stemming Aktif)
* **Stemming (PySastrawi):** Memangkas imbuhan kata bahasa Indonesia ke kata dasarnya (misal: *"mengecewakan"*, *"dikecewakan"* -> *"kecewa"*).
* **Justifikasi:** TF-IDF sangat sensitif terhadap *sparsity* (kerenggangan matriks). Stemming menyatukan berbagai variasi infleksi kata ke dalam satu dimensi fitur tunggal, sehingga meningkatkan nilai TF dari kata dasar tersebut dan mempermudah model linier menemukan pola.

#### Jalur B: Word2Vec (Stemming Nonaktif)
* **Tanpa Stemming:** Mempertahankan imbuhan asli pada kata.
* **Justifikasi:** Word2Vec melatih representasi vektor kata berdasarkan konteks kata-kata di sekitarnya. Imbuhan bahasa Indonesia membawa informasi sintaksis yang sangat kaya (misal: kata *"menipu"* [aktif] dan *"ditipu"* [pasif] memiliki nuansa semantik berbeda). Stemming yang berlebihan akan merusak informasi kontekstual ini dan menghasilkan ruang representasi vektor yang miskin nuansa tata bahasa.

---

## 7. Feature Engineering (Rekayasa Fitur)

### 7.1 TF-IDF (Term Frequency - Inverse Document Frequency)
* **Mengapa Digunakan:** TF-IDF memberikan pembobotan statistik yang mengukur tingkat kepentingan sebuah kata dalam suatu dokumen terhadap seluruh korpus.
* **Parameter:** Unigram + Bigram (`ngram_range=(1,2)`), frekuensi minimum kata (`min_df=2`), pembobotan sublinear term frequency untuk meredam pengaruh ulasan yang terlalu panjang.
* **Kelebihan:** Sangat efektif menangkap kata kunci emosi eksplisit (*"kecewa"*, *"bagus"*) yang memiliki nilai IDF tinggi karena kemunculannya yang spesifik pada kelas emosi tertentu.
* **Kelemahan:** Mengabaikan urutan kata, kesamaan semantik kata sinonim, serta rentan terhadap masalah *Out of Vocabulary* (OOV) pada kata-kata baru di luar set pelatihan.

### 7.2 Word2Vec Document Embeddings
* **Mengapa Digunakan:** Vektor kontinu menangkap kedekatan makna kata dalam ruang berdimensi rendah (dense representation), mengatasi kelemahan TF-IDF dalam menangani kata sinonim.
* **Parameter Model:** Skip-Gram kustom (`sg=1`), `vector_size=100`, `window=5`, `min_count=2`, dilatih **hanya** pada *Train split*.
* **Representasi Dokumen:**
  * **Simple Average:** Rata-rata aritmatika sederhana dari vektor seluruh kata dalam ulasan.
  * **TF-IDF Weighted Average:** Vektor kata dikalikan dengan bobot TF-IDF kata tersebut sebelum dirata-ratakan. Hal ini memberikan bobot representasi yang lebih besar pada kata kunci unik dibanding kata-kata umum.
* **Kelebihan:** Menghasilkan representasi dimensi rendah (100 dimensi vs. ribuan dimensi TF-IDF), menangkap kesamaan konsep semantik secara implisit.
* **Kelemahan:** Rata-rata vektor dokumen rentan mengalami pengenceran informasi (*information dilution*) pada ulasan yang lebih panjang, sehingga sinyal kata kunci emosi yang pendek menjadi kabur.

---

## 8. Model Selection (Seleksi Model)

### 8.1 Model Machine Learning Tradisional

#### A. LinearSVC (Support Vector Classifier)
* **Mengapa Dipilih:** SVM mencari hyperplane pemisah dengan margin maksimal di ruang fitur berdimensi sangat tinggi (seperti TF-IDF).
* **Kelebihan:** Sangat kokoh terhadap overfitting pada data sparse berdimensi tinggi.
* **Kelemahan:** Tidak menghasilkan probabilitas klasifikasi secara langsung (harus didekati melalui penskalaan Platt atau fungsi keputusan eksponensial).

#### B. Logistic Regression
* **Mengapa Dipilih:** Classifier linier probabilistik sederhana yang efisien dan bertindak sebagai baseline komparatif utama.
* **Kelebihan:** Komputasi sangat cepat, menghasilkan distribusi probabilitas kelas secara natural, dan interpretasi bobot koefisien kata yang sangat mudah.
* **Kelemahan:** Mengasumsikan linearitas hubungan antar fitur, kurang optimal menangkap interaksi fitur non-linier kompleks.

#### C. Complement Naive Bayes (CNB)
* **Mengapa Dipilih:** Varian Naive Bayes yang dirancang khusus untuk mengatasi ketidakseimbangan kelas (*class imbalance*).
* **Kelebihan:** Menghitung probabilitas berdasarkan komplemen dari setiap kelas, menyeimbangkan estimasi parameter pada kelas minoritas (*Anger*).
* **Kelemahan:** Mengasumsikan independensi fitur yang kuat (asumsi Naive Bayes), yang sering kali dilanggar pada teks (bigram kata saling berkorelasi).

#### D. Random Forest Classifier
* **Mengapa Dipilih:** Model ensemble pohon keputusan non-linier untuk menangkap interaksi fitur yang lebih kompleks.
* **Kelebihan:** Tidak sensitif terhadap penskalaan fitur, mampu memodelkan hubungan non-linier.
* **Kelemahan:** Sangat lambat dilatih pada matriks TF-IDF dimensi tinggi yang sparse, serta rentan mengalami overfitting jika kedalaman pohon (`max_depth`) tidak dibatasi.

### 8.2 Arsitektur Desain Deep Learning (IndoBERT Fine-Tuning)
Sebagai bahan kajian paper tingkat lanjut, berikut adalah desain teoritis implementasi arsitektur Deep Learning menggunakan **IndoBERT** untuk analisis emosi:

```
[Input Text Tokenization] ──> [IndoBERT Encoder Stack] ──> [CLS Token Vector] ──> [Classification Head] ──> [Softmax Probabilities]
```

* **Mengapa Transformer Sangat Kuat:** IndoBERT menggunakan mekanisme *self-attention* dua arah (bidirectional context) untuk memahami hubungan semantik antar kata berdasarkan posisinya di seluruh kalimat. Hal ini memungkinkannya menangkap makna negasi jarak jauh (misal: *"fitur ini sebenarnya tidak terlalu buruk"* -> dideteksi sebagai bukan emosi kemarahan meskipun ada kata *"buruk"*).
* **Prosedur Fine-Tuning:**
  1. Memuat model IndoBERT pra-latih (*pre-trained model*) dari Hugging Face Hub.
  2. Melakukan tokenisasi ulasan menggunakan tokeniser WordPiece bawaan IndoBERT.
  3. Mengambil vektor representasi dari token khusus `[CLS]` di lapisan encoder terakhir sebagai representasi dokumen.
  4. Menambahkan lapisan linier (*Classification Head*) berukuran `(768, 5)` di atas token CLS.
  5. Melakukan pelatihan ulang (*fine-tuning*) pada set data ulasan menggunakan pengoptimasi AdamW, skema learning rate decay, dan fungsi kerugian *Cross-Entropy Loss*.
* **Mengapa Ditunda dalam Fisik Implementasi Saat Ini:** Karena dataset berlabel otomatis menggunakan aturan leksikon deterministik, IndoBERT yang memiliki jutaan parameter hanya akan "menghafal" kamus leksikon tersebut secara cepat (*overfitting* pada aturan pencocokan kata leksikon) tanpa benar-benar belajar representasi bahasa alami yang mendalam.

---

## 9. Training Process (Proses Pelatihan & Optimasi)

### Skema Pembagian Data
Dataset dibagi secara acak terkontrol (`random_state=42`) menggunakan **Stratified Split 70:15:15**:
* **Data Latih (Train Set):** 30.951 sampel.
* **Data Validasi (Val Set):** 6.633 sampel.
* **Data Uji (Test Set):** 6.633 sampel.

### Strategi Validasi Silang & Tuning
* Optimasi model dijalankan menggunakan **GridSearchCV 5-Fold Stratified Cross Validation** pada data latih. Stratifikasi memastikan setiap fold memiliki proporsi kelas emosi yang seimbang dengan populasi aslinya.
* Metrik optimasi GridSearchCV dikunci pada **Macro F1-Score** untuk menjamin kestabilan performa pada kelas minoritas (*Anger*).
* **Pencegahan Overfitting:**
  * Penyetelan hyperparameter regularisasi `C` pada SVM dan Logistic Regression (pencarian nilai `[0.1, 1.0, 10.0]`).
  * Pembatasan kedalaman pohon `max_depth` pada Random Forest (`max_depth=[10, 20]` untuk TF-IDF dan `[10, 20, None]` untuk Word2Vec) untuk mencegah pohon tumbuh terlalu dalam dan menghafal kebisingan data latih.

---

## 10. Evaluation Metrics (Metrik Evaluasi)

Klasifikasi emosi ulasan adalah masalah multi-kelas tidak seimbang. Oleh karena itu, pemilihan metrik evaluasi yang tepat sangat menentukan validitas hasil:

* **Accuracy:** Menghitung total prediksi benar dibagi seluruh sampel. Metrik ini **tidak digunakan sebagai acuan utama** karena jika model memprediksi kelas mayoritas (*Fear*, *Joy*) dengan sempurna namun gagal total pada kelas minoritas (*Anger*), akurasi akan tetap terlihat tinggi. Hal ini menyamarkan kegagalan klasifikasi yang kritis.
* **Precision (Macro Avg):** Mengukur ketepatan prediksi emosi model (seberapa banyak ulasan yang diprediksi *Anger* yang benar-benar emosi *Anger*). Pendekatan Macro merata-ratakan nilai presisi antar kelas dengan bobot setara.
* **Recall (Macro Avg):** Mengukur cakupan deteksi model (seberapa banyak ulasan emosi *Anger* aktual yang berhasil diidentifikasi oleh model). Pendekatan Macro penting untuk mendeteksi sensitivitas model pada kelas minoritas.
* **Macro F1-Score (Acuan Utama):** Rata-rata harmonik dari Precision dan Recall dengan pendekatan rata-rata Macro. F1-Score Macro memberikan hukuman berat bagi model yang hanya unggul di kelas mayoritas dan mengabaikan kelas minoritas. Metrik ini merupakan standar baku evaluasi akademis untuk data tidak seimbang.
* **Confusion Matrix:** Matriks visualisasi 5x5 untuk mengidentifikasi arah pergeseran klasifikasi kesalahan (misal: kelas aktual apa yang paling sering disalahprediksi sebagai kelas apa oleh model).

---

## 11. Results (Hasil Eksperimen & Evaluasi)

### Tabel Komparasi Hasil Eksperimen Lengkap
Berikut adalah tabel perbandingan performa 10 konfigurasi eksperimen yang diuji secara empiris pada data uji (*Test set*):

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

### Analisis Kebocoran Data (Leakage Check)
Pada hasil eksperimen terdahulu (sebelum dilakukan perbaikan), representasi Word2Vec tampak memiliki F1-score yang tinggi (~85%). Setelah dilakukan pelacakan, ditemukan adanya *transductive data leakage* karena model Word2Vec Skip-Gram dilatih pada seluruh ulasan sebelum data dibagi menjadi Train/Val/Test. 

Setelah model Word2Vec Skip-Gram dibatasi secara ketat hanya dilatih menggunakan ulasan dari *Train set*, performa riil Word2Vec turun menjadi **79.90%** (untuk Average) dan **72.00%** (untuk Weighted). Temuan ini membuktikan secara ilmiah bahwa representasi leksikal **TF-IDF** lebih unggul secara mutlak untuk dataset ulasan pendek ini dibandingkan Word2Vec.

---

## 12. Visualizations (Integrasi Visual & Penjelasan)

Penyusunan visualisasi di folder `reports/figures/` dilakukan otomatis oleh pipeline kode. Berikut adalah pembahasan detail mengenai visualisasi utama:

### 12.1 Distribusi Rating Mentah & Perbandingan Panjang Karakter
Visualisasi ini memplot karakteristik fisik data sebelum dan sesudah proses pemfilteran:
* `reports/figures/raw_rating_distribution.png`: Menampilkan grafik batang distribusi rating ulasan mentah di Play Store. Grafik ini menunjukkan dominasi ulasan dengan rating ekstrem (rating 1 dan 5), yang membenarkan batas penyaringan leksikon emosi kami.
* `reports/figures/char_length_comparison.png`: Membandingkan histogram panjang karakter ulasan sebelum vs. sesudah pemfilteran. Ulasan yang terlalu pendek (di bawah 10 karakter) atau tidak memiliki muatan emosional disaring untuk meminimalisasi kebisingan ruang fitur.

### 12.2 Distribusi Kelas Emosi Leksikon
* `reports/figures/class_distribution.png`: Grafik batang jumlah ulasan pasca pelabelan leksikon. Menunjukkan ketidakseimbangan kelas di mana kelas *Fear* dan *Joy* mendominasi dataset, sementara kelas *Anger* bertindak sebagai kelas minoritas terkecil. Hal ini menuntut penggunaan Macro F1-score sebagai metrik acuan utama.

### 12.3 Word Clouds Per Kelas Emosi
* `reports/figures/wordclouds.png`: Menampilkan awan kata untuk masing-masing kelas emosi dengan parameter *collocations=False* (menghilangkan duplikasi kata yang muncul dari Bigram kolokasi). Visualisasi ini memperlihatkan kata-kata kunci utama kelas Joy (*"bagus"*, *"suka"*), kelas Anger (*"lemot"*, *"error"*, *"jelek"*), dan kelas Fear (*"ditangguhkan"*, *"bocor"*, *"hack"*).

### 12.4 Proyeksi t-SNE Kata Kunci Word2Vec
* `reports/figures/word2vec_tsne.png`: Menurunkan visualisasi proyeksi ruang vektor 2D kata-kata kunci emosi representatif dari Word2Vec Skip-Gram. Kluster kata-kata seperti *"kecewa"*, *"jelek"*, *"lemot"* mengelompok di satu wilayah semantik, menunjukkan keberhasilan model Word2Vec dalam mempelajari kemiripan kontekstual kata slang bahasa Indonesia dari data latih.

### 12.5 Grafik Batang Komparatif Performa Model
* `reports/figures/model_comparison_chart.png`: Visualisasi grouped bar chart yang membandingkan performa Macro F1-Score dari seluruh kombinasi eksperimen pada data validasi. Grafik ini secara jelas menunjukkan keunggulan LinearSVC + TF-IDF (0.941) di atas model lainnya.

### 12.6 Matriks Konfusi & Matriks Transisi Kesalahan
* `reports/figures/normalized_confusion_matrix.png`: Matriks konfusi ternormalisasi pada set uji untuk model LinearSVC terbaik. Diagonal utama menunjukkan nilai recall per kelas yang sangat tinggi (berkisar antara 89% hingga 99%).
* `reports/figures/error_transition_matrix.png`: Visualisasi kuantitatif khusus untuk prediksi yang salah. Matriks ini mengungkap hubungan transisi kesalahan prediksi terbesar, di mana kelas aktual *Anger* paling sering disalahprediksi sebagai kelas *Disgust* (29%), dan kelas aktual *Disgust* paling sering disalahprediksi sebagai kelas *Anger* (25%).

---

## 13. Analysis & Insights (Analisis & Pembahasan Mendalam)

### Mengapa LinearSVC + TF-IDF Unggul Mutlak?
1. **Pola Penulisan Ulasan Pendek:** Pengguna mengekspresikan emosi di Play Store secara langsung menggunakan istilah emosional kunci (seperti *"kecewa"*, *"bagus"*, *"susah login"*). Pembobotan TF-IDF memberikan bobot yang tinggi pada istilah emosi unik ini. Model linier seperti SVM sangat efisien dalam menentukan batas pemisahan linier (*hyperplane*) optimal pada dimensi fitur sparse yang dihasilkan.
2. **Keterbatasan Rata-rata Word2Vec (Information Dilution):** Ketika vektor kata dirata-ratakan (Simple Average Word2Vec) untuk menghasilkan vektor dokumen tunggal, bobot kata emosi yang pendek dan tajam menjadi melemah karena tercampur dengan representasi kata-kata fungsional netral lainnya. Pembobotan TF-IDF pada Word2Vec (Weighted Average) sedikit menolong performa (naik menjadi 0.720), tetapi masih jauh di bawah performa TF-IDF murni (0.941).

### Dampak Ketidakseimbangan Kelas (Class Imbalance)
Kelas *Anger* memiliki presisi 0.93 dan recall 0.89 (F1-score 0.91), yang merupakan skor terendah di antara seluruh kelas. Hal ini disebabkan oleh minimnya jumlah sampel latih untuk emosi *Anger* dibandingkan kelas *Fear* (F1-score 0.99) yang memiliki ribuan sampel. Penyeimbangan parameter internal algoritma (seperti penyetelan bobot kelas otomatis `class_weight='balanced'`) terbukti krusial untuk mencegah model condong memprediksi kelas mayoritas saja.

---

## 14. Strengths & Weaknesses (Kekuatan & Kelemahan Sistem)

### Kekuatan (Strengths)
* **Modularitas Tinggi:** Struktur kode python modular terpisah antara pengambilan data, pra-pengolahan dual-jalur, ekstraksi fitur, pemodelan, dan visualisasi sehingga memudahkan pemeliharaan kode.
* **Integritas Metodologi:** Isolasi data yang ketat pada pelatihan Word2Vec Skip-Gram mencegah bias evaluasi.
* **Rigoritas Ilmiah:** Menyertakan uji signifikansi statistik McNemar untuk pembuktian keunggulan model secara akademis.
* **Uji Inferensi Interaktif:** Menyertakan fungsi prediksi teks kustom interaktif langsung pada sel notebook untuk simulasi klasifikasi waktu nyata.

### Kelemahan (Weaknesses)
* **Sirkularitas Ground Truth Leksikon:** Karena pelabelan awal data latih menggunakan kamus leksikon otomatis, performa akurasi uji tinggi (96.75%) mencerminkan efisiensi model dalam menduplikasi pola kamus leksikon tersebut, bukan representasi kognitif emosi manusia asli.
* **Slang Dinamis:** Kamus normalisasi slang bersifat statis, sehingga performa dapat menurun bila diuji dengan kosakata slang internet Indonesia generasi terbaru.

---

## 15. Failure Cases (Studi Kasus Kegagalan Prediksi)

Berikut adalah contoh ulasan set uji yang gagal diprediksi oleh model terbaik:

### Kasus A: Sarkasme
* **Teks Ulasan:** *"Wah update baru aplikasinya keren banget ya, tiap buka langsung crash! Makasih developer atas kerja kerasnya."*
* **Label Aktual:** ANGER (Marah)
* **Prediksi Model:** JOY (Senang)
* **Penyebab Kegagalan:** Model TF-IDF LinearSVC mendeteksi kata-kata penguat emosi positif seperti *"keren banget"*, *"makasih"*, dan *"kerja keras"* sebagai indikator kuat untuk kelas *Joy*. Model linier tradisional tidak memiliki kemampuan untuk memodelkan hubungan kontradiksi semantik implisit (sarkasme) di mana kata *"crash"* sebenarnya meniadakan seluruh makna positif di awal kalimat.

### Kasus B: Konteks Ganda & Negasi Lemah
* **Teks Ulasan:** *"Akun saya tidak diblokir sih, cuma tiba-tiba log out sendiri terus pas mau masuk lagi ada notif eror jaringan."*
* **Label Aktual:** ANGER (Marah)
* **Prediksi Model:** FEAR (Takut)
* **Penyebab Kegagalan:** Kehadiran kata *"diblokir"* (yang biasanya diasosiasikan erat dengan kelas *Fear* karena penangguhan akun) membingungkan model. Meskipun ulasan dibersihkan dan memiliki kata negasi *"tidak"*, pembobotan bag-of-words leksikal TF-IDF sering kali gagal menangkap pengaruh pembalikan polaritas negasi jika jarak antar kata terlalu jauh.

---

## 16. Conclusion (Kesimpulan)

Eksperimen komparatif klasifikasi emosi multi-kelas ulasan Twitter/X Play Store Indonesia berhasil diselesaikan secara metodologis. Temuan utama menunjukkan bahwa model representasi fitur **TF-IDF unigram+bigram** yang dipasangkan dengan pengklasifikasi **LinearSVC** menghasilkan performa terbaik dengan skor Macro F1 **0.941** dan akurasi uji **96.75%**. 

Penelitian ini membuktikan secara ilmiah bahwa untuk ulasan pendek berbahasa Indonesia yang kaya akan bahasa slang informal, representasi leksikal tradisional TF-IDF jauh lebih kokoh dibandingkan embedding Word2Vec Skip-Gram yang mengalami pengenceran informasi semantik akibat proses perataan vektor dokumen.

---

## 17. Future Work (Rencana Lanjutan)

1. **Anotasi Data Ahli (Human Annotation):** Mengganti pelabelan otomatis berbasis kamus dengan anotasi manual oleh panel annotator manusia untuk menghilangkan bias sirkularitas leksikon dan menghitung nilai kesepakatan kesepahaman (*Inter-Annotator Agreement* Cohen's Kappa).
2. **Implementasi Transformer (IndoBERT):** Melakukan eksperimen fine-tuning penuh pada model IndoBERT pra-latih untuk memodelkan hubungan semantik kontekstual kompleks, negasi jarak jauh, dan sarkasme pada teks ulasan.
3. **Augmentasi Data Kelas Minoritas:** Menerapkan teknik augmentasi data teks (seperti *back-translation* atau substitusi sinonim berbasis Word2Vec) khusus untuk kelas minoritas *Anger* guna mendongkrak performa klasifikasi model.

---

## 18. Installation Guide (Panduan Instalasi)

Pastikan lingkungan lokal Anda menggunakan Python versi **3.12** atau di atasnya.

```bash
# 1. Clone repositori proyek Anda
git clone https://github.com/Rakanovan9/indonesian-reviews-emotion-analysis.git
cd indonesian-reviews-emotion-analysis

# 2. Buat environment virtual baru
python -m venv venv

# 3. Aktifkan environment virtual
# Untuk Windows:
.\venv\Scripts\activate
# Untuk macOS/Linux:
source venv/bin/activate

# 4. Install seluruh dependensi yang diperlukan
pip install -r requirements.txt
```

---

## 19. How to Run (Cara Menjalankan Pipeline)

Untuk memastikan konsistensi hasil dan kebersihan data, jalankan seluruh Jupyter Notebook di folder `notebooks/` secara berurutan:

```bash
# Opsi A: Jalankan antarmuka Jupyter Lab untuk memantau visualisasi secara langsung
jupyter lab

# Opsi B: Jalankan seluruh pipeline secara headless dari terminal menggunakan nbconvert
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_and_filtering.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_preprocessing.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_feature_extraction.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_model_training.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/05_error_analysis.ipynb
```

---

## 20. Project Structure (Struktur Folder Proyek)

```bash
emotion-analysis/
│
├── data/
│   ├── raw/                  # Berkas ulasan Play Store mentah hasil scraping (.csv)
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

## Referensi Akademis

1. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space*. arXiv preprint arXiv:1301.3781.
2. McNemar, Q. (1947). *Note on the sampling error of the difference between correlated proportions or percentages*. Psychometrika, 12(2), 153-157.
3. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). *Scikit-learn: Machine learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.
