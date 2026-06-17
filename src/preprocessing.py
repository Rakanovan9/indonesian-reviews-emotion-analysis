import re
import pandas as pd
from sklearn.model_selection import train_test_split

# Coba impor Sastrawi untuk Stemmer, gunakan fallback jika tidak terinstal
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    HAS_SASTRAWI = True
except ImportError:
    HAS_SASTRAWI = False
    print("[WARNING] Pustaka 'PySastrawi' tidak terinstal. Proses stemming akan dilewati menggunakan fallback.")

# Kamus pemetaan slang informal/emosi khusus Twitter/X
SLANG_DICT = {
    'yg': 'yang', 'dgn': 'dengan', 'jg': 'juga', 'klo': 'kalau', 'kalo': 'kalau',
    'tp': 'tapi', 'tpi': 'tapi', 'bgt': 'banget', 'bgtz': 'banget', 'skrg': 'sekarang',
    'gk': 'tidak', 'ga': 'tidak', 'gak': 'tidak', 'ndak': 'tidak', 'x': 'twitter',
    'kesel': 'kesal', 'kezel': 'kesal', 'kzyl': 'kesal', 'gedeg': 'kesal',
    'nyebelin': 'menyebalkan', 'sebel': 'kesal', 'benci': 'benci',
    'seneng': 'senang', 'hepi': 'senang', 'happy': 'senang', 'mantul': 'mantap',
    'sedii': 'sedih', 'keciwa': 'kecewa', 'nyesel': 'menyesal',
    'khawatir': 'khawatir', 'kuatir': 'khawatir', 'parno': 'khawatir',
    'waswas': 'khawatir', 'cemas': 'khawatir',
    'uninstall': 'copot', 'uninstal': 'copot', 'lelet': 'lambat', 'lemot': 'lambat'
}

# Daftar stopword kustom (membuang kata umum, mempertahankan negasi & intensitas)
CUSTOM_STOPWORDS = {
    'yang', 'di', 'ke', 'dari', 'dan', 'atau', 'ini', 'itu', 'adalah', 'yaitu',
    'oleh', 'untuk', 'pada', 'dengan', 'bahwa', 'tersebut', 'lah', 'kah', 'pun',
    'saya', 'anda', 'dia', 'mereka', 'kita', 'kami', 'kamu', 'aku', 'sih', 'kok'
}

def normalize_slang(text, slang_map=SLANG_DICT):
    """
    Mengubah kata-kata informal/slang menjadi kata baku berdasarkan kamus pemetaan.
    """
    if not isinstance(text, str):
        return ""
    words = text.split()
    normalized_words = [slang_map.get(word, word) for word in words]
    return " ".join(normalized_words)

def preprocess_for_tfidf(text):
    """
    Jalur Preprocessing TF-IDF:
    1. Slang Normalization
    2. Stemming (Sastrawi)
    3. Custom Stopword Removal (Negasi/Intensifikator dipertahankan)
    """
    text = normalize_slang(text)
    
    # Stemming jika Sastrawi terinstal
    if HAS_SASTRAWI:
        text = stemmer.stem(text)
        
    # Stopword removal
    words = text.split()
    filtered_words = [word for word in words if word not in CUSTOM_STOPWORDS]
    return " ".join(filtered_words)

def preprocess_for_word2vec(text):
    """
    Jalur Preprocessing Word2Vec:
    1. Slang Normalization saja (Menjaga keutuhan urutan kata dan imbuhan)
    """
    return normalize_slang(text)

def assign_lexicon_labels(df):
    """
    Melabeli ulasan berdasarkan frekuensi kata kunci emosi.
    Resolusi Konflik: Anger > Sadness > Disgust > Fear > Joy
    Membuang baris yang tidak memiliki kecocokan emosi.
    """
    # Kamus kata kunci untuk deteksi frekuensi
    lexicon = {
        'joy': {'bagus', 'keren', 'mantap', 'mantul', 'suka', 'puas', 'senang', 'seneng', 'hepi', 'terbantu', 'cinta', 'lucu', 'menghibur'},
        'anger': {'kesal', 'kesel', 'kezel', 'marah', 'benci', 'sialan', 'bangsat', 'anjing', 'anjg', 'gila', 'buruk', 'ampas', 'jelek', 'lambat', 'lelet', 'lemot', 'crash', 'eror', 'bug', 'force close', 'gagal'},
        'sadness': {'kecewa', 'keciwa', 'sedih', 'nyesel', 'menyesal', 'sayang', 'sayangnya', 'rugi', 'hilang', 'kangen', 'dulu', 'sebelumnya', 'berubah'},
        'fear': {'takut', 'khawatir', 'kuatir', 'parno', 'cemas', 'ngeri', 'bahaya', 'bocor', 'diretas', 'hack', 'keamanan', 'aman', 'penipuan', 'scam', 'ditangguhkan', 'suspend'},
        'disgust': {'muak', 'jijik', 'bosan', 'bosen', 'cape', 'capek', 'lelah', 'males', 'malas', 'risih', 'sampah', 'kotor', 'copot', 'uninstall', 'toxic', 'bot'}
    }
    
    labels = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        content = str(row['cleaned_content'])
        words = set(content.split())
        
        # Hitung skor kecocokan kata menggunakan pencarian substring berbasis batas kata (word boundaries)
        # Ini mendukung frasa multi-kata seperti "force close" atau "bintang lima"
        scores = {}
        for emotion, keywords in lexicon.items():
            score = 0
            for keyword in keywords:
                # Regex mencari batas kata utuh agar kata kunci tidak cocok di tengah kata lain
                if re.search(r'\b' + re.escape(keyword) + r'\b', content):
                    score += 1
            scores[emotion] = score
        
        # Cari nilai maksimum
        max_score = max(scores.values())
        
        if max_score == 0:
            # Tidak ada kecocokan emosi sama sekali
            labels.append(None)
        else:
            # Ambil semua emosi yang memiliki nilai kecocokan maksimum
            candidates = [emotion for emotion, score in scores.items() if score == max_score]
            
            if len(candidates) == 1:
                labels.append(candidates[0])
                valid_indices.append(idx)
            else:
                # Terapkan prioritas konflik: Anger > Sadness > Disgust > Fear > Joy
                priority = ['anger', 'sadness', 'disgust', 'fear', 'joy']
                chosen = next((emotion for emotion in priority if emotion in candidates), candidates[0])
                labels.append(chosen)
                valid_indices.append(idx)
                
    df['emotion'] = labels
    labeled_df = df.loc[valid_indices].copy()
    print(f"Data berlabel berhasil dibuat: {len(labeled_df)} baris.")
    print(labeled_df['emotion'].value_counts())
    
    return labeled_df
