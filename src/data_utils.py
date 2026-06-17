import os
import pandas as pd
import re

def clean_raw_text(text):
    """
    Melakukan pembersihan dasar pada teks mentah ulasan sebelum analisis kata kunci.
    """
    if not isinstance(text, str):
        return ""
    # Ubah ke lowercase
    text = text.lower()
    # Hapus URL
    text = re.sub(r'https?://\s*\S+|www\.\S+', '', text)
    # Hapus email
    text = re.sub(r'\S+@\S+', '', text)
    # Hapus user mentions (@username)
    text = re.sub(r'@\S+', '', text)
    # Hapus karakter non-alfabet (simpan spasi)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Gabungkan spasi ganda
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_filter_data(filepath):
    """
    Memuat dataset mentah dari CSV dan memfilter ulasan yang memiliki sinyal emosi.
    Menghapus ulasan teknis/faktual (netral) menggunakan leksikon emosi dasar.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Berkas tidak ditemukan: {filepath}")
        
    df = pd.read_csv(filepath)
    print(f"Total ulasan mentah dimuat: {len(df)}")
    
    # 1. Bersihkan teks secara ringan untuk mencocokkan kata kunci
    df['cleaned_content'] = df['content'].apply(clean_raw_text)
    
    # 2. Kamus leksikon kata kunci emosi (baku & informal/slang)
    emotion_lexicon = {
        'joy': [
            'bagus', 'keren', 'mantap', 'mantul', 'suka', 'puas', 'senang', 'seneng', 'hepi', 
            'terbantu', 'cinta', 'love', 'lucu', 'menghibur', 'rekomendasi', 'apresiasi', 'bintang lima'
        ],
        'anger': [
            'kesal', 'kesel', 'kezel', 'marah', 'benci', 'nyebelin', 'sialan', 'bangsat', 'anjing', 
            'anjg', 'gila', 'buruk', 'ampas', 'jelek', 'lambat', 'lelet', 'lemot', 'crash', 'eror', 
            'bug', 'kecewa', 'parah', 'hancur', 'rusak', 'susah login', 'force close', 'gagal'
        ],
        'sadness': [
            'kecewa', 'keciwa', 'sedih', 'sedii', 'nyesel', 'menyesal', 'sayang', 'sayangnya', 
            'rugi', 'hilang', 'kangen', 'dulu', 'sebelumnya', 'berubah', 'parah'
        ],
        'fear': [
            'takut', 'khawatir', 'kuatir', 'parno', 'cemas', 'ngeri', 'bahaya', 'bocor', 
            'diretas', 'hack', 'keamanan', 'aman', 'penipuan', 'scam', 'ditangguhkan', 'suspend'
        ],
        'disgust': [
            'muak', 'jijik', 'bosan', 'bosen', 'cape', 'capek', 'lelah', 'males', 'malas', 
            'risih', 'sampah', 'kotor', 'uninstall', 'copot', 'toksik', 'toxic', 'bot'
        ]
    }
    
    # Gabungkan semua kata kunci menjadi satu list unik
    all_emotional_keywords = set()
    for words in emotion_lexicon.values():
        all_emotional_keywords.update(words)
        
    # Helper function untuk mencocokkan kata kunci emosi
    def contains_emotion_keyword(text):
        tokens = text.split()
        return any(token in all_emotional_keywords for token in tokens)
    
    # 3. Kriteria pemfilteran:
    # Ulasan dipertahankan jika:
    # - Memiliki skor rating ekstrim (1 atau 5)
    # - ATAU memiliki kata kunci emosi dalam teksnya
    is_extreme_score = df['score'].isin([1, 5])
    has_keyword = df['cleaned_content'].apply(contains_emotion_keyword)
    
    filtered_df = df[is_extreme_score | has_keyword].copy()
    
    print(f"Ulasan berhasil disaring (memiliki sinyal emosi): {len(filtered_df)}")
    print(f"Ulasan non-emosional (dibuang): {len(df) - len(filtered_df)}")
    
    return filtered_df
