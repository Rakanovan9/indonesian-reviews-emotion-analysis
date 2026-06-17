import numpy as np
import pandas as pd
from gensim.models import Word2Vec

def train_custom_word2vec(sentences, vector_size=100, window=5, min_count=2, sg=1, epochs=10):
    """
    Melatih model Word2Vec kustom (Skip-Gram) dari korpus kalimat pelatihan.
    sentences: list of lists of strings (tokenized sentences)
    """
    print(f"Melatih Word2Vec model kustom dengan dimensi={vector_size}, window={window}...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        workers=4,
        seed=42
    )
    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)
    print(f"Word2Vec kustom berhasil dilatih! Jumlah kosakata unik: {len(model.wv)}")
    return model

def get_simple_average_vector(words, w2v_model, vector_size=100):
    """
    Menghitung vektor rata-rata sederhana dari sebuah ulasan.
    """
    valid_vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    if len(valid_vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(valid_vectors, axis=0)

def get_tfidf_weighted_vector(words, w2v_model, tfidf_dict, vector_size=100):
    """
    Menghitung rata-rata tertimbang vektor kata berdasarkan bobot TF-IDF.
    tfidf_dict: dictionary yang memetakan kata -> nilai TF-IDF dalam dokumen tersebut.
    """
    valid_vectors = []
    weights = []
    
    for word in words:
        if word in w2v_model.wv:
            # Ambil nilai TF-IDF kata tersebut, default ke 1.0 jika tidak ditemukan
            weight = tfidf_dict.get(word, 1.0)
            valid_vectors.append(w2v_model.wv[word] * weight)
            weights.append(weight)
            
    if len(valid_vectors) == 0:
        return np.zeros(vector_size)
        
    sum_weights = sum(weights)
    if sum_weights == 0:
        return np.zeros(vector_size)
        
    return np.sum(valid_vectors, axis=0) / sum_weights

def transform_to_word2vec_features(df, w2v_model, text_col='content_w2v', method='average', tfidf_vectorizer=None):
    """
    Mengubah teks kolom ulasan DataFrame menjadi matriks fitur Word2Vec dokumen (N x vector_size).
    """
    vector_size = w2v_model.vector_size
    features = []
    
    if method == 'average':
        for idx, row in df.iterrows():
            words = str(row[text_col]).split()
            vec = get_simple_average_vector(words, w2v_model, vector_size)
            features.append(vec)
            
    elif method == 'tfidf_weighted':
        if tfidf_vectorizer is None:
            raise ValueError("Untuk metode 'tfidf_weighted', objek 'tfidf_vectorizer' wajib disertakan.")
            
        # Dapatkan fitur nama kata kunci TF-IDF
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        # Transformasikan teks kolom ke representasi TF-IDF
        tfidf_matrix = tfidf_vectorizer.transform(df[text_col]).toarray()
        
        for idx, (df_idx, row) in enumerate(df.iterrows()):
            words = str(row[text_col]).split()
            # Bangun kamus TF-IDF untuk dokumen spesifik ini
            doc_tfidf_weights = {
                feature_names[i]: tfidf_matrix[idx, i] 
                for i in range(len(feature_names)) 
                if tfidf_matrix[idx, i] > 0
            }
            vec = get_tfidf_weighted_vector(words, w2v_model, doc_tfidf_weights, vector_size)
            features.append(vec)
            
    return np.array(features)
