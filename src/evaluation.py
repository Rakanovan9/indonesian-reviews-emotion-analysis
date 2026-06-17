import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_normalized_confusion_matrix(y_true, y_pred, classes, save_path=None):
    """
    Membuat dan menampilkan visualisasi 5x5 Normalized Confusion Matrix.
    Setiap baris dihitung proporsinya (menunjukkan recall di sepanjang diagonal).
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    # Normalisasi baris (pembagian berdasarkan jumlah sampel asli per kelas)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Oranges)
    ax.figure.colorbar(im, ax=ax)
    
    # Atur label sumbu
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes, 
        yticklabels=classes,
        title='5x5 Normalized Confusion Matrix (Recall Rates)',
        ylabel='True Label (Aktual)',
        xlabel='Predicted Label (Prediksi)'
    )
    
    # Putar label sumbu x
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Isi angka persentase di dalam kotak matriks
    fmt = '.2f'
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm_normalized[i, j], fmt),
                ha="center", va="center",
                color="white" if cm_normalized[i, j] > thresh else "black"
            )
            
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grafik Confusion Matrix disimpan ke: {save_path}")
        
    plt.show()

def extract_misclassifications(test_df, y_true, y_pred, target_true, target_pred, num_samples=3):
    """
    Mengambil sampel ulasan asli di mana model salah memprediksi target_true menjadi target_pred.
    Sangat membantu untuk ekstraksi studi kasus analisis eror linguistik di paper.
    """
    # Buat salinan sementara untuk pelacakan
    df_temp = test_df.copy()
    df_temp['true_label'] = y_true
    df_temp['pred_label'] = y_pred
    
    # Filter kondisi salah klasifikasi spesifik
    mask = (df_temp['true_label'] == target_true) & (df_temp['pred_label'] == target_pred)
    subset = df_temp[mask]
    
    print(f"Menampilkan {min(num_samples, len(subset))} dari {len(subset)} kasus salah prediksi ({target_true} -> {target_pred}):")
    print("=" * 80)
    
    for idx, row in subset.head(num_samples).iterrows():
        print(f"Review Asli : {row['content']}")
        print(f"TF-IDF Text : {row['content_tfidf']}")
        print(f"W2V Text   : {row['content_w2v']}")
        print("-" * 80)
