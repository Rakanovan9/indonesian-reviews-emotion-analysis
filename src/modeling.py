from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

def get_model_and_grid(model_name, feature_type):
    """
    Mengembalikan instance model ML dan grid hyperparameter-nya.
    Menerapkan aturan restriksi kedalaman pohon Random Forest berdasarkan jenis fitur.
    """
    if model_name == "LogisticRegression":
        estimator = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        param_grid = {
            'C': [0.1, 1.0, 10.0]
        }
        
    elif model_name == "LinearSVC":
        estimator = LinearSVC(class_weight='balanced', random_state=42, max_iter=2000)
        param_grid = {
            'C': [0.1, 1.0, 10.0]
        }
        
    elif model_name == "ComplementNB":
        if feature_type != "tfidf":
            raise ValueError("Complement Naive Bayes hanya mendukung representasi fitur non-negatif (TF-IDF).")
        estimator = ComplementNB()
        param_grid = {
            'alpha': [0.1, 0.5, 1.0]
        }
        
    elif model_name == "RandomForest":
        # Aturan kedalaman pohon RF berdasarkan dimensi fitur
        max_depth_options = [10, 20] if feature_type == "tfidf" else [10, 20, None]
        estimator = RandomForestClassifier(class_weight='balanced', random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': max_depth_options,
            'min_samples_split': [2, 5]
        }
    else:
        raise ValueError(f"Nama model tidak dikenal: {model_name}")
        
    return estimator, param_grid

def train_with_grid_search(model_name, feature_type, X_train, y_train):
    """
    Menjalankan GridSearchCV dengan 5-Fold Stratified CV, dioptimalkan untuk Macro F1.
    """
    estimator, param_grid = get_model_and_grid(model_name, feature_type)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=cv,
        n_jobs=-1,
        verbose=0
    )
    
    print(f"Menjalankan GridSearch untuk {model_name} ({feature_type})...")
    grid_search.fit(X_train, y_train)
    
    print(f"Hyperparameter terbaik: {grid_search.best_params_}")
    print(f"Best CV Macro F1: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_eval, y_eval):
    """
    Mengevaluasi model pada set evaluasi (Val atau Test) dan mengembalikan skor Macro F1 & Akurasi.
    """
    y_pred = model.predict(X_eval)
    macro_f1 = f1_score(y_eval, y_pred, average='macro')
    accuracy = accuracy_score(y_eval, y_pred)
    return macro_f1, accuracy
