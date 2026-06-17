# 06 — Modeling Strategy

## Model Selection and Hyperparameter Optimization

---

## 6.1 Machine Learning Classifiers

We will train and evaluate four machine learning classifiers. These models represent distinct learning paradigms (probabilistic, linear, margin-based, and ensemble).

| Model | Primary Strengths for NLP | Limitations |
|---|---|---|
| **Logistic Regression** | Highly interpretable, fast training, excellent baseline for linear text patterns | Assumes linear relationships between features |
| **Linear Support Vector Classifier (LinearSVC)** | Maximizes margin between classes; highly effective in high-dimensional sparse spaces (TF-IDF) | Output probabilities require calibration (sigmoid fitting) |
| **Multinomial Naive Bayes (MNB)** | Extremely fast, operates well on sparse word-count vectors | Assumes feature independence (which fails for text); cannot handle negative features (Word2Vec) |
| **Random Forest** | Non-linear tree ensemble; handles complex feature interactions, robust to outliers | Prone to overfitting on very high-dimensional sparse data (TF-IDF) |

---

## 6.2 Model-Feature Compatibility Rules

- **Rule 1: Naive Bayes Restriction**
  - **IF** feature extraction is **Word2Vec**:
    - **Action:** Exclude **Multinomial Naive Bayes** from the experiment.
    - **Why:** Word2Vec features contain negative coordinate values representing semantic directions. Multinomial Naive Bayes requires non-negative counts or TF-IDF weights. Training MNB on negative inputs will cause a runtime value error.
  - **IF** feature extraction is **TF-IDF**:
    - **Action:** Include Multinomial Naive Bayes.
- **Rule 2: Random Forest Tuning**
  - **IF** feature representation is **TF-IDF**:
    - **Action:** Constrain tree depth (`max_depth=20`) to prevent overfitting.
  - **IF** feature representation is **Word2Vec**:
    - **Action:** Allow deeper trees (`max_depth=None`) since dense dimensions are low ($100$ vs $10,000+$ for TF-IDF).

---

## 6.3 Hyperparameter Tuning Grid

Tuning must be performed using **GridSearchCV** with **5-Fold Stratified Cross-Validation**, optimizing for **Macro F1-Score**.

```python
param_grids = {
    "LogisticRegression": {
        "C": [0.1, 1.0, 10.0],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
        "max_iter": [1000]
    },
    "LinearSVC": {
        "C": [0.1, 1.0, 10.0]
    },
    "MultinomialNB": {
        "alpha": [0.1, 0.5, 1.0]
    },
    "RandomForest": {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5]
    }
}
```

---

## 6.4 Handling Class Imbalance in Models

To prevent models from prioritizing majority classes (e.g., *Joy*, *Anger*) over critical minority classes (e.g., *Fear*):

- **Action:** Set `class_weight='balanced'` for `LogisticRegression`, `LinearSVC`, and `RandomForest`.
- **Why it is done:** This dynamically adjusts weights inversely proportional to class frequencies in the input data:
  $$W_c = \frac{N_{samples}}{N_{classes} \times N_c}$$
  This ensures that misclassifying a rare emotion (like Fear) yields a much larger penalty in the loss function than misclassifying a common one.

---

## 6.5 Statistical Baseline

- **What to do:** Include a `DummyClassifier` with `strategy='stratified'` or `strategy='most_frequent'` as the baseline model.
- **Why it is done:** Establishes the performance of a zero-intelligence classifier. Any machine learning model must significantly outperform this baseline to prove statistical validity.
