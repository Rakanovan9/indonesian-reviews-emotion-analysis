# 03 — Dataset and Labeling Strategy

## From Rating Stars to Genuine Emotion Labels

---

## 3.1 The Labeling Challenge

In the sentiment analysis project, we used Google Play Store rating stars (1, 3, 5) as distant supervision labels. However, **ratings do not correlate cleanly with fine-grained emotions**. 

For example:
- **Rating 1** can express **Anger** ("*Aplikasi sampah, bikin HP lag!*") or **Sadness/Disappointment** ("*Kecewa sekali dengan update terbaru, padahal dulu bagus*") or **Fear** ("*Takut data saya dicuri karena aplikasi minta izin aneh*").
- **Rating 5** can express **Joy** ("*Keren banget aplikasinya!*") or be structurally neutral (automatic rating).

Therefore, we must employ a new strategy to acquire true emotion labels.

---

## 3.2 Label Acquisition Strategies (Conditional Choice)

Three options are proposed below, depending on available resources and time:

### Option A: Manual Annotation (Highly Recommended for Validation/Test Sets)
- **What:** The project group (2-3 members) manually annotates a subset of 3,000 reviews.
- **Why:** Human annotators understand context, sarcasm, and slang better than automated rules.
- **Methodology:**
  1. Define a strict annotation guide (defining Joy, Anger, Sadness, Fear, Disgust, and Neutral).
  2. Double-annotate 10% of the sample to measure inter-annotator agreement using **Cohen's Kappa** ($\kappa$).
- **Conditional Logic:**
  - **IF** $\kappa < 0.60$ (weak agreement):
    - **Action:** Halt annotation, review the annotation guide, hold a consensus alignment session to discuss disputable samples, and re-annotate.
  - **IF** $\kappa \ge 0.60$ (moderate-to-strong agreement):
    - **Action:** Proceed with individual annotation of the remaining dataset. Resolve conflicts on overlapping samples by majority vote or team lead decision.

### Option B: Lexicon-Based Distant Supervision (Weak Labeling)
- **What:** Use Indonesian emotion lexicons (e.g., translation of NRC Emotion Lexicon or custom emotion dictionaries) to auto-label reviews based on emotion-carrying keywords.
- **Why:** Scalable to the entire 49,000+ reviews.
- **Conditional Logic:**
  - **IF** a review contains keywords matching exactly ONE emotion class (e.g., only "marah", "kesal", "benci" for Anger):
    - **Action:** Assign that emotion label.
  - **IF** a review contains mixed keywords (e.g., "senang" and "kecewa") OR zero keywords:
    - **Action:** Discard from training set, or label as Neutral if it has rating 3.
  - **Why this logic:** Avoids injecting high-noise labels into the training set.

### Option C: Zero-Shot Transformer-Based Auto-Labeling
- **What:** Use a pre-trained multilingual model (e.g., `cardiffnlp/twitter-xlm-roberta-base-sentiment` or `lxyuan/distilbert-base-multilingual-cased-sentiments-student` adapted for emotions) to predict labels on the dataset.
- **Why:** Captures context and syntactic relations better than bag-of-words lexicons.
- **Validation Rule:** Manually annotate a validation subset of 500 samples. Run the zero-shot model on it and measure accuracy.
  - **IF** Zero-Shot Accuracy on validation subset $< 75\%$:
    - **Action:** Reject the zero-shot labeling approach. Fall back to manual annotation (Option A) or lexicon filtering (Option B).
  - **IF** Zero-Shot Accuracy on validation subset $\ge 75\%$:
    - **Action:** Use the zero-shot model to label the training set (weak supervision), but **always use the 100% human-annotated set for validation and testing**.

---

## 3.3 Data Splitting Strategy

The dataset splits must be performed **before** any preprocessing or resampling to prevent data leakage.

- **Split Ratio:** 70% Train, 15% Validation, 15% Test.
- **Splitting Method:** Stratified Split (retaining class proportions across splits).
- **Justification:** Emotion categories will be highly imbalanced. Simple random splitting might result in validation/test sets with zero or very few samples of rare classes (e.g., Fear).

---

## 3.4 Handling Class Imbalance

Emotion datasets are naturally imbalanced (Joy and Anger dominate; Fear and Disgust are rare).

### Conditional Imbalance Strategy:

- **IF** Class Imbalance is extreme (minority class $< 5\%$ of total samples):
  - **Action:** Avoid Undersampling (it will discard too much valuable data from majority classes). Instead:
    1. Apply **Class Weights** (`class_weight='balanced'`) in ML models (Logistic Regression, SVM, Random Forest).
    2. Use **Macro F1-Score** as the optimization objective in hyperparameter tuning (GridSearchCV), rather than Accuracy.
  - **Why:** Class weights penalize errors on minority classes proportionally to their scarcity, forcing the decision boundary to accommodate them without losing training samples.

- **IF** Class Imbalance is moderate (minority class is between $5\%$ and $15\%$ of total samples):
  - **Action:** Experiment with **Random Under-Sampling (RUS)** on the training set to create a balanced subset (e.g., 500 samples per class).
  - **Why:** Balances the decision boundary directly.
  - **Constraint:** Never resample the validation or test sets. They must remain in their natural distribution to reflect real-world performance.
