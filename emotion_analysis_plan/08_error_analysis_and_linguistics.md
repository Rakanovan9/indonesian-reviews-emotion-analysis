# 08 — Error Analysis and Linguistics

## Diagnostic Framework for Linguistic Challenges

---

## 8.1 Common Indonesian Linguistic Challenges in Reviews

Unlike standard Bahasa Indonesia, Play Store reviews are heavily colloquial, concise, and filled with grammatical compromises. We anticipate four primary linguistic challenges for our ML models:

### 1. Sarcasm & Irony
- **Example:** "*Bagus banget aplikasinya, baru buka langsung logout sendiri, mantap!*"
- **Linguistic Nature:** Positive words (*bagus*, *mantap*) are used with negative intentions.
- **Model Behavior:** TF-IDF models will likely classify this as **Joy** due to high term frequency of positive adjectives. Word2Vec might fail similarly if context vectors are averaged.

### 2. Negation Shifts
- **Example:** "*Saya tidak khawatir lagi kalau mau transaksi, aman.*" vs. "*Khawatir sekali kalau mau transaksi.*"
- **Linguistic Nature:** The word "*tidak*" flips the emotional valence of "*khawatir*" (Fear $\rightarrow$ Neutral/Joy).
- **Model Behavior:** Unigram TF-IDF fails because it treats "*tidak*" and "*khawatir*" as independent features. Bigram features are required to preserve the relationship.

### 3. Out-of-Vocabulary (OOV) Slang and Typos
- **Example:** "*anjg*", "*kesel bgt*", "*pny*".
- **Linguistic Nature:** Frequent character omissions, sound changes, or vulgar slang representing intense emotions.
- **Model Behavior:** Word2Vec cannot represent unknown tokens; TF-IDF creates separate, sparse columns for each typo, reducing generalization.

### 4. Blended Emotions
- **Example:** "*Kecewa banget [Sadness] dan kesel [Anger] kenapa transaksinya gagal terus.*"
- **Linguistic Nature:** The review expresses both sadness and anger simultaneously.

---

## 8.2 Error Typology Logging Framework

To perform a rigorous error analysis (Notebook 07), the developer must extract 100 randomly sampled misclassified cases from the validation set and classify them into the following error categories:

| Error Type | Description |
|---|---|
| **Type A: Sarcasm** | Text uses positive vocabulary to mock or criticize the application. |
| **Type B: Negation Failure** | Model missed negation indicators, predicting the inverse of the actual emotion. |
| **Type C: OOV Slang / Typo** | Critical emotion words were not normalized or recognized by the vocabulary. |
| **Type D: Label Noise** | The ground-truth label was incorrect (e.g. human annotator mistake or rating mismatch). |
| **Type E: Weak / Blended Signal** | Text is highly ambiguous or expresses multiple emotions with equal strength. |

---

## 8.3 Conditional Mitigation Strategy

Based on the distribution of the 100 logged errors, apply the following adjustments:

### Scenario 1: Slang is the primary culprit
- **IF** Type C (OOV Slang) represents $> 30\%$ of total misclassifications:
  - **Action:** Update the normalization dictionary in the preprocessing script. Run a frequency analysis on all words in the corpus that are OOV to identify missed slang variations (e.g., mapping *kezel*, *kzyl* to *kesal*).
  - **Why:** Re-concentrates semantic signal on root words.

### Scenario 2: Sarcasm or Negation is the primary culprit
- **IF** Type A (Sarcasm) or Type B (Negation) represents $> 40\%$ of total misclassifications:
  - **Action:** 
    1. Force the TF-IDF Vectorizer to use bigrams exclusively or a range of `ngram_range=(1,2)`.
    2. Try TF-IDF weighted Word2Vec instead of simple averages, or evaluate if a rule-based negation detector (e.g., prefixing words after "tidak" with `TIDAK_`) improves the ML classifiers.
  - **Why:** Bag-of-words models need contiguous word structures to capture context inversion.

### Scenario 3: Label Noise is the primary culprit
- **IF** Type D (Label Noise) represents $> 25\%$ of total misclassifications:
  - **Action:** Re-annotate the validation and test datasets. Do not retrain on noisy datasets.
  - **Why:** Model performance cannot exceed the quality of the evaluation set. Noisy test labels lead to misleading evaluation metrics.
