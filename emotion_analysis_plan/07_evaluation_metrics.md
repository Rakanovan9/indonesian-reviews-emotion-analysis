# 07 — Evaluation Metrics and Validation

## Assessing Model Performance on Class-Imbalanced Data

---

## 7.1 Selection of Metrics

Evaluating a multi-class model (5 classes) on highly imbalanced text data requires metrics that prioritize performance on minority classes as much as majority ones.

### 1. Macro F1-Score (Primary Optimization Metric)
- **Calculation:**
  $$\text{Macro F1} = \frac{1}{C} \sum_{c=1}^{C} F1_c$$
  where $F1_c$ is the F1-score for class $c$ (the harmonic mean of precision and recall for that specific class).
- **Why it is done:** Macro F1 treats all classes equally, regardless of their support size. A model that predicts the majority class (*Joy*) perfectly but fails on the minority class (*Fear*) will receive a low Macro F1-Score, alerting developers to class neglect.

### 2. Per-Class Precision and Recall
- **Why it is done:**
  - **Precision** ($\frac{TP}{TP + FP}$) measures purity: "When the model predicts *Anger*, how often is it correct?" Useful if false alarms are costly.
  - **Recall** ($\frac{TP}{TP + FN}$) measures coverage: "How many of the actual *Anger* reviews did the model capture?" Useful to avoid missing critical negative feedback.

### 3. Classification Accuracy
- **Why it is done:** Standard metric for general reference, but **never** used for model selection or comparison.
- **Why Accuracy is dangerous:** If the test set contains 70% *Joy* reviews, a dummy model that predicts *Joy* for everything will achieve 70% accuracy, while having a Macro F1 of only $\approx 0.16$.

---

## 7.2 Confusion Matrix Diagnostic Guide

A $5 \times 5$ confusion matrix must be plotted for the best performing model. The matrix should be normalized by row (true labels) to show recall rates along the diagonal.

```
                  PREDICTED LABELS
               Joy   Anger  Sadness  Fear   Disgust
       Joy    [ 85%    5%      5%     3%      2%   ]
T    Anger    [  5%   75%     10%     2%      8%   ]
R  Sadness    [  4%   15%     70%     8%      3%   ]
U     Fear    [  5%   10%     25%    55%      5%   ]
E  Disgust    [  5%   30%      5%     5%     55%   ]
```

### Diagnostic Decision Logic:

- **IF** there is high mutual confusion between **Anger** and **Disgust** ($> 20\%$):
  - **Action:** Check normalization dictionaries. Ensure that disgust terms (e.g. *muak*, *jijik*, *mual*) are not mapped to anger terms, and verify if human annotators are conflating the two.
- **IF** **Fear** is mostly misclassified as **Sadness**:
  - **Action:** Investigate text lengths. Fear statements in Play Store reviews (e.g. "*Takut uang hilang*") are often accompanied by disappointment markers ("*kecewa sekali*"). If the model is TF-IDF, the longer sentence may trigger the stronger TF-IDF weight of sadness keywords.
- **IF** **Sadness** is mostly misclassified as **Neutral**:
  - **Action:** Verify stopword lists. Some weak sadness markers (e.g., "*sayang sekali*", "*sayangnya*") might have been stripped as stopwords, leaving the text semantically empty (neutral). Add them back to the whitelist.

---

## 7.3 Model Comparison Standard

To draw valid conclusions for the paper, the final results table must take this format:

| Representation | Model | Macro F1 | Accuracy | Per-Class F1 (Joy / Anger / Sadness / Fear / Disgust) |
|---|---|---|---|---|
| *Baseline* | Dummy (Stratified) | | | |
| **TF-IDF Combined** | Logistic Regression | | | |
| **TF-IDF Combined** | LinearSVC | | | |
| **TF-IDF Combined** | Multinomial NB | | | |
| **TF-IDF Combined** | Random Forest | | | |
| **Word2Vec Avg** | Logistic Regression | | | |
| **Word2Vec Avg** | LinearSVC | | | |
| **Word2Vec Avg** | Random Forest | | | |
| **Word2Vec TF-IDF**| Logistic Regression | | | |
| **Word2Vec TF-IDF**| LinearSVC | | | |
| **Word2Vec TF-IDF**| Random Forest | | | |
