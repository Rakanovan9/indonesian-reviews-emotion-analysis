# 01 — Problem Definition

## From Sentiment to Emotion: A Redefinition

---

## 1.1 What Changed and Why It Matters

The prior project solved a **sentiment classification** problem:  
> "Does this review express a positive, negative, or neutral attitude?"

This project solves an **emotion detection** problem:  
> "What specific emotion does the author of this review express?"

These are fundamentally different tasks, even though both operate on the same type of text (Indonesian Play Store reviews):

| Dimension | Sentiment Analysis | Emotion Analysis |
|---|---|---|
| Granularity | Coarse (3 classes) | Fine-grained (4–6 classes) |
| Label signal | Rating stars (distant supervision) | Lexicon or manual annotation |
| Ambiguity | Low–medium | High (emotions co-occur) |
| Class distribution | Skewed (majority: Negative) | Highly skewed (majority: Joy/Anger) |
| Linguistic challenge | Negation, sarcasm | Blended emotions, irony |

---

## 1.2 Emotion Class Definition

### Selected Taxonomy: 5-Class Emotion Model

| Label | Indonesian Name | Typical Linguistic Signals in Play Store Reviews |
|---|---|---|
| **Joy** | Senang / Puas | "bagus", "keren", "mantap", "puas banget", "suka" |
| **Anger** | Marah / Kesal | "kesal", "nyebelin", "gak bisa", "benci", "sialan" |
| **Sadness** | Sedih / Kecewa | "kecewa", "sedih", "nyesel", "kenapa begini" |
| **Fear** | Takut / Khawatir | "khawatir", "takut", "bahaya", "data bocor" |
| **Disgust** | Jijik / Muak | "muak", "bosan", "capek", "gak betah" |

> **Why 5 classes?**  
> The NRC Emotion Lexicon (Plutchik's wheel) and most Indonesian emotion research (EmoT dataset) use a 5–6 class schema. "Surprise" is excluded because it rarely appears distinctly in Play Store reviews and is most often conflated with Joy or Fear.

### Class Consolidation Decision

If the chosen dataset provides a different number of classes:

- **IF 4 classes** (Joy, Anger, Sadness, Fear): Drop Disgust. Merge Disgust into Anger if needed. Justification: reducing noise from poorly defined class.
- **IF 6 classes** (add Surprise): Keep only if Surprise has ≥ 200 samples in training set, otherwise merge with Joy or Fear based on context.
- **IF 3 classes** (Positive/Negative/Neutral mapped to emotion): This is NOT emotion analysis. Reject and re-label.

---

## 1.3 Impact on Labels

### Previous Labeling Scheme (Sentiment):
- Rating 1 → Negative
- Rating 3 → Neutral  
- Rating 5 → Positive
- Ratings 2, 4 → Discarded

### New Labeling Requirement (Emotion):
Rating stars are **insufficient proxies** for emotion because:
- A 1-star review can express Anger ("Aplikasi ini bikin kesal!") or Sadness ("Kecewa sekali, dulu bagus...")
- A 5-star review can express Joy ("Mantap!") or neutral satisfaction (not emotion)
- Rating does not distinguish Disgust from Fear

**Conclusion:** The existing distant supervision scheme cannot be extended to emotion labels.  
A new labeling approach is required. See `03_dataset_strategy.md` for full decision logic.

---

## 1.4 Impact on Models

| Factor | Impact |
|---|---|
| More classes (5 vs 3) | Higher baseline classification difficulty; random chance drops from 33% to 20% |
| More class imbalance | Minority classes (Fear, Sadness) may be under-represented; weighted strategies required |
| Higher linguistic ambiguity | TF-IDF bag-of-words representations lose contextual nuance; word embedding becomes relatively more important |
| No Deep Learning requirement | Task scope per syllabus (Topic 2) only requires ML + word embedding comparison; no BERT fine-tuning needed |

---

## 1.5 Impact on Evaluation

- **Accuracy becomes misleading** at high class imbalance: a model predicting only "Joy" can achieve high accuracy on a dataset dominated by Joy reviews.
- **Macro F1 remains the primary metric** for the same reason as the sentiment project: equal weighting across all 5 classes.
- **Per-class F1 is mandatory** because the minority classes (Fear, Sadness) are the hardest and most informative to evaluate.
- **Confusion matrix** becomes more complex (5×5) and must be analyzed for systematic confusion between similar emotions (e.g., Anger ↔ Disgust, Sadness ↔ Fear).

---

## 1.6 Research Questions (for Paper)

1. Can TF-IDF-based ML models effectively classify fine-grained emotions in informal Indonesian text?
2. Does word embedding (Word2Vec) improve emotion classification performance over TF-IDF, particularly for minority emotional classes?
3. Which emotion classes are systematically confused by ML classifiers, and why?
4. How does class imbalance in emotion data affect model fairness across emotion categories?
