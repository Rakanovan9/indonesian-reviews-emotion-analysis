# 05 — Feature Extraction Strategy

## Representing Text for Machine Learning Classifiers

---

## 5.1 Overview

We must convert preprocessed Indonesian review text into numerical feature vectors that machine learning classifiers can process. The project syllabus (Topic 2) requires a comparative analysis between **TF-IDF** and **Word Embedding (Word2Vec)**.

---

## 5.2 TF-IDF Representation

TF-IDF measures how important a word is to a document in a collection.

### Configurations to Evaluate
1. **Unigram TF-IDF:** Capture single-word occurrences.
2. **Bigram TF-IDF:** Capture two-word sequences (e.g., "*tidak kecewa*", "*kurang bagus*").
3. **Combined TF-IDF (Unigram + Bigram):** Capture both individual words and immediate word pairs.

### Parameter Tuning Logic:
- **`sublinear_tf=True`:** Compress term frequency logarithmically ($1 + \log(tf)$) to prevent long reviews with repeated emotional words from dominating the feature space.
- **`min_df=2`:** Ignore words that appear in only 1 review. Prevents overfitting on typos.
- **`max_df=0.9`:** Ignore terms that appear in more than 90% of reviews. This filters out universal words that carry no discriminative emotional power.

---

## 5.3 Word Embedding (Word2Vec)

Word2Vec maps words to dense vector spaces where semantic similarities are represented as geometric distances.

### CBOW vs. Skip-Gram Selection Logic:
- **IF** the dataset is dominated by short reviews with frequent common slang words (informal style):
  - **Decision:** Use **Continuous Bag of Words (CBOW)** with a window size of $5$.
  - **Why:** CBOW is faster to train and has slightly better representation for highly frequent words.
- **IF** the dataset contains diverse vocabulary and rare emotional words (e.g., specific terms for Fear/Disgust):
  - **Decision:** Use **Skip-Gram** with a window size of $5$ and negative sampling.
  - **Why:** Skip-gram is better at representing rare words because it predicts context from target words.

### Training Strategy:
- **IF** Training Corpus Size $N \ge 10,000$ reviews:
  - **Action:** Train Word2Vec from scratch using the `gensim` library on the processed training set. 
  - **Parameters:** `vector_size=100`, `window=5`, `min_count=2`, `workers=4`.
  - **Why:** The corpus is large enough to learn domain-specific word associations for Play Store applications.
- **IF** Training Corpus Size $N < 10,000$ reviews:
  - **Action:** Load a pre-trained Indonesian Word2Vec model (e.g., Wikipedia-derived or fastText Indonesian vectors) and use it as a static embedding layer.
  - **Why:** Small datasets do not contain enough co-occurrence variety to build stable vector spaces, resulting in low-quality word clusters.

---

## 5.4 Vector Aggregation (Document Embedding)

Because ML models require a single vector per document, individual word vectors must be merged:

### Aggregation Choices:
1. **Simple Average Vector:**
   $$V_{doc} = \frac{1}{|D|} \sum_{w \in D} V_w$$
2. **TF-IDF Weighted Average Vector:**
   $$V_{doc} = \frac{1}{\sum_{w \in D} \text{TF-IDF}(w)} \sum_{w \in D} \text{TF-IDF}(w) \cdot V_w$$

### Conditional Aggregation Logic:
- **IF** median review length is short ($\le 15$ words):
  - **Decision:** Use **Simple Average Vector** as the baseline.
  - **Why:** With short texts, the vector space is dense and less susceptible to semantic dilution.
- **IF** median review length is longer ($> 15$ words):
  - **Decision:** Use **TF-IDF Weighted Average Vector**.
  - **Why:** It weights emotional keywords higher (which have higher TF-IDF weights) and downweights general words, preventing the document vector from drifting toward a neutral average.

---

## 5.5 Out-of-Vocabulary (OOV) Handling

Word2Vec cannot generate vectors for words unseen during training (or absent from pre-trained vocabularies).

### OOV Mitigation Logic:
- **Action:** When aggregating, ignore any OOV word (do not add it to the sum or count).
- **IF** a review contains *only* OOV words:
  - **Action:** Return a zero vector of dimension $D$ (e.g., $100$-dimension zero vector). Do not drop the row.
  - **Why:** Dropping rows during feature extraction breaks the evaluation pipeline alignment and causes array size mismatch in validation splits.
