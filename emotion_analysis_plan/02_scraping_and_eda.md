# 02 — Scraping and Exploratory Data Analysis (EDA)

## Scraping Strategy and EDA Targets

---

## 2.1 Scraping Strategy

The project must collect or utilize a dataset of Indonesian text with rich emotional signals. 

### Sourcing Options
- **Source A (Internal):** The existing Google Play Store reviews for Twitter/X (`com.twitter.android`).
- **Source B (External):** Multi-app Play Store reviews (e.g., banking, gaming, social media) to capture a broader range of emotions.
- **Source C (Public Dataset):** Indonesian Emotion Twitter Dataset (e.g., EmoT).

### Conditional Scraping Logic:
- **IF** the project constraints require reusing the Twitter/X dataset:
  - **Action:** Extract the raw review text from `data/raw/reviews_playstore_indonesia.csv`. Do not scrape new data.
  - **Why:** Saves computational time and allows direct comparison of pre-processing changes from the sentiment project.
- **IF** the project allows new scraping and aims for class balance across rare emotions (e.g., Fear, Disgust):
  - **Action:** Scrape 10,000 new reviews from diverse app categories:
    - *Finance/Banking Apps* (e.g., BCA Mobile, Bibit) to capture **Fear** (security, money loss) and **Anger** (system downtime).
    - *Mobile Gaming Apps* (e.g., Mobile Legends) to capture **Joy** (winning, fun) and **Anger/Disgust** (toxic players, lag).
    - *Social Media Apps* (e.g., TikTok, Instagram) to capture **Joy**, **Sadness**, and **Disgust**.
  - **Why:** Social media reviews of a single app (Twitter/X) are heavily biased towards tech complaints (Anger/Sadness) and spam (neutral), making it hard to find genuine Joy or Fear.

---

## 2.2 Exploratory Data Analysis (EDA) Framework

Before feature extraction or model training, the dataset's characteristics must be analyzed to guide preprocessing and modeling choices.

### 1. Label Distribution Analysis
- **What to do:** Plot the count and percentage of each emotion class in the labeled dataset.
- **Why it is done:** Identifies the level of class imbalance.
- **Conditional Logic:**
  - **IF** the ratio of the majority class to the minority class is $> 4:1$ (e.g., Joy has 2000 samples, Fear has 200):
    - **Decision:** Implement class-imbalance countermeasures during training (e.g., Random Under-Sampling, SMOTE, or class-weighted loss functions).
  - **IF** the ratio is $\le 4:1$:
    - **Decision:** Use standard stratified splitting without resampling, as the imbalance is mild enough for standard classifiers.

### 2. Text Length and Word Count Analysis
- **What to do:** Calculate the character and word length distribution of reviews, segmented by emotion class.
- **Why it is done:** Influences feature extraction parameters (like `max_df` and `min_df` in TF-IDF, or average vector logic in Word2Vec).
- **Conditional Logic:**
  - **IF** the median review length is $< 10$ words:
    - **Decision:** Avoid high n-grams (e.g., trigrams) in TF-IDF as they will be extremely sparse. Rely on unigrams and bigrams.
  - **IF** the median review length is $> 30$ words:
    - **Decision:** Average Word2Vec embeddings will likely lose specific emotional cues due to vector dilution. Implement TF-IDF weighted Word2Vec instead of simple average Word2Vec.

### 3. Emotion-Specific Keyword Association (Chi-Square or Mutual Information)
- **What to do:** Identify words most highly associated with each emotion category using Chi-Square ($\chi^2$) feature selection or Mutual Information scoring.
- **Why it is done:** Validates the semantic integrity of the dataset. If "kecewa" (disappointed) is highly associated with *Joy*, it flags systematic labeling errors.
- **Action on failure:** If keywords overlap heavily across distinct classes (e.g., "bagus" appearing in both *Joy* and *Anger* due to sarcasm), document this in `08_error_analysis_and_linguistics.md` and design custom normalizations.

### 4. Overlap and Co-occurrence Analysis
- **What to do:** In a subset of manually annotated data, check if reviews express multiple emotions (e.g., Anger + Sadness in "Aplikasi ini rusak [Anger], kecewa sekali saya [Sadness]").
- **Why it is done:** Determines if a single-label assumption is valid.
- **Conditional Logic:**
  - **IF** multiple emotions co-occur in $> 15\%$ of reviews:
    - **Decision:** Establish a hierarchy of dominance for labeling: `Anger > Sadness > Disgust > Fear > Joy`. Anger takes precedence for labeling because it represents the most critical actionable feedback.
  - **IF** co-occurrence is negligible ($< 5\%$):
    - **Decision:** Treat as mutually exclusive single-label classification.
