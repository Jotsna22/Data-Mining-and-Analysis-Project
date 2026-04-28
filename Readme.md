# Trust-Aware Fake News Detection using Semantic and Stylistic Features

Fake news spreads rapidly online and can influence public opinion, decisions, and real-world events. At scale, manual fact-checking is simply not feasible. Most existing detection systems focus only on **what is written** — the semantic meaning of the text. This project goes further by asking: does the **way** fake news is written also give it away? We build a hybrid deep learning pipeline that combines **DistilBERT semantic embeddings** with **7 handcrafted stylistic features** (sentiment, readability, punctuation patterns, capitalization, and lexical diversity) in a late-fusion architecture. The result is a model that understands both the *content* and the *writing behavior* of an article — and the results confirm that combining both signals meaningfully outperforms using semantics alone.

---

## Project Video

>  **[Watch the project walkthrough here](#https://youtu.be/OlQ9bhyapuk?si=dt9_oNR_eyZy3HIa)**


---

## Main Deliverable

The main deliverable is **`main_notebook.ipynb`** — it contains the full pipeline end-to-end: EDA, feature engineering, model architecture, training, evaluation, and results.

---

## Research Question

> **How do semantic representations (DistilBERT) compare with stylistic features in fake news detection, and does combining both lead to more accurate and interpretable predictions?**

Specifically, I investigated whether:
- **Semantic features** capture *meaning* (via DistilBERT `[CLS]` embeddings)
- **Stylistic features** capture *writing behavior* (sentiment, readability, punctuation, caps, lexical diversity)
- **Combining both** improves classification performance over either alone

---

## Results Summary

Our fusion model achieves **99.17% accuracy** and a **99.08% F1 score** on the held-out test set — outperforming the semantic-only baseline by over **4 percentage points** across every metric, confirming that writing style is a meaningful and measurable signal for fake news detection.

| Metric | Baseline (TF-IDF + Logistic Regression) | Ours (DistilBERT + Stylistic Fusion) |
|---|---|---|
| Accuracy | ~95.00% | **99.17%** |
| Precision | ~95.00% | **98.99%** |
| Recall | ~95.00% | **99.17%** |
| F1 Score | ~95.00% | **99.08%** |

---

## Repo Structure

```
├── main_notebook.ipynb        # 👈 Main deliverable — full pipeline
├── WELFake_Dataset.csv        # Dataset (download link below — not included due to size)
├── requirements.txt           # Full list of pinned dependencies
└── README.md
```

---

## 📦 Dataset

**WELFake Dataset**
- 72,134 news articles combining multiple prior fake-news sources to reduce overfitting to a single distribution
- **4 columns:** serial number, title, text, label
- **Label convention used in this project:** `0 = Real`, `1 = Fake`
- **Download:** [https://zenodo.org/records/4561253](https://zenodo.org/records/4561253)

### Preprocessing Steps
1. Standardized column names; dropped missing and duplicate rows
2. Combined **title + body text** into a single `combined_text` field
3. Extracted 7 stylistic features per article using TextBlob and textstat
4. Applied `StandardScaler` normalization to all stylistic features
5. Tokenized text with the DistilBERT tokenizer (max length 128, padding + truncation)
6. Stratified 80/20 train/validation split (`random_state=42`)

> ⚠️ The dataset CSV is not included in this repo due to file size. Download it from the Zenodo link above and place it in the root directory as `WELFake_Dataset.csv` before running.

---

## 🔁 How to Reproduce

This project was built and run on **Google Colab** with a GPU runtime (recommended).

1. Clone this repository
2. Download `WELFake_Dataset.csv` from [Zenodo](https://zenodo.org/records/4561253) and place it in the root folder
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Open **`main_notebook.ipynb`** in Colab or Jupyter and **run all cells top to bottom** — the notebook is fully self-contained with no separate scripts to run

> 💡 In Colab: go to **Runtime → Change runtime type → T4 GPU** before running for significantly faster DistilBERT training.

---

## 🛠️ Key Dependencies

| Package | Version |
|---|---|
| Python | 3.12.13 |
| torch | 2.2.0 |
| transformers | 4.x |
| scikit-learn | 1.4.1 |
| pandas | 2.2.0 |
| numpy | 1.26.x |
| textblob | 0.19.0 |
| textstat | 0.7.13 |
| seaborn | 0.13.x |
| matplotlib | 3.8.x |

> The full pinned list of every package is in **`requirements.txt`**.


---


## Pipeline Overview

### 1. Exploratory Data Analysis
- Class distribution check
- Article length analysis (title and body word count) by class
- Top unigrams and bigrams per class (TF-IDF based)

### 2. Feature Engineering

#### Semantic Features
- Text is tokenized and encoded using **DistilBERT** (`distilbert-base-uncased`), producing a 768-dimensional `[CLS]` embedding per article.

#### Stylistic Features (7 handcrafted signals)
| Feature | Description |
|---|---|
| Sentiment Polarity | Overall emotional tone of the text |
| Subjectivity | Degree of opinion vs. factual language |
| Readability | Flesch reading ease score |
| Exclamation Count | Number of `!` marks |
| Question Count | Number of `?` marks |
| Caps Ratio | Proportion of ALL CAPS words |
| Lexical Diversity | Type-token ratio (vocabulary variety) |

Fake news was found to have **higher subjectivity, simpler readability, more punctuation, greater ALL CAPS usage, and lower lexical diversity** than real news — consistent with content designed to provoke rather than inform.

### 3. Model Architecture: Late Fusion

```
Text Input
    │
    ▼
DistilBERT Encoder
    │
[CLS] Embedding (768-dim)
    │                         Stylistic Features (7-dim)
    │                                │
    │                         Linear(7 → 32) + ReLU + Dropout
    │                                │
    └──────────── Concat (800-dim) ──┘
                        │
               Linear(800 → 128) + ReLU + Dropout
                        │
               Linear(128 → 2)  →  Fake / Real
```

Each modality has a **dedicated encoding pathway** before fusion, ensuring stylistic signals are not drowned out by the high-dimensional transformer output.

### 4. Training

| Setting | Value |
|---|---|
| Base Model | `distilbert-base-uncased` |
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Batch Size | 16 |
| Epochs | 3 |
| Loss | CrossEntropyLoss |
| Train / Val Split | 80% / 20% (stratified) |
| Max Token Length | 128 |

Training loss: `0.0572 → 0.0169 → 0.0075` — steady convergence with no instability.

### 5. Uncertainty Estimation: MC Dropout
During inference, dropout is kept active and the model runs **T = 10** forward passes per sample. The mean probability is used as the final prediction and the variance serves as an **uncertainty score** — flagging borderline cases that may warrant human review.

---

## Results

| Metric | Baseline (TF-IDF + LR) | Our Model (DistilBERT + Fusion) |
|---|---|---|
| Accuracy | ~95.00% | **99.17%** |
| Precision | ~95.00% | **98.99%** |
| Recall | ~95.00% | **99.17%** |
| F1 Score | ~95.00% | **99.08%** |

---

## Limitations

- **Text-only:** The model does not process images or URLs, both of which are important vectors for misinformation.
- **Static dataset:** Performance may degrade on newer fake news styles without retraining.
- **No URL signals:** Domain reputation and URL structure are strong credibility indicators that are currently unused.

---

## Future Work

The natural next step is a **multimodal fake news detector** that extends this late fusion design with:

- **Vision encoder (CLIP / ViT)** — detect image manipulation and image-text inconsistencies
- **URL feature encoder** — extract domain reputation, TLD type, and structural URL signals
- **Cross-modal consistency modeling** — explicitly penalize mismatches between visual and textual content

---


## Key References

- Devlin et al. (2019) — *BERT: Pre-training of Deep Bidirectional Transformers*
- Sanh et al. (2019) — *DistilBERT, a distilled version of BERT*
- Gal & Ghahramani (2016) — *Dropout as a Bayesian Approximation (MC Dropout)*
- WELFake Dataset — Verma et al. (2021)
