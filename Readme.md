# Fake News Detection using Data Mining Techniques

## 📌 Project Overview
This project explores the **WELFake Fake News Dataset** using data mining and machine learning techniques.  
The goal is to analyze patterns in news articles and build models that can classify news as **fake or real**.

The project is part of a Data Mining and Analysis course and is designed to demonstrate:
- Exploratory Data Analysis (EDA)
- Feature extraction from text data
- Classification using traditional and advanced methods
- Interpretation of results using explainable AI techniques

This repository will be updated incrementally as the course progresses.

---

## 📂 Dataset
**WELFake: A Dataset for Fake News Detection in Text Data**  
Source: https://zenodo.org/records/4561253  

- Approximately 72,000 news articles  
- Columns: `id`, `title`, `text`, `label`  
- Label: `0 = fake`, `1 = real`

---

## 🎯 Project Objectives
- Perform exploratory data analysis (EDA) on fake and real news articles
- Study text length, word frequency, and class distribution
- Apply text mining techniques such as TF-IDF and n-grams
- Build baseline classification models (Naive Bayes, Logistic Regression, SVM)
- Explore external techniques such as:
  - Transformer-based models (BERT / RoBERTa)
  - Topic modeling (LDA / BERTopic)
  - Explainable AI (SHAP, attention visualization)
- Compare traditional and modern approaches

---

## 🛠️ Tools and Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- NLTK / spaCy
- Jupyter Notebook (Google Colab)

Optional (future work):
- PyTorch / HuggingFace Transformers
- SHAP
- BERTopic

---

## 📊 Current Progress
- [x] Dataset loading and preprocessing  
- [x] Exploratory Data Analysis (EDA)  
- [x] Class distribution and text length analysis  
- [ ] Feature extraction (TF-IDF, n-grams)  
- [ ] Baseline classification models  
- [ ] Advanced models (BERT / topic modeling)  
- [ ] Model evaluation and interpretation  


---

## 🔍 Example Research Questions
- How does article length differ between fake and real news?
- How does TF-IDF compare with transformer-based embeddings for classification?
- What words or phrases contribute most to predicting fake news?
- Can topic modeling reveal hidden themes in misinformation?

---

## ⚖️ Ethical Considerations
- Misclassification of real news as fake can harm credibility and trust.
- Dataset labels may contain bias based on sources or topics.
- Models should be interpreted carefully and not used for censorship or political manipulation.


