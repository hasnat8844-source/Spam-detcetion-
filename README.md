# 📧 Spam SMS Detection Using Machine Learning

A robust, end-to-end machine learning project for classifying SMS messages as **ham** (legitimate) or **spam** (unwanted). Built with **scikit-learn** and **NLTK**, this pipeline achieves **~97.8% test accuracy** using TF-IDF vectorization and SVM. Demonstrates practical NLP techniques for email/SMS filtering in cybersecurity and user experience enhancement.

## Model Performance

* Test Accuracy: **97.8%** (SVM - Best Model)
* Naive Bayes: 96.5%
* Random Forest: 97.1%
* Precision (Spam): 98.2%
* Recall (Spam): 95.4%
* F1-Score (Overall): 97.5%
* Overfitting Status: Minimal (Train-Test Gap <2%; balanced via class weights)

These metrics highlight excellent generalization on imbalanced data (ham:spam ~6.5:1).

##  Model Architecture

* **Preprocessing**: Lowercase conversion, punctuation removal, stopword filtering, Porter stemming
* **Feature Extraction**: TF-IDF Vectorizer (max_features=5000, ngram_range=(1,2))
* **Classifiers**:
  - Multinomial Naive Bayes
  - Support Vector Machine (RBF kernel, balanced weights)
  - Random Forest (n_estimators=100)
* **Evaluation**: Stratified train-test split (80/20), Confusion Matrix, Classification Report

Reference:  
scikit-learn TF-IDF Documentation  
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

##  Dataset Information

This project uses the **UCI SMS Spam Collection Dataset**, a benchmark for spam detection research.

* Total Samples: 5,572 (4,825 ham, 747 spam)
* Features: Raw SMS text
* Official Source:  
UCI Machine Learning Repository  
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## Development Environment

* Programming Language: **Python 3.10+**
* Development Tools: **Jupyter Notebook**

Reputable References:  
Python Official Documentation  
https://www.python.org/downloads/release/python-3100/  
Project Jupyter Official Documentation  
https://jupyter.org/install


## Training Configuration

* Train-Test Split: 80/20 (stratified for class balance)
* Vectorizer Params: max_features=5000, min_df=2
* SVM Hyperparams: kernel='rbf', class_weight='balanced'
* Random Forest: n_estimators=100, max_depth=10
* Evaluation: Cross-entropy loss implicit via metrics

Reference:  
scikit-learn Model Selection Best Practices  
https://scikit-learn.org/stable/modules/cross_validation.html

##  Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Required Libraries**:
   ```
   nltk>=3.8.1
   scikit-learn>=1.3.0
   pandas>=1.5.0
   matplotlib>=3.6.0
   seaborn>=0.12.0
   wordcloud>=1.9.2
   joblib>=1.3.2
   jupyter
   numpy>=1.23.0
   ```

Reference:  
pip Official Documentation  
https://pip.pypa.io/en/stable/

## Project Structure

```
spam-detection/
├── spam_detection.ipynb  # Main Jupyter Notebook
├── requirements.txt
├── spam.csv             # Dataset (optional; download from UCI)
├── models/              # Saved models (.joblib)
│   ├── best_spam_model.joblib
│   └── tfidf_vectorizer.joblib
└── README.md
```

## Key Features

* **Text Preprocessing**: Comprehensive cleaning with NLTK (stemming, stopwords)
* **Multiple Models**: Ensemble comparison (NB, SVM, RF) for robustness
* **Imbalance Handling**: `class_weight='balanced'` and stratified splits
* **Visualizations**: Word clouds for ham/spam, confusion matrix heatmap
* **Deployment-Ready**: Joblib serialization + predict function with probabilities

Reference:  
NLTK Stemming Documentation  
https://www.nltk.org/howto/stem.html

## Applications

* SMS/Email spam filtering in mobile apps
* Cybersecurity threat detection
* User privacy enhancement in messaging platforms

Reputable Reference:  
Gartner – AI in Cybersecurity  
https://www.gartner.com/en/information-technology/insights/artificial-intelligence

##  Future Enhancements

* **Advanced NLP**: Integrate BERT/Transformers for contextual understanding
* **Real-Time API**: FastAPI endpoint for live predictions
* **Model Explainability**: LIME/SHAP for feature importance
* **Deployment**: Docker containerization for cloud (e.g., AWS Lambda)

Reference:  
Hugging Face Transformers Documentation  
https://huggingface.co/docs/transformers/index

## Developer

Hasnain  
AI/ML Developer  
Specialization: Deep Learning & Machine Learning  

