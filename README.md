# 📚 Kindle Review Sentiment Analysis using Word2Vec & ML Algorithms

## Project Summary

This project aims to perform sentiment analysis on Kindle product reviews using various Natural Language Processing (NLP) techniques and machine learning models. The goal is to predict the **sentiment (positive or negative)** of a review based on the **reviewText** field.


##  Dataset Overview

- **Source**: Kaggle
- **Shape**: `12000 rows × 11 columns`
- **Target Feature**: `rating` (converted to binary: `1` for ratings > 3, `0` for ratings < 3)


##  Data Preprocessing & Cleaning

1. Converted `reviewText` to lowercase using `str.lower()`.
2. Removed:
   - Special characters
   - Stopwords (using NLTK)
   - URLs and HTML tags
   - Extra whitespace
3. Applied **lemmatization** to normalize words.


##  Feature Engineering

Performed separate feature extraction methods:

### 1. Bag of Words (BoW)
- Created BoW vectors using `CountVectorizer`
- Applied **Gaussian Naive Bayes**
- **Accuracy**: `58%`

### 2. TF-IDF (Term Frequency-Inverse Document Frequency)
- Used `TfidfVectorizer`
- Applied **Gaussian Naive Bayes**
- **Accuracy**: `59%
  

##  Word Embeddings

### 3. Word2Vec (Average Word Vectors)
- Used Gensim's `Word2Vec` model
- Computed average word vectors for each review
- Applied machine learning models:
  - **Logistic Regression**: `Accuracy = 76%`
  - **XGBoost Classifier**: `Accuracy = 75%`
  - **SVC (Support Vector Classifier)**: `Accuracy = 76%`


## ✅ Conclusion

- Word2Vec significantly outperformed BoW and TF-IDF.
- Accuracy using vector-based models (Word2Vec + ML) stayed within the range of **75–78%**.
- Final model comparison suggests:
  - Logistic Regression and SVC are most effective with Word2Vec embeddings in this task.


## 🚀 Future Improvements

- Try **pretrained Word2Vec** or **GloVe** embeddings
- Use **deep learning models** (LSTM, GRU)
- Experiment with **transformers** (like BERT)

