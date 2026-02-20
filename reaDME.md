

# 🛒 Digikala Sentiment Analysis: Classical ML vs. LSTM

This repository explores **Sentiment Analysis** on Persian (Farsi) user reviews from **Digikala**. The project demonstrates a complete NLP pipeline, transitioning from traditional machine learning (Logistic Regression) to deep learning (LSTM) using **PyTorch**.

## 🎯 Project Objective

The goal is to classify customer reviews into **Positive** or **Negative** sentiments. This involves handling the complexities of the Persian language, such as informal writing, word elongation, and specific orthographic characters.

## 🛠️ Key Features & Preprocessing

To achieve high accuracy, several specialized cleaning steps were implemented:

* **Exaggeration Removal**: Custom functions to reduce character repetitions (e.g., "عالییییی" → "عالیی") and excessive punctuation.
* **Persian Normalization**: Integration of the `Hazm` library for standardizing characters and tokenization.
* **Text Encoding**: Building a custom vocabulary with `<pad>` and `<unk>` tokens, filtering out rare words.
* **Sequence Handling**: Padding and truncating reviews to a fixed length of 200 tokens for neural network processing.

## 📊 Model Architectures & Performance

### Phase 1: Logistic Regression (Baseline)

* **Approach**: TF-IDF Vectorization + Logistic Regression.
* **Result**: Achieved a robust **86% Accuracy**.
* **Highlight**: Extremely fast training (< 1 second) and highly interpretable.

### Phase 2: LSTM (Deep Learning)

* **Approach**: PyTorch implementation with an Embedding layer, 2-layer LSTM, and Sigmoid output.
* **Result**: Peak **85.1% Accuracy** at Epoch 3.
* **Analysis**: While the model shows strong potential for capturing context, it exhibited signs of overfitting in later epochs, highlighting the importance of **Early Stopping**.

## 📈 Visualizations

The project includes:

* **Confusion Matrices**: To analyze misclassifications for each model.
* **Training Curves**: Visualizing the loss reduction and accuracy progression for the LSTM.
* **Performance Bar Charts**: Side-by-side comparison of baseline vs. deep learning results.

