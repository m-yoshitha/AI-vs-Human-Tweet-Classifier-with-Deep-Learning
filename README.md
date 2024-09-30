# AI-vs-Human-Tweet-Classifier-with-Deep-Learning

## Overview
This project involves building a machine learning model to classify whether a tweet is human-generated or AI-generated based on word embeddings and various tweet features. The dataset contains over 11,000 tweets with pre-processed features such as word count, punctuation count, and word embeddings. The primary goal is to design, train, and evaluate a model to accurately predict the origin of the tweet using techniques such as dimensionality reduction, feature engineering, and deep learning.

## Project Structure

1. **Data Collection:**
   - The data is divided into training and testing sets:
     - **Training Data**: 11,144 documents (tweets) with 772 columns.
     - **Test Data**: 2,786 documents with 771 columns (no target variable).
   - The target variable `ind` is binary:
     - `0`: Human-generated tweet.
     - `1`: AI-generated tweet.
   - Features include word embeddings (feature_0 to feature_767), word count, and punctuation count.

2. **Data Preparation:**
   - The dataset is highly imbalanced, with more human-generated tweets (class 0) than AI-generated (class 1).
   - Performed **train-test-validation split** to prepare the data for modeling.
   - Applied **scaling** on features to ensure proper range normalization.

3. **Exploratory Data Analysis (EDA):**
   - Discovered that human-generated tweets tend to have more words and punctuation than AI-generated tweets.
   - No significant correlation was found between word count and punctuation count or among word embeddings.

4. **Feature Engineering:**
   - Applied **Principal Component Analysis (PCA)** to reduce dimensionality while retaining 95% of the variance in the data.
   - Created **3-gram averages** to capture broader contextual features in the word embeddings.

5. **Balancing the Dataset:**
   - **SMOTE (Synthetic Minority Over-sampling Technique)** was used to address the class imbalance by generating synthetic samples for the minority class (AI-generated tweets).
   - Tried various sampling techniques like **random oversampling** and **undersampling**, but SMOTE yielded the best results.

## Models Implemented

### a. **Feedforward Neural Network (Best Performing Model)**
   - **Architecture**:
     - 4 Dense Layers with ReLU activation.
     - 2 Dropout Layers for regularization.
     - Sigmoid activation for the output layer.
   - **Optimization**:
     - Adam optimizer with early stopping based on validation loss.
   - **Results**:
     - **Accuracy**: 95%
     - **F1 Score**: 0.72
     - **AUC**: 0.82
   - **Confusion Matrix**:
     ```
     [[993  17]
      [ 37  68]]
     ```

### b. **Logistic Regression (Baseline Model)**
   - **Results**:
     - **Accuracy**: 83%
     - **F1 Score**: 0.48
     - **AUC**: 0.79

### c. **Recurrent Neural Networks (RNN and LSTM)**
   - **Challenges**:
     - Lower F1 score compared to FNN due to the non-sequential nature of the data.
     - LSTM struggled with retaining important patterns from the embeddings due to the large number of features and lack of sequential data.

## Model Evaluation

We evaluated our models based on several metrics:

- **AUC (Area Under the Curve)**: Evaluates the trade-off between true positive and false positive rates.
- **Accuracy**: Percentage of correct predictions out of total predictions.
- **F1 Score**: Harmonic mean of precision and recall, especially useful for imbalanced datasets.
- **Confusion Matrix**: Analyzed false positives and false negatives to assess model performance in identifying human vs AI-generated tweets.

| Metric        | FNN Model  | Logistic Regression |
| ------------- | ---------- | ------------------- |
| **Accuracy**  | 0.95       | 0.83                |
| **F1 Score**  | 0.72       | 0.48                |
| **AUC**       | 0.82       | 0.79                |

## Conclusion and Key Learnings

- **Feedforward Neural Networks (FNN)** performed the best in this task with high accuracy and an F1 score that balanced precision and recall effectively.
- **Logistic Regression** served as a good baseline but was outperformed by deep learning models due to its inability to capture complex patterns in the word embeddings.
- **Recurrent Neural Networks (RNNs and LSTM)**, while typically good for sequential data, did not perform as well on this non-sequential classification task.
- **SMOTE** was instrumental in addressing class imbalance, particularly helping the FNN model to better predict the minority class (AI-generated tweets).
- **Dimensionality Reduction (PCA)**: Although it helped reduce computational cost, it did not significantly improve model performance as important data patterns were potentially lost.
- **Normalization vs Scaling**: We found that **scaling** the data performed better than normalization due to the presence of outliers and high variance in the features.
- **Hyperparameter Tuning**: Tuning dropout rates, batch sizes, number of layers, and epochs significantly impacted the model’s accuracy and generalization ability.

