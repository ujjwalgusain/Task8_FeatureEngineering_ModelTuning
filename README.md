# Task8: FeatureEngineering_ModelTuning

This project is part of Data Analysis with Python (Task 8). It focuses on:
Feature Engineering & Hyperparameter Tuning (generic example with student dataset).
Fraud Detection using Decision Trees (synthetic dataset).

---

## 🚀 Project Structure

### Section 1: Feature Engineering & Model Tuning
- Created new features (Total_Score) in a student dataset.
- Tuned a Random Forest model using GridSearchCV.

### Section 2: Fraud Detection with Decision Trees
- Generated a synthetic fraud dataset (fraud_detection.csv).
- Encoded categorical features (credit/debit).
- Engineered new features (Amount_Squared, Log_Amount).
- Trained & tuned a Decision Tree using GridSearchCV.
- Evaluated with Precision, Recall, F1-score.

---

## 📊 Results

### Section 1 (Random Forest – Student Dataset)
- Best Params: {'max_depth': 3, 'n_estimators': 50}
- Accuracy: ~60%

### Section 2 (Decision Tree – Fraud Detection)
- Best Params: {'criterion': 'gini', 'max_depth': 10, 'min_samples_split': 5}
- Accuracy: ~90%

### ⚠️ Accuracy is high because most transactions are legitimate, but the model struggles with detecting fraud due to class imbalance.

---

## 💡 Recommendations to Improve Fraud Detection Accuracy

### Handle Class Imbalance
- Use SMOTE (Synthetic Minority Oversampling) or undersample legitimate transactions.
- Try class weights (class_weight="balanced") in Decision Trees.

### Try Advanced Models
- Random Forest, XGBoost, or LightGBM often perform better than a single Decision Tree.
- Ensemble methods can reduce overfitting and capture complex fraud patterns.

### Feature Engineering
- Create time-based features (e.g., transactions per hour/day).
- Calculate average transaction amount per user.
- Flag unusual transactions (very high or very frequent).

### Anomaly Detection
- Use algorithms like Isolation Forest or One-Class SVM to detect rare frauds.

## 🛠 Requirements
### Install dependencies with:
- pip install pandas numpy scikit-learn

### ▶️ How to Run
- python Task8_FeatureEngineering_ModelTuning.py

### This will:
- Run Section 1 (Random Forest on student dataset).
- Run Section 2 (Decision Tree on fraud detection dataset).

## 📂 Output
### fraud_detection.csv → synthetic fraud dataset generated automatically.
### Console output shows:
- Best hyperparameters
- Accuracy (Section 1)
- Classification Report (Section 2)
This repository is now managed by Ujjwal Gusain.
