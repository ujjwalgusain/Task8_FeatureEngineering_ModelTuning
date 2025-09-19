# Task 8: Feature Engineering & Model Tuning
# Covers Section 1 (Generic Example) + Section 2 (Fraud Detection + Confusion Matrix)

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Ensure images folder exists
os.makedirs("images", exist_ok=True)

# --------------------------------------------------------
# SECTION 1: Feature Engineering & Model Tuning (Generic)
# --------------------------------------------------------
print("\n================ SECTION 1: Feature Engineering & Model Tuning ================\n")

# Create synthetic student dataset
np.random.seed(42)
data = {
    "Math": np.random.randint(30, 100, 100),
    "Science": np.random.randint(30, 100, 100),
    "English": np.random.randint(30, 100, 100),
    "Pass": np.random.choice([0, 1], size=100)  # 0 = Fail, 1 = Pass
}
df1 = pd.DataFrame(data)

# Feature Engineering → new feature
df1["Total_Score"] = df1["Math"] + df1["Science"] + df1["English"]

# Train-test split
X1 = df1.drop("Pass", axis=1)
y1 = df1["Pass"]
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

# Random Forest + Hyperparameter Tuning
rf = RandomForestClassifier(random_state=42)
param_grid1 = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, None]
}
grid1 = GridSearchCV(rf, param_grid1, cv=5, scoring="accuracy")
grid1.fit(X1_train, y1_train)

best_rf = grid1.best_estimator_
y1_pred = best_rf.predict(X1_test)

print("✅ Section 1 - Best Params:", grid1.best_params_)
print("✅ Section 1 - Accuracy:", accuracy_score(y1_test, y1_pred))

# --------------------------------------------------------
# SECTION 2: Fraud Detection with Decision Trees
# --------------------------------------------------------
print("\n================ SECTION 2: Fraud Detection with Decision Trees ================\n")

# Create synthetic fraud detection dataset
n_samples = 1000
data2 = {
    "Transaction ID": range(1, n_samples + 1),
    "Amount": np.random.randint(10, 1000, size=n_samples),
    "Type": np.random.choice(["credit", "debit"], size=n_samples),
    "Is Fraud": np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])  # 10% fraud
}
df2 = pd.DataFrame(data2)
df2.to_csv("fraud_detection.csv", index=False)  # Save file for reference

# Encode categorical column
le = LabelEncoder()
df2["Type"] = le.fit_transform(df2["Type"])  # credit=0, debit=1

# Feature Engineering
df2["Amount_Squared"] = df2["Amount"] ** 2
df2["Log_Amount"] = df2["Amount"].apply(lambda x: 0 if x <= 0 else np.log(x + 1))

# Train-test split
X2 = df2.drop(["Transaction ID", "Is Fraud"], axis=1)
y2 = df2["Is Fraud"]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Decision Tree + Hyperparameter Tuning
dt = DecisionTreeClassifier(random_state=42)
param_grid2 = {
    "max_depth": [3, 5, 7, 10],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"]
}
grid2 = GridSearchCV(dt, param_grid2, cv=5, scoring="f1")
grid2.fit(X2_train, y2_train)

best_dt = grid2.best_estimator_
y2_pred = best_dt.predict(X2_test)

print("✅ Section 2 - Best Params:", grid2.best_params_)
print("\n📊 Section 2 - Classification Report:\n")
report = classification_report(y2_test, y2_pred)
print(report)

# Save classification report to file
with open("images/classification_report.txt", "w") as f:
    f.write(report)

# Confusion Matrix Plot
cm = confusion_matrix(y2_test, y2_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_dt.classes_)
disp.plot(cmap=plt.cm.Blues, values_format="d")
plt.title("Confusion Matrix - Fraud Detection")
plt.savefig("images/confusion_matrix.png")
plt.show()
