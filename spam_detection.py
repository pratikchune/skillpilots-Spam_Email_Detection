
import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR, "dataset", "spam_emails.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    OUTPUT_DIR, "spam_detection_model.pkl"
)

# ==========================================
# 2. LOAD DATASET
# ==========================================

print("\n====================================")
print("       EMAIL SPAM DETECTION")
print("====================================")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        "Dataset not found. Place spam_emails.csv "
        "inside the dataset folder."
    )

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nAvailable columns:")
print(df.columns.tolist())

# ==========================================
# 3. IDENTIFY TEXT AND LABEL COLUMNS
# ==========================================

def find_column(data, candidates):

    for column in data.columns:
        if column.lower().strip() in candidates:
            return column

    return None



text_column = "Email Text"
label_column = "Label"


if text_column is None or label_column is None:
    raise ValueError(
        "Could not identify the text and label columns. "
        "Rename them to 'text' and 'label'."
    )

# ==========================================
# 4. DATA CLEANING
# ==========================================

df = df[[text_column, label_column]].copy()

df.columns = ["text", "label"]

df.dropna(inplace=True)

df["text"] = df["text"].astype(str).str.strip()

df["label"] = (
    df["label"].astype(str).str.strip().str.lower()
)

df = df[df["text"] != ""]

# Convert different label formats into spam/ham

spam_labels = ["spam", "1", "junk", "unsolicited"]
ham_labels = [
    "ham", "0", "not spam", "legitimate", "normal"
]

def convert_label(value):

    if value in spam_labels:
        return "spam"

    if value in ham_labels:
        return "ham"

    return None


df["label"] = df["label"].apply(convert_label)

df.dropna(inplace=True)

if df["label"].nunique() != 2:
    raise ValueError(
        "Dataset must contain both Spam and Ham labels."
    )

df.drop_duplicates(subset=["text"], inplace=True)

print("\nData cleaning completed!")
print("Cleaned dataset shape:", df.shape)

print("\nLabel distribution:")
print(df["label"].value_counts())

# ==========================================
# 5. SEPARATE FEATURES AND TARGET
# ==========================================

X = df["text"]
y = df["label"]

# ==========================================
# 6. SPLIT DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# ==========================================
# 7. TF-IDF VECTORIZATION
# ==========================================

# Pipeline applies TF-IDF to training data
# and uses the same vocabulary for test data.

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        MultinomialNB()
    )
])

print("\nConverting email text using TF-IDF...")

# ==========================================
# 8. TRAIN NAIVE BAYES MODEL
# ==========================================

print("\nTraining Naive Bayes classifier...")

model.fit(X_train, y_train)

print("Model training completed successfully!")

# ==========================================
# 9. MODEL PREDICTION
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# 10. PERFORMANCE EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, pos_label="spam", zero_division=0
)

recall = recall_score(
    y_test, y_pred, pos_label="spam", zero_division=0
)

f1 = f1_score(
    y_test, y_pred, pos_label="spam", zero_division=0
)

print("\n====================================")
print("       MODEL PERFORMANCE")
print("====================================")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=["ham", "spam"],
        zero_division=0
    )
)

# ==========================================
# 11. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["ham", "spam"]
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"]
)

plt.title("Email Spam Detection - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrix.png")
)

plt.show()

# ==========================================
# 12. PERFORMANCE METRICS GRAPH
# ==========================================

metrics = {
    "Accuracy": accuracy * 100,
    "Precision": precision * 100,
    "Recall": recall * 100,
    "F1-Score": f1 * 100
}

plt.figure(figsize=(8, 5))

plt.bar(metrics.keys(), metrics.values())

plt.title("Spam Detection Performance Metrics")
plt.ylabel("Score (%)")
plt.ylim(0, 100)

for index, value in enumerate(metrics.values()):
    plt.text(index, value + 1, f"{value:.2f}%",
             ha="center")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "performance_metrics.png")
)

plt.show()

# ==========================================
# 13. SAVE TRAINED MODEL
# ==========================================

joblib.dump(model, MODEL_PATH)

print("\nTrained model saved successfully!")

# ==========================================
# 14. TEST NEW EMAILS
# ==========================================

print("\n====================================")
print("       TEST YOUR EMAIL")
print("====================================")

while True:

    email = input(
        "\nEnter email text (or type exit): "
    ).strip()

    if email.lower() == "exit":
        print("\nThank you for using Spam Detection!")
        break

    if not email:
        print("Please enter some email text.")
        continue

    prediction = model.predict([email])[0]

    probability = model.predict_proba([email])[0]

    spam_index = list(model.classes_).index("spam")

    spam_probability = probability[spam_index] * 100

    if prediction == "spam":

        print("\nResult: SPAM EMAIL")

    else:

        print("\nResult: NOT SPAM (HAM)")

    print(
        f"Estimated spam probability: "
        f"{spam_probability:.2f}%"
    )
