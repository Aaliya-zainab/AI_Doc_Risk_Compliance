import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = "data"

CATEGORIES = [
    "low",
    "medium",
    "high"
]


# ============================================================
# LOAD DATASET
# ============================================================

texts = []
labels = []


for category in CATEGORIES:

    category_path = os.path.join(
        DATASET_PATH,
        category
    )

    for filename in os.listdir(category_path):

        file_path = os.path.join(
            category_path,
            filename
        )

        if filename.endswith(".txt"):

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:

                text = file.read()

            texts.append(text)
            labels.append(category)


print("=" * 50)
print("DATASET INFORMATION")
print("=" * 50)

print("Total documents:", len(texts))

for category in CATEGORIES:

    print(
        category.upper(),
        ":",
        labels.count(category)
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.30,
    random_state=42,
    stratify=labels
)


print("\nTraining documents:", len(X_train))
print("Testing documents:", len(X_test))


# ============================================================
# MACHINE LEARNING PIPELINE
# ============================================================

model = Pipeline([

    (
        "tfidf",

        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )
    ),

    (
        "classifier",

        LogisticRegression(
            max_iter=1000
        )
    )

])


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# PREDICTION
# ============================================================

predictions = model.predict(
    X_test
)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)


print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions,
        labels=CATEGORIES,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=CATEGORIES
)


print("Confusion Matrix:")

print(
    "             Low  Medium  High"
)

for category, row in zip(
    CATEGORIES,
    matrix
):

    print(
        f"{category:<10}",
        row
    )


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    "models/risk_classifier.pkl"
)


print(
    "\nModel saved successfully:"
)

print(
    "models/risk_classifier.pkl"
)

print("\n" + "=" * 50)