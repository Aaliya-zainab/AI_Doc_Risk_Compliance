# AI Document Risk & Compliance Classifier

An AI-based system that classifies business documents into Low, Medium, and High risk categories and flags uncertain predictions for manual review.

## Features

- PDF text extraction
- TF-IDF feature extraction
- Logistic Regression classification
- Low / Medium / High risk classification
- Confidence-based manual review
- Important-term explainability
- Risk probability visualization
- Document analysis history
- Flagged-document dashboard

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- TF-IDF
- Logistic Regression
- Pandas
- NumPy
- Matplotlib
- pypdf
- Joblib

## Dataset

The project contains 30 sample documents:

- 10 Low-risk documents
- 10 Medium-risk documents
- 10 High-risk documents

The dataset is divided into training and testing sets during model training.

## Model Evaluation

The model achieved:

**Accuracy: 88.89%**

Evaluation also includes:

- Precision
- Recall
- F1-score
- Confusion Matrix

## How to Run

1. Install dependencies

```bash
pip install -r requirements.txt


2. Run the Streamlit application
streamlit run app.py


3. Train the model

If the model needs to be retrained:

python train_model.py


4.System Workflow
PDF Document
     ↓
Text Extraction
     ↓
TF-IDF Feature Extraction
     ↓
Logistic Regression
     ↓
Risk Classification
     ↓
Confidence Check
     ↓
Manual Review if Confidence < 70%
     ↓
Dashboard




5.Project Structure
AI_Doc_Risk_Compliance/
│
├── data/
│   ├── low/
│   ├── medium/
│   └── high/
│
├── models/
│   └── risk_classifier.pkl
│
├── outputs/
│   └── analysis_history.csv
│
├── app.py
├── extract_text.py
├── predict.py
├── train_model.py
├── requirements.txt
└── README.md