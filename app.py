import streamlit as st
import joblib
import os
import pandas as pd
from datetime import datetime

from extract_text import extract_text_from_pdf


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="AI Document Risk Classifier",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    "models/risk_classifier.pkl"
)

CONFIDENCE_THRESHOLD = 0.70

HISTORY_FILE = "outputs/analysis_history.csv"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# SAVE ANALYSIS FUNCTION
# ============================================================

def save_analysis(filename, risk, confidence, manual_review):

    new_record = pd.DataFrame([{
        "Document": filename,
        "Risk": risk.upper(),
        "Confidence": f"{confidence * 100:.2f}%",
        "Manual Review": (
            "REQUIRED"
            if manual_review
            else "NOT REQUIRED"
        ),
        "Date": datetime.now().strftime(
            "%Y-%m-%d %H:%M"
        )
    }])

    if os.path.exists(HISTORY_FILE):

        history = pd.read_csv(
            HISTORY_FILE
        )

        history = pd.concat(
            [history, new_record],
            ignore_index=True
        )

    else:

        history = new_record

    history.to_csv(
        HISTORY_FILE,
        index=False
    )


# ============================================================
# TITLE
# ============================================================

st.title(
    "AI Document Risk & Compliance Classifier"
)

st.write(
    "Upload a business PDF to classify its potential "
    "risk level and identify documents that may need "
    "manual review."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "System Information"
)

st.sidebar.write(
    "Model: Logistic Regression"
)

st.sidebar.write(
    "Features: TF-IDF"
)

st.sidebar.write(
    "Categories: Low / Medium / High"
)

st.sidebar.write(
    "Manual Review Threshold: 70%"
)


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

st.subheader(
    "Upload Document"
)

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


# ============================================================
# ANALYZE DOCUMENT
# ============================================================

if uploaded_file is not None:

    st.success(
        f"Selected: {uploaded_file.name}"
    )

    if st.button(
        "Analyze Document"
    ):

        # ----------------------------------------------------
        # Save uploaded PDF
        # ----------------------------------------------------

        os.makedirs(
            "uploads",
            exist_ok=True
        )

        file_path = os.path.join(
            "uploads",
            uploaded_file.name
        )

        with open(
            file_path,
            "wb"
        ) as file:

            file.write(
                uploaded_file.getbuffer()
            )


        # ----------------------------------------------------
        # Extract text from PDF
        # ----------------------------------------------------

        try:

            text = extract_text_from_pdf(
                file_path
            )

        except Exception as error:

            st.error(
                f"Error reading PDF: {error}"
            )

            st.stop()


        if not text.strip():

            st.error(
                "No readable text was found in this PDF."
            )

            st.stop()


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            [text]
        )[0]

        probabilities = model.predict_proba(
            [text]
        )[0]

        classes = model.classes_

        confidence = max(
            probabilities
        )


        # ----------------------------------------------------
        # Manual Review Decision
        # ----------------------------------------------------

        manual_review = (
            confidence < CONFIDENCE_THRESHOLD
        )


        # ----------------------------------------------------
        # Save Analysis History
        # ----------------------------------------------------

        save_analysis(
            uploaded_file.name,
            prediction,
            confidence,
            manual_review
        )


        # ----------------------------------------------------
        # Analysis Result
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Analysis Result"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Risk Level",
                prediction.upper()
            )


        with col2:

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )


        with col3:

            if manual_review:

                st.metric(
                    "Manual Review",
                    "REQUIRED"
                )

            else:

                st.metric(
                    "Manual Review",
                    "NOT REQUIRED"
                )


        # ----------------------------------------------------
        # Manual Review Message
        # ----------------------------------------------------

        if manual_review:

            st.warning(
                "Manual Review Required: "
                "The model confidence is below 70%."
            )

        else:

            st.success(
                "Classification completed successfully."
            )


        # ----------------------------------------------------
        # Risk Probability
        # ----------------------------------------------------

        st.subheader(
            "Risk Probability"
        )

        probability_data = {}

        for class_name, probability in zip(
            classes,
            probabilities
        ):

            probability_data[
                class_name.upper()
            ] = probability


        st.bar_chart(
            probability_data
        )


        # ----------------------------------------------------
        # Explainability
        # ----------------------------------------------------

        st.subheader(
            "Important Terms"
        )

        vectorizer = model.named_steps[
            "tfidf"
        ]

        classifier = model.named_steps[
            "classifier"
        ]


        # Convert document into TF-IDF values

        tfidf_values = vectorizer.transform(
            [text]
        )

        feature_names = (
            vectorizer.get_feature_names_out()
        )


        # Find predicted class index

        predicted_class_index = list(
            classifier.classes_
        ).index(
            prediction
        )


        # Get classifier coefficients

        coefficients = (
            classifier.coef_[
                predicted_class_index
            ]
        )


        # Calculate term contribution

        contributions = (
            tfidf_values.toarray()[0]
            * coefficients
        )


        # Sort terms by contribution

        top_indices = (
            contributions.argsort()[::-1]
        )


        important_terms = []


        for index in top_indices:

            if contributions[index] > 0:

                term = feature_names[index]

                # Show individual words only

                if " " not in term:

                    if term not in important_terms:

                        important_terms.append(
                            term
                        )

                if len(
                    important_terms
                ) == 6:

                    break


        if important_terms:

            st.write(
                "Terms that contributed to the prediction:"
            )

            st.write(
                ", ".join(
                    important_terms
                )
            )

        else:

            st.write(
                "No strong individual terms identified."
            )


        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        st.subheader(
            "Summary"
        )

        st.write(
            f"**Document:** "
            f"{uploaded_file.name}"
        )

        st.write(
            f"**Classification:** "
            f"{prediction.upper()} RISK"
        )

        st.write(
            f"**Confidence:** "
            f"{confidence * 100:.2f}%"
        )

        if manual_review:

            st.write(
                "**Decision:** "
                "Manual review required."
            )

        else:

            st.write(
                "**Decision:** "
                "Manual review not required."
            )


# ============================================================
# DOCUMENT RISK DASHBOARD
# ============================================================

st.divider()

st.header(
    "Document Risk Dashboard"
)


if os.path.exists(
    HISTORY_FILE
):

    history = pd.read_csv(
        HISTORY_FILE
    )


    # --------------------------------------------------------
    # Dashboard Counts
    # --------------------------------------------------------

    total_documents = len(
        history
    )

    low_count = len(
        history[
            history["Risk"] == "LOW"
        ]
    )

    medium_count = len(
        history[
            history["Risk"] == "MEDIUM"
        ]
    )

    high_count = len(
        history[
            history["Risk"] == "HIGH"
        ]
    )

    review_count = len(
        history[
            history["Manual Review"]
            == "REQUIRED"
        ]
    )


    # --------------------------------------------------------
    # Dashboard Metrics
    # --------------------------------------------------------

    col1, col2, col3, col4, col5 = (
        st.columns(5)
    )


    with col1:

        st.metric(
            "Total Analyzed",
            total_documents
        )


    with col2:

        st.metric(
            "Low Risk",
            low_count
        )


    with col3:

        st.metric(
            "Medium Risk",
            medium_count
        )


    with col4:

        st.metric(
            "High Risk",
            high_count
        )


    with col5:

        st.metric(
            "Manual Review",
            review_count
        )


    # --------------------------------------------------------
    # Flagged Documents
    # --------------------------------------------------------

    st.subheader(
        "Flagged Documents"
    )


    flagged = history[
        history["Manual Review"]
        == "REQUIRED"
    ]


    if len(flagged) > 0:
        st.table(flagged)

    else:

        st.success(
            "No documents currently require manual review."
        )


    # --------------------------------------------------------
    # Complete Analysis History
    # --------------------------------------------------------

    st.subheader(
        "Analysis History"
    )


    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True
    )


else:

    st.info(
        "No documents have been analyzed yet."
    )