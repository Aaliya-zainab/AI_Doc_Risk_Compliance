import joblib
from extract_text import extract_text_from_pdf


MODEL_PATH = "models/risk_classifier.pkl"

CONFIDENCE_THRESHOLD = 0.70

model = joblib.load(MODEL_PATH)

pdf_path = "uploads/test.pdf"


# Extract text
text = extract_text_from_pdf(pdf_path)


if not text.strip():

    print("ERROR: No text could be extracted from the PDF.")

else:

    # Prediction
    prediction = model.predict([text])[0]

    # Probabilities
    probabilities = model.predict_proba([text])[0]

    classes = model.classes_

    # Highest probability
    confidence = max(probabilities)

    print("\n==============================")
    print("DOCUMENT RISK ANALYSIS")
    print("==============================")

    print("\nPredicted Risk:", prediction.upper())

    print(f"Confidence: {confidence * 100:.2f}%")

    print("\nClass Probabilities:")

    for class_name, probability in zip(classes, probabilities):
        print(
            f"{class_name}: {probability * 100:.2f}%"
        )

    if confidence < CONFIDENCE_THRESHOLD:

        print("\n MANUAL REVIEW REQUIRED")
        print(
            "The model confidence is below the "
            f"{CONFIDENCE_THRESHOLD * 100:.0f}% threshold."
        )

    else:

        print("\nNo manual review required.")