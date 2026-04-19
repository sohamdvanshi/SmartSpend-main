import os
import joblib
import pandas as pd

from models import EnhancedExpenseClassifier


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "Expense_model", "models")


def get_model_path(filename: str) -> str:
    return os.path.join(MODELS_DIR, filename)


def load_expense_classifier():
    """
    Load the trained model artifacts from Expense_model/models/
    Returns:
        EnhancedExpenseClassifier if all artifacts exist
        base sklearn model if only model exists
        None if model missing
    """
    model_path = get_model_path("expense_model.pkl")
    tfidf_path = get_model_path("tfidf_vectorizer.pkl")
    scaler_path = get_model_path("feature_scaler.pkl")
    features_path = get_model_path("numeric_features.csv")

    if not os.path.exists(model_path):
        print(f"⚠️ Model file not found: {model_path}")
        return None

    try:
        model = joblib.load(model_path)
        print("✅ expense_model.pkl loaded")

        tfidf_exists = os.path.exists(tfidf_path)
        scaler_exists = os.path.exists(scaler_path)
        features_exists = os.path.exists(features_path)

        if tfidf_exists and scaler_exists and features_exists:
            tfidf = joblib.load(tfidf_path)
            scaler = joblib.load(scaler_path)
            numeric_features = pd.read_csv(features_path)["feature"].tolist()

            print("✅ TF-IDF, scaler, and numeric feature list loaded")
            return EnhancedExpenseClassifier(
                model=model,
                tfidf=tfidf,
                scaler=scaler,
                numeric_features=numeric_features,
            )

        print("⚠️ Only base model found, enhanced artifacts missing")
        return model

    except Exception as e:
        print(f"❌ Error loading model artifacts: {e}")
        return None


def predict_expense_category(model_wrapper, description, amount):
    """
    Unified prediction helper.
    Works with:
    1. EnhancedExpenseClassifier
    2. plain sklearn model
    """
    if model_wrapper is None:
        return "Miscellaneous"

    try:
        amount = float(amount or 0)
    except Exception:
        amount = 0.0

    description = str(description or "").strip()

    if isinstance(model_wrapper, EnhancedExpenseClassifier):
        return model_wrapper.predict({
            "Note": description,
            "Amount": amount
        })

    try:
        df = pd.DataFrame({
            "Note": [description],
            "Amount": [amount]
        })
        pred = model_wrapper.predict(df)[0]
        return str(pred)
    except Exception as e:
        print(f"❌ Base model prediction failed: {e}")
        return "Miscellaneous"
