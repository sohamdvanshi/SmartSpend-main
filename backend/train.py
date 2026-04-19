import os
import re
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from scipy.sparse import hstack
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Expense_model", "exp.csv")
MODELS_DIR = os.path.join(BASE_DIR, "Expense_model", "models")

os.makedirs(MODELS_DIR, exist_ok=True)


def find_column(df, candidates, required=True):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(f"Could not find any of these columns: {candidates}")
    return None


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = " ".join(text.split())
    return text


def get_amount_range(amount):
    if amount < 50:
        return 0
    elif amount < 200:
        return 1
    elif amount < 500:
        return 2
    elif amount < 1000:
        return 3
    elif amount < 5000:
        return 4
    return 5


KEYWORD_CATEGORIES = {
    "food": [
        "food", "restaurant", "cafe", "meal", "lunch", "dinner", "breakfast",
        "snacks", "grocery", "milk", "bread", "delivery", "dining", "pizza",
        "burger", "chicken", "rice", "curry", "naan", "biryani", "angara"
    ],
    "transport": [
        "auto", "taxi", "train", "bus", "metro", "fuel", "gas", "parking",
        "uber", "ola", "transport", "vehicle", "car"
    ],
    "bills": [
        "bill", "electric", "electricity", "water", "internet", "phone",
        "mobile", "subscription", "utility", "service", "recharge"
    ],
    "shopping": [
        "shopping", "store", "mall", "amazon", "flipkart", "clothes",
        "electronics", "purchase"
    ],
    "health": [
        "hospital", "doctor", "pharmacy", "medical", "health", "clinic", "medicine"
    ],
    "entertainment": [
        "movie", "cinema", "game", "netflix", "entertainment", "show"
    ],
}


def count_keywords(text, keywords):
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def build_features(df, text_col, amount_col):
    notes = df[text_col].fillna("").astype(str)
    amounts = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0)

    note_clean = notes.apply(clean_text)

    feature_rows = []
    for raw_note, note, amount in zip(notes, note_clean, amounts):
        row = {
            "Amount": float(amount),
            "LogAmount": float(np.log1p(max(amount, 0))),
            "AmountRange": get_amount_range(float(amount)),
            "DayOfWeek": pd.Timestamp.now().dayofweek,
            "Month": pd.Timestamp.now().month,
            "Day": pd.Timestamp.now().day,
            "IsWeekend": 1 if pd.Timestamp.now().dayofweek >= 5 else 0,
            "IsMonthEnd": 1 if pd.Timestamp.now().day >= 25 else 0,
            "IsMonthStart": 1 if pd.Timestamp.now().day <= 5 else 0,
            "TextLength": len(note),
            "WordCount": len(note.split()),
            "UpperCaseRatio": sum(1 for c in raw_note if c.isupper()) / len(raw_note) if raw_note else 0,
            "DigitRatio": sum(1 for c in raw_note if c.isdigit()) / len(raw_note) if raw_note else 0,
            "HasAmountPattern": 1 if re.search(r"\d+\s*(rs|rupees|inr|\$)", raw_note.lower()) else 0,
            "HasTimePattern": 1 if re.search(r"\d{1,2}:\d{2}", raw_note) else 0,
            "HasPlacePattern": 1 if re.search(r"place\s+\d+", raw_note.lower()) else 0,
        }

        for category, keywords in KEYWORD_CATEGORIES.items():
            row[f"{category}_keywords"] = count_keywords(note, keywords)

        feature_rows.append(row)

    numeric_df = pd.DataFrame(feature_rows)
    return note_clean, numeric_df


def main():
    warnings.filterwarnings("ignore")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Training dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    text_col = find_column(df, ["Note", "Description", "description", "note", "Item", "Expense"])
    amount_col = find_column(df, ["Amount", "amount", "Price", "price", "Total", "total"])
    target_col = find_column(df, ["Category", "category", "Label", "label", "Type", "type"])

    df = df[[text_col, amount_col, target_col]].copy()
    df[text_col] = df[text_col].fillna("").astype(str)
    df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0)
    df[target_col] = df[target_col].fillna("").astype(str)

    df = df[(df[text_col].str.strip() != "") & (df[target_col].str.strip() != "")]
    df = df.reset_index(drop=True)

    if len(df) < 10:
        raise ValueError("Dataset is too small. Add more labeled rows to exp.csv.")

    X_text, X_numeric = build_features(df, text_col, amount_col)
    y = df[target_col]

    numeric_features = X_numeric.columns.tolist()

    stratify_y = y if y.value_counts().min() >= 2 else None

    X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
        X_text,
        X_numeric,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_y
    )

    tfidf = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )
    X_text_train_tfidf = tfidf.fit_transform(X_text_train)
    X_text_test_tfidf = tfidf.transform(X_text_test)

    scaler = StandardScaler()
    X_num_train_scaled = scaler.fit_transform(X_num_train)
    X_num_test_scaled = scaler.transform(X_num_test)

    X_train = hstack([X_text_train_tfidf, X_num_train_scaled])
    X_test = hstack([X_text_test_tfidf, X_num_test_scaled])

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    full_text_tfidf = tfidf.transform(X_text)
    full_num_scaled = scaler.transform(X_numeric)
    X_full = hstack([full_text_tfidf, full_num_scaled])

    if len(df) >= 15 and y.value_counts().min() >= 2:
        cv_folds = min(5, int(y.value_counts().min()))
        cv_scores = cross_val_score(model, X_full, y, cv=cv_folds, scoring="accuracy")
        cv_mean = float(np.mean(cv_scores))
    else:
        cv_folds = 0
        cv_mean = 0.0

    model_path = os.path.join(MODELS_DIR, "expense_model.pkl")
    tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    scaler_path = os.path.join(MODELS_DIR, "feature_scaler.pkl")
    features_path = os.path.join(MODELS_DIR, "numeric_features.csv")
    report_path = os.path.join(MODELS_DIR, "classification_report.json")

    joblib.dump(model, model_path)
    joblib.dump(tfidf, tfidf_path)
    joblib.dump(scaler, scaler_path)
    pd.DataFrame({"feature": numeric_features}).to_csv(features_path, index=False)

    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    mlflow.set_experiment("SmartSpend-Expense-Classification")

    with mlflow.start_run(run_name="logreg_tfidf_numeric"):
        mlflow.log_param("algorithm", "LogisticRegression")
        mlflow.log_param("max_iter", 2000)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("random_state", 42)
        mlflow.log_param("tfidf_max_features", 3000)
        mlflow.log_param("tfidf_ngram_range", "(1,2)")
        mlflow.log_param("train_rows", int(len(X_text_train)))
        mlflow.log_param("test_rows", int(len(X_text_test)))
        mlflow.log_param("num_classes", int(y.nunique()))
        mlflow.log_param("numeric_feature_count", int(len(numeric_features)))
        mlflow.log_metric("accuracy", float(accuracy))
        mlflow.log_metric("weighted_f1", float(weighted_f1))
        mlflow.log_metric("cv_accuracy_mean", float(cv_mean))
        mlflow.log_metric("dataset_rows", int(len(df)))

        mlflow.log_artifact(model_path)
        mlflow.log_artifact(tfidf_path)
        mlflow.log_artifact(scaler_path)
        mlflow.log_artifact(features_path)
        mlflow.log_artifact(report_path)

        mlflow.sklearn.log_model(model, artifact_path="model")

    print("\nTraining completed successfully.")
    print(f"Dataset rows used: {len(df)}")
    print(f"Classes: {sorted(y.unique().tolist())}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    if cv_folds > 0:
        print(f"CV Accuracy ({cv_folds}-fold): {cv_mean:.4f}")
    print(f"Saved model: {model_path}")
    print(f"Saved TF-IDF: {tfidf_path}")
    print(f"Saved scaler: {scaler_path}")
    print(f"Saved numeric features: {features_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
