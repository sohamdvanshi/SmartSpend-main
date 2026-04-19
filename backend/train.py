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
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Expense_model", "exp_cleaned.csv")
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
    "food": ["food","restaurant","cafe","meal","lunch","dinner","pizza","burger","biryani"],
    "transport": ["uber","ola","bus","train","fuel","metro","auto"],
    "bills": ["bill","electricity","water","internet","recharge"],
    "shopping": ["amazon","flipkart","clothes","shopping","store"],
    "health": ["doctor","medicine","hospital","pharmacy"],
    "entertainment": ["movie","netflix","game","show"]
}


def count_keywords(text, keywords):
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def build_features(df, text_col, amount_col, date_col=None):
    notes = df[text_col].fillna("").astype(str)
    amounts = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0)

    if date_col and date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce")
    else:
        dates = pd.Series([pd.NaT] * len(df))

    note_clean = notes.apply(clean_text)

    feature_rows = []
    for i, (raw_note, note, amount) in enumerate(zip(notes, note_clean, amounts)):
        expense_date = dates.iloc[i] if dates.iloc[i] is not pd.NaT else pd.Timestamp.now()

        row = {
            "Amount": float(amount),
            "LogAmount": float(np.log1p(max(amount, 0))),
            "AmountRange": get_amount_range(float(amount)),
            "DayOfWeek": expense_date.dayofweek,
            "Month": expense_date.month,
            "IsWeekend": 1 if expense_date.dayofweek >= 5 else 0,
            "TextLength": len(note),
            "WordCount": len(note.split())
        }

        for category, keywords in KEYWORD_CATEGORIES.items():
            row[f"{category}_keywords"] = count_keywords(note, keywords)

        feature_rows.append(row)

    numeric_df = pd.DataFrame(feature_rows)
    return note_clean, numeric_df


def main():
    warnings.filterwarnings("ignore")

    df = pd.read_csv(DATA_PATH)

    text_col = find_column(df, ["Note"])
    amount_col = find_column(df, ["Amount"])
    target_col = find_column(df, ["Category"])
    date_col = find_column(df, ["Date"], required=False)

    df = df[[text_col, amount_col, target_col] + ([date_col] if date_col else [])].copy()
    df = df[(df[text_col].str.strip() != "") & (df[target_col].str.strip() != "")]

    X_text, X_numeric = build_features(df, text_col, amount_col, date_col)
    y = df[target_col]

    stratify_y = y if y.value_counts().min() >= 2 else None

    X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
        X_text, X_numeric, y, test_size=0.2, random_state=42, stratify=stratify_y
    )

    # 🔥 BEST TF-IDF CONFIG (BIGRAM)
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    X_text_train_tfidf = tfidf.fit_transform(X_text_train)
    X_text_test_tfidf = tfidf.transform(X_text_test)

    scaler = StandardScaler()
    X_num_train_scaled = scaler.fit_transform(X_num_train)
    X_num_test_scaled = scaler.transform(X_num_test)

    X_train = hstack([X_text_train_tfidf, X_num_train_scaled])
    X_test = hstack([X_text_test_tfidf, X_num_test_scaled])

    # 🔥 BEST MODEL
    model = LinearSVC(
        C=0.5,
        class_weight="balanced",
        max_iter=5000
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    # Save
    joblib.dump(model, os.path.join(MODELS_DIR, "expense_model.pkl"))
    joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "feature_scaler.pkl"))

    # MLflow
    mlflow.set_experiment("SmartSpend-Expense-Classification")

    with mlflow.start_run(run_name="linearsvc_bigram_c05"):
        mlflow.log_param("algorithm", "LinearSVC")
        mlflow.log_param("C", 0.5)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("weighted_f1", weighted_f1)
        mlflow.sklearn.log_model(model, "model")

    print("\nTraining completed")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")


if __name__ == "__main__":
    main()