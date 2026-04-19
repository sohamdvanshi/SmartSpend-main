import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Re-use helpers from train.py
from train import find_column, clean_text, build_features, get_amount_range, KEYWORD_CATEGORIES

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Expense_model", "exp.csv")
MODELS_DIR = os.path.join(BASE_DIR, "Expense_model", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Define all candidates to sweep ──────────────────────────────────────────

CANDIDATES = [
    {
        "run_name": "rf_baseline_n200",
        "model": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
        "tfidf_params": {"max_features": 5000, "ngram_range": (1, 3), "sublinear_tf": True},
    },
    {
        "run_name": "rf_tuned_n300_depth20",
        "model": RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1),
        "tfidf_params": {"max_features": 5000, "ngram_range": (1, 3), "sublinear_tf": True},
    },
    {
        "run_name": "extratrees_n300",
        "model": ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
        "tfidf_params": {"max_features": 5000, "ngram_range": (1, 3), "sublinear_tf": True},
    },
    {
        "run_name": "gradientboost_n200_lr01",
        "model": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
        "tfidf_params": {"max_features": 5000, "ngram_range": (1, 2), "sublinear_tf": True},
    },
    {
        "run_name": "logreg_tfidf_bigram",
        "model": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "tfidf_params": {"max_features": 8000, "ngram_range": (1, 2), "sublinear_tf": True},
    },
    {
        "run_name": "linearsvc_tfidf_trigram",
        "model": LinearSVC(class_weight="balanced", random_state=42, max_iter=2000),
        "tfidf_params": {"max_features": 8000, "ngram_range": (1, 3), "sublinear_tf": True},
    },
    {
    "run_name": "logreg_tfidf_trigram_c5",
    "model": LogisticRegression(C=5, max_iter=1000, class_weight="balanced", random_state=42),
    "tfidf_params": {"max_features": 10000, "ngram_range": (1, 3), "sublinear_tf": True},
},
{
    "run_name": "linearsvc_bigram_c05",
    "model": LinearSVC(C=0.5, class_weight="balanced", random_state=42, max_iter=2000),
    "tfidf_params": {"max_features": 8000, "ngram_range": (1, 2), "sublinear_tf": True},
},
{
    "run_name": "sgd_hinge_tfidf_trigram",
    "model": SGDClassifier(loss="hinge", alpha=1e-4, class_weight="balanced", random_state=42, max_iter=1000, tol=1e-3),
    "tfidf_params": {"max_features": 8000, "ngram_range": (1, 3), "sublinear_tf": True},
},
{
    "run_name": "gradientboost_n300_lr005",
    "model": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42),
    "tfidf_params": {"max_features": 5000, "ngram_range": (1, 2), "sublinear_tf": True},
},
]

# ── Load & prep data (once) ──────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(DATA_PATH)
    text_col   = find_column(df, ["Note", "Description", "description", "note", "Item", "Expense"])
    amount_col = find_column(df, ["Amount", "amount", "Price", "price", "Total", "total"])
    target_col = find_column(df, ["Category", "category", "Label", "label", "Type", "type"])
    date_col   = find_column(df, ["Date", "date", "DateTime", "datetime", "Timestamp"], required=False)

    cols = [text_col, amount_col, target_col] + ([date_col] if date_col else [])
    df = df[cols].copy()
    df[text_col]   = df[text_col].fillna("").astype(str)
    df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0)
    df[target_col] = df[target_col].fillna("").astype(str)
    df = df[(df[text_col].str.strip() != "") & (df[target_col].str.strip() != "")].reset_index(drop=True)

    return df, text_col, amount_col, target_col, date_col


def run_sweep():
    warnings.filterwarnings("ignore")
    print(f"\n{'='*60}")
    print("  SmartSpend — Hyperparameter Sweep")
    print(f"{'='*60}\n")

    df, text_col, amount_col, target_col, date_col = load_data()
    print(f"✅ Dataset loaded: {len(df)} rows, {df[target_col].nunique()} classes")
    print(f"   Classes: {sorted(df[target_col].unique().tolist())}\n")

    X_text, X_numeric = build_features(df, text_col, amount_col, date_col)
    y = df[target_col]

    stratify_y = y if y.value_counts().min() >= 2 else None

    X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
        X_text, X_numeric, y,
        test_size=0.2, random_state=42, stratify=stratify_y
    )

    mlflow.set_experiment("SmartSpend-Expense-Classification")

    results = []

    for i, candidate in enumerate(CANDIDATES):
        run_name = candidate["run_name"]
        model    = candidate["model"]
        tp       = candidate["tfidf_params"]

        print(f"[{i+1}/{len(CANDIDATES)}] Running: {run_name} ...")

        # TF-IDF
        tfidf = TfidfVectorizer(**tp, min_df=1, max_df=0.9)
        X_tr_tfidf = tfidf.fit_transform(X_text_train)
        X_te_tfidf = tfidf.transform(X_text_test)

        # Scaler
        scaler = StandardScaler()
        X_tr_num = scaler.fit_transform(X_num_train)
        X_te_num = scaler.transform(X_num_test)

        X_train = hstack([X_tr_tfidf, X_tr_num])
        X_test  = hstack([X_te_tfidf, X_te_num])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy    = accuracy_score(y_test, y_pred)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Cross-validation
        X_full_tfidf = tfidf.transform(X_text)
        X_full_num   = scaler.transform(X_numeric)
        X_full       = hstack([X_full_tfidf, X_full_num])

        cv_folds  = min(5, int(y.value_counts().min()))
        cv_scores = cross_val_score(model, X_full, y, cv=cv_folds, scoring="accuracy") if cv_folds >= 2 else np.array([accuracy])
        cv_mean   = float(np.mean(cv_scores))

        print(f"   Accuracy: {accuracy:.4f} | Weighted F1: {weighted_f1:.4f} | CV Mean: {cv_mean:.4f}")

        # Log to MLflow
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("algorithm",         type(model).__name__)
            mlflow.log_param("run_name",          run_name)
            mlflow.log_param("tfidf_max_features",tp["max_features"])
            mlflow.log_param("tfidf_ngram_range", str(tp["ngram_range"]))
            mlflow.log_param("tfidf_sublinear_tf",tp["sublinear_tf"])
            mlflow.log_param("train_rows",        len(X_text_train))
            mlflow.log_param("test_rows",         len(X_text_test))
            mlflow.log_param("num_classes",       int(y.nunique()))

            # Log model-specific params
            params = model.get_params()
            for k, v in params.items():
                try:
                    mlflow.log_param(k, v)
                except Exception:
                    pass

            mlflow.log_metric("accuracy",        float(accuracy))
            mlflow.log_metric("weighted_f1",     float(weighted_f1))
            mlflow.log_metric("cv_accuracy_mean",float(cv_mean))

            mlflow.sklearn.log_model(model, artifact_path="model")

        results.append({
            "run_name":   run_name,
            "algorithm":  type(model).__name__,
            "accuracy":   round(accuracy, 4),
            "weighted_f1":round(weighted_f1, 4),
            "cv_mean":    round(cv_mean, 4),
            "model":      model,
            "tfidf":      tfidf,
            "scaler":     scaler,
        })

    # ── Print leaderboard ────────────────────────────────────────────────────
    results.sort(key=lambda x: (x["weighted_f1"], x["cv_mean"]), reverse=True)

    print(f"\n{'='*60}")
    print("  LEADERBOARD (sorted by weighted_f1)")
    print(f"{'='*60}")
    print(f"{'Rank':<5} {'Run Name':<35} {'Accuracy':<10} {'F1':<10} {'CV Mean'}")
    print("-" * 60)
    for rank, r in enumerate(results, 1):
        marker = "  ⭐ BEST" if rank == 1 else ""
        print(f"{rank:<5} {r['run_name']:<35} {r['accuracy']:<10} {r['weighted_f1']:<10} {r['cv_mean']}{marker}")

    # ── Auto-save the best model ─────────────────────────────────────────────
    best = results[0]
    print(f"\n🏆 Best model: {best['run_name']}")
    print(f"   Saving as expense_model.pkl ...")

    joblib.dump(best["model"],  os.path.join(MODELS_DIR, "expense_model.pkl"))
    joblib.dump(best["tfidf"],  os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(best["scaler"], os.path.join(MODELS_DIR, "feature_scaler.pkl"))

    print(f"✅ Best model saved! Restart app.py to use it.\n")


if __name__ == "__main__":
    run_sweep()