import json
import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import joblib

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "data", "smart_manufacturing_dataset.csv")
MODEL_DIR = os.path.join(HERE, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "condition_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

NUMERIC_FEATURES = [
    "Temperature_C",
    "Vibration_mm_s",
    "Pressure_bar",
    "Humidity_%",
    "Energy_Consumption_kWh",
    "Machine_Speed_RPM",
    "Downtime_Minutes",
    "Throughput_units_hr",
    "Product_Quality_Score",
    "Reject_Rate_%",
]
CATEGORICAL_FEATURES = ["Operation_Mode", "Machine_ID", "Fault_Type"]
TARGET = "Target_Class"

SCALED_MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=SEED,
    ),
}

HGB_MODEL = HistGradientBoostingClassifier(
    learning_rate=0.08,
    max_iter=600,
    max_depth=None,
    l2_regularization=1.0,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=SEED,
)


def build_preprocessor(use_scaling: bool) -> ColumnTransformer:
    if use_scaling:
        num_pipe = Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
    else:
        num_pipe = SimpleImputer(strategy="median")
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", num_pipe, NUMERIC_FEATURES), ("cat", cat_pipe, CATEGORICAL_FEATURES)]
    )


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Fault_Type"] = df["Fault_Type"].fillna("None")
    return df


def evaluate(name, pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    return {
        "model": name,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "predictions": y_pred,
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_data()

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    results = []

    for name, est in SCALED_MODELS.items():
        pipe = Pipeline([("prep", build_preprocessor(use_scaling=True)), ("clf", est)])
        pipe.fit(X_train, y_train)
        results.append(evaluate(name, pipe, X_test, y_test))

    hgb_pipe = Pipeline([("prep", build_preprocessor(use_scaling=False)), ("clf", HGB_MODEL)])
    hgb_pipe.fit(X_train, y_train)
    results.append(evaluate("HistGradientBoosting", hgb_pipe, X_test, y_test))

    results.sort(key=lambda r: (r["accuracy"], r["macro_f1"]), reverse=True)
    best = results[0]

    print("\n=== Model comparison (test set) ===")
    for r in results:
        print(f"{r['model']:<22} acc={r['accuracy']:.4f}  macro_f1={r['macro_f1']:.4f}")
    print(f"\nSelected best: {best['model']} (acc={best['accuracy']})")

    if best["model"] == "HistGradientBoosting":
        final_pipe = Pipeline(
            [("prep", build_preprocessor(use_scaling=False)), ("clf", HGB_MODEL)]
        )
    else:
        final_pipe = Pipeline(
            [("prep", build_preprocessor(use_scaling=True)), ("clf", SCALED_MODELS[best["model"]])]
        )
    final_pipe.fit(X, y)

    best_pipe = Pipeline(
        [
            ("prep", build_preprocessor(use_scaling=(best["model"] != "HistGradientBoosting"))),
            (
                "clf",
                HGB_MODEL
                if best["model"] == "HistGradientBoosting"
                else SCALED_MODELS[best["model"]],
            ),
        ]
    )
    best_pipe.fit(X_train, y_train)
    y_pred = best_pipe.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    classes = list(final_pipe.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=classes).tolist()
    metrics = {
        "best_model": best["model"],
        "accuracy": best["accuracy"],
        "macro_f1": best["macro_f1"],
        "classification_report": report,
        "classes": classes,
        "confusion_matrix": cm,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_order": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    joblib.dump(final_pipe, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")
    print(f"Final test accuracy: {accuracy_score(y_test, y_pred):.4f}")


if __name__ == "__main__":
    main()
