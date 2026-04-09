"""
Mushroom Classification — EDA & Machine Learning Pipeline
Author: Luna Sbahtu | Arizona State University CSE 572 Data Mining

Classifies mushrooms as edible or poisonous using:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - AdaBoost (Gradient Boosting)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score)


# ─────────────────────────────────────────────
# 1. Load & Explore Data
# ─────────────────────────────────────────────
def load_data(path="mushrooms.csv"):
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nClass distribution:\n{df['class'].value_counts()}")
    return df


# ─────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────
def preprocess(df):
    le = LabelEncoder()
    df_enc = df.apply(le.fit_transform)
    X = df_enc.drop("class", axis=1)
    y = df_enc["class"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ─────────────────────────────────────────────
# 3. Model Evaluation
# ─────────────────────────────────────────────
def evaluate(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f"\n{'='*40}")
    print(f"Model: {name}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Edible", "Poisonous"]))
    return {"model": name, "accuracy": acc, "auc": auc, "predictions": y_pred}


# ─────────────────────────────────────────────
# 4. Visualizations
# ─────────────────────────────────────────────
def plot_confusion_matrix(name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Edible", "Poisonous"],
                yticklabels=["Edible", "Poisonous"])
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"cm_{name.lower().replace(' ', '_')}.png", dpi=120)
    plt.show()


def plot_feature_importance(model, feature_names, title="Feature Importance"):
    importance = model.feature_importances_
    idx = np.argsort(importance)[::-1][:15]
    plt.figure(figsize=(10, 5))
    plt.bar(range(15), importance[idx], color="steelblue")
    plt.xticks(range(15), [feature_names[i] for i in idx], rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=120)
    plt.show()


def plot_model_comparison(results):
    names = [r["model"] for r in results]
    accs = [r["accuracy"] for r in results]
    aucs = [r["auc"] for r in results]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, accs, 0.4, label="Accuracy", color="steelblue")
    ax.bar(x + 0.2, aucs, 0.4, label="ROC-AUC", color="tomato")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0.9, 1.01)
    ax.set_title("Model Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=120)
    plt.show()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data("mushrooms.csv")
    X_train, X_test, y_train, y_test = preprocess(df)

    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("Decision Tree",       DecisionTreeClassifier(random_state=42)),
        ("Random Forest",       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ("AdaBoost",            AdaBoostClassifier(n_estimators=200, random_state=42)),
    ]

    results = []
    for name, model in models:
        res = evaluate(name, model, X_train, X_test, y_train, y_test)
        plot_confusion_matrix(name, y_test, res["predictions"])
        results.append(res)

    # Feature importance from Random Forest
    rf_model = [m for n, m in models if n == "Random Forest"][0]
    plot_feature_importance(rf_model, list(X_train.columns))

    plot_model_comparison(results)
    print("\nBest model:", max(results, key=lambda r: r["accuracy"])["model"])
