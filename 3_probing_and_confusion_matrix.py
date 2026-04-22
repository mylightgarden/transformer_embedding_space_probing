import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    accuracy_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns


# Config: embedding files + metadata CSV
# -------------------------------------------------------------------
EMBEDDING_FILES = {
    "BAAI_bge_large_en_v1_5": "sentence_embeddings_BAAI_bge_large_en_v1_5.npy",
    "all-mpnet-base-v2": "sentence_embeddings_sentence_transformers_all_mpnet_base_v2.npy",
    "all-MiniLM-L6-v2": "sentence_embeddings_sentence_transformers_all_MiniLM_L6_v2.npy",
}
METADATA_CSV = "masked_dataset_with_tiers_scores.csv"

TIER_ORDER = [
        "Shadow",
        "Striving",
        "Conflict",
        "Activation",
        "Growth",
        "Clarity",
        "Unity",
    ]

RIDGE_ALPHA = 1.0

N_SPLITS = 30
TEST_SIZE = 0.2

# Data loading
# -------------------------------------------------------------------
def load_dataframe(csv_path):
    df = pd.read_csv(csv_path)

    df = df.dropna(subset=["sentence", "score", "tier"]).reset_index(drop=True)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"]).reset_index(drop=True)

    print(f"Loaded dataframe: shape = {df.shape}")
    return df

def load_embeddings_dict(embedding_files):
    loaded ={}
    for model_name, path in embedding_files.items():
        X = np.load(path)
        loaded[model_name] = X
    return loaded


# Helper to aggregate metrics
# -------------------------------------------------------------------
def aggregate_metrics(metrics_list):
    return {
        k + "_mean": float(np.mean([m[k] for m in metrics_list]))
        for k in metrics_list[0]
    }


# Probes: Ridge Regression
# -------------------------------------------------------------------
def run_ridge_regression_probe(df, X, model_name, seed):
    y= df['score'].values.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)

    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"\n=== [Ridge Regression] on `score` for model: {model_name} ===")
    print(f"seed: {seed}")
    print(f"R²  : {r2:.4f}")
    print(f"MSE : {mse:.4f}")

    return model, {"r2": r2, "mse": mse}


# Probes: MLP Regression
# -------------------------------------------------------------------
def run_mlp_regression_probe(df, X, model_name, seed):
    y = df["score"].values.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)

    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64), 
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=seed,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.1,
    )
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"\n=== [MLP Regression] on `score` for model: {model_name} ===")
    print(f"seed: {seed}")
    print(f"R²  : {r2:.4f}")
    print(f"MSE : {mse:.4f}")

    return mlp, {"r2": r2, "mse": mse}


# Plot confusion matrix
# -------------------------------------------------------------------
def plot_confusion_matrix(cm, model_name: str):
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=TIER_ORDER,
        yticklabels=TIER_ORDER,
    )
    plt.title(f"{model_name} — Confusion Matrix (Tier Classification)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    fname = f"confusion_{model_name}.png"
    plt.savefig(fname, dpi=150)
    plt.close()


# Tier Classifier + Confusion
# -------------------------------------------------------------------
def run_tier_classifier(df, X, model_name, seed):
    # Map string tier -> index
    def tier_to_idx(t):
        t = str(t)
        if t not in TIER_ORDER:
            raise ValueError(f"Tier:'{t}' not in TIER_ORDER.")
        return TIER_ORDER.index(t)
    
    y = df['tier'].apply(tier_to_idx).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)

    # Train logistic regression classifier
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\n=== [Logistic Regression] on `tier` for model: {model_name} ===")
    print(f"seed: {seed}")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")

    # Confusion matrix
    if seed == 0:
        labels = list(range(len(TIER_ORDER)))
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        plot_confusion_matrix(cm, model_name)

    return clf, {"accuracy": acc, "f1": f1}

# Main
# -------------------------------------------------------------------
def main():
    df = load_dataframe(METADATA_CSV)
    embeddings_dict = load_embeddings_dict(EMBEDDING_FILES)
    results_summary = []

    for model_name, X in embeddings_dict.items():
        if X.shape[0] != df.shape[0]:
            raise ValueError(
                f"[ERROR] Row mismatch for model '{model_name}': "
                f"{X.shape[0]} embeddings vs {df.shape[0]} dataframe rows."
            )

        ridge_metrics_all = []
        mlp_metrics_all = []
        tier_metrics_all = []

        for seed in range(N_SPLITS):
            np.random.seed(seed)

            _, ridge_metrics = run_ridge_regression_probe(df, X, model_name, seed)
            _, mlp_metrics   = run_mlp_regression_probe(df, X, model_name, seed)
            _, tier_metrics  = run_tier_classifier(df, X, model_name, seed)

            ridge_metrics_all.append(ridge_metrics)
            mlp_metrics_all.append(mlp_metrics)
            tier_metrics_all.append(tier_metrics)

        ridge_avg = aggregate_metrics(ridge_metrics_all)
        mlp_avg   = aggregate_metrics(mlp_metrics_all)
        tier_avg  = aggregate_metrics(tier_metrics_all)

        results_summary.append(
            {
                "model": model_name,
                "n_samples": len(df),
                "n_splits": N_SPLITS,
                "test_size": TEST_SIZE,
                "ridge_r2_mean": ridge_avg["r2_mean"],
                "ridge_mse_mean": ridge_avg["mse_mean"],
                "mlp_r2_mean": mlp_avg["r2_mean"],
                "mlp_mse_mean": mlp_avg["mse_mean"],
                "tier_acc_mean": tier_avg["accuracy_mean"],
                "tier_f1_mean": tier_avg["f1_mean"],
            }
        )

    print("\n\n====== Summary across models (Repeated Splits) ======")
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
