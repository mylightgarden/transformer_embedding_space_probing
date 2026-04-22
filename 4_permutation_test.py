import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder


# Config
# ------------------------------------------------------------
CSV_PATH   = "masked_dataset_with_tiers_scores.csv"
EMBED_PATH = "sentence_embeddings_BAAI_bge_large_en_v1_5.npy"

OUT_DIR = "probe_perm_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Repeated train/test splits
N_SPLITS = 30
SPLIT_SEEDS = list(range(N_SPLITS))   # [0, 1, ..., 29]
TEST_SIZE = 0.2

# Permutation tests
N_PERM_SCORE = 200
N_PERM_TIER  = 200

# Models
RIDGE_ALPHA = 1.0
LOGREG_MAX_ITER = 2000

# RNG seed (permutation reproducibility)
PERM_RNG_SEED = 12345


# Data loading
# ------------------------------------------------------------
def load_data(csv_path, embed_path):
    df = pd.read_csv(csv_path)

    df = df.dropna(subset=["sentence", "score", "tier"]).reset_index(drop=True)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"]).reset_index(drop=True)

    X = np.load(embed_path)
    if X.shape[0] != df.shape[0]:
        raise ValueError(f"Row mismatch: embeddings={X.shape[0]} vs df={df.shape[0]}")

    return df, X


# Evaluation helpers
# ------------------------------------------------------------
def eval_once_regression(X, y_score, seed):
    X_train, X_test, ys_train, ys_test = train_test_split(X, y_score, test_size=TEST_SIZE, random_state=seed)
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(X_train, ys_train)
    ys_pred = model.predict(X_test)
    return{
        "ridge_r2": r2_score(ys_test, ys_pred),
        "ridge_mse": mean_squared_error(ys_test, ys_pred),
    }

def eval_once_classification(X, y_tier, seed):
    X_train, X_test, yt_train, yt_test = train_test_split(X, y_tier, test_size=TEST_SIZE, random_state=seed, stratify=y_tier)
    model = LogisticRegression(max_iter=LOGREG_MAX_ITER)
    model.fit(X_train, yt_train)
    yt_pred = model.predict(X_test)
    return{
        "tier_acc": accuracy_score(yt_test, yt_pred),
        "tier_f1": f1_score(yt_test, yt_pred, average="weighted"),
    }

def eval_repeated_splits(X, y_score, y_tier):
    rows = []
    for seed in SPLIT_SEEDS:
        m_reg = eval_once_regression(X, y_score, seed)
        m_clf = eval_once_classification(X, y_tier, seed)
        rows.append({"seed": seed, **m_reg, **m_clf})
    dfm = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    summary = dfm[["ridge_r2","ridge_mse","tier_acc","tier_f1"]].agg(["mean","std"])
    return dfm, summary

def perm_p_value(null_vals, observed):
    return (float(np.sum(null_vals>=observed))+1.0) / (len(null_vals) + 1.0)

def plot_hist(null_vals, observed, title, xlabel, out_png):
    plt.figure(figsize=(10,7))
    plt.hist(null_vals, bins=35, alpha=0.9)
    plt.axvline(observed, linewidth=3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# Permutation tests
# ------------------------------------------------------------
def score_permutation_test(X, y_score, y_tier, observed_mean_r2, rng):
    null_r2=[]
    for _ in range(N_PERM_SCORE):
        ys = rng.permutation(y_score)   # shuffle score only
        dfm, _ = eval_repeated_splits(X,ys, y_tier)
        null_r2.append(float(dfm["ridge_r2"].mean()))
    null_r2 = np.array(null_r2, dtype=float)
    p = perm_p_value(null_r2, observed_mean_r2)
    return null_r2, p


def tier_permutation_test(X, y_score, y_tier, observed_mean_f1: float, rng: np.random.Generator):
    null_f1=[]
    for _ in range(N_PERM_TIER):
        yt = rng.permutation(y_tier)    # shuffle tier only
        dfm, _ = eval_repeated_splits(X, y_score, yt)
        null_f1.append(float(dfm["tier_f1"].mean()))
    null_f1=np.array(null_f1, dtype=float)
    p=perm_p_value(null_f1, observed_mean_f1)
    return null_f1, p


# Main
# ------------------------------------------------------------
def main():
    df, X = load_data(CSV_PATH, EMBED_PATH)
    y_score = df["score"].to_numpy(dtype=float)
    le = LabelEncoder()
    y_tier = le.fit_transform(df["tier"].astype(str).to_numpy())

    print("[Observed metrics: repeated splits]") 
    per_split, summary = eval_repeated_splits(X,y_score, y_tier)

    obs_r2_mean = float(per_split["ridge_r2"].mean())
    obs_mse_mean = float(per_split["ridge_mse"].mean())
    obs_acc_mean = float(per_split["tier_acc"].mean())
    obs_f1_mean = float(per_split["tier_f1"].mean())  

    # Save observed
    per_split.to_csv(os.path.join(OUT_DIR, "observed_per_split.csv"), index=False)
    summary.to_csv(os.path.join(OUT_DIR, "observed_summary_mean_std.csv"))

    observed_row = pd.DataFrame([{
        "model": "BAAI_bge_large_en_v1_5",
        "n_samples": int(len(df)),
        "n_splits": int(N_SPLITS),
        "test_size": float(TEST_SIZE),
        "ridge_r2_mean": obs_r2_mean,
        "ridge_mse_mean": obs_mse_mean,
        "tier_acc_mean": obs_acc_mean,
        "tier_f1_mean": obs_f1_mean,
    }])
    observed_row.to_csv(os.path.join(OUT_DIR, "observed_metrics.csv"), index=False)
    print(observed_row.to_string(index=False))

    rng = np.random.default_rng(PERM_RNG_SEED)

    # Score-permutation test (R²)
    print(f"\n[Score permutation test] N_PERM_SCORE={N_PERM_SCORE}")
    null_r2, p_r2 = score_permutation_test(X, y_score, y_tier, obs_r2_mean, rng)
    pd.DataFrame({"null_mean_r2": null_r2}).to_csv(os.path.join(OUT_DIR, "score_perm_null_r2.csv"), index=False)
    pd.DataFrame([{
        "perm_N": int(N_PERM_SCORE),
        "obs_r2_mean": obs_r2_mean,
        "perm_p_r2": p_r2,
    }]).to_csv(os.path.join(OUT_DIR, "score_perm_pvalues.csv"), index=False)

    plot_hist(
        null_r2, obs_r2_mean,
        title="BGE — Score Permutation Test (Mean Ridge R² across splits)",
        xlabel="Mean R² under shuffled score labels",
        out_png=os.path.join(OUT_DIR, "score_perm_r2_hist.png"),
    )

    # Tier-permutation test (F1)
    print(f"\n[Tier permutation test] N_PERM_TIER={N_PERM_TIER}")
    null_f1, p_f1 = tier_permutation_test(X, y_score, y_tier, obs_f1_mean, rng)
    pd.DataFrame({"null_mean_f1": null_f1}).to_csv(os.path.join(OUT_DIR, "tier_perm_null_f1.csv"), index=False)
    pd.DataFrame([{
        "perm_N": int(N_PERM_TIER),
        "obs_f1_mean": obs_f1_mean,
        "perm_p_f1": p_f1,
    }]).to_csv(os.path.join(OUT_DIR, "tier_perm_pvalues.csv"), index=False)

    plot_hist(
        null_f1, obs_f1_mean,
        title="BGE — Tier Permutation Test (Mean weighted F1 across splits)",
        xlabel="Mean weighted F1 under shuffled tier labels",
        out_png=os.path.join(OUT_DIR, "tier_perm_f1_hist.png"),
    )

    print("\n[Permutation p-values]")
    print(f"score_perm_p_r2 = {p_r2:.6f}")
    print(f"tier_perm_p_f1  = {p_f1:.6f}")

    print(f"\n[Saved outputs] {OUT_DIR}/")
    print(" - observed_metrics.csv")
    print(" - observed_per_split.csv")
    print(" - observed_summary_mean_std.csv")
    print(" - score_perm_null_r2.csv, score_perm_pvalues.csv, score_perm_r2_hist.png")
    print(" - tier_perm_null_f1.csv, tier_perm_pvalues.csv, tier_perm_f1_hist.png")

    print("\n[Tier label mapping (LabelEncoder order)]")
    for i, name in enumerate(le.classes_):
        print(f"  {i}: {name}")

if __name__ == "__main__":
    main()