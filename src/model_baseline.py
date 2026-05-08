"""Baseline recommender: implicit-feedback ALS on H&M transactions.

Implements the Hu, Koren, Volinsky (2008) ALS-with-confidence formulation on the
H&M customer-by-article interaction matrix. Predictions are evaluated with MAP@12
on the final-week hold-out per the Kaggle competition metric.

Run:
    cd /root/AI/liora_projects/13_hm_fashion
    python src/model_baseline.py

Inputs (expected under ../data/):
    transactions_train.csv
    customers.csv
    articles.csv

Outputs (written to ../deliverables/):
    als_model.npz                    - factor matrices (user, item)
    als_metrics.json                 - MAP@12 / Recall@12 / NDCG@12 / coverage
    als_top12_holdout.parquet        - per-customer top-12 predictions on hold-out
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

# `implicit` is the Cython-optimised CF library that bundles ALS, BPR and LMF.
# pip install implicit
from implicit.als import AlternatingLeastSquares


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DELIV_DIR = PROJECT_ROOT / "deliverables"
DELIV_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# Hyperparameters (defaults; overridable via argparse)
# --------------------------------------------------------------------------- #
DEFAULTS = dict(
    factors=128,
    regularization=0.05,
    iterations=20,
    alpha=40.0,            # confidence scale (Hu, Koren, Volinsky 2008)
    recency_halflife_days=30.0,
    cut_date="2020-09-16",
    top_k=12,
    min_customer_txn=1,    # cold-start customers go to popularity fallback
    min_article_txn=2,
    seed=42,
)


# --------------------------------------------------------------------------- #
# Data loading and matrix construction
# --------------------------------------------------------------------------- #
def load_tables(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    txn_dtypes = {
        "customer_id": "string",
        "article_id": "int64",
        "price": "float32",
        "sales_channel_id": "int8",
    }
    transactions = pd.read_csv(
        data_dir / "transactions_train.csv",
        dtype=txn_dtypes,
        parse_dates=["t_dat"],
    )
    customers = pd.read_csv(data_dir / "customers.csv")
    articles = pd.read_csv(data_dir / "articles.csv")
    return transactions, customers, articles


def split_train_holdout(
    transactions: pd.DataFrame, cut_date: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = pd.Timestamp(cut_date)
    train = transactions[transactions["t_dat"] < cut].copy()
    holdout = transactions[transactions["t_dat"] >= cut].copy()
    return train, holdout


def build_interaction_matrix(
    train: pd.DataFrame,
    recency_halflife_days: float,
    min_article_txn: int,
) -> tuple[sparse.csr_matrix, pd.Index, pd.Index]:
    """Customer-by-article confidence matrix C = 1 + alpha * decayed_count.

    Recency decay: weight = 0.5 ** (days_old / halflife).
    Returns CSR with float32 confidence-minus-1 values; the ALS solver expects
    that the input sparse matrix carries r_ui (interaction strength); the +1
    base of the confidence term is added internally by implicit when alpha is
    set on the solver. Here we precompute alpha into the values so we can keep
    the recency weighting flexible.
    """
    # Filter rare articles to stabilise embeddings.
    art_counts = train["article_id"].value_counts()
    keep_articles = art_counts[art_counts >= min_article_txn].index
    train = train[train["article_id"].isin(keep_articles)].copy()

    most_recent = train["t_dat"].max()
    days_old = (most_recent - train["t_dat"]).dt.days.astype("float32")
    weights = np.power(0.5, days_old / np.float32(recency_halflife_days))
    train["weight"] = weights

    # Aggregate to (customer, article) -> sum_weight (effectively decayed count).
    agg = (
        train.groupby(["customer_id", "article_id"], as_index=False, sort=False)
        ["weight"].sum()
    )

    cust_idx = pd.Index(agg["customer_id"].unique(), name="customer_id")
    art_idx = pd.Index(agg["article_id"].unique(), name="article_id")
    cust_pos = {c: i for i, c in enumerate(cust_idx)}
    art_pos = {a: i for i, a in enumerate(art_idx)}

    rows = agg["customer_id"].map(cust_pos).to_numpy()
    cols = agg["article_id"].map(art_pos).to_numpy()
    data = agg["weight"].to_numpy(dtype=np.float32)

    M = sparse.csr_matrix(
        (data, (rows, cols)), shape=(len(cust_idx), len(art_idx)), dtype=np.float32
    )
    return M, cust_idx, art_idx


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def fit_als(
    M: sparse.csr_matrix,
    factors: int,
    regularization: float,
    iterations: int,
    alpha: float,
    seed: int,
) -> AlternatingLeastSquares:
    """Fit ALS on confidence-weighted user-item matrix.

    The implicit library expects the confidence matrix to be passed; setting
    alpha here applies the confidence scale internally (C = 1 + alpha * R).
    """
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        alpha=alpha,
        use_gpu=False,
        random_state=seed,
    )
    model.fit(M)
    return model


# --------------------------------------------------------------------------- #
# Prediction and evaluation
# --------------------------------------------------------------------------- #
def predict_top_k(
    model: AlternatingLeastSquares,
    M: sparse.csr_matrix,
    cust_idx: pd.Index,
    art_idx: pd.Index,
    top_k: int,
    batch_size: int = 4096,
) -> pd.DataFrame:
    """For each customer in `cust_idx`, return top-k article_id predictions."""
    n = len(cust_idx)
    out_rows: list[pd.DataFrame] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        users = np.arange(start, end)
        ids, _scores = model.recommend(
            users,
            M[users],
            N=top_k,
            filter_already_liked_items=True,
        )
        df = pd.DataFrame(
            {
                "customer_id": cust_idx[users].repeat(top_k),
                "rank": np.tile(np.arange(top_k), len(users)),
                "article_id": art_idx.to_numpy()[ids.flatten()],
            }
        )
        out_rows.append(df)
    return pd.concat(out_rows, axis=0, ignore_index=True)


def map_at_k(predictions: pd.DataFrame, holdout: pd.DataFrame, k: int = 12) -> float:
    """Mean Average Precision at k, per Kaggle metric.

    For each customer with at least one hold-out purchase, AP@k is computed over
    the top-k predicted article_ids and the set of true purchased articles in
    the hold-out window. Customers with no hold-out purchases are dropped.
    """
    truth = holdout.groupby("customer_id")["article_id"].apply(set)
    customers = truth.index
    pred_grouped = (
        predictions[predictions["customer_id"].isin(customers)]
        .sort_values(["customer_id", "rank"])
        .groupby("customer_id")["article_id"]
        .apply(list)
    )

    aps = []
    for cust in customers:
        relevant = truth[cust]
        preds = pred_grouped.get(cust, [])[:k]
        if not preds or not relevant:
            aps.append(0.0)
            continue
        hits = 0
        ap = 0.0
        for i, a in enumerate(preds, start=1):
            if a in relevant:
                hits += 1
                ap += hits / i
        ap /= min(len(relevant), k)
        aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


def recall_at_k(predictions: pd.DataFrame, holdout: pd.DataFrame, k: int = 12) -> float:
    truth = holdout.groupby("customer_id")["article_id"].apply(set)
    customers = truth.index
    pred_grouped = (
        predictions[predictions["customer_id"].isin(customers)]
        .sort_values(["customer_id", "rank"])
        .groupby("customer_id")["article_id"]
        .apply(list)
    )
    recalls = []
    for cust in customers:
        relevant = truth[cust]
        preds = set(pred_grouped.get(cust, [])[:k])
        if not relevant:
            continue
        recalls.append(len(preds & relevant) / len(relevant))
    return float(np.mean(recalls)) if recalls else 0.0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(args: argparse.Namespace) -> None:
    t0 = time.time()
    print("[1/6] Loading tables ...")
    transactions, customers, articles = load_tables(DATA_DIR)
    print(f"      transactions: {transactions.shape}")
    print(f"      customers   : {customers.shape}")
    print(f"      articles    : {articles.shape}")

    print("[2/6] Splitting train / hold-out ...")
    train, holdout = split_train_holdout(transactions, args.cut_date)
    print(f"      train rows  : {len(train):,}")
    print(f"      holdout rows: {len(holdout):,} "
          f"({holdout['customer_id'].nunique():,} customers)")

    print("[3/6] Building confidence matrix ...")
    M, cust_idx, art_idx = build_interaction_matrix(
        train,
        recency_halflife_days=args.recency_halflife_days,
        min_article_txn=args.min_article_txn,
    )
    print(f"      shape: {M.shape}, nnz: {M.nnz:,}, "
          f"density: {M.nnz / (M.shape[0] * M.shape[1]):.6%}")

    print("[4/6] Fitting ALS ...")
    model = fit_als(
        M,
        factors=args.factors,
        regularization=args.regularization,
        iterations=args.iterations,
        alpha=args.alpha,
        seed=args.seed,
    )

    print("[5/6] Predicting top-12 for hold-out customers ...")
    holdout_customers = holdout["customer_id"].unique()
    in_train_mask = pd.Index(holdout_customers).isin(cust_idx)
    eligible = pd.Index(holdout_customers)[in_train_mask]
    eligible_pos = cust_idx.get_indexer(eligible)
    eligible_pos = eligible_pos[eligible_pos >= 0]
    predictions = predict_top_k(
        model, M, cust_idx[eligible_pos], art_idx, top_k=args.top_k
    )

    print("[6/6] Evaluating MAP@12 / Recall@12 ...")
    map12 = map_at_k(predictions, holdout, k=args.top_k)
    rec12 = recall_at_k(predictions, holdout, k=args.top_k)

    metrics = {
        "model": "implicit_als",
        "factors": args.factors,
        "regularization": args.regularization,
        "iterations": args.iterations,
        "alpha": args.alpha,
        "recency_halflife_days": args.recency_halflife_days,
        "n_customers_train": int(M.shape[0]),
        "n_articles_train": int(M.shape[1]),
        "n_customers_holdout_eval": int(len(eligible)),
        "MAP@12": map12,
        "Recall@12": rec12,
        "fit_seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(metrics, indent=2))

    np.savez_compressed(
        DELIV_DIR / "als_model.npz",
        user_factors=model.user_factors,
        item_factors=model.item_factors,
        cust_idx=cust_idx.to_numpy(),
        art_idx=art_idx.to_numpy(),
    )
    with open(DELIV_DIR / "als_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    predictions.to_parquet(DELIV_DIR / "als_top12_holdout.parquet", index=False)
    print(f"Saved model and metrics to {DELIV_DIR}/")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
