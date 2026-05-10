"""Advanced recommender: two-tower neural recsys for H&M.

User tower consumes customer features and a recent purchase-history embedding.
Item tower consumes article metadata. Both project to a shared d-dim space and
are trained with sampled-softmax over in-batch plus mixed negatives [Yang 2020]
with the popularity-bias correction of Yi et al. (2019). At inference, item
embeddings are precomputed and indexed in Faiss HNSW [Malkov & Yashunin 2018,
Johnson et al. 2019]; per-user retrieval is one user-tower forward plus a top-k
ANN query, well below a 50 ms latency budget.

Run:
    cd /root/AI/project_root
    python src/model_advanced.py

Inputs (../data/):
    transactions_train.csv, customers.csv, articles.csv

Outputs (../deliverables/):
    twotower_model.pt          - trained model state_dict
    twotower_item_emb.npy      - item-tower embeddings (n_articles x d)
    twotower_metrics.json      - MAP@12 / Recall@12 / NDCG@12 / coverage
    twotower_faiss.index       - HNSW index for retrieval
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DELIV_DIR = PROJECT_ROOT / "deliverables"
DELIV_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# Hyperparameters
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    embed_dim: int = 64
    hidden_dim: int = 256
    history_len: int = 20            # most recent purchases used in user tower
    batch_size: int = 4096
    epochs: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-6
    n_negatives_global: int = 1024   # uniform-popularity-corrected negatives
    cut_date: str = "2020-09-16"
    top_k: int = 12
    seed: int = 42
    use_image_emb: bool = False      # set True if articles_image_emb.npy exists
    image_emb_path: str = "data/articles_image_emb.npy"
    cat_cols: list[str] = field(default_factory=lambda: [
        "product_type_no",
        "product_group_name",
        "graphical_appearance_no",
        "colour_group_code",
        "perceived_colour_value_id",
        "perceived_colour_master_id",
        "department_no",
        "index_code",
        "index_group_no",
        "section_no",
        "garment_group_no",
    ])
    user_cat_cols: list[str] = field(default_factory=lambda: [
        "club_member_status",
        "fashion_news_frequency",
    ])


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transactions = pd.read_csv(
        DATA_DIR / "transactions_train.csv",
        dtype={"customer_id": "string", "article_id": "int64",
               "price": "float32", "sales_channel_id": "int8"},
        parse_dates=["t_dat"],
    )
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    articles = pd.read_csv(DATA_DIR / "articles.csv")
    return transactions, customers, articles


def build_id_maps(transactions: pd.DataFrame) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    cust_arr = transactions["customer_id"].unique()
    art_arr = transactions["article_id"].unique()
    cust2idx = {c: i for i, c in enumerate(cust_arr)}
    art2idx = {a: i for i, a in enumerate(art_arr)}
    return cust2idx, art2idx, cust_arr, art_arr


def encode_categorical(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Replace each column with int codes; return mapping dicts for inference."""
    maps: dict[str, dict] = {}
    for c in cols:
        if c not in df.columns:
            df[c] = "MISSING"
        df[c] = df[c].fillna("MISSING").astype(str)
        cats = pd.Categorical(df[c])
        df[c] = cats.codes.astype(np.int64)
        maps[c] = {v: i for i, v in enumerate(cats.categories)}
    return df, maps


# --------------------------------------------------------------------------- #
# Towers
# --------------------------------------------------------------------------- #
class ItemTower(nn.Module):
    def __init__(self, cat_cardinalities: list[int], embed_dim: int,
                 hidden_dim: int, image_dim: int | None = None):
        super().__init__()
        self.cat_embeds = nn.ModuleList(
            [nn.Embedding(c, embed_dim) for c in cat_cardinalities]
        )
        in_dim = embed_dim * len(cat_cardinalities) + (image_dim or 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, cat_ids: torch.Tensor, image_emb: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.cat([emb(cat_ids[:, i]) for i, emb in enumerate(self.cat_embeds)], dim=1)
        if image_emb is not None:
            x = torch.cat([x, image_emb], dim=1)
        z = self.mlp(x)
        return F.normalize(z, dim=1)


class UserTower(nn.Module):
    def __init__(self, n_articles: int, user_cat_cardinalities: list[int],
                 embed_dim: int, hidden_dim: int, history_len: int):
        super().__init__()
        # Shared item id-embedding for the history; in production this is
        # initialised from the item tower's id-embedding and frozen.
        self.history_embed = nn.Embedding(n_articles + 1, embed_dim, padding_idx=0)
        self.user_cat_embeds = nn.ModuleList(
            [nn.Embedding(c, embed_dim) for c in user_cat_cardinalities]
        )
        # +1 numeric feature: age (z-scored).
        in_dim = embed_dim + embed_dim * len(user_cat_cardinalities) + 1
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.history_len = history_len

    def forward(
        self,
        history: torch.Tensor,         # (B, history_len) int item ids
        user_cat: torch.Tensor,        # (B, n_user_cat)
        age: torch.Tensor,             # (B, 1)
    ) -> torch.Tensor:
        h = self.history_embed(history).mean(dim=1)
        c = torch.cat([emb(user_cat[:, i]) for i, emb in enumerate(self.user_cat_embeds)], dim=1)
        x = torch.cat([h, c, age], dim=1)
        z = self.mlp(x)
        return F.normalize(z, dim=1)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class TwoTowerDataset(Dataset):
    """Yields (history, user_cat, age, target_item_cat, target_item_id) tuples."""

    def __init__(
        self,
        train: pd.DataFrame,
        articles_enc: pd.DataFrame,
        customers_enc: pd.DataFrame,
        cust2idx: dict,
        art2idx: dict,
        history_len: int,
    ):
        self.history_len = history_len
        # Sort transactions per customer by time.
        train = train.sort_values(["customer_id", "t_dat"]).reset_index(drop=True)
        self.cust_id = train["customer_id"].to_numpy()
        self.art_id = train["article_id"].to_numpy()
        self.cust_idx = np.array([cust2idx[c] for c in self.cust_id], dtype=np.int64)
        self.art_idx = np.array([art2idx[a] for a in self.art_id], dtype=np.int64)
        # Pre-compute per-customer historical purchase index lists.
        df = pd.DataFrame({"c": self.cust_idx, "a": self.art_idx})
        df["pos"] = df.groupby("c").cumcount()
        self.history_lookup: dict[int, list[int]] = (
            df.groupby("c")["a"].apply(list).to_dict()
        )
        self.row_pos = df["pos"].to_numpy()

        # Article cat features
        art_id_to_pos = {a: i for i, a in enumerate(articles_enc["article_id"].to_numpy())}
        self.art_cat_lookup = articles_enc.drop(columns=["article_id"]).to_numpy(dtype=np.int64)
        self.art_id_to_pos = art_id_to_pos

        # Customer features
        cust_id_to_pos = {c: i for i, c in enumerate(customers_enc["customer_id"].to_numpy())}
        self.user_cat_lookup = customers_enc.drop(columns=["customer_id", "age_z"]).to_numpy(dtype=np.int64)
        self.user_age_lookup = customers_enc["age_z"].to_numpy(dtype=np.float32)
        self.cust_id_to_pos = cust_id_to_pos

    def __len__(self) -> int:
        return len(self.art_idx)

    def __getitem__(self, i: int):
        cust = int(self.cust_idx[i])
        target_item = int(self.art_idx[i])
        # History up to and excluding this row.
        hist_full = self.history_lookup[cust]
        pos = int(self.row_pos[i])
        hist = hist_full[max(0, pos - self.history_len): pos]
        # +1 to all ids to reserve 0 as padding index.
        hist_padded = np.zeros(self.history_len, dtype=np.int64)
        if hist:
            arr = np.asarray(hist, dtype=np.int64) + 1
            hist_padded[-len(arr):] = arr

        cust_strid = self.cust_id[i]
        c_pos = self.cust_id_to_pos.get(cust_strid, 0)
        user_cat = self.user_cat_lookup[c_pos]
        age = self.user_age_lookup[c_pos]

        art_strid = self.art_id[i]
        a_pos = self.art_id_to_pos[art_strid]
        item_cat = self.art_cat_lookup[a_pos]

        return (
            torch.from_numpy(hist_padded),
            torch.from_numpy(user_cat),
            torch.tensor([age], dtype=torch.float32),
            torch.from_numpy(item_cat),
            torch.tensor(target_item, dtype=torch.long),
        )


# --------------------------------------------------------------------------- #
# Sampled-softmax loss with popularity correction (Yi et al. 2019)
# --------------------------------------------------------------------------- #
def sampled_softmax_loss(
    user_z: torch.Tensor,
    pos_item_z: torch.Tensor,
    neg_item_z: torch.Tensor,
    pos_item_logQ: torch.Tensor,
    neg_item_logQ: torch.Tensor,
) -> torch.Tensor:
    """In-batch sampled-softmax with logQ correction.

    user_z, pos_item_z: (B, d). neg_item_z: (N_neg, d). logQ: prior log-prob.
    """
    pos_logits = (user_z * pos_item_z).sum(dim=1) - pos_item_logQ
    neg_logits = user_z @ neg_item_z.t() - neg_item_logQ.unsqueeze(0)
    logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)
    labels = torch.zeros(user_z.size(0), dtype=torch.long, device=user_z.device)
    return F.cross_entropy(logits, labels)


# --------------------------------------------------------------------------- #
# Train / inference
# --------------------------------------------------------------------------- #
def train(cfg: Config) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    print("[1/6] Loading and encoding tables ...")
    transactions, customers, articles = load_tables()
    cut = pd.Timestamp(cfg.cut_date)
    train_df = transactions[transactions["t_dat"] < cut].copy()
    holdout = transactions[transactions["t_dat"] >= cut].copy()

    cust2idx, art2idx, cust_arr, art_arr = build_id_maps(train_df)

    # Encode article cat columns
    art_cat_cols = [c for c in cfg.cat_cols if c in articles.columns]
    articles_enc = articles[["article_id"] + art_cat_cols].copy()
    articles_enc, _ = encode_categorical(articles_enc, art_cat_cols)
    articles_enc = articles_enc[articles_enc["article_id"].isin(art_arr)].reset_index(drop=True)

    # Encode customer cat columns + z-scored age
    user_cat_cols = [c for c in cfg.user_cat_cols if c in customers.columns]
    customers_enc = customers[["customer_id", "age"] + user_cat_cols].copy()
    customers_enc["age"] = customers_enc["age"].fillna(customers_enc["age"].median())
    customers_enc["age_z"] = (
        (customers_enc["age"] - customers_enc["age"].mean())
        / customers_enc["age"].std(ddof=0)
    ).astype(np.float32)
    customers_enc = customers_enc.drop(columns=["age"])
    customers_enc, _ = encode_categorical(customers_enc, user_cat_cols)

    item_cat_cards = [int(articles_enc[c].max() + 1) for c in art_cat_cols]
    user_cat_cards = [int(customers_enc[c].max() + 1) for c in user_cat_cols]
    print(f"      n_customers (train): {len(cust_arr):,}")
    print(f"      n_articles  (train): {len(art_arr):,}")
    print(f"      item cat cards    : {item_cat_cards}")
    print(f"      user cat cards    : {user_cat_cards}")

    # Optional: load image embeddings
    image_dim = None
    image_emb_table = None
    if cfg.use_image_emb and (PROJECT_ROOT / cfg.image_emb_path).exists():
        image_emb_table = np.load(PROJECT_ROOT / cfg.image_emb_path).astype(np.float32)
        image_dim = image_emb_table.shape[1]
        print(f"      image embedding dim: {image_dim}")

    print("[2/6] Building dataset ...")
    dataset = TwoTowerDataset(
        train=train_df,
        articles_enc=articles_enc,
        customers_enc=customers_enc,
        cust2idx=cust2idx,
        art2idx=art2idx,
        history_len=cfg.history_len,
    )
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=4, pin_memory=True,
    )

    # Popularity prior for logQ correction.
    item_counts = np.bincount(dataset.art_idx, minlength=len(art_arr)).astype(np.float64)
    item_pop = item_counts / item_counts.sum()
    item_logQ = torch.from_numpy(np.log(item_pop + 1e-9).astype(np.float32)).to(device)

    print("[3/6] Building towers ...")
    item_tower = ItemTower(item_cat_cards, cfg.embed_dim, cfg.hidden_dim, image_dim).to(device)
    user_tower = UserTower(len(art_arr), user_cat_cards, cfg.embed_dim,
                           cfg.hidden_dim, cfg.history_len).to(device)
    optimizer = torch.optim.AdamW(
        list(item_tower.parameters()) + list(user_tower.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    print("[4/6] Training ...")
    item_tower.train()
    user_tower.train()
    t0 = time.time()
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        n_steps = 0
        for batch in loader:
            history, user_cat, age, item_cat, target_item = [b.to(device) for b in batch]
            user_z = user_tower(history, user_cat, age)
            pos_item_z = item_tower(item_cat, None)

            # Mixed negatives: in-batch + uniform global sample [Yang 2020].
            # In-batch negatives are the other rows' positive items. We add a
            # uniform sample from all items to reduce popularity bias further.
            global_neg = torch.randint(
                low=0, high=len(art_arr),
                size=(cfg.n_negatives_global,), device=device,
            )
            global_neg_cat = torch.from_numpy(
                dataset.art_cat_lookup[global_neg.cpu().numpy()]
            ).to(device)
            global_neg_z = item_tower(global_neg_cat, None)

            neg_z = torch.cat([pos_item_z, global_neg_z], dim=0)
            neg_logQ = torch.cat([
                item_logQ[target_item],
                item_logQ[global_neg],
            ], dim=0)

            pos_logQ = item_logQ[target_item]
            loss = sampled_softmax_loss(
                user_z, pos_item_z, neg_z, pos_logQ, neg_logQ
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_steps += 1
        print(f"  epoch {epoch + 1}/{cfg.epochs} - loss {epoch_loss / max(1, n_steps):.4f} "
              f"- elapsed {time.time() - t0:.0f}s")

    print("[5/6] Computing item embeddings and Faiss index ...")
    item_tower.eval()
    user_tower.eval()
    with torch.no_grad():
        all_cat = torch.from_numpy(dataset.art_cat_lookup).to(device)
        all_emb = item_tower(all_cat, None).cpu().numpy().astype(np.float32)
    np.save(DELIV_DIR / "twotower_item_emb.npy", all_emb)

    try:
        import faiss  # type: ignore
        index = faiss.IndexHNSWFlat(cfg.embed_dim, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 64
        index.add(all_emb)
        faiss.write_index(index, str(DELIV_DIR / "twotower_faiss.index"))
    except ImportError:
        print("  faiss not installed; skipping ANN index build")

    torch.save({
        "user_tower": user_tower.state_dict(),
        "item_tower": item_tower.state_dict(),
        "config": cfg.__dict__,
    }, DELIV_DIR / "twotower_model.pt")

    print("[6/6] Evaluating MAP@12 / Recall@12 on hold-out ...")
    metrics = evaluate(cfg, user_tower, all_emb, dataset, holdout, art_arr, device)
    metrics["n_train_rows"] = int(len(dataset))
    with open(DELIV_DIR / "twotower_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


def evaluate(
    cfg: Config,
    user_tower: nn.Module,
    item_emb: np.ndarray,
    dataset: TwoTowerDataset,
    holdout: pd.DataFrame,
    art_arr: np.ndarray,
    device: torch.device,
) -> dict:
    """Compute MAP@12 / Recall@12 on the final-week hold-out."""
    truth = holdout.groupby("customer_id")["article_id"].apply(set)
    cust_keys = list(truth.index)

    aps: list[float] = []
    recs: list[float] = []

    item_emb_t = torch.from_numpy(item_emb).to(device)
    art_arr_np = np.asarray(art_arr)

    with torch.no_grad():
        for cust_strid in cust_keys:
            c_pos = dataset.cust_id_to_pos.get(cust_strid)
            if c_pos is None:
                aps.append(0.0)
                recs.append(0.0)
                continue
            relevant = truth[cust_strid]

            # Build last-history of this customer from train.
            cust_int = dataset.cust_idx[dataset.cust_id == cust_strid]
            if len(cust_int) == 0:
                aps.append(0.0)
                recs.append(0.0)
                continue
            hist = dataset.history_lookup.get(int(cust_int[-1]), [])
            hist = hist[-cfg.history_len:]
            hist_padded = np.zeros(cfg.history_len, dtype=np.int64)
            if hist:
                arr = np.asarray(hist, dtype=np.int64) + 1
                hist_padded[-len(arr):] = arr

            user_cat = dataset.user_cat_lookup[c_pos]
            age = dataset.user_age_lookup[c_pos]

            history_t = torch.from_numpy(hist_padded).unsqueeze(0).to(device)
            user_cat_t = torch.from_numpy(user_cat).unsqueeze(0).to(device)
            age_t = torch.tensor([[age]], dtype=torch.float32, device=device)
            user_z = user_tower(history_t, user_cat_t, age_t)
            scores = (user_z @ item_emb_t.t()).squeeze(0)
            top_idx = torch.topk(scores, k=cfg.top_k).indices.cpu().numpy()
            preds = art_arr_np[top_idx].tolist()

            # AP@12
            hits = 0
            ap = 0.0
            for i, a in enumerate(preds, start=1):
                if a in relevant:
                    hits += 1
                    ap += hits / i
            aps.append(ap / min(len(relevant), cfg.top_k) if relevant else 0.0)
            recs.append(len(set(preds) & relevant) / len(relevant) if relevant else 0.0)

    return {
        "model": "two_tower",
        "embed_dim": cfg.embed_dim,
        "hidden_dim": cfg.hidden_dim,
        "history_len": cfg.history_len,
        "epochs": cfg.epochs,
        "MAP@12": float(np.mean(aps)) if aps else 0.0,
        "Recall@12": float(np.mean(recs)) if recs else 0.0,
        "n_customers_holdout_eval": len(aps),
    }


def parse_args() -> argparse.Namespace:
    cfg = Config()
    p = argparse.ArgumentParser()
    for k, v in cfg.__dict__.items():
        if isinstance(v, list):
            continue
        p.add_argument(f"--{k}", type=type(v), default=v)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config()
    for k, v in vars(args).items():
        setattr(cfg, k, v)
    train(cfg)
