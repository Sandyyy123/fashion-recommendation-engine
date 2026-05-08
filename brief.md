# Project 13 - H&M Personalized Fashion Recommendations

**Track:** Machine Learning Engineer / Recommender Systems
**Difficulty:** 8/10
**Status:** Phase 1 - scaffolded code-only

## Goal

Build a personalized fashion recommender for H&M customers that, given a customer ID, returns a ranked list of up to 12 article IDs the customer is most likely to purchase in the next seven days. The Kaggle competition metric is Mean Average Precision at 12 (MAP@12) on the held-out final week of transactions.

## Business framing (DACH retail / Zalando-class)

H&M Group operates roughly 4,400 stores across 70+ markets, with a digital arm that competes directly with Zalando, About You, and Otto in the DACH region. Personalization at H&M-scale needs to handle:

- Sparse implicit feedback (purchases, no explicit ratings)
- Heavy cold-start on both sides (new fashion drops every week, new customers daily)
- Strong recency / seasonality signal (one season's bestseller is dead the next)
- Rich item-side features (article metadata, product image, text description, color, garment group)
- Latency budget under 50 ms per request at retrieval time

The same pattern applies to Zalando, About You, Otto, Bonprix, and Breuninger in the DACH market.

## Data

- **transactions_train.csv** approx 31 million rows: customer_id, article_id, t_dat, price, sales_channel_id (2018-09-20 to 2020-09-22)
- **customers.csv** approx 1.4 million rows: customer_id, age, postal_code, FN/Active/club_member_status/fashion_news_frequency
- **articles.csv** approx 105k rows: article_id, product_type, product_group, garment_group, colour, perceived_colour, index_name, section, department, detail_desc
- **images/** approx 26 GB of JPGs, one per article_id, hierarchical folder structure

Source: Kaggle competition `h-and-m-personalized-fashion-recommendations`. See `data/README.md` for download command and storage notes. Total dataset is roughly 30 GB; tabular CSVs alone are roughly 1 GB compressed. **Phase 1 documents the dataset only and does not download it.**

## Methodology - hybrid recommender

Two complementary models are scaffolded. Both will be executed in a later session.

### Baseline (`src/model_baseline.py`)

Implicit-feedback collaborative filtering using Alternating Least Squares (ALS) on the customer-by-article interaction matrix [Hu et al. 2008]. The `implicit` Python library provides a Cython-optimised ALS implementation that handles sparse confidence-weighted matrices at H&M scale (1.4M customers x 105k articles) in minutes on a single machine. The baseline uses transaction count as the implicit confidence signal, with a recency decay so that older purchases contribute less. Recall@12 and MAP@12 are reported on the final-week hold-out per the competition metric.

### Advanced (`src/model_advanced.py`)

Two-tower neural recommender (DSSM-style) [Huang et al. 2013, Yi et al. 2019] in PyTorch. The user tower consumes customer features (age bucket, club membership, fashion-news frequency, postcode region embedding) plus a sequence embedding of the customer's recent purchase history. The item tower consumes article metadata (product type, garment group, colour, section, department) plus an optional CLIP-style image embedding. Both towers project to a shared `d`-dimensional space, trained with sampled-softmax over in-batch negatives plus mixed negative sampling [Yang et al. 2020] to correct popularity bias [Yi et al. 2019]. At inference, item-tower embeddings are precomputed and indexed in a Faiss HNSW ANN structure [Johnson et al. 2019, Malkov & Yashunin 2018]; per-user retrieval is then a single user-tower forward plus a top-k ANN query, well below the 50 ms latency budget. An optional session-aware reranker (SASRec [Kang & McAuley 2018] or BERT4Rec [Sun et al. 2019]) reorders the top-100 retrieved items using the customer's most recent session.

## Evaluation

- Primary: MAP@12 on transactions in the final week (2020-09-16 to 2020-09-22), per the Kaggle competition metric
- Secondary: Recall@12, NDCG@12, coverage, intra-list diversity, calibration (Steck 2018)
- Cold-start variant: customers with fewer than three historical purchases handled by popularity baseline plus content-tower-only retrieval

## Deliverables in this scaffold

- `data/README.md` - exact `kaggle competitions download` command, extraction notes, storage layout
- `notebooks/01_EDA.ipynb` - non-executed EDA skeleton (load, schema, target distribution, sparsity, recency, seasonality narrative cells)
- `reports/references.md` - 30+ verified academic references (CrossRef-resolved)
- `src/model_baseline.py` - ALS baseline, runnable, not executed
- `src/model_advanced.py` - two-tower neural recommender, runnable, not executed
- `manuscripts/manuscript.md` - 4000-5000 word IMRaD manuscript
- `deliverables/presentation.html` - self-contained HTML deck
- `checkpoint.json` - phase status JSON
