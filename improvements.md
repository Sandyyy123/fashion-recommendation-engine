# Improvements - Project 13 - H&M Personalized Fashion Recommendations

Role B (IMPROVER) review. Recommendations only. No file modifications.

## Top recommendation

**Add a learning-to-rank GBDT reranker (LightGBM LambdaMART or CatBoost YetiRank) on top of the two-tower retrieval, trained on the top-100 ANN candidates with hand-engineered customer-article-context features.** The single highest-leverage move. Every public H&M Kaggle solution at gold/silver level (Radek Osmulski, Chris Deotte, the Kaggle Grandmaster ensembles) shows the same lift pattern: ALS or two-tower retrieval gets you to MAP@12 around 0.022-0.026, and a LambdaMART reranker over the top-100 candidates with features like article age, customer-article cosine, days-since-customer-last-purchase, garment-group affinity, price-quantile match, and weekly trend lifts MAP@12 to 0.034-0.038 (a 40-60% relative gain). The current scaffold names a sequential reranker (SASRec / BERT4Rec) as a Phase 3 extension only; in practice the GBDT reranker is the bigger win, ships faster, and is what the leaderboard winners actually used. Concrete next steps: add `src/model_reranker.py` with a LightGBM ranker (objective="lambdarank", group=customer), persist `lgbm_reranker.txt`, and report MAP@12 with and without rerank in Table 2 of the manuscript.

---

## Weakness 1 - No GBDT reranker (HIGH)

**Gap.** Manuscript Section 5.2 names "no learned reranker" as a limitation but punts to Phase 3. The two-tower retriever alone is well below what the dataset supports.

**Fix.** Add a LightGBM LambdaMART reranker over the top-100 candidates from the union of ALS and two-tower retrieval. Features per (customer, article) candidate row: ALS score, two-tower cosine, popularity in last 7/14/28 days, customer-garment-group affinity (count of past purchases in that garment group), customer-section affinity, article age in days, customer's median price quantile vs article price, channel match (does the customer mostly shop online vs store), customer-recency (days since last purchase), week-of-year. Train on the second-to-last week's positives with sampled negatives from the candidate pool. Expect MAP@12 lift of 30-60% relative.

---

## Weakness 2 - No image embeddings actually computed (HIGH)

**Gap.** The scaffold supports an optional `articles_image_emb.npy` but provides no code to generate it. Manuscript notes that image embeddings "roughly double" lift, yet there is no precompute path.

**Fix.** Add `src/precompute_image_embeddings.py` that loads CLIP ViT-B/32 (or OpenCLIP `laion/CLIP-ViT-L-14-laion2B-s32B-b82K`), batches the 105k article JPGs at resolution 224 with 8-bit half-precision on GPU, and saves `articles_image_emb.npy` at float16 dim 512. Roughly 30 minutes on a single L4 / RTX 4090. Document a fallback to `timm` ResNet-50 ImageNet embeddings for CPU-only runs. Then flip `Config.use_image_emb=True` as the default, not an opt-in.

---

## Weakness 3 - No text features from `prod_name` and `detail_desc` (MEDIUM)

**Gap.** The item tower throws away the two free-text columns despite manuscript Section 2 acknowledging they carry signal. `detail_desc` is multi-sentence, English, and discriminative for fashion (sleeve length, fit, fabric).

**Fix.** Add a sentence-transformer pass (`sentence-transformers/all-MiniLM-L6-v2`, dim 384) over `prod_name + " " + detail_desc` and persist `articles_text_emb.npy`. Concatenate with the categorical embeddings and image embedding in `ItemTower.forward`. MiniLM runs at roughly 5k articles/sec on a 16-core CPU, so the 105k catalogue takes about 30 seconds. Cite Reimers & Gurevych 2019 (Sentence-BERT, EMNLP, DOI:10.18653/v1/D19-1410).

---

## Weakness 4 - No `requirements.txt` and no environment pin (HIGH)

**Gap.** The scaffold imports `implicit`, `torch`, `faiss`, `pandas`, `scipy`, `numpy`, but the project root has no `requirements.txt`, no `pyproject.toml`, and no `environment.yml`. A reviewer or future-you cannot reproduce the runtime.

**Fix.** Add a single `requirements.txt` at project root with pinned versions: `implicit==0.7.2`, `torch==2.3.0`, `faiss-cpu==1.8.0` (or `faiss-gpu`), `pandas==2.2.2`, `scipy==1.13.1`, `numpy==1.26.4`, `scikit-learn==1.5.0`, `lightgbm==4.4.0`, `pyarrow==16.1.0`, `sentence-transformers==2.7.0`, `open_clip_torch==2.24.0`, `tqdm==4.66.4`. Add a one-line `python -m venv .venv && pip install -r requirements.txt` block to `data/README.md`.

---

## Weakness 5 - No ablation matrix or statistical test on the metric (MEDIUM)

**Gap.** Manuscript Section 4 lists only the four point estimates of MAP@12 across models. No bootstrap CI, no significance test on per-customer AP@12, no ablation breakdown of which two-tower component (history pooling, image features, mixed negatives, logQ correction) drives the lift.

**Fix.** Add a bootstrap-percentile 95% CI on MAP@12 (resample customers with replacement, 1000 iterations) and a paired Wilcoxon signed-rank test on per-customer AP@12 between (i) ALS vs two-tower and (ii) two-tower vs hybrid. Add a 2x2x2 ablation table that toggles {image-emb, text-emb, mixed-negatives}; this is what reviewers expect for a Phase 2 paper.

---

## Weakness 6 - Cold-start fallback is named but not implemented in code (MEDIUM)

**Gap.** Manuscript Section 3.2 and 3.5 describe a "popularity baseline" for cold customers and a content-only retrieval for cold articles. Neither is in `src/model_baseline.py` or `src/model_advanced.py`. The baseline silently scores zero for cold customers.

**Fix.** Add `src/popularity_fallback.py` that computes a 14-day rolling top-12 by `garment_group_no` and a global top-12 over the same window. In `predict_top_k` and the two-tower `evaluate`, check `customer_id in cust_idx` first; if not, return the popularity list segmented by `club_member_status`. For cold articles, ensure the two-tower item tower is given every article in the catalogue (not just trained ones) so retrieval can surface new drops; the current `articles_enc = articles_enc[articles_enc["article_id"].isin(art_arr)]` filter explicitly drops them.

---

## Weakness 7 - Diversity, calibration, and fairness only named in prose (MEDIUM)

**Gap.** Manuscript Section 3.6 lists Steck calibration and Adomavicius-Kwon intra-list diversity as secondary metrics, but neither is computed in the evaluation harness in either model script. The Table 2 columns will stay TBD because the code doesn't compute them.

**Fix.** Add three functions in a shared `src/eval_utils.py`: `intra_list_diversity(top12, articles_df, cat_col="garment_group_no")` (1 minus mean pairwise category equality), `coverage(predictions, n_articles)` (unique articles surfaced / catalogue size), `kl_calibration(predictions, history, cat_col)` (Steck KL of recommended vs historical genre share). Wire them into both model scripts so Table 2 fills in.

---

## Weakness 8 - Notebook EDA does not look at returns or repeat purchases (LOW-MEDIUM)

**Gap.** The notebook has good schema, sparsity, and seasonality cells, but no analysis of repeat-purchase behaviour (a fashion-specific quirk: customers re-buy the same article id in different sizes / colours via the `product_code` parent), no return-rate proxy (price spikes / negative price events), and no analysis of the time gap between first and last purchase per customer (loyalty cohort).

**Fix.** Add three EDA cells: (i) repeat-purchase rate by `product_code` (the parent of size/colour variants) so the recommender can score the parent and surface the right variant, (ii) per-customer purchase-gap histogram (informs the recency half-life), (iii) per-section new-article arrival rate per week (motivates the cold-article handler). Cite the `product_code` aggregation trick from the public Radek Osmulski H&M Kaggle write-up.

---

## Weakness 9 - No fairness / bias audit on age and `club_member_status` (LOW)

**Gap.** Manuscript Section 5.3 mentions calibration and diversity but nothing on demographic parity. With age and club membership as user-tower inputs, the model can amplify subgroup biases (e.g. older customers under-served by trend-driven retrieval).

**Fix.** Add a per-subgroup MAP@12 breakdown (age decile x club_member_status). If the gap between best and worst subgroup MAP@12 exceeds 30% relative, flag in the discussion and consider per-subgroup popularity priors. One-time analysis, no model change required, gives the manuscript a fairness paragraph that is currently missing.

---

## Weakness 10 - Manuscript prose: future-tense "will be populated" leaks throughout Results (LOW)

**Gap.** Section 4 mixes spec language with results-tense language ("Headline figures are reported as `<TBD>` and will be populated...") which reads as a project plan, not a paper. Reviewers of a Phase 1 deliverable still expect tighter framing.

**Fix.** Rewrite Section 4 in the present tense as a "Planned evaluation" section, move it under Methods (3.6), and replace the empty Results section with a "Phase 1 deliverable summary" table that lists what is shipped (code, refs, EDA cells, manuscript) versus what is deferred. This is the convention used by Liora-style scaffolded papers and avoids the awkward TBD tables in the body.

---

## Priority summary

| # | Weakness | Priority |
|---|----------|----------|
| 1 | LightGBM LambdaMART reranker | HIGH |
| 2 | CLIP image embeddings precompute | HIGH |
| 4 | requirements.txt with pins | HIGH |
| 3 | Sentence-transformer text features | MEDIUM |
| 5 | Bootstrap CI + ablation table | MEDIUM |
| 6 | Implement cold-start fallback in code | MEDIUM |
| 7 | Compute diversity / coverage / calibration | MEDIUM |
| 8 | Repeat-purchase + product_code EDA | LOW-MEDIUM |
| 9 | Per-subgroup MAP@12 fairness audit | LOW |
| 10 | Tighten Results section prose | LOW |

End of improvements.md.
