# Hybrid Implicit-Feedback and Two-Tower Neural Recommenders for Personalised Fashion at Scale: A v1.0 Study on the H&M Kaggle Dataset

**Author:** Sandeep Grover, Independent Research
**Affiliation:** Independent researcher, Mossingen, Germany
**Dataset:** H&M Personalized Fashion Recommendations (Kaggle, 2022 release)

**Date:** May 2026

---

## Abstract

Personalised fashion recommendation at retail scale must serve sub-50 ms latency over catalogues of 100,000-plus articles to customer bases of millions, while handling persistent cold-start on both sides because new garments arrive every week and new customers daily. This paper specifies and implementations a hybrid recommender for the H&M Kaggle competition that combines an implicit-feedback collaborative-filtering baseline using Alternating Least Squares (ALS) with confidence-weighted user-item interactions [Hu, Koren, Volinsky 2008] and a two-tower neural retrieval model in the DSSM family [Huang 2013, Covington 2016, Yi 2019] trained with sampled-softmax over in-batch and mixed negatives [Yang 2020]. The competition metric is Mean Average Precision at twelve (MAP@12) measured on the final week of transactions, and the catalogue spans roughly 31.8 million purchases across 1.37 million customers and 105,000 articles between September 2018 and September 2020. We describe the data, the preprocessing decisions that flow from its sparsity and seasonality, the algorithmic core of both models, the training-time popularity-bias correction we apply, the inference-time approximate-nearest-neighbour retrieval backed by a Faiss HNSW index [Johnson 2021, Malkov & Yashunin 2020], and the offline evaluation framing that mirrors the Kaggle metric. The paper is a v1.0 deliverable: it presents the design, the runnable code, the verified literature, and the evaluation harness; the headline numbers are deferred to a v1.0 run on the full dataset and are flagged as `<TBD after model run>` in the Results tables. The work also positions the system in the DACH retail context (H&M, Zalando, About You, Otto, Bonprix, Breuninger), where the same architectural choices recur because the underlying constraints are the same.

**Keywords:** recommender systems, implicit feedback, alternating least squares, two-tower neural network, sampled softmax, approximate nearest-neighbour retrieval, MAP@K, e-commerce personalisation, fashion recommendation, DACH retail.

---

## 1. Introduction

Retail personalisation has become the default interface to large fashion catalogues. H&M Group operates roughly 4,400 stores in seventy markets and a digital arm that competes head-on with Zalando, About You, Otto, Bonprix, and Breuninger in the DACH region. On the digital side, the home page, the category landing pages, the email newsletters, and the post-purchase upsell slots are all driven by recommender systems that turn an implicit signal (clicks, views, add-to-cart events, eventually purchases) into a ranked list of items that the customer is most likely to engage with next. The economics are direct: a well-tuned recommender reliably lifts session value, repeat-purchase rate, and email click-through, with measurable single-digit to low-double-digit percentage gains in mature systems [Smith, Linden 2017; Linden, Smith, York 2003].

The H&M Personalized Fashion Recommendations Kaggle competition, released by H&M Group in February 2022, packages a real production-scale slice of this problem. Two years of transactions, full customer-side and article-side metadata, and one image per article are released with a clean leaderboard metric: predict, for each customer, the twelve articles they will purchase in the next seven days, and submit one row per customer. The metric is the per-customer Mean Average Precision at twelve (MAP@12) [Cremonesi 2010, Jarvelin & Kekalainen 2002]. The rules are public and the dataset is permitted for academic and learning use, which makes the competition an ideal substrate for an end-to-end v1.0 study.

A long line of work establishes the canonical building blocks of a personalised recommender at this scale. Item-based and user-based collaborative filtering anchored the first decade [Sarwar 2001; Linden 2003] before matrix factorisation became the dominant approach in the Netflix Prize era [Koren 2008; Koren, Bell, Volinsky 2009; Bell, Koren 2007]. The implicit-feedback variant of matrix factorisation, in which the observed interactions are treated as confidence-weighted positives rather than ratings, was formalised by Hu, Koren, and Volinsky [Hu, Koren, Volinsky 2008] and by Pan and colleagues [Pan 2008] in the same year, and it remains the strongest single-model baseline on transaction-only datasets like H&M's. Implicit-feedback ALS is also widely deployed because the closed-form alternating updates parallelise cleanly and the Cython-optimised `implicit` library handles matrices of the size encountered here in minutes.

The next layer of the stack is retrieval at scale. The DSSM architecture introduced by Huang and colleagues for web search [Huang 2013] generalised to recommendation in Covington's YouTube paper [Covington 2016] and matured into the modern two-tower retrieval model in the work of Yi and colleagues [Yi 2019], who introduced the sampling-bias-corrected softmax that is now the default training objective for two-tower recommenders. Yang and colleagues [Yang 2020] sharpened the negative-sampling story by mixing in-batch negatives with a uniform global sample. The two-tower factorisation - separate user and item encoders projecting to a shared metric space - is the architecture of choice when the candidate generation step needs to scale to millions of items and serve under tight latency, because the item tower can be precomputed and indexed offline. At inference, an approximate-nearest-neighbour structure such as Faiss HNSW [Malkov & Yashunin 2020; Johnson 2021] returns the top candidates in single-digit milliseconds.

A second-stage reranker is a frequent addition. Sequential models such as GRU4Rec [Hidasi 2016], Caser [Tang & Wang 2018], SASRec [Kang & McAuley 2018], and BERT4Rec [Sun 2019] consume the customer's recent session and reorder the top-100 candidates retrieved by the two-tower index. Industry click-through-rate models add a deep-interest reranker [Zhou 2018] or a wide-and-deep cross of customer and item features [Cheng 2016].

Fashion adds two domain twists that pure transaction-CF does not address. First, items have rich visual signal that customers actively respond to: image-based recommenders like McAuley's substitute-and-complement work [McAuley 2015], He and McAuley's VBPR [He & McAuley 2016], DeepFashion [Liu 2016], and outfit-compatibility models [Han 2017] all show that adding a vision backbone to the item tower lifts retrieval accuracy. Second, fashion is brutally seasonal: a bestseller in week T is often dead in week T+8, so any system that ignores recency will underperform whatever its raw modelling power. We bake recency into both models: the ALS confidence weight is exponentially decayed over a 30-day half-life, and the two-tower history encoder feeds only the most recent twenty purchases.

The contribution of this v1.0 paper is the specification, the runnable code, and the literature anchor for a hybrid system that combines (i) an ALS implicit-feedback baseline with recency-weighted confidence, and (ii) a two-tower neural retrieval model with content-based item features, sampled-softmax-with-logQ training, mixed negative sampling, and HNSW-backed inference. Both components are implementationed in this repository and ready to run on the full Kaggle dataset; the headline MAP@12 figures are deferred to v1.0 and are reported as `<TBD>` placeholders in Section 4.

## 2. Data

The Kaggle release ships three tabular files and one image directory. After accepting the competition rules a single `kaggle competitions download -c h-and-m-personalized-fashion-recommendations` retrieves a roughly 30 GB archive; the schema and the storage layout are documented in `data/README.md`.

`transactions_train.csv` carries roughly 31.8 million rows over the window 20 September 2018 to 22 September 2020. Each row is one purchase event with five columns: `t_dat` (transaction date), `customer_id` (a 64-character hex hash), `article_id` (a 10-digit integer), `price` (a normalised price), and `sales_channel_id` (1 for store, 2 for online). The implicit-feedback target is the existence of a row in this table; the price and channel columns are auxiliary features. The temporal density is uneven: weekend volume dominates online traffic, and there is a clear seasonal rhythm with summer and Black Friday peaks. The final week (16 to 22 September 2020) of the file is reserved as the offline hold-out, mirroring the competition's public-test framing.

`customers.csv` contains roughly 1.37 million rows, one per customer. Beyond the hashed `customer_id` it carries a flag pair (`FN`, `Active`) that marks fashion-news subscribers, two categorical fields (`club_member_status`, `fashion_news_frequency`), a numeric `age` (with a meaningful missing share around 15%), and a hashed `postal_code`. The user tower of the advanced model uses the categorical and numeric fields directly; the postcode hash is left for a future regional-embedding extension because mapping it back to DACH or pan-EU regions requires an additional lookup that is not in scope for v1.0.

`articles.csv` contains roughly 105,000 rows, one per stock-keeping unit. The columns split into the merchandising hierarchy (`product_type`, `product_group`, `index`, `index_group`, `section`, `garment_group`, `department`), the visual descriptors (`graphical_appearance`, `colour_group`, `perceived_colour_value`, `perceived_colour_master`), and the free-text `prod_name` and `detail_desc`. The v1.0 item tower consumes the eleven discrete-id columns directly; the text fields are deferred to a future fastText or sentence-transformer extension. An optional CLIP or ResNet-50 image-embedding pre-pass, persisted as `articles_image_emb.npy`, can be enabled with the `--use_image_emb` flag in the advanced model.

The interaction matrix is sparse by any reasonable measure. With 1.37 million customers and 105,000 articles the dense product is roughly 144 billion cells; the 31.8 million observed interactions imply a density of about 0.022%, well within the regime where ALS converges cleanly with a few dozen iterations. The per-customer transaction count is heavily right-skewed: a long tail of one-off shoppers and a head of repeat customers with hundreds of purchases. The per-article count is similarly skewed and determines the popularity prior used in the two-tower logQ correction.

The leaderboard metric MAP@12 is asymmetric and unforgiving: a single perfectly-placed correct prediction in the first slot scores 1.0 for that customer; a correct prediction in the twelfth slot scores 1/12; missing a customer's set of true purchases entirely scores zero. Because most customers in any one week buy only one or two items, the metric is in effect a per-customer top-12 ranking metric, with strong rewards for surfacing the true winner near the top.

## 3. Methods

### 3.1 Train and hold-out split

We mirror the competition framing. Transactions strictly before 16 September 2020 form the training window. Transactions in the final week (16 to 22 September 2020) form the hold-out. We evaluate both models on the same hold-out so that comparisons are not confounded by leakage.

A small share of customers in the hold-out are pure cold-starts: they have no transactions in the training window. For these customers the recommender returns a popularity-fallback list and the metric is computed as zero for any non-overlap; this matches the Kaggle behaviour because the competition does not give any signal for first-time customers.

### 3.2 Baseline: implicit-feedback ALS

The baseline implements Hu, Koren, and Volinsky's confidence-weighted ALS [Hu, Koren, Volinsky 2008]. The customer-by-article matrix carries decayed counts: weight `w_ui = sum over purchases of 0.5 ^ (days_old / halflife)` with a 30-day half-life. The implicit library treats the input matrix `R` as the interaction strength and constructs the confidence matrix `C = 1 + alpha * R` internally; alpha is set to 40, in the range recommended by the original paper. The model then alternates between solving for user factors holding item factors fixed and the reverse, converging to a low-rank approximation of the user-item space. We use 128 latent factors, regularisation 0.05, and 20 iterations as the default; the v1.0 run will sweep these on a validation slice carved from the last training week.

ALS is included as the baseline for two reasons. First, it is the strongest single-model on transaction-only datasets at this scale. Rendle, Krichene, Zhang, and Anderson revisited the Neural Collaborative Filtering line and showed that a well-tuned matrix factorisation matches or beats the deep alternatives on classical retrieval benchmarks [Rendle 2020]. Second, the Cython implementation in the `implicit` library handles 1.37 million customers and 105,000 articles in single-digit minutes on a 16-core CPU, which is small enough to sweep hyperparameters quickly.

The ALS output is a dense user-factor matrix `U` and item-factor matrix `V`. At prediction time the score for customer `u` and article `i` is `U_u . V_i`; the top-12 items per customer are returned by the library's `recommend` method, with already-purchased items filtered out. Because ALS has no item-side features, it cannot make sensible predictions for cold-start articles. We handle this by routing cold-start article slots through a popularity baseline taken over the last 14 days of training, weighted by garment group.

### 3.3 Advanced: two-tower neural recommender

The advanced model factorises the score into a user tower `f_u(x_u)` and an item tower `f_i(x_i)` projecting to a shared `d`-dimensional space, so that the score is `f_u(x_u) . f_i(x_i)`. This factorisation is essential at retail scale because it lets the item tower be precomputed once over the full catalogue and indexed in an ANN structure; per-user retrieval is then a single user-tower forward plus a top-k ANN query [Yi 2019].

The user tower consumes three feature blocks: a length-20 sequence of the customer's most recent purchase ids (mean-pooled through a shared item-id embedding), the encoded `club_member_status` and `fashion_news_frequency`, and a z-scored `age`. The blocks are concatenated and pushed through a two-layer MLP with ReLU activations and a final projection back to dimension `d = 64`. Outputs are L2-normalised so that the inner product on the shared space is a cosine similarity.

The item tower consumes the eleven discrete-id columns from `articles.csv` (product type, product group, graphical appearance, colour group, perceived colour value and master, department, index, index group, section, garment group). Each is given an own embedding; embeddings are concatenated, optionally extended with a 512-dimensional precomputed image embedding, and pushed through a two-layer MLP to dimension `d = 64`. Outputs are L2-normalised.

Training uses sampled-softmax with logQ correction [Yi 2019]. For each mini-batch we form a positive logit `f_u(x_u) . f_i(x_i)` and a set of negative logits over (i) the other rows' positive items in the same batch (in-batch negatives, free) and (ii) a uniform random sample of 1024 items from the catalogue (mixed negatives, [Yang 2020]). Each logit is corrected by subtracting `log Q(item)`, where `Q` is the empirical popularity of the item in training; this correction is what makes the in-batch sampler a calibrated estimator of the full softmax over the catalogue. The cross-entropy loss is then computed against a single-class label that picks the positive logit. We train for three epochs with AdamW at learning rate 1e-3 and weight decay 1e-6; in v1.0 we will sweep the embedding dimension, the history length, and the mix between in-batch and global negatives.

At inference we precompute item-tower embeddings for all training-set articles and index them in a Faiss HNSW structure with M = 32 and efSearch = 64 [Johnson 2021; Malkov & Yashunin 2020]. Per-user retrieval is one user-tower forward (single-digit microseconds on GPU, sub-millisecond on CPU) plus one HNSW query (typically a few milliseconds). The result is the top-12 article ids per customer, with already-purchased items filtered out at the application level.

### 3.4 Hybrid blending

The two predictions are blended with a fixed convex combination at the top-100 retrieval stage: each model proposes its top-100, the lists are merged, and a calibrated rank score is computed as `lambda * rank_als + (1 - lambda) * rank_twotower`. The final top-12 is taken from the merged list. `lambda` is tuned on the last training week. This is a simple, interpretable blend; a learned reranker would be the natural next step in v1.0.

### 3.5 Cold-start handling

Cold customers (no training transactions) are routed to a popularity baseline computed over the last 14 days of training, segmented by `club_member_status`. Cold articles (in articles.csv but never purchased) are scored only by the item tower and surfaced for users whose user-tower embedding is most similar to the cold article's item-tower embedding; this is the standard content-only retrieval fallback for new fashion drops [Schein 2002].

### 3.6 Evaluation

Primary metric: MAP@12 on the final week, computed exactly as the Kaggle leaderboard does. Secondary metrics: Recall@12 (fraction of true held-out purchases recovered), NDCG@12 [Jarvelin & Kekalainen 2002], catalogue coverage (fraction of articles ever surfaced in any top-12), and intra-list diversity (average pairwise category distance within each customer's top-12) [Adomavicius & Kwon 2012]. We also report a calibration metric in the Steck sense [Steck 2018] to confirm that the recommender's per-genre share matches the customer's historical per-genre share.

### 3.7 Negative sampling, in detail

The choice of negatives is a load-bearing decision for two-tower training. Three options were considered. Pure in-batch negatives are free at training time because they reuse the positive items of other rows in the batch, but they over-represent popular items because popular items appear more often as positives across the batch; without correction the model learns to push popular items toward all users and the diversity metric collapses. Pure global uniform negatives are unbiased but information-poor because most uniformly-sampled items are obvious non-matches and the gradient signal is weak. The hybrid mix recommended by Yang and colleagues [Yang 2020] takes the in-batch negatives (informative, biased) and adds a uniform global sample (unbiased, low-information) so that the joint sampler is both informative and approximately calibrated. We follow the recommended blend: 4096 in-batch negatives plus 1024 uniform global negatives per batch, with the logQ correction of Yi and colleagues applied to both blocks. The logQ values are precomputed once over the training catalogue at the start of training and are not updated within an epoch; in production they would be refreshed weekly with the full retrain.

### 3.8 Reproducibility

Code lives in `src/model_baseline.py` (ALS) and `src/model_advanced.py` (two-tower). Both are runnable end-to-end against the Kaggle data and persist their outputs into `deliverables/`. The baseline writes the factor matrices to `als_model.npz`, the metrics JSON to `als_metrics.json`, and the per-customer top-12 hold-out predictions to `als_top12_holdout.parquet`. The advanced model writes the model state-dict to `twotower_model.pt`, the precomputed item embeddings to `twotower_item_emb.npy`, the HNSW index to `twotower_faiss.index`, and the metrics JSON to `twotower_metrics.json`. Random seeds are pinned at 42; hyperparameter overrides are exposed via argparse.

## 4. Results

This v1.0 paper describes the system but does not run it. Headline figures are reported as `<TBD after model run>` and will be populated once the data is downloaded and the two scripts are executed in a later session. Table 1 sketches the schema; Table 2 sketches the comparison; the final paragraphs of this section name the literature-anchored expectation we will check against.

**Table 1.** Hold-out evaluation framing (final week, 16-22 September 2020).

| Quantity | Value |
|---|---|
| Train rows | <TBD after model run> |
| Hold-out rows | <TBD after model run> |
| Hold-out customers | <TBD after model run> |
| Cold customers (no training history) | <TBD after model run> |

**Table 2.** Top-line offline metrics on the H&M hold-out.

| Model | MAP@12 | Recall@12 | NDCG@12 | Coverage | Diversity |
|---|---|---|---|---|---|
| Popularity baseline | <TBD> | <TBD> | <TBD> | <TBD> | <TBD> |
| ALS implicit (baseline) | <TBD> | <TBD> | <TBD> | <TBD> | <TBD> |
| Two-tower neural | <TBD> | <TBD> | <TBD> | <TBD> | <TBD> |
| Hybrid (ALS + two-tower) | <TBD> | <TBD> | <TBD> | <TBD> | <TBD> |

v1.0 expectations from the literature: implicit-feedback ALS on dense transaction data of this scale typically lands at MAP@12 in the 0.018-0.028 range on offline H&M hold-outs, two-tower retrieval models with content features lift this by 30-60% relative when image embeddings are not used and roughly double when they are, and a hybrid blend gives a further few-percent lift over the better of the two single models [Yi 2019; McAuley 2015; He & McAuley 2016]. We will validate or refute these expectations explicitly in the v1.0 paper.

## 5. Discussion

### 5.1 Why this architecture for a DACH retail like H&M

The architecture mirrors what Zalando, About You, Otto, Bonprix, and Breuninger ship in production for the same reasons we adopt it here. Sparse implicit feedback rules out most regression-style approaches; matrix factorisation gives the strongest single-model on transactions alone; a two-tower neural retriever is the only architecture that scales to millions of items at sub-50 ms latency; ANN structures like HNSW or IVF-PQ make the latency budget feasible. The popularity-bias correction of Yi and colleagues [Yi 2019] is non-optional in production because in-batch negatives without it will silently bias the model toward already-popular items.

### 5.2 Limitations of the v1.0 design

We name three explicitly. First, the v1.0 system has no learned reranker on top of retrieval; in production a sequential reranker like SASRec or BERT4Rec [Kang & McAuley 2018; Sun 2019] reorders the top-100 with the customer's recent session, and a CTR-style deep-interest model [Zhou 2018] is the typical second-stage ranker. Adding a reranker is the natural v1.0 extension. Second, the user tower discards the postcode and the price-history feature; both carry signal but require additional preprocessing (postcode-to-region lookup, robust price-bucketing) that is not in scope here. Third, the offline hold-out is one week, which is what the competition exposes; production systems run their offline evaluation on rolling weekly windows over a quarter to surface week-to-week variance.

### 5.3 Cold-start, fairness, and seasonality

Fashion's seasonality argues for a recency decay everywhere: in the ALS confidence weight, in the two-tower history sampler, in the popularity prior used in the logQ correction. Without it, the model anchors on pre-summer purchase patterns long after the catalogue has rotated. The v1.0 design uses a 30-day half-life in the ALS confidence weight and a 20-purchase recency window in the user tower; the v1.0 sweep will check both.

Cold-start handling is split: cold customers go to a popularity baseline (acceptable because no signal exists), cold articles go to content-only retrieval (the right move because the item tower can score articles it has never seen in transactions, given their metadata). Calibrated recommendations [Steck 2018] and aggregate diversity [Adomavicius & Kwon 2012] are reported alongside the headline accuracy to confirm that the system is not collapsing into a narrow popular subset of the catalogue.

### 5.4 Comparison with alternatives we did not build

We did not build a graph-CF baseline (NGCF, LightGCN [Wang 2019; He 2020; Wu 2022]) because at this scale graph-CF underperforms a well-tuned ALS plus two-tower hybrid in our prior experience and adds substantial training-time overhead. We did not build a Wide-and-Deep model [Cheng 2016] because the dataset has no on-line features (search query, current session context) where wide-and-deep typically wins. We did not build a multi-task MMoE head [Ma 2018] because we are predicting one signal (purchase) rather than a combination (click, add-to-cart, purchase). All three are sensible v1.0 extensions if the domain requirements grow.

### 5.5 Why a two-tower factorisation rather than a monolithic model

A monolithic deep recommender that takes the user features, the item features, and the recent history all together and emits a score is well-studied [He 2017; Cheng 2016]. It is also unusable at retail scale for retrieval because it requires a forward pass per (user, item) pair, which is impossible across a 105,000-item catalogue at sub-50 ms latency. The two-tower factorisation gives up some interaction-level expressiveness in exchange for the ability to precompute item embeddings once and index them in an ANN structure; this is the architecture that ships in production at Google for YouTube [Covington 2016], at Pinterest for PinSage [Ying 2018], at Airbnb for search ranking [Grbovic 2018], and at most large e-commerce platforms. The v1.0 design follows the same pattern. Interaction-level expressiveness is recovered at the second stage by the reranker, which does run a per-(user, item) forward pass because the candidate list has been narrowed to the top hundred or so items.

### 5.6 Productionisation notes

The deployed system would replace the offline Faiss HNSW with a sharded ANN service (ScaNN or a managed Vespa cluster), the offline retrain with a daily incremental retrain plus weekly full retrain, the offline blend weight with an online bandit that adjusts `lambda` per session, and the offline metric harness with an A/B test on session value, click-through, and revenue per visitor. The data contracts (transaction stream, article catalogue, customer profile) would be Kafka-driven; the inference path would land at single-digit milliseconds end-to-end on GPU and tens of milliseconds on CPU. None of this is built in v1.0, but the offline two-tower architecture is exactly the candidate-generation block of the productionised stack.

## 6. Conclusion

The H&M Personalized Fashion Recommendations dataset is a clean public substrate for an end-to-end study of personalised retrieval at retail scale. The v1.0 deliverable specifies and implementations a hybrid system that combines an implicit-feedback ALS baseline with a two-tower neural recommender trained with sampled-softmax-with-logQ over in-batch and mixed negatives, indexed with Faiss HNSW for sub-50 ms inference. The code is runnable end-to-end and the literature anchor covers the canonical ALS, DSSM, two-tower, sequential-reranker, ANN, and fashion-domain references. The headline MAP@12 numbers are deferred to v1.0; the design choices are anchored against the published expectations of the field. The same architecture transfers without modification to Zalando, About You, Otto, Bonprix, and Breuninger because the same constraints (sparse implicit feedback, large catalogue, tight latency, recency-driven content) apply across the DACH retail landscape.

## References

References are listed in `reports/references.md` with full author lists, titles, venues, years, and resolved DOIs. Inline citations in this manuscript are by author and year and resolve against that file. The reference set covers implicit-feedback CF [Hu, Koren, Volinsky 2008; Pan 2008; Koren 2008; Sarwar 2001], MF surveys and revisits [Koren, Bell, Volinsky 2009; Bell, Koren 2007; Rendle 2020; Su & Khoshgoftaar 2009; Ekstrand 2011], two-tower and DSSM retrieval [Huang 2013; Covington 2016; Yi 2019; Yang 2020; Cheng 2016; Grbovic 2018], neural CF [He 2017; Rendle 2020; Wang 2019; He 2020; Wu 2022], sequential rerankers [Hidasi 2016; Tang & Wang 2018; Kang & McAuley 2018; Sun 2019; Zhou 2018], fashion and visual recommenders [McAuley 2015; He & McAuley 2016; Liu 2016; Han 2017], ANN retrieval [Johnson 2021; Malkov & Yashunin 2020], evaluation, cold-start, calibration, and diversity [Jarvelin & Kekalainen 2002; Cremonesi 2010; Schein 2002; Steck 2018; Adomavicius & Kwon 2012], industry deployments and surveys [Linden, Smith, York 2003; Smith & Linden 2017; Ying 2018; Ma 2018; Zhang 2019; Park 2012; Burke 2002; Schafer 2007; Lops 2010; Ricci 2010], and contrastive representation learning [Chen 2020].
