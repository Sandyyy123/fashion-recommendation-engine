# Additional References - Project 13 (H&M Personalized Fashion Recommendations)

Role: Literature Scout. All entries verified live against `https://api.crossref.org/works/{doi}` on 2026-05-08. Volume/issue/page numbers intentionally omitted per project policy. Existing `reports/references.md` is untouched.

## State-of-the-art callout: gaps in the current `reports/references.md`

The existing reference list is strong on the canonical 2008-2020 stack (Hu/Koren ALS, DSSM, Yi 2019 two-tower with logQ, Yang 2020 mixed negatives, SASRec, BERT4Rec, VBPR, DeepFashion, Faiss/HNSW, Steck calibration). It under-covers five fronts that are now the live state of the art and that a Phase 2 H&M run should cite:

1. **LLM-based and generative recommenders** - Inductive Generative Recommendation via Retrieval-based Speculation (Ding et al., AAAI 2026), LARR (Wan et al., RecSys 2024), and Bridging Search and Recommendation in Generative Retrieval (Penha et al., RecSys 2024) define the new "decode item ids as tokens" paradigm that pure two-tower retrieval no longer dominates on. The current list has none of these.
2. **Vision-language fashion encoders** - UniFashion (Zhao et al., EMNLP 2024) is the modern unified VLM successor to VBPR/DeepFashion and is what an image-tower upgrade should be benchmarked against. Not cited.
3. **Diffusion-based recommenders and their reproducibility caveats** - Benigni, Ferrari Dacrema and Jannach (TORS 2026) explicitly call out reproducibility issues in diffusion recsys; Chen et al. (Neural Networks 2025) propose conditional diffusion. The current list cites no diffusion recsys work.
4. **Up-to-date calibration / fairness / negative-sampling theory** - The Calibrated Recommendations survey (da Silva and Jannach, TORS 2026) supersedes Steck 2018 for any production calibration discussion; the Item-Sampling Evaluation paper (Li et al., TORS 2024) is the recent reference on biased top-k evaluation.
5. **Industrial two-tower production accounts (2024-2026)** - Allegro RecSys 2025 and the Off-Policy Evaluation paper (Wang et al., RecSys 2025) are the freshest production-grade analogues to the YouTube/Yi 2019 Two-Tower story; both belong in any client-facing DACH-retail framing.

The 29 references below cover these five gaps and a few adjacent corners (cold-start metadata alignment, sequential SASRec-vs-BERT4Rec re-evaluation, graph-attention outfit compatibility).

---

## Architectures and retrieval at scale (two-tower / candidate generation, 2024-2026)

Osowska-Kurczab A, Nazarko K, Marzec M, Wojciechowska L, Kremenova E. Suggest, Complement, Inspire: Story of Two-Tower Recommendations at Allegro.com. Proceedings of the Nineteenth ACM Conference on Recommender Systems. 2025. DOI:10.1145/3705328.3748135

Wang P, Shi Z, Shabbeer A, London B. Off-Policy Evaluation of Candidate Generators in Two-Stage Recommender Systems. Proceedings of the Nineteenth ACM Conference on Recommender Systems. 2025. DOI:10.1145/3705328.3748057

An Z, Joe I. TMH: Two-Tower Multi-Head Attention neural network for CTR prediction. PLOS ONE. 2024. DOI:10.1371/journal.pone.0295440

Kekuda A, Zhang Y, Udayashankar A. Embedding based retrieval for long tail search queries in ecommerce. 18th ACM Conference on Recommender Systems. 2024. DOI:10.1145/3640457.3688039

Chen E, Wang B. One backpropagation in two tower recommendation models. Neurocomputing. 2026. DOI:10.1016/j.neucom.2026.133148

## Generative retrieval and LLM-based recommenders

Penha G, Vardasbi A, Palumbo E, De Nadai M, Bouchard H. Bridging Search and Recommendation in Generative Retrieval: Does One Task Help the Other? 18th ACM Conference on Recommender Systems. 2024. DOI:10.1145/3640457.3688123

Ding Y, Li J, McAuley J, Hou Y. Inductive Generative Recommendation via Retrieval-based Speculation. Proceedings of the AAAI Conference on Artificial Intelligence. 2026. DOI:10.1609/aaai.v40i17.38486

Wan Z, Yin B, Xie J, Jiang F, Li X, Lin W. LARR: Large Language Model Aided Real-time Scene Recommendation with Semantic Understanding. 18th ACM Conference on Recommender Systems. 2024. DOI:10.1145/3640457.3688135

Na H, Gang M, Ko Y, Seol J, Lee S. Enhancing Large Language Model Based Sequential Recommender Systems with Pseudo Labels Reconstruction. Findings of the Association for Computational Linguistics: EMNLP. 2024. DOI:10.18653/v1/2024.findings-emnlp.423

Kunstmann H, Ollier J, Persson J, von Wangenheim F. EventChat: Implementation and user-centric evaluation of a large language model-driven conversational recommender system. ACM Transactions on Recommender Systems. 2026. DOI:10.1145/3803546

## Sequential and session-based recommendation (rerankers)

Chen M, Pan W, Ming Z. Explicit and Implicit Modeling via Dual-Path Transformer for Behavior Set-informed Sequential Recommendation. Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2024. DOI:10.1145/3637528.3671755

Kim H, Yoon M, Lee J. SASRec vs. BERT4Rec: Performance Analysis of Transformer-based Sequential Recommendation Models. Journal of KIISE. 2024. DOI:10.5626/jok.2024.51.4.352

Zhao Z, Zhang J, Jia C, Li C, Yu Y, Zeng Q. CoLA-Former: Graph Transformer Using Communal Linear Attention for Lightweight Sequential Recommendation. Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence. 2024. DOI:10.24963/ijcai.2024/410

Zou J, Sun A, Long C, Kanoulas E. Knowledge-Enhanced Conversational Recommendation via Transformer-Based Sequential Modeling. ACM Transactions on Information Systems. 2024. DOI:10.1145/3677376

Madangarli S, Suganeshwari G, Syed Ibrahim S, Kavitha M, Setozaki N. Semantic and temporal-aware hybrid embedding for transformer-based sequential recommendation. Artificial Intelligence Review. 2026. DOI:10.1007/s10462-025-11483-5

## Cold-start handling

Monteil J, Vaskovych V, Lu W, Majumder A, van den Hengel A. MARec: Metadata Alignment for cold-start Recommendation. 18th ACM Conference on Recommender Systems. 2024. DOI:10.1145/3640457.3688125

Moscati M. Multimodal Representation Learning for High-Quality Recommendations in Cold-Start and Beyond-Accuracy. 18th ACM Conference on Recommender Systems. 2024. DOI:10.1145/3640457.3688009

## Fashion, outfit compatibility, and multimodal vision-language

Zhao X, Zhang Y, Zhang W, Wu X. UniFashion: A Unified Vision-Language Model for Multimodal Fashion Retrieval and Generation. Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. 2024. DOI:10.18653/v1/2024.emnlp-main.89

Kim E, Kim S, Jung C, Hahm Y, Cho S. Fashion Image Retrieval With Vision-Language Model Guided Fine-Grained Textual Attributes and Cross-Domain Contrastive Optimization. IEEE Access. 2025. DOI:10.1109/access.2025.3630632

Suvarna B, Balakrishna S. Enhanced content-based fashion recommendation system through deep ensemble classifier with transfer learning. Fashion and Textiles. 2024. DOI:10.1186/s40691-024-00382-y

Lu M, Guo Q, Zhang Y, Yi X. Heterogeneous Graph Neural Network for Fashion Outfit Compatibility Prediction. Journal of Computer-Aided Design and Computer Graphics. 2024. DOI:10.3724/sp.j.1089.2024.19974

Chang C, Chen Y, Jiang D. Using large multimodal models to predict outfit compatibility. Decision Support Systems. 2025. DOI:10.1016/j.dss.2025.114457

Saed S, Teimourpour B. Hybrid-hierarchical fashion graph attention network for compatibility-oriented and personalized outfit recommendation. Machine Learning with Applications. 2026. DOI:10.1016/j.mlwa.2025.100802

## Negative sampling, evaluation, calibration, fairness

Li D, Jin R, Liu Z, Ren B, Gao J, Liu Z. On Item-Sampling Evaluation for Recommender System. ACM Transactions on Recommender Systems. 2024. DOI:10.1145/3629171

da Silva D, Jannach D. Calibrated Recommendations: Survey and Future Directions. ACM Transactions on Recommender Systems. 2026. DOI:10.1145/3789266

Rahmani H, Naghiaei M, Deldjoo Y. A Personalized Framework for Consumer and Producer Group Fairness Optimization in Recommender Systems. ACM Transactions on Recommender Systems. 2024. DOI:10.1145/3651167

Xuan Y, Sokol K, Sanderson M, Chan J. Diversity-Augmented Negative Sampling for Implicit Collaborative Filtering. Proceedings of the ACM Web Conference 2026. 2026. DOI:10.1145/3774904.3792346

## Diffusion-based recommenders

Benigni M, Ferrari Dacrema M, Jannach D. Diffusion Recommender Models and the Illusion of Progress: A Concerning Study of Reproducibility and a Conceptual Mismatch. ACM Transactions on Recommender Systems. 2026. DOI:10.1145/3795792

Chen R, Fan J, Wu M, Ma S. Conditional diffusion model for recommender systems. Neural Networks. 2025. DOI:10.1016/j.neunet.2025.107204

---

## Verification note

29 candidates queried at `https://api.crossref.org/works/{doi}` on 2026-05-08; all 29 returned HTTP 200 with matching titles and author lists. None were padded; entries that did not resolve cleanly were dropped before this list was finalised.
