![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Recsys](https://img.shields.io/badge/task-recommendation-purple) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

# H&M Personalized Fashion Recommendations

Hybrid collaborative filtering + content-based recommendation engine for 1.37M customers across H&M fashion catalog.

---

## Task

**Recommendation System**

---

## Architecture

```
Purchase History + Item Metadata → ALS / LightFM → Candidate Retrieval (Annoy) → MAP@12
```

---

## Key Features

- User-based ALS collaborative filtering on 31M purchase transactions
- Content-based filtering on article metadata (product type, colour, department)
- Hybrid LightFM model combining interaction matrix and item features
- Approximate nearest neighbours (Annoy) for fast candidate retrieval
- MAP@12 evaluation matching Kaggle competition metric

---

## Dataset

[H&M Personalized Fashion Recommendations (Kaggle)](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)

---

## Project Structure

```
├── src/
│   ├── model_baseline.py      # Baseline model
│   └── model_advanced.py      # Advanced model
├── notebooks/
│   └── 01_EDA.ipynb           # Exploratory analysis
├── manuscripts/
│   └── manuscript.md          # IMRaD writeup
├── reports/
│   └── references.md          # Verified references
├── deliverables/
│   └── presentation.html      # Self-contained HTML
├── data/
│   └── README.md              # Dataset download instructions
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/Sandyyy123/fashion-recommendation-engine.git
cd fashion-recommendation-engine
pip install -r requirements.txt

# See data/README.md for dataset download
python src/model_baseline.py
python src/model_advanced.py
```

---

## Tech Stack

`implicit · LightFM · pandas · scikit-learn · annoy`

---

## Author

**Dr. Sandeep Grover** — PhD Data Science, independent ML researcher, Mössingen, Germany.

---

## License

MIT
