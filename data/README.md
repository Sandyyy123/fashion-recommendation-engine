# Data - H&M Personalized Fashion Recommendations

## Source

Kaggle competition: [`h-and-m-personalized-fashion-recommendations`](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)

Released by H&M Group, Stockholm, in February 2022. Open for academic and learning use under the competition rules. The dataset accompanies a closed Kaggle competition; you must accept the rules at the competition page once before downloading.

## v1.0 status

**Not downloaded.** Total payload is roughly 30 GB once extracted (transactions plus customer plus article metadata plus 105k product images). Per the Portfolio rules, anything above 2 GB or behind a competition acceptance gate is documented only at this stage.

## Download command

```bash
# Prereq once: accept the competition rules at
#   https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules
# Prereq once: kaggle CLI installed and ~/.kaggle/kaggle.json populated.

cd /root/AI/project_root/data
kaggle competitions download -c h-and-m-personalized-fashion-recommendations
unzip h-and-m-personalized-fashion-recommendations.zip
# Optional, image-heavy: this unpacks the images/ tree (~26 GB).
# To stay under 5 GB and skip images for tabular-only modelling:
#   unzip -x h-and-m-personalized-fashion-recommendations.zip 'images/*'
```

## Files after extraction

| File | Approx. size | Rows / contents |
|------|--------------|-----------------|
| `transactions_train.csv` | ~3.4 GB uncompressed (~470 MB gz) | ~31.8M purchase events (2018-09-20 to 2020-09-22) |
| `customers.csv` | ~200 MB | ~1.37M customer profiles |
| `articles.csv` | ~30 MB | ~105k article metadata rows |
| `sample_submission.csv` | ~70 MB | template prediction file |
| `images/` | ~26 GB | one JPG per article in nested folders `images/0NN/0NNxxx/<article_id>.jpg` |

## Schema

### `transactions_train.csv`

| Column | Type | Description |
|--------|------|-------------|
| `t_dat` | date | transaction date (YYYY-MM-DD) |
| `customer_id` | str (64 hex) | hashed customer identifier |
| `article_id` | int (10) | article identifier |
| `price` | float | normalised price |
| `sales_channel_id` | int | 1 = store, 2 = online |

### `customers.csv`

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | str | hashed customer identifier |
| `FN` | float | fashion-news subscription flag (0/1/NaN) |
| `Active` | float | active flag (0/1/NaN) |
| `club_member_status` | str | ACTIVE / PRE-CREATE / LEFT CLUB |
| `fashion_news_frequency` | str | NONE / Regularly / Monthly |
| `age` | float | customer age (with NaN) |
| `postal_code` | str | hashed postcode |

### `articles.csv`

| Column | Type | Description |
|--------|------|-------------|
| `article_id` | int | article identifier |
| `product_code` | int | product code (parent of size/colour variants) |
| `prod_name` | str | product name |
| `product_type_no` / `product_type_name` | int / str | e.g. Trousers, T-shirt |
| `product_group_name` | str | e.g. Garment Upper body |
| `graphical_appearance_*` | int / str | pattern category |
| `colour_group_*`, `perceived_colour_*` | int / str | colour metadata |
| `department_*`, `index_*`, `index_group_*`, `section_*`, `garment_group_*` | int / str | merchandising hierarchy |
| `detail_desc` | str | free-text product description |

## Storage layout once downloaded

```
data/
  transactions_train.csv
  customers.csv
  articles.csv
  sample_submission.csv
  images/
    010/
      0108775015.jpg
      ...
    011/
    ...
```

For modelling, downsample transactions to the last 6 to 12 weeks unless the full window is needed; the full 31M-row table loads in roughly 90 s on a 32 GB machine with `pandas.read_csv(parse_dates=['t_dat'])`. For the two-tower advanced model, cache article metadata as a Parquet file (`articles.parquet`, ~12 MB) before training.

## Image handling

Images are optional for the baseline but recommended for the two-tower advanced model. Recommended flow if disk-bound:

1. Skip the `images/` folder during initial unzip.
2. After the metadata model is trained, download images on-demand for the top-100k most-purchased articles only (covers >95% of recent transactions).
3. Precompute CLIP or ResNet-50 embeddings once, persist as `articles_image_emb.npy` (~50 MB for 100k items at 512-d float16).

## Competition metric

[MAP@12](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/overview/evaluation): for each customer, the precision is averaged over the first 12 predicted article IDs, then averaged across customers. Customers in the public test set who do not appear in the prediction file get zero. Submission is one row per customer with up to 12 space-separated article IDs.

## License and access

The data is released under the Kaggle competition rules. It is free to download for participating users, including post-competition for academic use; redistribution is not permitted. For this Initial implementation no data is stored in the repository; only the schema and the download command are checked in.
