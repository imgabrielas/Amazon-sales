# Amazon Product Ratings & Discount Strategy Analysis

Analysis of ~1,000 Amazon product listings to examine whether heavy discounting is associated with lower customer ratings, and whether rating can be predicted from price, discount, and category.

## Business Question

Pricing and category management teams want to know whether **heavy discounting is masking weak product quality**, or whether discounts and customer satisfaction are largely independent. The analysis looks at how `discount_percentage` relates to `rating` across categories, and builds a model that predicts a product's rating from price, discount, and category.

**Goal:** RMSE ≤ 0.5 on a 1–5 rating scale (or precision ≥ 0.75 for a "well-rated" ≥4.0 classifier).

## Data

[Amazon Sales Dataset](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset) (via `kagglehub`, downloaded and cached automatically on first run). Includes product pricing, discount, category, rating, and review fields.

## Approach

1. **Cleaning** — drop identifier/text columns (`product_id`, `user_id`, `user_name`, `review_id`, `img_link`, `product_link`, `about_product`, `review_title`, `review_content`); collapse the 221 raw `category` values down to 9 top-level categories (`category_main`); strip currency/percent symbols and coerce `price`, `discount_percentage`, `rating`, and `rating_count` to numeric; drop rows missing `rating`/`rating_count`.
2. **EDA** — correlation and regression plot of discount % vs. rating, plus mean discount/rating by category.
3. **Modeling** — one-hot encode category, then compare a Linear Regression and a Random Forest Regressor (80/20 train/test split) predicting `rating` from price, discount, and category.

## Findings

- Discount % vs. rating shows only a **weak negative relationship** (~4.2 → ~4.0 across 0–90% discount), with heavy scatter and ratings tightly clustered around 3.9–4.3.
- `Electronics` and `Computers&Accessories` combine the deepest average discounts (~51–54%) with the lowest average ratings (~4.08–4.16); `OfficeProducts` has a much smaller average discount (~12%) and the highest average rating (4.31).
- **Linear Regression:** R sq = 0.11, RMSE = 0.27
- **Random Forest:** R sq = 0.17, RMSE = 0.26

Both models meet the RMSE ≤ 0.5 target, but that's mostly because ratings themselves are tightly clustered (low variance to explain) rather than because the models are capturing real signal — both explain well under a fifth of the variance (R sq < 0.2). This reinforces the conclusion that **price, discount, and category are weak predictors of rating** — discounting and customer satisfaction appear largely independent rather than discounting masking poor quality.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open and run `amazon_sales.ipynb`. The dataset is downloaded automatically via `kagglehub` on first run (requires a Kaggle account/API token configured per [kagglehub's auth docs](https://github.com/Kaggle/kagglehub)).

## Project Structure

```
amazon_sales.ipynb   # full analysis: cleaning, EDA, modeling
requirements.txt     # pinned dependencies
```