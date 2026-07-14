# STEG Fraud Detection in Electricity and Gas Consumption

This competition is hosted on Zindi, a machine learning platform for data science challenges.  
Here is the link to the competition: [Fraud Detection in Electricity and Gas Consumption Challenge 🌾 - AI Hack Tunisia](https://zindi.global/competitions/fraud-detection-in-electricity-and-gas-consumption-challenge)

Ranked 6th position (20 continuous hours, only 54 succeed to submit among 191 competitors)!

---

**Competition:** Zindi — Detect fraudulent meter manipulation for Tunisian utility STEG
**Target:** Binary fraud classification (5.6% positive rate)
**Metric:** AUC
**Data:** Client metadata + 15 years of billing history (2005–2019)

## The Problem

STEG lost 200M Tunisian Dinars to fraudulent meter manipulation. Given a client's full billing history (invoices, counter readings, consumption levels), predict which clients are involved in fraud.

The challenge: invoice-level data must be aggregated to client-level features, the fraud rate is heavily imbalanced (5.6%), and categorical variables have inconsistent levels between train/test.

## Architecture

```
00_config.R              → Constants, libraries
01_data_loading.R        → Load client + invoice CSVs, join, harmonize levels
02_feature_engineering.R → Aggregate invoices → client-level features (consumption, frequency, diffs)
03_encoding_selection.R  → Target/WOE/James-Stein encoders + random subset feature selection
04_model_lightgbm.R      → 3 LightGBM variants (different seeds/boosters)
05_model_xgboost.R       → 4 XGBoost variants (Bayesian-tuned + grid + caret)
06_model_catboost_rf.R   → 3 CatBoost + H2O AutoML + Random Forest
07_ensemble.R            → Meta-learner blending (elastic net on 14 base predictions)
MAIN.R                   → Run all steps sequentially
```

## Key Engineering Decisions

### 1. Random Subset Feature Selection (100 × LightGBM)

Instead of traditional forward/backward selection, run 100 iterations where each randomly samples 12–35 features, trains LightGBM, and records AUC. The features appearing in the top-5 performing subsets form the final set.

This explores the feature interaction space stochastically — a single greedy search can get trapped in local optima.

```r
for (i in 1:100) {
    set.seed(i)
    m <- sample(num_features, sample(12:35, 1))
    bst <- lgb.train(data = lgb.Dataset(x[, m], label = y), ...)
    higher_auc_list[[i]] <- AUC(predict(bst, xt[, m]), ytest)
    parameters_list[[i]] <- m
}
best_features <- Reduce(union, parameters_list[top_5_indices])
```

### 2. Multiple Target Encoders Compared

For the `region` variable (high cardinality), five encoding strategies were tested:
- **Target mean**: `P(fraud | region)`
- **WOE**: Weight of Evidence
- **James-Stein**: Shrinkage toward global mean
- **M-estimator**: Bayesian smoothing with prior
- **Leave-one-out**: Prevent self-encoding leakage

Each was kept as a separate feature — the model learns which encoding captures the most signal.

### 3. Level Harmonization Before Encoding

Train/test had mismatched factor levels (e.g., `counter_statue` had values in train that didn't exist in test). Rather than dropping them, rare levels were merged into meaningful groups based on fraud rate:

```r
tot$counter_statue[tot$counter_statue %in% c("269375","420","46","618","769","A")] <- "2"
```

This preserves signal while ensuring prediction works on unseen data.

### 4. 14-Model Diverse Ensemble

Diversity via three axes:
- **Algorithm diversity**: LightGBM, XGBoost, CatBoost, H2O AutoML, Random Forest
- **Hyperparameter diversity**: 3-4 configs per algorithm (deep/shallow, high/low LR, DART/gbdt)
- **Seed diversity**: Different random seeds produce different bootstrap samples

The meta-learner (elastic net on 14 probability predictions) learns the optimal blend:

```r
meta_features = [lgbm1, lgbm2, lgbm3, xgb1, xgb2, xgb3, xgb4, cat1, cat2, cat3, h2o, rf1, rf2, rf3]
enet_meta = train(meta_features ~ target, method = "glmnet")
```

### 5. Imbalanced Sampling Strategy

At 5.6% fraud, the model default would predict all-zero. Multiple sampling approaches were tested:
- **Stratified undersampling** (keeping fraud proportion fixed while reducing majority class)
- **SMOTE** (synthetic minority oversampling)
- **Under 50/50** for feature selection (fast iteration)
- **Full data** for final models (with internal class weighting)

## Feature Groups

| Source | Features | Examples |
|--------|----------|----------|
| Consumption | 8 | Cons_tot, Cons_1–4, proportions, monthly average |
| Invoice patterns | 6 | nb_fact, nb_months, avg_months, low-consumption count |
| Counter metadata | 5 | nb_counter, nb_counter_type, nb_read, gas indicator |
| Temporal diffs | 8 | mean/sd of invoice gaps, reading changes, index diffs |
| Encoded categoricals | 7 | Region (5 encoders), district/category dummies |
| Seasonality | 2 | nb_fact_lev4_summer, level 4 summer proportion |

