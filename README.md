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
# Sea Turtle Rescue Forecast Challenge

**Competition:** Zindi — Predict weekly sea turtle rescues per capture site for 2019
**Target:** `Capture_Number` (count per site per week)
**Metric:** RMSE
**Data:** 29 capture sites, weekly rescue counts from 1998–2018

## The Problem

Local Ocean Conservation rescues sea turtles caught in fishing nets across 29 sites in Kenya. The goal: forecast how many turtles each site will rescue each week in 2019, so staff and budget can be allocated in advance.

This is a multi-site time series forecasting problem with strong seasonality, sparse counts (many site-weeks with zero rescues), and a structural data gap in 2016.

## Architecture

```
00_config.R              → Constants, libraries, parameters
01_data_loading.R        → Load train.csv, build site×week grids for train/test
02_feature_engineering.R → Site aggregates, week aggregates, frequency, elasticity, clustering
03_time_series.R         → Per-site STL+ETS forecasts with expanding-window backtesting
04_model_supervised.R    → CatBoost + XGBoost using TS forecasts as features
05_ensemble.R            → Weighted blend (70% TS + 30% CatBoost) → submission
MAIN.R                   → Run all steps sequentially
```

## Key Engineering Decisions

### 1. Hybrid TS + Supervised (Time Series as a Feature)

A pure time series model (STL+ETS per site) captures seasonality well but cannot learn cross-site patterns. A pure supervised model can use site features but struggles with temporal structure in sparse count data.

The solution: **use the time series forecast as a feature** in the supervised model. The supervised model learns when and how to deviate from the TS baseline.

But there's a training problem: if you only generate a TS forecast for 2019, the supervised model has no training rows with that feature. The fix:

### 2. Expanding-Window Backtesting for Training Data

Generate time series forecasts for **every historical year** using only data available before that year:

- Forecast 2014 using ≤2013 data
- Forecast 2015 using ≤2014 data
- Forecast 2017 using ≤2015 data (2016 excluded)
- Forecast 2018 using ≤2017 data
- Forecast 2019 using ≤2018 data

Now the supervised model can train on 2014–2018 rows where both the TS prediction and the actual outcome are known. This is the key insight that makes the hybrid work.

### 3. Year 2016 Exclusion

2016 shows a structural drop in rescue counts. Rather than let this contaminate seasonal patterns, it was excluded entirely from both the time series fitting and the supervised training.

### 4. Outlier Capping in Time Series

ETS can produce extreme forecasts for sites with high variance. A guard rail caps predictions at `mean + 5×sd` per site, scaled proportionally to preserve the shape of the forecast.

```r
gard[j-4, 2] <- mean(df_ts[, j]) + (sd(df_ts[, j]) * 5)
if (fore_x[i] > gard[j-4, 2]) {
    fore_x[i] <- round(fore_x[i] * gard[j-4, 2] / max(fore_x))
}
```

### 5. Year-over-Year Elasticity

Captures the trend direction per site×week — whether rescues are accelerating or declining across years:

```r
elast = (((diff_2+1) - (diff_1+1)) / (diff_2+1) +
         ((last_diff+1) - (diff_2+1)) / (last_diff+1)) / 2
```

### 6. Final Ensemble: 70% TS + 30% CatBoost

The time series model dominates because this is fundamentally a seasonal problem with 44-week periodicity. The CatBoost model contributes corrections based on site category, cluster membership, and cross-site rescue frequency patterns.

## Data Structure

| Feature Source | Examples |
|---|---|
| Site aggregates | Total rescues per site, monthly average, weekly total |
| Week aggregates | Total across all sites per week, monthly average per week |
| Frequency | Mean/max/min gap between rescue events (2018) |
| Elasticity | Year-over-year change rate per site×week |
| Clustering | K-means (k=5) on site rescue profiles |
| TS forecast | STL+ETS per-site prediction (expanding window) |
# IEEE Escalations in Customer Support

**Competition:** IEEE/Kaggle — Predict time-to-next-escalation for enterprise customer support cases
**Target:** `INV_TIME_TO_NEXT_ESCALATION` (inverse time, regression)
**Metric:** RMSE

## The Problem

Given a customer support case with metadata, status history events, milestone logs, and comment threads — all with anonymized/tokenized text — predict whether and when the case will escalate next.

The core challenge: **you can only use information available before the escalation moment**, but the dataset gives you the full timeline. You have to construct your own training labels.

## Architecture

```
00_config.R              → Constants, file paths, libraries
01_data_loading.R        → Load 7 CSVs (metadata, history, milestones, comments, test, dictionary, lemma)
02_target_construction.R → Temporal cut-point sampling for observation windows
03_feature_engineering.R → Target-encoded categoricals + event sequence features
04_nlp_features.R        → Decode anonymized text + NLP sub-model
05_feature_selection.R   → IV, correlation filtering, final feature set
06_model_catboost.R      → CatBoost with native categorical support
07_model_xgboost.R       → XGBoost with sparse matrices
08_ensemble.R            → Weighted blend → final submission
MAIN.R                   → Run all steps sequentially
```

## Key Engineering Decisions

### 1. Temporal Cut-Point Sampling

The biggest design decision. For escalated cases, we cut the event timeline just before the escalation starts — that's our "decision point." For non-escalated cases, we need to simulate when a prediction would be made.

Solution: sample random cut-points using the **same distribution** as real escalation timing:

```r
escalation_points_dist <- quantile(
    escalation_starts[, escalation_start / case_end],
    seq(0, 1, 0.01)
)
other_cases[, time_cut := case_end * sample(escalation_points_dist, .N, replace = TRUE)]
```

This ensures the model sees realistic observation windows — not artificially early or late.

### 2. NLP Sub-Model as Meta-Feature

Text is anonymized with ID tokens. A lemma mapping file lets us decode IDs back to real words. Rather than dumping decoded TF-IDF features into the main model (thousands of sparse columns), we:

1. Decode text via `stri_replace_all_fixed(text, lemma$id, lemma$lemma)`
2. Extract only the words that actually decoded (set difference)
3. Build TF-IDF → XGBoost binary classifier (escalation yes/no)
4. Use the **probability output** as a single feature in the final model

This is model stacking at the feature level — the NLP model's confidence becomes signal.

### 3. Target Encoding of High-Cardinality Categoricals

13 categorical variables (site, company, product, user group, etc.) with hundreds of levels each. Instead of one-hot encoding or label encoding:

```r
rates <- train[, .(
    not_esc = sum(!(REFERENCEID %in% ref_ids_esc)),
    esc = sum(REFERENCEID %in% ref_ids_esc)
), by = col_name]
rates[, enc_col := (not_esc) / (not_esc + esc) * 100]
```

Each category becomes its historical non-escalation rate. Computed only on training data to avoid leakage.

### 4. Event Sequence Features (Cut at Decision Time)

All temporal features are computed only on events **before the cut-point**:
- **History:** severity change count, initial vs. current severity, event frequency
- **Milestones:** unique milestone IDs, note repetition patterns, zero-time batch count
- **Comments:** unique note count, issue type diversity, text length patterns

### 5. POS/NER Dictionary Features

The challenge provides a dictionary mapping anonymized IDs to POS tags and NER types. We count linguistic patterns per case:
```r
nb_pos_noun_verb, nb_pos_adj_adv, nb_ner_cardinal, nb_ner_other
```

These capture whether the case discussion is about entities (organizations, products) vs. actions.

## Data Sources

| File | Description | Rows |
|------|-------------|------|
| Case metadata | Static attributes per case (site, product, customer) | ~30K |
| Status history | Temporal events with severity, escalation flags | ~500K |
| Milestones | Timestamped milestone logs with anonymized notes | ~300K |
| Comments | Timestamped comments with anonymized text | ~200K |
| Dictionary | POS/NER tags for anonymized token IDs | ~50K |
| Lemma mapping | ID → real word translations | ~30K |

## Final Model

Ensemble of CatBoost (55%) + XGBoost (45%), both trained on the same feature set augmented with the NLP sub-model probability score.
# Nairobi Ambulance Deployment Optimization

**Competition**: Zindi – Ambulance Deployment Optimization
**Date**: 2019-2020
**Language**: R

## Task

Position 6 ambulances across Nairobi every 3 hours to minimize distance to reported traffic crashes during the test period (Jul-Dec 2019). Training data: 6,318 crashes from Jan 2018 to Jun 2019.

## Approach

A **two-stage system**: predict where crashes will happen, then place ambulances optimally.

1. **Stage 1 (Prediction)**: Binary classification at the (road segment × 3-hour window) level — will a crash occur here, now?
2. **Stage 2 (Deployment)**: Cluster predicted hotspots into 6 positions per time window.

## Engineering Decisions

### 1. Segment-Level ABT (Analytical Base Table)

Instead of predicting at individual lat/long points, the city is decomposed into ~800 road segments. Each segment becomes a row in the ABT, crossed with every 3-hour time window. This converts a spatial problem into a structured tabular classification.

### 2. Multi-Source Feature Fusion

Five data sources merged per segment per time window:
- **Crash history**: segment frequency, monthly elasticity, seasonal patterns
- **Road survey** (228 features): reduced to 2 dimensions via t-SNE
- **Uber Movement**: hourly speeds per OSM way, matched to segments
- **Weather**: temperature, humidity, wind, precipitation (simulated forward for test period)
- **Temporal**: hour bin, weekend, holidays

### 3. t-SNE for Obfuscated Road Survey Features

The 228 obfuscated road survey columns are reduced to 2 features via t-SNE (after NZV + correlation filtering). This captures the latent structure of road characteristics (crosswalks, obstacles, traffic behavior) without needing column semantics.

```r
tsne_out <- Rtsne(dr_final, dims = 2, perplexity = 25, max_iter = 7000)
```

### 4. Crash Elasticity as a Temporal Feature

Instead of using raw crash counts (which are always zero for the test period), compute month-over-month evolution per segment — the *trend*. Segments with increasing crash rates get higher predicted probabilities even without future crash data.

```r
elastic[, diff := (nb_crash_monthly - shift(nb_crash_monthly)) /
                  pmax(shift(nb_crash_monthly), 1), by = segment_id]
```

### 5. Weather Simulation for Test Period

Weather data only covers the training period. Future weather is simulated using year-over-year monthly adjustment: `weather_2019_m = weather_2018_m × (recent_trend_ratio)`. This preserves seasonal patterns without using future data.

### 6. Prediction → Deployment Conversion

The model outputs P(crash) per segment per window. Converting this to 6 ambulance positions:
- Take top-50 segments by predicted probability
- Cluster their midpoints into 6 groups (K-means)
- Place ambulances at cluster centroids

This ensures ambulances are distributed across predicted hotspots rather than concentrated at a single high-risk point.

### 7. Extreme Class Imbalance Handling

With ~0.1% positive rate (crash in a specific segment-window), stratified downsampling of negatives creates manageable training sets while preserving signal. Multiple subsets with different negative ratios feed into the ensemble.

## Pipeline

| Step | File | Purpose |
|------|------|---------|
| 0 | `00_config.R` | Configuration, libraries |
| 1 | `01_data_loading.R` | Load crashes, segments, weather |
| 2 | `02_segment_features.R` | Crash-to-segment matching, elasticity, t-SNE |
| 3 | `03_uber_speeds.R` | Uber Movement speed integration |
| 4 | `04_clustering.R` | Segment zones + crash clusters |
| 5 | `05_abt_construction.R` | Full segment × time grid |
| 6 | `06_feature_selection.R` | IV, RF, Elastic Net, random search |
| 7 | `07_model_lightgbm.R` | Primary LightGBM model |
| 8 | `08_model_ensemble.R` | XGBoost + CatBoost ensemble |
| 9 | `09_ambulance_deployment.R` | Predictions → ambulance positions |

## Run

```r
source("MAIN.R")
```

## Key Libraries

`data.table`, `rgdal`, `geosphere`, `lightgbm`, `xgboost`, `catboost`, `Rtsne`, `caret`, `text2vec`
# Akeed Restaurant Recommendation

**Competition**: Zindi – Akeed Restaurant Recommendation Challenge
**Date**: 2020
**Language**: R

## Task

Predict which of 100 restaurants each customer will order from, given customer locations, vendor information, and order history. ~35k train customers, ~10k test customers. Submission format: one row per (customer × location × vendor) triplet with a binary prediction.

## Approach

A **geography-first segmented recommendation engine** instead of standard collaborative filtering.

Why: with only 100 vendors, a strong delivery-radius constraint, and sparse interactions (most customers ordered from 1-2 vendors), the dominant signal is **spatial proximity within a demographic segment**, not user-item similarity.

## Engineering Decisions

### 1. Vendor Geographic Zones (K-Means)

Cluster vendors into 4 geographic zones based on coordinates. This creates the spatial structure that constrains recommendations — a customer won't order from a vendor 30km away.

```r
km_vend <- kmeans(
    x = vend_valid[, .(long_clean, lat_clean)],
    centers = 4,
    algorithm = "MacQueen",
    iter.max = 5000,
    nstart = 25
)
```

### 2. Customer Segmentation (18 segments)

Segment customers by: **gender** (3 levels) × **location count** (3 buckets) × **account tenure** (old/new). This captures behavioral groupings — a new male user with 1 address behaves differently from a long-tenured female with multiple delivery locations.

### 3. Vendor Tag Grouping

Original vendor tags are too granular (50+ unique tags). Grouped into 8 cuisine categories (Arabic, Indian, International, Desserts, Drinks, Sandwiches, Breakfast, Others) to make vendor profiles useful for matching.

### 4. Segment × Zone Recommendation

For each (segment, zone) pair in the test set:
- Find train customers in the same segment and zone who ordered
- Count vendor frequency (which vendors they ordered from)
- Score vendors for each test customer: `frequency / manhattan_distance`
- Recommend top-scoring vendors

### 5. Order Feature Engineering

Aggregate order history per (customer × location × vendor) triplet: order count, basket size, payment mode, ratings, delivery times, promo usage. These features characterize the customer-vendor relationship.

## Pipeline

| Step | File | Purpose |
|------|------|---------|
| 0 | `00_config.R` | Seed, paths, libraries |
| 1 | `01_data_loading.R` | Load CSVs, fix casing |
| 2 | `02_order_features.R` | Aggregate order history per triplet |
| 3 | `03_vendor_processing.R` | Tag grouping, coordinate cleanup |
| 4 | `04_customer_features.R` | Customer-location profiles |
| 5 | `05_clustering.R` | Vendor zones + customer segments |
| 6 | `06_recommendation.R` | Segment × zone recommendation |

## Run

```r
source("MAIN.R")
```

## Key Libraries

`data.table`, `dplyr`, `lubridate`, `sqldf`, `caret`
# Mental Health Text Classification

**Competition**: Zindi – AI4D iCompass Social Media Sentiment Analysis for Mental Health
**Date**: Late 2021
**Language**: R

## Task

Classify short free-text statements from Kenyan university students into four mental health categories: Depression, Alcohol, Suicide, Drugs. Statements respond to "What is on your mind?" and contain spelling errors, slang, and code-switching.

## Engineering Decisions

### 1. Spelling Correction Before Tokenization

Student text is noisy—misspellings fragment the vocabulary ("depressd", "deppressed", "depressed" → three separate tokens in TF-IDF). The pipeline applies UDPipe tokenization followed by hunspell correction, unifying variants before feature extraction. Kenyan slang terms (bhang, miraa) are whitelisted to avoid false corrections.

```r
tokens$is_correct <- hunspell_check(tokens$token)
slang_terms <- c("bhang", "miraa", "muguka", "shisha", "weed", "meth")
tokens$is_correct[tokens$token %in% slang_terms] <- TRUE
```

### 2. One-vs-Rest Decomposition

Rather than training a single multi-class model, each class gets its own binary classifier. This lets different models specialize—depression language patterns differ structurally from drug references. The ensemble averages probabilities across models independently per class.

### 3. TF-IDF + LSA (4 latent topics)

Sublinear TF-IDF with L2 normalization handles the short, variable-length statements. LSA with exactly 4 latent dimensions (matching the 4 classes) adds semantic structure to the sparse matrix, giving gradient-boosted models a denser signal.

### 4. Multiple Representation Strategies

The pipeline tests multiple embedding approaches in parallel:
- **TF-IDF + XGBoost**: sparse bag-of-words with boosting
- **GLMNet (Ridge)**: regularized logistic regression, strong on sparse text
- **Keras embedding**: learns task-specific 16-dim embeddings
- **StarSpace (ruimtehol)**: learns joint text-label embedding space

### 5. Simple Average Ensemble

With a small dataset (~600 train samples), a stacking meta-learner risks overfitting. Simple averaging of probability outputs from diverse model families (linear, tree, embedding-based) reduces variance while avoiding that risk.

## Pipeline

| Step | File | Purpose |
|------|------|---------|
| 0 | `00_config.R` | Seed, paths, libraries |
| 1 | `01_data_loading.R` | Load CSV, create binary targets |
| 2 | `02_text_preprocessing.R` | UDPipe + hunspell correction + stemming |
| 3 | `03_feature_engineering.R` | TF-IDF (ngram 1-2) + LSA |
| 4 | `04_model_glmnet.R` | Ridge logistic regression per class |
| 5 | `05_model_xgboost.R` | XGBoost per class with CV |
| 6 | `06_model_keras.R` | Keras text embedding network |
| 7 | `07_model_ruimtehol.R` | StarSpace joint embeddings |
| 8 | `08_ensemble_submission.R` | Average and write submission |

## Run

```r
source("MAIN.R")
```

## Key Libraries

`data.table`, `text2vec`, `udpipe`, `hunspell`, `xgboost`, `glmnet`, `keras`, `ruimtehol`
# Malawi Flood Extent Prediction

Predicting the fraction of 1km² grid squares flooded in southern Malawi during Cyclone Idai (March 2019). Training data is the 2015 flood event. Evaluation: RMSE on flood fraction [0, 1]. Zindi competition sponsored by UNICEF and Arm.

## Key Engineering Decisions

**Train on one flood event, predict another.** The fundamental challenge is not generalization across rows — it's generalization across events. The model trained on 2015 data must predict 2019 (Cyclone Idai), a different storm with a different path, different rainfall distribution, and larger extent. Event-specific features from 2015 (individual week rainfall values) carry limited signal for 2019. The features that generalize are physical landscape invariants: terrain shape, soil properties, proximity to water. The pipeline prioritizes these.

**Terrain derivative features from elevation raster.** Elevation alone doesn't predict flooding — slope, flow direction, and terrain position do. A low-elevation flat plain pools water differently than a low-elevation hillside that drains rapidly. From the raw elevation grid, six terrain derivatives are computed:

```r
terrain_stack <- raster::terrain(
    rast_elev,
    opt = c("slope", "aspect", "tpi", "tri", "roughness", "flowdir"),
    unit = "degrees", neighbors = 8
)
```

- **TPI** (Terrain Position Index): is this cell a depression or a ridge?
- **TRI** (Terrain Ruggedness Index): how rough is the local surface?
- **flowdir**: which of the 8 cardinal directions does water drain toward?
- **hillshade**: a proxy for local exposure and shadow patterns

**Zero-inflated target decomposition.** ~70% of grid squares have zero flood extent. A regression model minimizing RMSE on zero-inflated data will be pulled toward near-zero predictions everywhere. The pipeline creates two auxiliary classification variables from the training target: `bin` (did it flood at all?) and `zero` (no / partial / severe), used for EDA, stratified splitting, and understanding where the regression error originates.

**Prediction clipping.** The target is a bounded fraction [0, 1]. Gradient boosting models can predict outside this range. All predictions are clipped before evaluation and submission:

```r
clip_predictions <- function(x) pmax(0, pmin(1, x))
```

**Soil infiltration variables as primary external data.** Soil drainage class, parent material, erosion type, and texture (from FAO/ISRIC soil maps) directly control the rate at which water infiltrates vs. pools. These are joined to each grid square and treated as first-class features alongside terrain.

## Project Structure

```
00_config.R              # Global settings, libraries, utility functions
01_data_loading.R        # Load train/test, separate 2015/2019 rainfall columns
02_spatial_features.R    # Terrain derivation from elevation raster
03_external_data.R       # Soil data cleaning, zero-inflation auxiliary targets
04_train_test_split.R    # Train / validation / test split
05_model_lightgbm.R      # LightGBM with early stopping
06_model_xgboost.R       # XGBoost with CV-tuned nrounds
07_model_catboost.R      # CatBoost with native categorical support
08_model_h2o.R           # H2O AutoML
09_generate_submission.R # Inverse-RMSE weighted ensemble, submission file
MAIN.R                   # Pipeline orchestration
```

## Running the Pipeline

```r
source("MAIN.R")
```

Or run individual steps:
```r
source("00_config.R")
source("01_data_loading.R")
# ... etc.
```

## Data Files Required

| File | Description |
|---|---|
| `Train.csv` | Grid squares with 2015 and 2019 rainfall features + 2015 flood target |
| `SampleSubmission.csv` | Submission format template |

External soil data (FAO/ISRIC) must be separately downloaded and joined — see `03_external_data.R`.

## Configuration

Key settings in `00_config.R`:

| Variable | Default | Purpose |
|---|---|---|
| `GLOBAL_SEED` | 7 | Reproducibility |
| `CV_FOLDS` | 5 | Cross-validation folds |
| `NUM_THREADS` | 4 | Parallelism for boosting models |
| `DATA_DIR` | `"."` | Directory containing Train.csv |
# 5G Base Station Energy Consumption Prediction

Predicting energy consumption for 5G base stations from network load, energy-saving mode states, hardware configuration (antenna count, RU type, frequency/bandwidth), and time-of-day patterns. Competition dataset (2023), tabular regression.

## Key Engineering Decisions

**Target construction via frequency-weighted energy disaggregation.** The raw data records total energy at the base station level. For multi-cell stations (two cells operating at different frequencies), no per-cell energy measurement exists. To enable cell-level modeling, energy is attributed to each cell using frequency-specific proportions derived from hardware characteristics — `AVG_365`, `AVG_426`, `AVG_155`, `AVG_189`. Predicting cell energy is only possible after this attribution step. Without it, the label itself is undefined at the modeling granularity.

```r
energy <- energy %>%
  mutate(
    perct = case_when(
      nb_cell == 2 & freq_cat == '365'    ~ AVG_365,
      nb_cell == 2 & freq_cat == '426.98' ~ AVG_426,
      nb_cell == 2 & freq_cat == '155.6'  ~ AVG_155,
      nb_cell == 2 & freq_cat == '189'    ~ AVG_189,
      TRUE ~ 1
    ),
    target = Energy * perct
  )
```

**Domain-derived indicator features from hardware rules.** Features like `es1_load_low`, `es3_load_high`, `tree_energy_pack_F` are explicit interaction rules between energy-saving mode states, load levels, antenna counts, and frequencies. These encode known hardware behavior (e.g. ES1 activates under very low load with large antenna arrays) as binary flags rather than relying on the model to rediscover the interaction.

**Per-variable encoding strategy.** Each categorical variable receives an encoding matched to its structure: Helmert contrasts for `day_of_week` (ordered, sequential), polynomial contrasts for `RUType` (ordered by hardware generation), backward difference for `type_anten`, WoE for binary indicator flags, M-estimator smoothed encoding for `load_hour`. A separate dataset (`05_build_lightgbm_dataset.R`) is built with these pre-encoded representations specifically for LightGBM.

**Two-level modeling: base station and cell.** Models are trained independently at both granularities. Base-level models use aggregated cell features; cell-level models use individual cell observations. Final submission blends predictions from both levels.

**Three-way train / freeze / test split.** A dedicated freeze set is held out for ensemble weight optimization and model comparison, separate from both training and final test.

## Project Structure

```
00_config.R                  # Global settings and utility functions
01_data_loading.R            # Load raw data, apply hardware fixes, aggregate base info
02_feature_engineering.R     # Encoding functions and domain feature constructors
03_build_base_dataset.R      # Base-level dataset with aggregated cell features
04_build_cell_dataset.R      # Cell-level dataset with frequency-weighted energy split
05_build_lightgbm_dataset.R  # Pre-encoded dataset variant for LightGBM
06_train_test_split.R        # Train / freeze / test splits (base and cell)
07_model_lightgbm.R          # LightGBM with grid search
08_model_xgboost.R           # XGBoost with Bayesian tuning via tidymodels
09_model_catboost.R          # CatBoost with K-fold CV averaging
10_model_caret.R             # RF, GLMNet, GAM via caret
11_model_h2o.R               # H2O AutoML
12_model_stacking.R          # Tidymodels stacking ensemble
13_model_sl3.R               # SuperLearner (sl3) with NNLS metalearner
14_clustering_analysis.R     # Cluster features from load/ES mode patterns
15_model_cell_level.R        # LightGBM and XGBoost at cell granularity
16_generate_submission.R     # Ensemble and format final predictions
17_eda_and_visualization.R   # EDA reports
18_evaluate_freeze.R         # Model comparison on freeze set
MAIN.R                       # Pipeline orchestration
```

## Running the Pipeline

```r
source("MAIN.R")
run_pipeline()          # All steps

run_data_prep()         # Steps 1-6 only
run_training()          # Steps 7-15 only
run_evaluation()        # Steps 16-18 only

run_pipeline(steps = c(1, 3, 7))  # Specific steps
```

## Data Files Required

| File | Contents |
|---|---|
| `imgs_202307101549519358.csv` | Submission template |
| `imgs_2023071012133740345.csv` | Base station energy measurements |
| `imgs_2023071012130978799.csv` | Cell-level load and ES mode states |
| `imgs_2023071012123392536.csv` | Base station hardware configuration |

## Configuration

Key settings in `00_config.R`:

| Variable | Purpose |
|---|---|
| `GLOBAL_SEED` | Reproducibility seed |
| `FILE_SUBMISSION` | Path to submission template |
| `FILE_ENERGY` | Path to energy data |
| `FILE_CELL` | Path to cell data |
| `FILE_BASE` | Path to base station data |

Edit `00_config.R` to modify:
- `GLOBAL_SEED`: Random seed for reproducibility (default: 1618)
- `CV_FOLDS`: Number of cross-validation folds (default: 5)
- `TRAIN_SAMPLE_SIZE`: Training set size (default: 75000)
- `FREEZE_SAMPLE_SIZE`: Holdout validation size (default: 10000)
- `AVG_*`: Energy split proportions for multi-cell bases

## Output Files

### Intermediate Data Files
- `data_base_cleaned.RData`: Cleaned base station data
- `df_base.RData`: Base-level feature dataset
- `df_cell.RData`: Cell-level feature dataset
- `df_ligh.RData`: LightGBM-optimized dataset
- `splits_base.RData`: Train/test/freeze splits (base level)
- `splits_cell.RData`: Train/test/freeze splits (cell level)

### Model Files
- `model_lgb_*.txt`: LightGBM models
- `model_xgb_*.model`: XGBoost models
- `model_catboost_*.cbm`: CatBoost models
- `model_*_caret.rds`: Caret models
- `model_stacks.rds`: Stacked ensemble
- `model_sl3.rds`: Super Learner model

### Prediction Files
- `freez_*_*.csv`: Freeze set predictions for each model
- `submission_*.csv`: Final submission files

### Analysis Files
- `model_comparison_results.csv`: Model performance comparison
- `eda_results.rds`: EDA analysis results
- `cluster_summary.csv`: Clustering analysis results

## Dependencies

### Core Packages
- data.table, dplyr, tidyr, purrr

### Machine Learning
- caret, tidymodels, recipes, stacks
- xgboost, lightgbm, catboost
- ranger, glmnet, mgcv
- h2o, sl3
- treesnip, bonsai

### Parallel Processing
- parallel, doParallel, future

### Visualization & Analysis
- ggplot2, rpart, rpart.plot
- effectsize, ltm

## Original File Mapping

| Original File       | Reorganized To                           |
|---------------------|------------------------------------------|
| First.R             | 01, 02, 03, 17                           |
| Second.R            | 04                                       |
| Build_DF.R          | 03, 04, 05                               |
| Encoding.R          | 02, 05                                   |
| Freez.R             | 06, 18                                   |
| Caret.R             | 10                                       |
| Catboost.R          | 09                                       |
| draft.R, draf_2.R   | 07, 08                                   |
| ligh_random*.R      | 07                                       |
| ligh_recipes.R      | 07, 15                                   |
| h2o.R               | 11                                       |
| Model_Base.R        | 07, 08, 10, 11, 12, 13                   |
| Model_Cell.R        | 15                                       |
| Stacking.R          | 12, 18                                   |
| Clustering.R        | 14                                       |
| Submit.R            | 16                                       |
| Graphs.R            | 17                                       |

## Best Practices Applied

1. **Modular Design**: Each script has a single responsibility
2. **Configuration Separation**: All constants in 00_config.R
3. **Consistent Naming**: Files numbered in execution order
4. **Documentation**: Detailed comments and headers
5. **Error Handling**: Graceful handling of missing files
6. **Reproducibility**: Seeds set for all random operations
7. **Reusable Functions**: Helper functions in 02_feature_engineering.R
8. **Pipeline Orchestration**: MAIN.R for easy execution

## Notes

- The freeze set is used for local validation before submission
- Cell-level predictions need aggregation to base level for submission
- Some models (H2O, SL3) may require additional installation steps
- Run times vary significantly by model (LightGBM ~5min, H2O AutoML ~30min)
