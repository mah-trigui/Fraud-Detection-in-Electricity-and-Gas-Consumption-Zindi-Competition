# =============================================================================
# 04_model_lightgbm.R - LightGBM Models (Multiple Seeds/Configs)
# STEG Fraud Detection
# =============================================================================
# Train 3 LightGBM variants with different seeds and hyperparameters.
# LightGBM handles categoricals natively and runs fast for ensemble diversity.
# =============================================================================

source("00_config.R")
print_section("Step 4: LightGBM Models")

train_50 <- readRDS(file.path(OUTPUT_DIR, "train_50.rds"))
test_holdout <- readRDS(file.path(OUTPUT_DIR, "test_holdout.rds"))
submit <- readRDS(file.path(OUTPUT_DIR, "submit_encoded.rds"))
final_features <- readRDS(file.path(OUTPUT_DIR, "final_features.rds"))

categ <- c("region", "disrict", "client_catg")
all_features <- c(final_features, categ)
all_features <- intersect(all_features, names(train_50))

ytest <- test_holdout$target

x <- as.matrix(train_50[, all_features, drop = FALSE])
xt <- as.matrix(test_holdout[, all_features, drop = FALSE])
xs <- as.matrix(submit[, all_features, drop = FALSE])

dtrain <- lgb.Dataset(
    data = x, label = train_50$target,
    free_raw_data = FALSE, categorical_feature = categ
)
dtest <- lgb.Dataset.create.valid(dtrain, data = xt, label = ytest)
valids <- list(train = dtrain, test = dtest)

# -----------------------------------------------------------------------------
# MODEL 1: Default tuned
# -----------------------------------------------------------------------------

print_step("LightGBM Model 1...")

set.seed(1989)
bst1 <- lgb.train(
    data = dtrain, nrounds = 1000,
    objective = "binary", eval = "auc", metric = "auc",
    valids = valids, early_stopping_round = 50,
    nthread = NUM_THREADS, seed = 1989,
    num_leaves = 31, min_data_in_leaf = 38,
    bagging_fraction = 0.7, feature_fraction = 0.4,
    feature_fraction_bynode = 0.8,
    lambda_l1 = 2.81, lambda_l2 = 3.01,
    feature_pre_filter = FALSE
)
pred1_test <- predict(bst1, xt)
pred1_submit <- predict(bst1, xs)
cat(sprintf("  Model 1 AUC: %.4f\n", MLmetrics::AUC(pred1_test, ytest)))

# -----------------------------------------------------------------------------
# MODEL 2: Different seed + params
# -----------------------------------------------------------------------------

print_step("LightGBM Model 2...")

set.seed(721)
bst2 <- lgb.train(
    data = dtrain, nrounds = 1000,
    objective = "binary", eval = "auc", metric = "auc",
    valids = valids, early_stopping_round = 50,
    nthread = NUM_THREADS, seed = 721,
    num_leaves = 31, min_data_in_leaf = 47,
    min_data_per_group = 59,
    bagging_fraction = 0.7, feature_fraction = 0.5,
    feature_fraction_bynode = 0.7,
    lambda_l1 = 3.0, lambda_l2 = 2.5,
    feature_pre_filter = FALSE
)
pred2_test <- predict(bst2, xt)
pred2_submit <- predict(bst2, xs)
cat(sprintf("  Model 2 AUC: %.4f\n", MLmetrics::AUC(pred2_test, ytest)))

# -----------------------------------------------------------------------------
# MODEL 3: DART booster
# -----------------------------------------------------------------------------

print_step("LightGBM Model 3 (DART)...")

set.seed(1618)
bst3 <- lgb.train(
    data = dtrain, nrounds = 500,
    objective = "binary", boosting = "dart",
    eval = "auc", metric = "auc",
    valids = valids, early_stopping_round = 50,
    nthread = NUM_THREADS, seed = 1618,
    num_leaves = 10, max_depth = 6, learning_rate = 0.1
)
pred3_test <- predict(bst3, xt)
pred3_submit <- predict(bst3, xs)
cat(sprintf("  Model 3 AUC: %.4f\n", MLmetrics::AUC(pred3_test, ytest)))

# -----------------------------------------------------------------------------
# SAVE
# -----------------------------------------------------------------------------

preds_light <- list(
    test   = cbind(pred1_test, pred2_test, pred3_test),
    submit = cbind(pred1_submit, pred2_submit, pred3_submit)
)
saveRDS(preds_light, file.path(OUTPUT_DIR, "preds_lightgbm.rds"))

print_step("LightGBM models complete.")
