# =============================================================================
# 07_ensemble.R - Meta-Learner Blending of 14 Base Models
# STEG Fraud Detection
# =============================================================================
# Architecture:
# Level 0: 14 base models (3 LightGBM + 4 XGBoost + 3 CatBoost + H2O + 3 RF)
# Level 1: Meta-learner (elastic net / neural net / weighted average)
#
# Key insight: base models are trained on different subsamples and with
# different hyperparameters for diversity. The meta-learner learns
# optimal combination weights.
# =============================================================================

source("00_config.R")
print_section("Step 7: Meta-Learner Ensemble")

preds_light <- readRDS(file.path(OUTPUT_DIR, "preds_lightgbm.rds"))
preds_xgb <- readRDS(file.path(OUTPUT_DIR, "preds_xgboost.rds"))
preds_cat <- readRDS(file.path(OUTPUT_DIR, "preds_catboost.rds"))
preds_other <- readRDS(file.path(OUTPUT_DIR, "preds_h2o_rf.rds"))
test_holdout <- readRDS(file.path(OUTPUT_DIR, "test_holdout.rds"))
submit <- readRDS(file.path(OUTPUT_DIR, "submit_encoded.rds"))

ytest <- test_holdout$target

# -----------------------------------------------------------------------------
# 1. BUILD META-FEATURES MATRIX
# -----------------------------------------------------------------------------

print_step("Assembling meta-feature matrix (14 base models)...")

meta_test <- data.table(
    light_1 = preds_light$test[, 1], light_2 = preds_light$test[, 2],
    light_3 = preds_light$test[, 3],
    xgb_1 = preds_xgb$test[, 1], xgb_2 = preds_xgb$test[, 2],
    xgb_3 = preds_xgb$test[, 3], xgb_4 = preds_xgb$test[, 4],
    cat_1 = preds_cat$test[, 1], cat_2 = preds_cat$test[, 2],
    cat_3 = preds_cat$test[, 3],
    h2o = preds_other$test[, 1], rf_1 = preds_other$test[, 2],
    target = ytest
)

meta_submit <- data.table(
    light_1 = preds_light$submit[, 1], light_2 = preds_light$submit[, 2],
    light_3 = preds_light$submit[, 3],
    xgb_1 = preds_xgb$submit[, 1], xgb_2 = preds_xgb$submit[, 2],
    xgb_3 = preds_xgb$submit[, 3], xgb_4 = preds_xgb$submit[, 4],
    cat_1 = preds_cat$submit[, 1], cat_2 = preds_cat$submit[, 2],
    cat_3 = preds_cat$submit[, 3],
    h2o = preds_other$submit[, 1], rf_1 = preds_other$submit[, 2]
)

# Per-family averages
meta_test[, `:=`(
    light_avg = (light_1 + light_2 + light_3) / 3,
    xgb_avg   = (xgb_1 + xgb_2 + xgb_3 + xgb_4) / 4,
    cat_avg   = (cat_1 + cat_2 + cat_3) / 3
)]
meta_submit[, `:=`(
    light_avg = (light_1 + light_2 + light_3) / 3,
    xgb_avg   = (xgb_1 + xgb_2 + xgb_3 + xgb_4) / 4,
    cat_avg   = (cat_1 + cat_2 + cat_3) / 3
)]

# -----------------------------------------------------------------------------
# 2. WEIGHTED AVERAGE (performance-based)
# -----------------------------------------------------------------------------

print_step("Computing performance-weighted average...")

# Individual AUCs
aucs <- sapply(meta_test[, 1:12], function(p) MLmetrics::AUC(p, ytest))
cat("  Individual model AUCs:\n")
print(round(aucs, 4))

# Normalize to weights
weights <- aucs / sum(aucs)
pred_wavg <- as.matrix(meta_submit[, 1:12]) %*% weights

cat(sprintf(
    "  Weighted average AUC on holdout: %.4f\n",
    MLmetrics::AUC(as.matrix(meta_test[, 1:12]) %*% weights, ytest)
))

# -----------------------------------------------------------------------------
# 3. ELASTIC NET META-LEARNER
# -----------------------------------------------------------------------------

print_step("Training elastic net meta-learner...")

x_meta <- as.matrix(meta_test[, 1:12])
y_meta <- as.factor(ifelse(ytest == 1, "A", "B"))
s_meta <- as.matrix(meta_submit[, 1:12])

control <- trainControl(
    method = "repeatedcv", number = 5, repeats = 3,
    classProbs = TRUE, summaryFunction = twoClassSummary
)

enet_meta <- train(x_meta, y_meta,
    method = "glmnet",
    trControl = control, tuneLength = 10, metric = "ROC"
)

pred_enet <- predict(enet_meta, s_meta, type = "prob")$A
cat(sprintf(
    "  Elastic net meta AUC: %.4f\n",
    MLmetrics::AUC(predict(enet_meta, x_meta, type = "prob")$A, ytest)
))

# -----------------------------------------------------------------------------
# 4. FINAL SUBMISSION
# -----------------------------------------------------------------------------

print_step("Generating final submission...")

# Use best meta-learner
final_pred <- pred_enet

submission <- data.table(
    client_id = submit$client_id,
    target = final_pred
)

fwrite(submission, file.path(OUTPUT_DIR, "submission_ensemble.csv"),
    quote = FALSE, row.names = FALSE
)

cat(sprintf("  Final submission: %d rows\n", nrow(submission)))
cat(sprintf(
    "  Predicted fraud (>0.5): %d (%.1f%%)\n",
    sum(final_pred > 0.5), 100 * mean(final_pred > 0.5)
))

print_step("Ensemble complete. Pipeline finished.")
