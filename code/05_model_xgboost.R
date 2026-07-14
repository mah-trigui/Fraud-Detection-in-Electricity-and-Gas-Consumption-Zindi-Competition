# =============================================================================
# 05_model_xgboost.R - XGBoost Models (Multiple Configs + Bayesian Tuning)
# STEG Fraud Detection
# =============================================================================

source("00_config.R")
print_section("Step 5: XGBoost Models")

train_50 <- readRDS(file.path(OUTPUT_DIR, "train_50.rds"))
test_holdout <- readRDS(file.path(OUTPUT_DIR, "test_holdout.rds"))
submit <- readRDS(file.path(OUTPUT_DIR, "submit_encoded.rds"))
final_features <- readRDS(file.path(OUTPUT_DIR, "final_features.rds"))

# Prepare matrices (XGBoost needs numeric only)
num_features <- intersect(final_features, names(train_50))
ytest <- test_holdout$target

x <- as.matrix(train_50[, num_features, drop = FALSE])
xt <- as.matrix(test_holdout[, num_features, drop = FALSE])
xs <- as.matrix(submit[, num_features, drop = FALSE])

dtrain <- xgb.DMatrix(data = x, label = train_50$target)
dtest <- xgb.DMatrix(data = xt, label = ytest)
dsubmit <- xgb.DMatrix(data = xs)

# -----------------------------------------------------------------------------
# MODEL 1: Bayesian-optimized parameters
# -----------------------------------------------------------------------------

print_step("XGBoost Model 1 (Bayesian-tuned)...")

params1 <- list(
    booster = "gbtree", objective = "binary:logistic",
    eval_metric = "auc",
    eta = 0.05, max_depth = 6,
    min_child_weight = 3, subsample = 0.8,
    colsample_bytree = 0.7, gamma = 1,
    alpha = 2, lambda = 3,
    nthread = NUM_THREADS
)

cv1 <- xgb.cv(
    params = params1, data = dtrain,
    nrounds = 5000, nfold = 5,
    early_stopping_rounds = 150,
    maximize = TRUE, verbose = 0
)

model_xgb1 <- xgb.train(
    params = params1, data = dtrain,
    nrounds = cv1$best_iteration, verbose = 0
)
pred_xgb1_test <- predict(model_xgb1, dtest)
pred_xgb1_submit <- predict(model_xgb1, dsubmit)
cat(sprintf(
    "  Model 1 AUC: %.4f (rounds: %d)\n",
    MLmetrics::AUC(pred_xgb1_test, ytest), cv1$best_iteration
))

# -----------------------------------------------------------------------------
# MODEL 2: Higher regularization
# -----------------------------------------------------------------------------

print_step("XGBoost Model 2...")

params2 <- list(
    booster = "gbtree", objective = "binary:logistic",
    eval_metric = "auc",
    eta = 0.01, max_depth = 4,
    min_child_weight = 5, subsample = 0.75,
    colsample_bytree = 0.8, gamma = 0.5,
    nthread = NUM_THREADS
)

cv2 <- xgb.cv(
    params = params2, data = dtrain,
    nrounds = 7000, nfold = 5,
    early_stopping_rounds = 200,
    maximize = TRUE, verbose = 0
)

model_xgb2 <- xgb.train(
    params = params2, data = dtrain,
    nrounds = cv2$best_iteration, verbose = 0
)
pred_xgb2_test <- predict(model_xgb2, dtest)
pred_xgb2_submit <- predict(model_xgb2, dsubmit)
cat(sprintf(
    "  Model 2 AUC: %.4f (rounds: %d)\n",
    MLmetrics::AUC(pred_xgb2_test, ytest), cv2$best_iteration
))

# -----------------------------------------------------------------------------
# MODEL 3: Shallow + aggressive subsampling
# -----------------------------------------------------------------------------

print_step("XGBoost Model 3...")

params3 <- list(
    booster = "gbtree", objective = "binary:logistic",
    eval_metric = "auc",
    eta = 0.1, max_depth = 4,
    min_child_weight = 1, subsample = 0.6,
    colsample_bytree = 0.5, gamma = 2,
    nthread = NUM_THREADS
)

cv3 <- xgb.cv(
    params = params3, data = dtrain,
    nrounds = 3000, nfold = 5,
    early_stopping_rounds = 100,
    maximize = TRUE, verbose = 0
)

model_xgb3 <- xgb.train(
    params = params3, data = dtrain,
    nrounds = cv3$best_iteration, verbose = 0
)
pred_xgb3_test <- predict(model_xgb3, dtest)
pred_xgb3_submit <- predict(model_xgb3, dsubmit)
cat(sprintf(
    "  Model 3 AUC: %.4f (rounds: %d)\n",
    MLmetrics::AUC(pred_xgb3_test, ytest), cv3$best_iteration
))

# -----------------------------------------------------------------------------
# MODEL 4: Caret-tuned (grid search)
# -----------------------------------------------------------------------------

print_step("XGBoost Model 4 (caret grid)...")

train_50$fraud_f <- as.factor(ifelse(train_50$target == 1, "yes", "no"))
fitControl <- trainControl(
    method = "cv", number = 5,
    classProbs = TRUE, summaryFunction = twoClassSummary
)

xgb_caret <- train(
    x = train_50[, num_features], y = train_50$fraud_f,
    method = "xgbTree", trControl = fitControl,
    tuneLength = 5, maximize = TRUE, metric = "ROC", verbose = FALSE
)

pred_xgb4_test <- predict(xgb_caret, test_holdout[, num_features], type = "prob")$yes
pred_xgb4_submit <- predict(xgb_caret, submit[, num_features], type = "prob")$yes
cat(sprintf("  Model 4 AUC: %.4f\n", MLmetrics::AUC(pred_xgb4_test, ytest)))

# -----------------------------------------------------------------------------
# SAVE
# -----------------------------------------------------------------------------

preds_xgb <- list(
    test   = cbind(pred_xgb1_test, pred_xgb2_test, pred_xgb3_test, pred_xgb4_test),
    submit = cbind(pred_xgb1_submit, pred_xgb2_submit, pred_xgb3_submit, pred_xgb4_submit)
)
saveRDS(preds_xgb, file.path(OUTPUT_DIR, "preds_xgboost.rds"))

print_step("XGBoost models complete.")
