# =============================================================================
# 06_model_catboost_rf.R - CatBoost + Random Forest + H2O AutoML
# STEG Fraud Detection
# =============================================================================

source("00_config.R")
print_section("Step 6: CatBoost, Random Forest & H2O Models")

train_50 <- readRDS(file.path(OUTPUT_DIR, "train_50.rds"))
test_holdout <- readRDS(file.path(OUTPUT_DIR, "test_holdout.rds"))
submit <- readRDS(file.path(OUTPUT_DIR, "submit_encoded.rds"))
final_features <- readRDS(file.path(OUTPUT_DIR, "final_features.rds"))

ytest <- test_holdout$target
num_features <- intersect(final_features, names(train_50))

# -----------------------------------------------------------------------------
# CATBOOST (3 variants)
# -----------------------------------------------------------------------------

print_step("Training CatBoost models...")

X_train <- train_50[, num_features]
X_test <- test_holdout[, num_features]
X_sub <- submit[, num_features]

train_pool <- catboost.load_pool(data = X_train, label = train_50$target)
test_pool <- catboost.load_pool(data = X_test)
submit_pool <- catboost.load_pool(data = X_sub)

# CatBoost 1: Manual tuned
params_cb1 <- list(
    iterations = 500, learning_rate = 0.05,
    l2_leaf_reg = 3, loss_function = "CrossEntropy",
    eval_metric = "AUC", random_seed = 721,
    logging_level = "Silent",
    random_strength = 0.2, bagging_temperature = 0,
    border_count = 10, rsm = 1
)
cb1 <- catboost.train(train_pool, params = params_cb1)
pred_cb1_test <- catboost.predict(cb1, test_pool, prediction_type = "Probability")
pred_cb1_submit <- catboost.predict(cb1, submit_pool, prediction_type = "Probability")
cat(sprintf("  CatBoost 1 AUC: %.4f\n", MLmetrics::AUC(pred_cb1_test, ytest)))

# CatBoost 2: Auto early stopping
set.seed(GLOBAL_SEED)
idx <- createDataPartition(train_50$target, p = 0.8, list = FALSE)
dataset <- train_50[idx, ]
validation <- train_50[-idx, ]

train_pool2 <- catboost.load_pool(data = dataset[, num_features], label = dataset$target)
valid_pool2 <- catboost.load_pool(data = validation[, num_features], label = validation$target)

params_cb2 <- list(
    iterations = 5000, depth = 6,
    loss_function = "CrossEntropy", eval_metric = "AUC",
    random_seed = 721, logging_level = "Silent",
    od_type = "IncToDec", od_wait = 200,
    use_best_model = TRUE, bagging_temperature = 0,
    border_count = 10, rsm = 1
)
cb2 <- catboost.train(train_pool2, valid_pool2, params = params_cb2)
pred_cb2_test <- catboost.predict(cb2, test_pool, prediction_type = "Probability")
pred_cb2_submit <- catboost.predict(cb2, submit_pool, prediction_type = "Probability")
cat(sprintf("  CatBoost 2 AUC: %.4f\n", MLmetrics::AUC(pred_cb2_test, ytest)))

# CatBoost 3: Caret wrapper
train_50$fraud_f <- as.factor(ifelse(train_50$target == 1, "yes", "no"))
fitControl <- trainControl(
    method = "cv", number = 5,
    classProbs = TRUE, summaryFunction = twoClassSummary
)
cb3_caret <- train(
    x = X_train, y = train_50$fraud_f,
    method = "catboost", trControl = fitControl, metric = "ROC"
)
pred_cb3_test <- predict(cb3_caret, X_test, type = "prob")$yes
pred_cb3_submit <- predict(cb3_caret, X_sub, type = "prob")$yes
cat(sprintf("  CatBoost 3 AUC: %.4f\n", MLmetrics::AUC(pred_cb3_test, ytest)))

# -----------------------------------------------------------------------------
# H2O AUTOML
# -----------------------------------------------------------------------------

print_step("Running H2O AutoML...")

invisible(h2o.init(nthreads = -1))
train_h <- as.h2o(train_50[, c(num_features, "fraud_f")])
test_h <- as.h2o(test_holdout[, c(num_features, "target")])
submit_h <- as.h2o(submit[, num_features])

aml <- h2o.automl(
    x = num_features, y = "fraud_f",
    training_frame = train_h,
    max_models = 50, stopping_metric = "AUC",
    sort_metric = "AUC", seed = 1989,
    max_runtime_secs = 1800, nfolds = 5
)

pred_h2o_test <- as.data.table(h2o.predict(aml@leader, test_h))$p1
pred_h2o_submit <- as.data.table(h2o.predict(aml@leader, submit_h))$p1
cat(sprintf("  H2O AutoML AUC: %.4f\n", MLmetrics::AUC(pred_h2o_test, ytest)))

# Random Forest from H2O
my_rf <- h2o.randomForest(
    x = num_features, y = "fraud_f",
    training_frame = train_h, ntrees = 500,
    seed = 1989, nfolds = 5
)
pred_rf1_test <- as.data.table(h2o.predict(my_rf, test_h))$p1
pred_rf1_submit <- as.data.table(h2o.predict(my_rf, submit_h))$p1
cat(sprintf("  H2O RF AUC: %.4f\n", MLmetrics::AUC(pred_rf1_test, ytest)))

# -----------------------------------------------------------------------------
# SAVE
# -----------------------------------------------------------------------------

preds_cat <- list(
    test   = cbind(pred_cb1_test, pred_cb2_test, pred_cb3_test),
    submit = cbind(pred_cb1_submit, pred_cb2_submit, pred_cb3_submit)
)
preds_other <- list(
    test   = cbind(pred_h2o_test, pred_rf1_test),
    submit = cbind(pred_h2o_submit, pred_rf1_submit)
)

saveRDS(preds_cat, file.path(OUTPUT_DIR, "preds_catboost.rds"))
saveRDS(preds_other, file.path(OUTPUT_DIR, "preds_h2o_rf.rds"))

h2o.shutdown(prompt = FALSE)

print_step("CatBoost/RF/H2O models complete.")
