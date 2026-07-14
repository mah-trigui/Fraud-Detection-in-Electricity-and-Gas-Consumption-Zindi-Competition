# =============================================================================
# 03_encoding_selection.R - Encoding Strategies + Feature Selection
# STEG Fraud Detection
# =============================================================================
# Key design decisions:
# 1. Multiple target encoders compared: target mean, WOE, James-Stein, M-estimator
# 2. Feature selection via random subset search (100 iterations × LightGBM)
#    → keep features that appear across multiple winning subsets
# 3. Also: elastic net for variable importance, IV analysis, correlation filtering
# =============================================================================

source("00_config.R")
source("Encoders.R") # Custom encoding functions
print_section("Step 3: Encoding & Feature Selection")

client <- readRDS(file.path(OUTPUT_DIR, "client_features.rds"))

# Split
train <- client[!is.na(client$target), ]
submit <- client[is.na(client$target), ]

cat(sprintf("  Train: %d | Submit: %d\n", nrow(train), nrow(submit)))

# -----------------------------------------------------------------------------
# 1. TARGET ENCODING OF REGION
# -----------------------------------------------------------------------------

print_step("Applying multiple target encoders to region...")

df <- train
df$reg <- encode_target(df[["region"]], df[["target"]])
df$reg_woe <- encode_woe(df[["region"]], df[["target"]])
df$reg_james <- encode_james_stein(df[["region"]], df[["target"]])
df$reg_estim <- encode_m_estimator(df[["region"]], df[["target"]])
df$reg_woe[!is.finite(df$reg_woe)] <- 0

# Map to submit set
unique_region <- unique(df[, c("region", "reg", "reg_woe", "reg_james", "reg_estim")])
train <- df
submit <- submit %>% left_join(unique_region, by = "region")

# Client category and district dummies
for (ds in list(train, submit)) {
    ds$cat_12 <- as.integer(ds$client_catg == "12")
    ds$cat_51 <- as.integer(ds$client_catg == "51")
    ds$dist_60 <- as.integer(ds$disrict == "60")
    ds$dist_63 <- as.integer(ds$disrict == "63")
    ds$dist_69 <- as.integer(ds$disrict == "69")
}
train$cat_12 <- as.integer(train$client_catg == "12")
train$cat_51 <- as.integer(train$client_catg == "51")
train$dist_60 <- as.integer(train$disrict == "60")
train$dist_63 <- as.integer(train$disrict == "63")
train$dist_69 <- as.integer(train$disrict == "69")
submit$cat_12 <- as.integer(submit$client_catg == "12")
submit$cat_51 <- as.integer(submit$client_catg == "51")
submit$dist_60 <- as.integer(submit$disrict == "60")
submit$dist_63 <- as.integer(submit$disrict == "63")
submit$dist_69 <- as.integer(submit$disrict == "69")

# -----------------------------------------------------------------------------
# 2. STRATIFIED SAMPLING (imbalanced: 5.6% fraud)
# -----------------------------------------------------------------------------

print_step("Creating stratified training subsets...")

set.seed(GLOBAL_SEED)
index <- createDataPartition(train$target, p = 0.1, list = FALSE)
test_holdout <- train[index, ]
train_full <- train[-index, ]

# Undersampled 50/50 for fast feature selection
strat_50 <- stratified(train_full[train_full$target == 0, ],
    c("nb_fact", "nb_counter", "nb_read"),
    size = 0.07
)
train_50 <- rbind(strat_50, train_full[train_full$target == 1, ])

cat(sprintf(
    "  Holdout: %d | Train_50 (balanced): %d\n",
    nrow(test_holdout), nrow(train_50)
))

# -----------------------------------------------------------------------------
# 3. FEATURE SELECTION: RANDOM SUBSET SEARCH
# -----------------------------------------------------------------------------

print_step("Running random subset search (100 iterations × LightGBM)...")

num_features <- setdiff(
    names(train_50)[sapply(train_50, is.numeric)],
    c("client_id", "target", "fraud")
)
categ <- c("region", "disrict", "client_catg")

ytest <- test_holdout$target
higher_auc_list <- list()
parameters_list <- list()

for (i in 1:100) {
    set.seed(i)
    n <- sample(12:35, 1)
    m <- sample(num_features, n)

    x <- as.matrix(train_50[, c(m, categ), drop = FALSE])
    xt <- as.matrix(test_holdout[, c(m, categ), drop = FALSE])

    dtrain <- lgb.Dataset(
        data = x, label = train_50$target,
        free_raw_data = FALSE, categorical_feature = categ
    )
    dtest <- lgb.Dataset.create.valid(dtrain, data = xt, label = ytest)
    valids <- list(train = dtrain, test = dtest)

    bst <- lgb.train(
        data = dtrain, nrounds = 2000,
        objective = "binary", eval = "auc", metric = "auc",
        valids = valids, early_stopping_round = 50,
        nthread = NUM_THREADS, verbose = -1
    )

    higher_auc_list[[i]] <- MLmetrics::AUC(predict(bst, xt), ytest)
    parameters_list[[i]] <- m
}

# Find best subsets and intersect
best_idx <- order(unlist(higher_auc_list), decreasing = TRUE)[1:5]
best_features <- Reduce(union, parameters_list[best_idx])

cat(sprintf(
    "  Best AUC: %.4f | Union of top-5 subsets: %d features\n",
    max(unlist(higher_auc_list)), length(best_features)
))

# -----------------------------------------------------------------------------
# 4. ELASTIC NET VALIDATION
# -----------------------------------------------------------------------------

print_step("Elastic net for confirmation...")

x_enet <- as.matrix(train_50[, num_features])
y_enet <- as.factor(train_50$target)
grid <- expand.grid(.alpha = seq(0, 1, by = 0.2), .lambda = seq(0.01, 0.03, by = 0.002))
control <- trainControl(method = "cv", number = 5)

enet <- train(x_enet, y_enet,
    method = "glmnet",
    trControl = control, tuneGrid = grid, metric = "Kappa"
)

best_enet <- glmnet::glmnet(x_enet, y_enet,
    alpha = enet$bestTune$alpha,
    lambda = enet$bestTune$lambda, family = "binomial"
)
enet_coefs <- coef(best_enet)
enet_features <- rownames(enet_coefs)[which(enet_coefs != 0)][-1]

cat(sprintf("  Elastic net selected: %d features\n", length(enet_features)))

# Final feature list (union of random subset + elastic net)
final_features <- unique(c(best_features, enet_features))

# -----------------------------------------------------------------------------
# 5. SAVE
# -----------------------------------------------------------------------------

saveRDS(train, file.path(OUTPUT_DIR, "train_encoded.rds"))
saveRDS(submit, file.path(OUTPUT_DIR, "submit_encoded.rds"))
saveRDS(train_50, file.path(OUTPUT_DIR, "train_50.rds"))
saveRDS(test_holdout, file.path(OUTPUT_DIR, "test_holdout.rds"))
saveRDS(final_features, file.path(OUTPUT_DIR, "final_features.rds"))

print_step("Encoding & selection complete.")
