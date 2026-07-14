# =============================================================================
# 00_config.R - Configuration and Setup
# STEG Fraud Detection in Electricity and Gas Consumption
# =============================================================================
# Task: Detect fraudulent meter manipulation in Tunisian utility clients
# Data: Client metadata + billing/invoice history (2005-2019)
# Target: Binary fraud (5.6% positive rate)
# Metric: AUC
# Approach: Invoice aggregation to client-level → multiple encoders →
#           massive ensemble (14 base models) → meta-learner blending
# =============================================================================

GLOBAL_SEED <- 1618
NUM_THREADS <- 10

DATA_DIR <- "."
OUTPUT_DIR <- file.path(DATA_DIR, "output")

if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)

# Libraries
suppressPackageStartupMessages({
    library(data.table)
    library(dplyr)
    library(sqldf)
    library(Matrix)
    library(caret)
    library(xgboost)
    library(lightgbm)
    library(catboost)
    library(h2o)
    library(glmnet)
    library(MLmetrics)
    library(Information)
    library(fifer)
    library(lubridate)
})
options(sqldf.driver = "RSQLite")
set.seed(GLOBAL_SEED)

print_section <- function(msg) {
    cat("\n", strrep("=", 70), "\n", msg, "\n", strrep("=", 70), "\n\n", sep = "")
}
print_step <- function(msg) cat("  >>", msg, "\n")
# =============================================================================
# 00_config.R - Configuration and Setup
# Sea Turtle Rescue Forecast Challenge
# =============================================================================
# Task: Predict number of turtles rescued per week per capture site for 2019
# Data: 29 capture sites, weekly rescue counts from 1998-2018
# Approach: Hybrid time series (STL+ETS per site) + supervised learning
#           (CatBoost/XGBoost) with TS forecasts as features
# =============================================================================

GLOBAL_SEED <- 7
NUM_THREADS <- 10
FREQ <- 44 # 44 weeks per prediction year
YEAR_EXCLUDE <- 2016 # Data quality issue — big drop
TRAIN_START <- "2013-12-30"
TEST_START <- "2018-12-31"
TEST_END <- "2019-10-31"

DATA_DIR <- "."
OUTPUT_DIR <- file.path(DATA_DIR, "output")

if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)

# Libraries
suppressPackageStartupMessages({
    library(data.table)
    library(dplyr)
    library(sqldf)
    library(Matrix)
    library(xgboost)
    library(catboost)
    library(caret)
    library(useful)
    library(forecast)
    library(lubridate)
    library(xts)
    library(ggplot2)
    library(skimr)
})
options(sqldf.driver = "RSQLite")
set.seed(GLOBAL_SEED)

print_section <- function(msg) {
    cat("\n", strrep("=", 70), "\n", msg, "\n", strrep("=", 70), "\n\n", sep = "")
}
print_step <- function(msg) cat("  >>", msg, "\n")
# =============================================================================
# 00_config.R - Configuration and Setup
# IEEE Escalations in Customer Support
# =============================================================================
# Task: Predict time-to-next-escalation for customer support cases
# Data: Case metadata, status history, milestones, comments (anonymized text)
# Approach: Feature engineering from event sequences + NLP on decoded text +
#           target encoding of categorical variables + ensemble (CatBoost/XGBoost)
# =============================================================================

GLOBAL_SEED <- 1618
NUM_THREADS <- 10
TIME_INTERVAL <- 24 * 3600 # 1 day in seconds
MIN_TIME_GAP <- 3600 # 1 hour
MAX_TIME_GAP <- 14 * 24 * 3600 # 14 days

DATA_DIR <- "."
OUTPUT_DIR <- file.path(DATA_DIR, "output")

if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)

# File paths
FILE_SUMMARY <- "IBI_case_metadata_anonymized.csv"
FILE_HISTORY <- "IBI_case_status_history_v2.csv"
FILE_MILESTONES <- "IBI_case_milestones_anonymized.csv"
FILE_COMMENTS <- "IBI_case_comments_anonymized.csv"
FILE_TEST <- "IBI_test_cases_no_target.csv"
FILE_DICTIONARY <- "challenge_dictionary_info.csv"
FILE_LEMMA <- "id_to_lemma_public_translations.csv"

# Libraries
suppressPackageStartupMessages({
    library(data.table)
    library(dplyr)
    library(caret)
    library(stringi)
    library(tm)
    library(tidytext)
    library(text2vec)
    library(xgboost)
    library(catboost)
    library(Matrix)
    library(Information)
    library(fifer)
})
set.seed(GLOBAL_SEED)

print_section <- function(msg) {
    cat("\n", strrep("=", 70), "\n", msg, "\n", strrep("=", 70), "\n\n", sep = "")
}
print_step <- function(msg) cat("  >>", msg, "\n")
NUM_AMBULANCES <- 6

DATA_DIR <- "."
OUTPUT_DIR <- file.path(DATA_DIR, "output")

if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)

# Special days and holidays (Kenya 2018-2019)
SP_DAYS_COMMON <- c(
    "11", "51", "61", "621", "923", "1010", "1020",
    "1212", "1222", "1224", "1225", "1226", "1231"
)
SP_DAYS_YEAR <- c(
    "2018320", "2018330", "201841", "201842", "2018513",
    "2018616", "2018822", "2018117",
    "2019321", "2019419", "2019421", "2019422", "2019512",
    "201965", "2019812", "20191027"
)

# Libraries
suppressPackageStartupMessages({
    library(data.table)
    library(dplyr)
    library(caret)
    library(Matrix)
    library(rgdal)
    library(raster)
    library(geosphere)
    library(lubridate)
    library(lightgbm)
    library(xgboost)
    library(sqldf)
    library(cluster)
    library(fpc)
    library(Rtsne)
})
options(sqldf.driver = "RSQLite")

print_section <- function(msg) {
    cat("\n", strrep("=", 70), "\n", msg, "\n", strrep("=", 70), "\n\n", sep = "")
}
print_step <- function(msg) cat("  >>", msg, "\n")
# =============================================================================
# 00_config.R - Configuration and Setup
# Akeed Restaurant Recommendation Challenge
# =============================================================================
# Task: Predict which of 100 vendors each test customer will order from
# Approach: Geographic zone clustering + customer segmentation + distance-based
#           recommendation (segment × zone matching)
# =============================================================================

GLOBAL_SEED <- 1618
REFERENCE_DATE <- as.Date("2020-06-30")

DATA_DIR <- "."
OUTPUT_DIR <- file.path(DATA_DIR, "output")

if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)

# Libraries
suppressPackageStartupMessages({
    library(data.table)
    library(dplyr)
    library(tidyr)
    library(caret)
    library(stringr)
    library(lubridate)
    library(sqldf)
})
options(sqldf.driver = "RSQLite")

print_section <- function(msg) {
    cat("\n", strrep("=", 70), "\n", msg, "\n", strrep("=", 70), "\n\n", sep = "")
}

print_step <- function(msg) cat("  >>", msg, "\n")
# =============================================================================
# 00_config.R - Configuration and Setup
# Mental Health Text Classification Pipeline
# =============================================================================
# Task: Classify student statements into 4 categories:
#       Depression, Alcohol, Suicide, Drugs
# Format: Multi-label probabilities per class
# Metric: Log loss
# =============================================================================

GLOBAL_SEED <- 216
NUM_THREADS <- 4

DATA_DIR <- "."
OUTPUT_DIR <- file.path(DATA_DIR, "output")

TRAIN_FILE <- file.path(DATA_DIR, "Train.csv")
TEST_FILE <- file.path(DATA_DIR, "Test.csv")

LABELS <- c("Depression", "Alcohol", "Suicide", "Drugs")

if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)

# Libraries
suppressPackageStartupMessages({
    library(data.table)
    library(dplyr)
    library(tidyr)
    library(text2vec)
    library(tm)
    library(SnowballC)
    library(udpipe)
    library(hunspell)
    library(xgboost)
    library(glmnet)
    library(caret)
    library(Matrix)
    library(Metrics)
})

print_section <- function(msg) {
    cat("\n", strrep("=", 70), "\n", msg, "\n", strrep("=", 70), "\n\n", sep = "")
}

print_step <- function(msg) cat("  >>", msg, "\n")
# =============================================================================
# 00_config.R - Configuration and Setup
# Malawi Flood Extent Prediction Pipeline
# =============================================================================
# Train: 2015 flood extent (southern Malawi)
# Test:  2019 flood extent (Cyclone Idai)
# Target: fraction of 1km² square flooded [0, 1]
# Metric: RMSE
# =============================================================================

# -----------------------------------------------------------------------------
# 1. GLOBAL SETTINGS
# -----------------------------------------------------------------------------

GLOBAL_SEED <- 7
CV_FOLDS <- 5
NUM_THREADS <- 4

DATA_DIR <- "."
OUTPUT_DIR <- file.path(DATA_DIR, "output")

TRAIN_FILE <- file.path(DATA_DIR, "Train.csv")
SUBMISSION_FILE <- file.path(DATA_DIR, "SampleSubmission.csv")

if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)

# -----------------------------------------------------------------------------
# 2. LIBRARIES
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
    # Data manipulation
    library(data.table)
    library(dplyr)
    library(tidyr)

    # Spatial / raster
    library(raster)
    library(sf)
    library(sp)
    library(rgdal)
    library(rgeos)
    library(geosphere)
    library(nngeo)
    library(spatialEco)

    # Machine learning
    library(caret)
    library(lightgbm)
    library(xgboost)
    library(catboost)
    library(Matrix)
    library(h2o)
    library(gbm)

    # Feature selection
    library(FSinR)

    # EDA / visualization
    library(ggplot2)
    library(corrplot)
    library(correlationfunnel)
    library(DataExplorer)
    library(funModeling)
    library(summarytools)
    library(inspectdf)
    library(matrixStats)

    # Utilities
    library(Metrics)
})

# -----------------------------------------------------------------------------
# 3. UTILITY FUNCTIONS
# -----------------------------------------------------------------------------

print_section <- function(msg) {
    cat("\n", strrep("=", 70), "\n", msg, "\n", strrep("=", 70), "\n\n", sep = "")
}

print_step <- function(msg) {
    cat("  >>", msg, "\n")
}

evaluate_model <- function(actual, predicted, model_name = "Model") {
    predicted <- pmax(0, pmin(1, predicted))
    rmse_val <- Metrics::rmse(actual, predicted)
    mae_val <- Metrics::mae(actual, predicted)
    cat(sprintf("  [%s] RMSE: %.5f | MAE: %.5f\n", model_name, rmse_val, mae_val))
    invisible(list(rmse = rmse_val, mae = mae_val))
}

clip_predictions <- function(x) pmax(0, pmin(1, x))
# ============================================================================
# 00_CONFIG.R - Configuration and Library Management
# ============================================================================
# Description: Central configuration file for the Energy Consumption Prediction
#              project. Contains all library imports and global settings.
# Updated: Comprehensive version with all dependencies from original scripts
# ============================================================================

# ============================================================================
# 1. REQUIRED LIBRARIES
# ============================================================================

cat("\n")
cat("═══════════════════════════════════════════════════════════════\n")
cat("  Loading Required Packages for Energy Prediction Pipeline     \n")
cat("═══════════════════════════════════════════════════════════════\n\n")

suppressPackageStartupMessages({
    # -- Data Manipulation --
    library(data.table)
    library(dplyr)
    library(tidyr)
    library(purrr)
    library(tibble)

    # -- Visualization & EDA --
    library(ggplot2)
    library(gwalkr) # Alternative: GWalkR
    library(explore)
    library(dataMaid) # For data reports
    library(SmartEDA) # For EDA reports
    library(dlookr) # For diagnostics

    # -- Factor Manipulation --
    library(forcats)

    # -- Statistical Analysis --
    library(MASS) # For boxcox, statistical tests
    library(ltm) # For biserial correlation
    library(broom) # Tidy model outputs
    library(effectsize) # Effect size calculations

    # -- Machine Learning - Core Frameworks --
    library(caret)
    library(tidymodels)
    library(recipes)
    library(stacks)
    library(workflows)
    library(tune)
    library(yardstick)
    library(rsample)
    library(parsnip)

    # -- Machine Learning - Tree-Based Algorithms --
    library(xgboost)
    library(lightgbm)
    library(catboost)
    library(ranger)
    library(rpart)
    library(rpart.plot)

    # -- Machine Learning - Advanced Frameworks --
    library(h2o)
    library(sl3) # Super Learner
    library(treesnip) # Additional tidymodels engines

    # -- Machine Learning - Linear/Statistical Models --
    library(glmnet) # Elastic net, lasso, ridge
    library(mgcv) # GAM models
    library(kernlab) # SVM
    library(kknn) # KNN

    # -- Clustering --
    library(cluster)
    library(FNN)

    # -- Feature Engineering & Encoding --
    library(vtreat)

    # -- Parallel Processing --
    library(parallel)
    library(doParallel)

    # -- Evaluation Metrics --
    library(Metrics)
})

cat("✓ All packages loaded successfully.\n\n")

# ============================================================================
# 2. GLOBAL SETTINGS
# ============================================================================

# -- Random Seed for Reproducibility --
GLOBAL_SEED <- 1618
set.seed(GLOBAL_SEED)

# -- File Paths --
DATA_PATH <- getwd() # Adjust as needed

# Input files (update these paths as needed)
FILE_SUBMISSION <- "imgs_202307101549519358.csv"
FILE_ENERGY <- "imgs_2023071012133740345.csv"
FILE_CELL <- "imgs_2023071012130978799.csv"
FILE_BASE <- "imgs_2023071012123392536.csv"

# -- Model Parameters --
CV_FOLDS <- 5
TRAIN_SAMPLE_SIZE <- 75000
FREEZE_SAMPLE_SIZE <- 10000

# -- Energy Split Proportions (for 2-cell bases) --
# Best performing proportions from analysis
AVG_189 <- 0.4328334
AVG_155 <- 0.5671666
AVG_426 <- 0.3533747
AVG_365 <- 0.6466253

# -- Global Options --
options(
    scipen = 999, # Disable scientific notation
    digits = 6,
    tidymodels.dark = TRUE, # Better console output for tidymodels
    warn = -1 # Suppress warnings (set to 0 to enable)
)

# ============================================================================
# 3. CREATE OUTPUT DIRECTORIES
# ============================================================================

if (!dir.exists("models")) dir.create("models")
if (!dir.exists("predictions")) dir.create("predictions")
if (!dir.exists("submissions")) dir.create("submissions")
if (!dir.exists("reports")) dir.create("reports")
if (!dir.exists("data")) dir.create("data")

# ============================================================================
# 4. HELPER FUNCTIONS
# ============================================================================

#' Set up parallel processing cluster
#' @param n_cores Number of cores to use (default: 2)
#' @return Cluster object
setup_parallel <- function(n_cores = 2) {
    cl <- makePSOCKcluster(n_cores)
    doParallel::registerDoParallel(cl)
    cat(sprintf("✓ Parallel cluster started with %d cores.\n", n_cores))
    return(cl)
}

#' Stop parallel processing cluster
#' @param cl Cluster object
stop_parallel <- function(cl) {
    if (!is.null(cl)) {
        stopCluster(cl)
        cat("✓ Parallel cluster stopped.\n")
    }
}

#' Custom MAPE summary function for caret
#' @param data Data frame with obs and pred columns
#' @param lev Levels (not used for regression)
#' @param model Model name (not used)
#' @return Named vector with MAPE value
mapeSummary <- function(data, lev = NULL, model = NULL) {
    mape <- mean(abs((data$obs - data$pred) / data$obs))
    c(MAPE = mape)
}

#' Print model evaluation metrics
#' @param actual Actual values
#' @param predicted Predicted values
#' @param model_name Name of the model for display
#' @return List with MAE and MAPE values
evaluate_model <- function(actual, predicted, model_name = "Model") {
    mae_val <- Metrics::mae(actual, predicted)
    mape_val <- Metrics::mape(actual, predicted)
    rmse_val <- sqrt(mean((actual - predicted)^2))

    cat(sprintf("\n%s Performance:\n", model_name))
    cat(sprintf("  MAE:  %.6f\n", mae_val))
    cat(sprintf("  MAPE: %.6f (%.2f%%)\n", mape_val, mape_val * 100))
    cat(sprintf("  RMSE: %.6f\n", rmse_val))

    return(list(MAE = mae_val, MAPE = mape_val, RMSE = rmse_val))
}

#' Save predictions to CSV file
#' @param predictions Data frame with predictions
#' @param filename Output filename
#' @param subfolder Subfolder within predictions/ (default: "")
save_predictions <- function(predictions, filename, subfolder = "") {
    output_dir <- file.path("predictions", subfolder)
    if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

    filepath <- file.path(output_dir, filename)
    write.csv(predictions, filepath, quote = FALSE, row.names = FALSE)
    cat(sprintf("✓ Predictions saved to: %s\n", filepath))
}

#' Create train/validation split with stratification
#' @param data Data frame
#' @param target_col Target column name
#' @param prop Proportion for training (default: 0.8)
#' @return List with train and valid data frames
create_split <- function(data, target_col = "Energy", prop = 0.8) {
    set.seed(GLOBAL_SEED)

    # Create stratification variable based on target quantiles
    data$strata_temp <- cut(data[[target_col]],
        breaks = quantile(data[[target_col]],
            probs = seq(0, 1, 0.2),
            na.rm = TRUE
        ),
        include.lowest = TRUE,
        labels = FALSE
    )

    split_idx <- createDataPartition(data$strata_temp, p = prop, list = FALSE)

    train_data <- data[split_idx, ]
    valid_data <- data[-split_idx, ]

    # Remove temporary stratification column
    train_data$strata_temp <- NULL
    valid_data$strata_temp <- NULL

    return(list(train = train_data, valid = valid_data))
}

# ============================================================================
# 5. PRINT CONFIGURATION SUMMARY
# ============================================================================

cat("\n")
cat("═══════════════════════════════════════════════════════════════\n")
cat("          Energy Consumption Prediction - Configuration        \n")
cat("═══════════════════════════════════════════════════════════════\n")
cat(sprintf("  Random Seed:           %d\n", GLOBAL_SEED))
cat(sprintf("  CV Folds:              %d\n", CV_FOLDS))
cat(sprintf("  Training Sample Size:  %d\n", TRAIN_SAMPLE_SIZE))
cat(sprintf("  Freeze Sample Size:    %d\n", FREEZE_SAMPLE_SIZE))
cat(sprintf("  Working Directory:     %s\n", getwd()))
cat("═══════════════════════════════════════════════════════════════\n\n")
