# =============================================================================
# MAIN.R - Full Pipeline Orchestration
# STEG Fraud Detection in Electricity and Gas Consumption
# =============================================================================

cat("\n========== STEG Fraud Detection Pipeline ==========\n")
cat("Pipeline started:", format(Sys.time()), "\n\n")

source("01_data_loading.R")
source("02_feature_engineering.R")
source("03_encoding_selection.R")
source("04_model_lightgbm.R")
source("05_model_xgboost.R")
source("06_model_catboost_rf.R")
source("07_ensemble.R")

cat("\n========== Pipeline complete:", format(Sys.time()), "==========\n")
# =============================================================================
# MAIN.R - Full Pipeline Orchestration
# Sea Turtle Rescue Forecast Challenge
# =============================================================================

cat("\n========== Sea Turtle Rescue Forecast ==========\n")
cat("Pipeline started:", format(Sys.time()), "\n\n")

source("01_data_loading.R")
source("02_feature_engineering.R")
source("03_time_series.R")
source("04_model_supervised.R")
source("05_ensemble.R")

cat("\n========== Pipeline complete:", format(Sys.time()), "==========\n")
# =============================================================================
# MAIN.R - Full Pipeline Orchestration
# IEEE Escalations in Customer Support
# =============================================================================
# Run this script to execute the full pipeline end-to-end.
# Each step saves intermediate outputs to ./output/ so individual scripts
# can also be run independently after prior steps have completed.
# =============================================================================

cat("\n========== IEEE Escalations in Customer Support ==========\n")
cat("Pipeline started:", format(Sys.time()), "\n\n")

source("01_data_loading.R")
source("02_target_construction.R")
source("03_feature_engineering.R")
source("04_nlp_features.R")
source("05_feature_selection.R")
source("06_model_catboost.R")
source("07_model_xgboost.R")
source("08_ensemble.R")

cat("\n========== Pipeline complete:", format(Sys.time()), "==========\n")
# =============================================================================
# MAIN.R - Nairobi Ambulance Deployment Pipeline
# =============================================================================
# Competition: Zindi - Ambulance Deployment Optimization (2019-2020)
#
# Task:    Position 6 ambulances across Nairobi every 3 hours to minimize
#          response distance to traffic crashes during the test period.
# Approach: Predict crash probability per road segment per time window,
#           then cluster predicted hotspots into 6 ambulance positions.
# =============================================================================
# PIPELINE:
#   00_config.R              -> Configuration, libraries, constants
#   01_data_loading.R        -> Load crashes, segments, weather
#   02_segment_features.R    -> Crash-to-segment matching, elasticity, t-SNE
#   03_uber_speeds.R         -> Uber Movement speed integration
#   04_clustering.R          -> Segment zones + crash clusters (6 = ambulances)
#   05_abt_construction.R    -> Full time × segment grid with all features
#   06_feature_selection.R   -> IV, RF, Elastic Net, random subset search
#   07_model_lightgbm.R      -> Primary LightGBM model
#   08_model_ensemble.R      -> XGBoost + CatBoost ensemble members
#   09_ambulance_deployment.R -> Predictions → clustered ambulance positions
# =============================================================================

cat("\n")
cat(strrep("=", 70), "\n")
cat("  NAIROBI AMBULANCE DEPLOYMENT PIPELINE\n")
cat(strrep("=", 70), "\n\n")

source("01_data_loading.R")
source("02_segment_features.R")
source("03_uber_speeds.R")
source("04_clustering.R")
source("05_abt_construction.R")
source("06_feature_selection.R")
source("07_model_lightgbm.R")
source("08_model_ensemble.R")
source("09_ambulance_deployment.R")

cat("\n", strrep("=", 70), "\n")
cat("  PIPELINE COMPLETE - submission.csv written to output/\n")
cat(strrep("=", 70), "\n\n")
# =============================================================================
# MAIN.R - Akeed Restaurant Recommendation Pipeline
# =============================================================================
# Competition: Zindi - Akeed Restaurant Recommendation Challenge (2020)
#
# Task:    Predict which of 100 vendors each customer will order from,
#          given customer location, vendor info, and order history.
# Approach: Geographic zone clustering + customer segmentation.
#           Within each (segment, zone) pair, recommend vendors based on
#           historical ordering frequency weighted by proximity.
# =============================================================================
# PIPELINE:
#   00_config.R            -> Configuration and libraries
#   01_data_loading.R      -> Load CSVs, basic corrections
#   02_order_features.R    -> Aggregate order history per triplet
#   03_vendor_processing.R -> Tag grouping, coordinate cleanup
#   04_customer_features.R -> Customer-location profiles
#   05_clustering.R        -> Vendor zones (K-means) + customer segments
#   06_recommendation.R    -> Segment × zone matching + distance scoring
# =============================================================================

cat("\n")
cat(strrep("=", 70), "\n")
cat("  AKEED RESTAURANT RECOMMENDATION PIPELINE\n")
cat(strrep("=", 70), "\n\n")

source("01_data_loading.R")
source("02_order_features.R")
source("03_vendor_processing.R")
source("04_customer_features.R")
source("05_clustering.R")
source("06_recommendation.R")

cat("\n", strrep("=", 70), "\n")
cat("  PIPELINE COMPLETE - submission.csv written to output/\n")
cat(strrep("=", 70), "\n\n")
# =============================================================================
# MAIN.R - Kenya Mental Health Text Classification Pipeline
# =============================================================================
# Competition: Zindi - AI4D iCompass Social Media Sentiment Analysis
#              for Mental Health (Late 2021)
#
# Task:    Classify student free-text statements into 4 mental health
#          categories: Depression, Alcohol, Suicide, Drugs
# Input:   Short text responses to "What is on your mind?"
# Output:  Probabilities per class (multi-label format)
# =============================================================================
# PIPELINE:
#   00_config.R              -> Configuration and libraries
#   01_data_loading.R        -> Load CSV data, create binary targets
#   02_text_preprocessing.R  -> UDPipe tokenization + hunspell spelling correction
#   03_feature_engineering.R -> TF-IDF (sublinear) + LSA (4 topics)
#   04_model_glmnet.R        -> Ridge logistic regression (per class)
#   05_model_xgboost.R       -> XGBoost with CV-tuned nrounds (per class)
#   06_model_keras.R         -> Keras text embedding + dense network
#   07_model_ruimtehol.R     -> StarSpace joint embeddings
#   08_ensemble_submission.R -> Average predictions, write submission
# =============================================================================

cat("\n")
cat(strrep("=", 70), "\n")
cat("  MENTAL HEALTH TEXT CLASSIFICATION PIPELINE\n")
cat(strrep("=", 70), "\n\n")

source("01_data_loading.R")
source("02_text_preprocessing.R")
source("03_feature_engineering.R")
source("04_model_glmnet.R")
source("05_model_xgboost.R")
# source("06_model_keras.R")       # Requires keras/tensorflow
# source("07_model_ruimtehol.R")   # Requires ruimtehol
source("08_ensemble_submission.R")

cat("\n", strrep("=", 70), "\n")
cat("  PIPELINE COMPLETE - submission.csv written to output/\n")
cat(strrep("=", 70), "\n\n")
# =============================================================================
# MAIN.R - Pipeline Orchestration
# Malawi Flood Extent Prediction Pipeline
# =============================================================================

source("00_config.R")

run_step <- function(script, description) {
    cat(sprintf("\n%s\nRunning: %s\n%s\n", strrep("-", 60), description, strrep("-", 60)))
    t0 <- Sys.time()
    tryCatch(
        {
            source(script, local = FALSE)
            cat(sprintf("  Completed in %.1f min\n", as.numeric(difftime(Sys.time(), t0, units = "mins"))))
        },
        error = function(e) {
            cat(sprintf("  ERROR in %s: %s\n", script, e$message))
        }
    )
}

run_step("01_data_loading.R", "Data Loading")
run_step("02_spatial_features.R", "Spatial Feature Extraction")
run_step("03_external_data.R", "External Data Integration")
run_step("04_train_test_split.R", "Train / Test Split")
run_step("05_model_lightgbm.R", "LightGBM Model")
run_step("06_model_xgboost.R", "XGBoost Model")
run_step("07_model_catboost.R", "CatBoost Model")
run_step("08_model_h2o.R", "H2O AutoML")
run_step("09_generate_submission.R", "Ensemble and Submission")

cat("\nPipeline complete.\n")
# ============================================================================
# MAIN.R - Master Script for Energy Consumption Prediction Pipeline
# ============================================================================
# Description: This is the main orchestration script that runs the entire
#              pipeline from data loading to submission generation.
#
# Usage:
#   1. Set your working directory to the organized folder
#   2. Source this file: source("MAIN.R")
#   3. Call run_pipeline() to run everything, or call individual steps
#
# Author: Energy Prediction Project
# ============================================================================

# Set working directory (modify as needed)
# setwd("c:/Users/mtrigui2/Desktop/sss/organized")

# ============================================================================
# PIPELINE STEPS
# ============================================================================

#' Run the complete pipeline
#' @param steps Vector of step numbers to run (1-18), or "all" for everything
run_pipeline <- function(steps = "all") {
    start_time <- Sys.time()

    cat("\n")
    cat("╔══════════════════════════════════════════════════════════════╗\n")
    cat("║     ENERGY CONSUMPTION PREDICTION PIPELINE                   ║\n")
    cat("╚══════════════════════════════════════════════════════════════╝\n\n")

    all_steps <- 1:18
    if (identical(steps, "all")) {
        steps <- all_steps
    }

    # Step 1: Data Loading
    if (1 %in% steps) {
        cat("\n[STEP 1/18] Loading and cleaning data...\n")
        cat("─────────────────────────────────────────\n")
        source("01_data_loading.R")
    }

    # Step 2: Feature Engineering Functions (loaded by other scripts)
    if (2 %in% steps) {
        cat("\n[STEP 2/18] Loading feature engineering functions...\n")
        cat("─────────────────────────────────────────────────────\n")
        source("02_feature_engineering.R")
        cat("Feature engineering functions loaded.\n")
    }

    # Step 3: Build Base Dataset
    if (3 %in% steps) {
        cat("\n[STEP 3/18] Building base-level dataset...\n")
        cat("───────────────────────────────────────────\n")
        source("03_build_base_dataset.R")
    }

    # Step 4: Build Cell Dataset
    if (4 %in% steps) {
        cat("\n[STEP 4/18] Building cell-level dataset...\n")
        cat("───────────────────────────────────────────\n")
        source("04_build_cell_dataset.R")
    }

    # Step 5: Build LightGBM Dataset
    if (5 %in% steps) {
        cat("\n[STEP 5/18] Building LightGBM-optimized dataset...\n")
        cat("───────────────────────────────────────────────────\n")
        source("05_build_lightgbm_dataset.R")
    }

    # Step 6: Train/Test Split
    if (6 %in% steps) {
        cat("\n[STEP 6/18] Creating train/test/freeze splits...\n")
        cat("─────────────────────────────────────────────────\n")
        source("06_train_test_split.R")
    }

    # Step 7: LightGBM Model
    if (7 %in% steps) {
        cat("\n[STEP 7/18] Training LightGBM models...\n")
        cat("────────────────────────────────────────\n")
        source("07_model_lightgbm.R")
    }

    # Step 8: XGBoost Model
    if (8 %in% steps) {
        cat("\n[STEP 8/18] Training XGBoost models...\n")
        cat("───────────────────────────────────────\n")
        source("08_model_xgboost.R")
    }

    # Step 9: CatBoost Model
    if (9 %in% steps) {
        cat("\n[STEP 9/18] Training CatBoost models...\n")
        cat("────────────────────────────────────────\n")
        source("09_model_catboost.R")
    }

    # Step 10: Caret Models
    if (10 %in% steps) {
        cat("\n[STEP 10/18] Training Caret models (RF, GLMNet, GAM)...\n")
        cat("────────────────────────────────────────────────────────\n")
        source("10_model_caret.R")
    }

    # Step 11: H2O Models
    if (11 %in% steps) {
        cat("\n[STEP 11/18] Training H2O AutoML models...\n")
        cat("───────────────────────────────────────────\n")
        source("11_model_h2o.R")
    }

    # Step 12: Stacking
    if (12 %in% steps) {
        cat("\n[STEP 12/18] Training stacked ensemble models...\n")
        cat("─────────────────────────────────────────────────\n")
        source("12_model_stacking.R")
    }

    # Step 13: SL3 Super Learner
    if (13 %in% steps) {
        cat("\n[STEP 13/18] Training Super Learner (SL3)...\n")
        cat("─────────────────────────────────────────────\n")
        source("13_model_sl3.R")
    }

    # Step 14: Clustering Analysis
    if (14 %in% steps) {
        cat("\n[STEP 14/18] Performing clustering analysis...\n")
        cat("───────────────────────────────────────────────\n")
        source("14_clustering_analysis.R")
    }

    # Step 15: Cell-Level Models
    if (15 %in% steps) {
        cat("\n[STEP 15/18] Training cell-level models...\n")
        cat("───────────────────────────────────────────\n")
        source("15_model_cell_level.R")
    }

    # Step 16: Generate Submissions
    if (16 %in% steps) {
        cat("\n[STEP 16/18] Generating submission files...\n")
        cat("─────────────────────────────────────────────\n")
        source("16_generate_submission.R")
    }

    # Step 17: EDA and Visualization
    if (17 %in% steps) {
        cat("\n[STEP 17/18] Running EDA and visualization...\n")
        cat("───────────────────────────────────────────────\n")
        source("17_eda_and_visualization.R")
    }

    # Step 18: Evaluate Freeze Set
    if (18 %in% steps) {
        cat("\n[STEP 18/18] Evaluating models on freeze set...\n")
        cat("─────────────────────────────────────────────────\n")
        source("18_evaluate_freeze.R")
    }

    end_time <- Sys.time()
    duration <- difftime(end_time, start_time, units = "mins")

    cat("\n")
    cat("╔══════════════════════════════════════════════════════════════╗\n")
    cat("║                    PIPELINE COMPLETE                         ║\n")
    cat(sprintf("║            Total time: %.2f minutes                        ║\n", duration))
    cat("╚══════════════════════════════════════════════════════════════╝\n\n")
}

#' Run only data preparation steps (1-6)
run_data_prep <- function() {
    run_pipeline(steps = 1:6)
}

#' Run only model training steps (7-15)
run_training <- function() {
    run_pipeline(steps = 7:15)
}

#' Run only evaluation and submission steps (16-18)
run_evaluation <- function() {
    run_pipeline(steps = 16:18)
}

# ============================================================================
# QUICK START GUIDE
# ============================================================================

cat("\n")
cat("╔══════════════════════════════════════════════════════════════╗\n")
cat("║       ENERGY CONSUMPTION PREDICTION - QUICK START            ║\n")
cat("╠══════════════════════════════════════════════════════════════╣\n")
cat("║                                                              ║\n")
cat("║  Available commands:                                         ║\n")
cat("║                                                              ║\n")
cat("║  run_pipeline()       - Run complete pipeline                ║\n")
cat("║  run_pipeline(c(1:6)) - Run specific steps                   ║\n")
cat("║  run_data_prep()      - Run data preparation only            ║\n")
cat("║  run_training()       - Run model training only              ║\n")
cat("║  run_evaluation()     - Run evaluation only                  ║\n")
cat("║                                                              ║\n")
cat("║  Pipeline Steps:                                             ║\n")
cat("║   1. Data Loading          10. Caret Models                  ║\n")
cat("║   2. Feature Engineering   11. H2O AutoML                    ║\n")
cat("║   3. Build Base Dataset    12. Stacking                      ║\n")
cat("║   4. Build Cell Dataset    13. Super Learner                 ║\n")
cat("║   5. Build LightGBM Data   14. Clustering                    ║\n")
cat("║   6. Train/Test Split      15. Cell-Level Models             ║\n")
cat("║   7. LightGBM Models       16. Generate Submissions          ║\n")
cat("║   8. XGBoost Models        17. EDA & Visualization           ║\n")
cat("║   9. CatBoost Models       18. Evaluate Freeze Set           ║\n")
cat("║                                                              ║\n")
cat("╚══════════════════════════════════════════════════════════════╝\n\n")
