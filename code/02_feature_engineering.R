# =============================================================================
# 02_feature_engineering.R - Invoice Aggregation to Client-Level Features
# STEG Fraud Detection
# =============================================================================
# Features built from invoice history per client:
# 1. Consumption: total, per-level, monthly average, proportions
# 2. Invoice patterns: count, frequency diffs, reading changes
# 3. Counter metadata: distinct counters, statuses, types, gas indicator
# 4. Temporal diffs: index changes, invoice gaps, reading_remarque changes
# =============================================================================

source("00_config.R")
print_section("Step 2: Feature Engineering")

tot <- readRDS(file.path(OUTPUT_DIR, "tot.rds"))
cl_tot <- readRDS(file.path(OUTPUT_DIR, "cl_tot.rds"))

# -----------------------------------------------------------------------------
# 1. LAG-BASED FEATURES (per client, ordered by invoice date)
# -----------------------------------------------------------------------------

print_step("Computing lag-based temporal features...")

df_fact <- as.data.table(sqldf("
    SELECT DISTINCT client_id, invoice_date, reading_remarque, consom, old_index, new_index
    FROM tot ORDER BY client_id, invoice_date"))

df_fact <- df_fact %>%
    group_by(client_id) %>%
    mutate(
        date_2  = dplyr::lag(invoice_date, n = 1, default = NA),
        read_2  = dplyr::lag(reading_remarque, n = 1, default = NA)
    ) %>%
    ungroup()

df_fact$diff_fact <- as.numeric(difftime(df_fact$invoice_date, df_fact$date_2, units = "days"))
df_fact$diff_read <- abs(as.integer(df_fact$reading_remarque) - as.integer(df_fact$read_2))
df_fact$diff_index <- df_fact$new_index - df_fact$old_index

df_client <- as.data.table(df_fact)[, .(
    mean_consom     = mean(consom, na.rm = TRUE),
    sd_consom       = sd(consom, na.rm = TRUE),
    mean_diff_fact  = mean(diff_fact, na.rm = TRUE),
    sd_diff_fact    = sd(diff_fact, na.rm = TRUE),
    mean_diff_read  = mean(diff_read, na.rm = TRUE),
    sd_diff_read    = sd(diff_read, na.rm = TRUE),
    mean_diff_index = mean(diff_index, na.rm = TRUE),
    sd_diff_index   = sd(diff_index, na.rm = TRUE)
), by = client_id]

# -----------------------------------------------------------------------------
# 2. AGGREGATE INVOICE FEATURES (SQL-based)
# -----------------------------------------------------------------------------

print_step("Aggregating invoice-level features per client...")

part_1 <- as.data.table(sqldf("SELECT
    client_id,
    MAX(Anc) AS Anc,
    COUNT(DISTINCT invoice_date) AS nb_fact,
    COUNT(DISTINCT tarif_type) AS nb_tarif_type,
    COUNT(DISTINCT counter_number) AS nb_counter,
    COUNT(DISTINCT counter_statue) AS nb_counter_sta,
    COUNT(DISTINCT counter_code) AS nb_counter_code,
    COUNT(DISTINCT reading_remarque) AS nb_read,
    AVG(CAST(reading_remarque AS REAL)) AS avg_read,
    COUNT(DISTINCT counter_coefficient) AS nb_coef,
    COUNT(DISTINCT counter_type) AS nb_counter_type,
    SUM(consommation_level_1 + consommation_level_2 + consommation_level_3 + consommation_level_4) AS Cons_tot,
    SUM(consommation_level_1) AS Cons_1,
    SUM(consommation_level_2) AS Cons_2,
    SUM(consommation_level_3) AS Cons_3,
    SUM(consommation_level_4) AS Cons_4,
    AVG(months_number) AS avg_months,
    SUM(DISTINCT months_number) AS nb_months,
    ROUND(SUM(consommation_level_1 + consommation_level_2 + consommation_level_3 + consommation_level_4) /
          NULLIF(SUM(months_number), 0), 2) AS Cons_Mens
FROM tot GROUP BY client_id"))

# Low-consumption invoices count
df_inf_10 <- as.data.table(sqldf("SELECT client_id, COUNT(*) AS nb_inf_10
    FROM tot WHERE (consommation_level_1+consommation_level_2+consommation_level_3+consommation_level_4) <= 10
    GROUP BY client_id"))

# Level 4 consumption count (high usage)
df_lev_4 <- as.data.table(sqldf("SELECT client_id, COUNT(*) AS nb_fact_lev4
    FROM tot WHERE consommation_level_4 > 0 GROUP BY client_id"))

# Summer level 4 (seasonal high usage)
df_prct <- as.data.table(sqldf("SELECT client_id, COUNT(*) AS nb_fact_lev4_summer
    FROM tot WHERE consommation_level_4 > 0 AND Month IN ('06','07','08')
    GROUP BY client_id"))

# Gas indicator
df_gaz <- as.data.table(sqldf("SELECT DISTINCT client_id, 1 AS gaz
    FROM tot WHERE counter_type = 'GAZ'"))

# Real counter count (excluding gas counters)
df_gaz_count <- as.data.table(sqldf("SELECT client_id, COUNT(DISTINCT counter_number) AS nb_counter_gaz
    FROM tot WHERE counter_type = 'GAZ' GROUP BY client_id"))

# -----------------------------------------------------------------------------
# 3. JOIN ALL FEATURES
# -----------------------------------------------------------------------------

print_step("Joining all feature sets...")

client <- part_1 %>%
    left_join(df_inf_10, by = "client_id") %>%
    left_join(df_lev_4, by = "client_id") %>%
    left_join(df_prct, by = "client_id") %>%
    left_join(df_gaz, by = "client_id") %>%
    left_join(df_gaz_count, by = "client_id") %>%
    left_join(df_client, by = "client_id")

# Derived features
client[is.na(client)] <- 0
client$prct_1 <- client$Cons_1 / pmax(client$Cons_tot, 1)
client$prct_2 <- client$Cons_2 / pmax(client$Cons_tot, 1)
client$prct_3 <- client$Cons_3 / pmax(client$Cons_tot, 1)
client$prct_4 <- client$Cons_4 / pmax(client$Cons_tot, 1)
client$Cons_Moy <- client$Cons_Mens / pmax(client$nb_months, 1)
client$nb_real_counter <- client$nb_counter - client$nb_counter_gaz

# Join client metadata (district, region, category)
client <- client %>%
    left_join(cl_tot[, c("client_id", "disrict", "client_catg", "region", "target", "fraud")],
        by = "client_id"
    )
client[is.na(client)] <- 0

cat(sprintf("  Client feature matrix: %d rows × %d columns\n", nrow(client), ncol(client)))

# -----------------------------------------------------------------------------
# 4. SAVE
# -----------------------------------------------------------------------------

saveRDS(client, file.path(OUTPUT_DIR, "client_features.rds"))

print_step("Feature engineering complete.")
# =============================================================================
# 02_feature_engineering.R - Build Features from Rescue History
# Sea Turtle Rescue Forecast Challenge
# =============================================================================
# Features at site × week level:
# 1. Site-level: total rescues, monthly average, weekly total per site
# 2. Week-level: total across all sites, monthly average per week
# 3. Frequency: gap between rescue events (2018 only)
# 4. Year-over-year change: elasticity of growth/decline per site×week
# 5. Clustering: k-means on site profiles
# =============================================================================

source("00_config.R")
print_section("Step 2: Feature Engineering")

aux_df <- readRDS(file.path(OUTPUT_DIR, "aux_df.rds"))
capt <- readRDS(file.path(OUTPUT_DIR, "capt.rds"))
org <- readRDS(file.path(OUTPUT_DIR, "grid_train.rds"))
test <- readRDS(file.path(OUTPUT_DIR, "grid_test.rds"))

# -----------------------------------------------------------------------------
# 1. SITE-LEVEL FEATURES
# -----------------------------------------------------------------------------

print_step("Building site-level aggregates...")

aux_capt <- as.data.table(sqldf("
    SELECT CaptureSite, year, month, week,
           COUNT(DISTINCT Rescue_ID) AS nb_turt_s
    FROM aux_df
    GROUP BY CaptureSite, year, month, week"))

aux_capt <- aux_capt %>%
    group_by(CaptureSite) %>%
    mutate(nb_turt_capt_tot = sum(nb_turt_s)) %>%
    group_by(CaptureSite, month) %>%
    mutate(nb_turt_capt_avg_m = mean(nb_turt_s)) %>%
    group_by(CaptureSite, week) %>%
    mutate(nb_turt_tot_c_w = sum(nb_turt_s)) %>%
    ungroup()
aux_capt$week <- as.Date(aux_capt$week)

# -----------------------------------------------------------------------------
# 2. WEEK-LEVEL FEATURES
# -----------------------------------------------------------------------------

print_step("Building week-level aggregates...")

aux_week <- as.data.table(sqldf("
    SELECT week, CaptureSite, year, month,
           COUNT(DISTINCT Rescue_ID) AS nb_turt_s
    FROM aux_df
    GROUP BY week, CaptureSite, year, month"))
aux_week$week_ <- week(aux_week$week)

aux_week <- aux_week %>%
    group_by(week_) %>%
    mutate(nb_turt_week_tot = sum(nb_turt_s)) %>%
    group_by(week_, month) %>%
    mutate(nb_turt_week_avg_m = mean(nb_turt_s)) %>%
    ungroup()
aux_week$week <- as.Date(aux_week$week)

# -----------------------------------------------------------------------------
# 3. RESCUE FREQUENCY FEATURES (2018 only)
# -----------------------------------------------------------------------------

print_step("Computing rescue frequency patterns...")

df_full <- rbind(org, test)
df_full <- df_full[df_full$year != YEAR_EXCLUDE, ]
df_full$day_year <- yday(df_full$Date)
df_full <- df_full %>% left_join(capt, by = "CaptureSite")
df_full$week <- as.Date(df_full$week)
df_full$week_ <- week(df_full$Date)

# Count per site×week (for frequency calc)
aux_freq <- as.data.table(sqldf("
    SELECT DISTINCT CaptureSite, week, SUM(nb_turt_s) AS nb
    FROM aux_capt WHERE year = 2018
    GROUP BY CaptureSite, week"))
aux_freq <- aux_freq[aux_freq$nb > 0, ]
aux_freq$week_ <- week(aux_freq$week)
aux_freq <- aux_freq %>%
    group_by(CaptureSite) %>%
    mutate(Diff = week_ - lag(week_)) %>%
    ungroup()
aux_freq$Diff[is.na(aux_freq$Diff)] <- 0
aux_freq <- aux_freq[aux_freq$Diff > 0, ]

aux_freq_ <- aux_freq %>%
    group_by(CaptureSite) %>%
    summarise(
        freq_c_mean = mean(Diff),
        freq_c_max = max(Diff),
        freq_c_min = min(Diff),
        freq_c_sd = sd(Diff),
        .groups = "drop"
    )

# -----------------------------------------------------------------------------
# 4. YEAR-OVER-YEAR ELASTICITY
# -----------------------------------------------------------------------------

print_step("Computing year-over-year growth elasticity...")

aux_prct_14 <- as.data.table(sqldf("SELECT DISTINCT CaptureSite, week, SUM(nb_turt_s) AS nb_14
    FROM aux_capt WHERE year=2014 GROUP BY CaptureSite, week"))
aux_prct_15 <- as.data.table(sqldf("SELECT DISTINCT CaptureSite, week, SUM(nb_turt_s) AS nb_15
    FROM aux_capt WHERE year=2015 GROUP BY CaptureSite, week"))
aux_prct_17 <- as.data.table(sqldf("SELECT DISTINCT CaptureSite, week, SUM(nb_turt_s) AS nb_17
    FROM aux_capt WHERE year=2017 GROUP BY CaptureSite, week"))
aux_prct_18 <- as.data.table(sqldf("SELECT DISTINCT CaptureSite, week, SUM(nb_turt_s) AS nb_18
    FROM aux_capt WHERE year=2018 GROUP BY CaptureSite, week"))

aux_prct_14$week_ <- week(aux_prct_14$week)
aux_prct_15$week_ <- week(aux_prct_15$week)
aux_prct_17$week_ <- week(aux_prct_17$week)
aux_prct_18$week_ <- week(aux_prct_18$week)

aux_prct <- aux_prct_14[, -c(2)] %>%
    left_join(aux_prct_15[, -c(2)], by = c("CaptureSite", "week_")) %>%
    left_join(aux_prct_17[, -c(2)], by = c("CaptureSite", "week_")) %>%
    left_join(aux_prct_18[, -c(2)], by = c("CaptureSite", "week_"))

aux_prct <- aux_prct %>%
    mutate(
        diff_1 = nb_15 - nb_14,
        diff_2 = nb_17 - nb_15,
        last_diff = nb_18 - nb_17,
        elast1 = (((diff_2 + 1) - (diff_1 + 1)) / (diff_2 + 1) +
            ((last_diff + 1) - (diff_2 + 1)) / (last_diff + 1)) / 2,
        elast2 = (((diff_2 + 2) - (diff_1 + 2)) / (diff_2 + 2) +
            ((last_diff + 2) - (diff_2 + 2)) / (last_diff + 2)) / 2
    )
aux_prct$elast1[!is.finite(aux_prct$elast1)] <- -10
aux_prct$elast2[!is.finite(aux_prct$elast2)] <- -10
aux_prct$elast <- pmax(aux_prct$elast1, aux_prct$elast2)

# -----------------------------------------------------------------------------
# 5. K-MEANS CLUSTERING OF SITES
# -----------------------------------------------------------------------------

print_step("Clustering capture sites...")

# Join features to full grid first (for clustering input)
df <- df_full %>%
    left_join(aux_capt[, c(1:5)], by = c("CaptureSite", "year", "month", "week")) %>%
    left_join(unique(aux_capt[, c(1, 6)]), by = "CaptureSite") %>%
    left_join(unique(aux_capt[, c(1, 3, 7)]), by = c("CaptureSite", "month")) %>%
    left_join(unique(aux_capt[, c(1, 4, 8)]), by = c("CaptureSite", "week")) %>%
    left_join(unique(aux_week[, c(6, 7)]), by = "week_") %>%
    left_join(unique(aux_week[, c(6, 4, 8)]), by = c("week_", "month"))

df$nb_turt_s[is.na(df$nb_turt_s)] <- 0
df$nb_turt_tot_c_w[is.na(df$nb_turt_tot_c_w)] <- 0
df$nb_turt_capt_avg_m[is.na(df$nb_turt_capt_avg_m)] <- 0

# Cluster on site profiles
clust_input <- df[, c("nb_turt_capt_tot", "nb_turt_capt_avg_m", "nb_turt_tot_c_w")]
clust_input[is.na(clust_input)] <- 0
res_kmeans <- kmeans(clust_input, centers = 5, nstart = 7)
df$cluster <- res_kmeans$cluster

# Join remaining features
df <- df %>%
    left_join(aux_freq_, by = "CaptureSite") %>%
    left_join(aux_prct[, c("CaptureSite", "nb_15", "week_", "elast")],
        by = c("CaptureSite", "week_")
    )

# Impute NAs with sensible defaults
df$freq_c_mean[is.na(df$freq_c_mean)] <- 17
df$freq_c_max[is.na(df$freq_c_max)] <- 20
df$freq_c_min[is.na(df$freq_c_min)] <- 17
df$freq_c_sd[is.na(df$freq_c_sd)] <- 7
df$last_diff[is.na(df$last_diff)] <- -20
df$elast[is.na(df$elast)] <- -12

# Site category dummies
df$categ_0 <- as.integer(df$CaptureSiteCategory == "CaptureSiteCategory_0")
df$categ_1 <- as.integer(df$CaptureSiteCategory == "CaptureSiteCategory_1")

# Week alignment fix
for (i in seq_len(nrow(df))) {
    if (df$year[i] == 2014) df$week_[i] <- df$week_[i] + 1
}
df$week_[df$year == 2013] <- 1
df$month[df$year == 2013] <- 1
df$year[df$year == 2013] <- 2014
df <- df[df$week_ <= FREQ, ]

cat(sprintf("  Final feature matrix: %d rows × %d columns\n", nrow(df), ncol(df)))

# -----------------------------------------------------------------------------
# 6. SAVE
# -----------------------------------------------------------------------------

saveRDS(df, file.path(OUTPUT_DIR, "df_features.rds"))
saveRDS(aux_capt, file.path(OUTPUT_DIR, "aux_capt.rds"))

print_step("Feature engineering complete.")
# ============================================================================
# 02_FEATURE_ENGINEERING.R - Feature Engineering Pipeline
# ============================================================================
# Description: Creates engineered features for both base-level and cell-level
#              datasets including time features, domain-specific indicators,
#              and statistical transformations.
# ============================================================================

source("00_config.R")

# Load cleaned data if not in memory
if (!exists("base")) {
    load("data_base_cleaned.RData")
}

# ============================================================================
# 1. HELPER FUNCTIONS FOR ENCODING
# ============================================================================

#' Helmert Contrast Encoding
#' @param n Number of levels
helmert_matrix <- function(n) {
    m <- t((diag(seq(n - 1, 0)) - upper.tri(matrix(1, n, n)))[-n, ])
    t(apply(m, 1, rev))
}

#' Encode categorical variable using Helmert contrasts
#' @param df Data frame
#' @param var Variable name to encode
encode_helmert <- function(df, var) {
    x <- unique(df[[var]])
    n <- length(x)
    d <- as.data.frame(helmert_matrix(n))
    d[[var]] <- rev(x)
    names(d) <- c(paste0(var, 1:(n - 1)), var)
    return(d)
}

#' Polynomial Contrast Encoding
#' @param df Data frame
#' @param var Variable name to encode
encode_polynomial <- function(df, var) {
    x <- unique(df[[var]])
    n <- length(x)
    d <- as.data.frame(contr.poly(n))
    d[[var]] <- x
    names(d) <- c(paste0(var, 1:(n - 1)), var)
    return(d)
}

#' Backward Difference Encoding Matrix
#' @param n Number of levels
backward_difference_matrix <- function(n) {
    m <- matrix(0:(n - 1), nrow = n, ncol = n)
    m <- m + upper.tri(matrix(1, n, n))
    m2 <- matrix(-(n - 1), n, n)
    m2[upper.tri(m2)] <- 0
    m <- (m + m2) / n
    m <- (t(m))[, -n]
    return(m)
}

#' Encode categorical variable using backward difference contrasts
#' @param df Data frame
#' @param var Variable name to encode
encode_backward_difference <- function(df, var) {
    x <- unique(df[[var]])
    n <- length(x)
    d <- as.data.frame(backward_difference_matrix(n))
    d[[var]] <- x
    names(d) <- c(paste0(var, 1:(n - 1)), var)
    return(d)
}

#' Target Encoding (Mean Encoding)
#' @param x Categorical variable
#' @param y Target variable
#' @param sigma Optional noise for regularization
encode_target <- function(x, y, sigma = NULL) {
    d <- aggregate(y, list(factor(x, exclude = NULL)), mean, na.rm = TRUE)
    m <- d[is.na(as.character(d[, 1])), 2]
    l <- d[, 2]
    names(l) <- d[, 1]
    l <- l[x]
    l[is.na(l)] <- m

    if (!is.null(sigma)) {
        l <- l * rnorm(length(l), mean = 1, sd = sigma)
    }
    return(l)
}

#' M-Estimator Encoding (Smoothed Target Encoding)
#' @param x Categorical variable
#' @param y Target variable
#' @param m Smoothing parameter
#' @param sigma Optional noise for regularization
encode_m_estimator <- function(x, y, m = 1, sigma = NULL) {
    p_all <- mean(y, na.rm = TRUE)

    d <- aggregate(y, list(factor(x, exclude = NULL)), sum, na.rm = TRUE)
    d2 <- aggregate(y, list(factor(x, exclude = NULL)), length)
    g <- names(d)[1]
    d <- merge(d, d2, by = g, all = TRUE)
    d[, 4] <- (d[, 2] + p_all * m) / (d[, 3] + m)

    m_default <- d[is.na(as.character(d[, 1])), 4]
    l <- d[, 4]
    names(l) <- d[, 1]
    l <- l[x]
    l[is.na(l)] <- m_default

    if (!is.null(sigma)) {
        l <- l * rnorm(length(l), mean = 1, sd = sigma)
    }

    list(
        encoded = l,
        encode_new = function(new_x) {
            new_l <- l[new_x]
            new_l[is.na(new_l)] <- m_default
            if (!is.null(sigma)) {
                new_l <- new_l * rnorm(length(new_l), mean = 1, sd = sigma)
            }
            return(new_l)
        }
    )
}

#' Weight of Evidence Encoding
#' @param x Categorical variable
#' @param y Target variable
#' @param sigma Optional noise for regularization
encode_woe <- function(x, y, sigma = NULL) {
    d <- aggregate(y, list(factor(x, exclude = NULL)), mean, na.rm = TRUE)
    d[["woe"]] <- log(((1 / d[, 2]) - 1) * (sum(y, na.rm = TRUE) / sum(1 - y, na.rm = TRUE)))

    m <- d[is.na(as.character(d[, 1])), 3]
    l <- d[, 3]
    names(l) <- d[, 1]
    l <- l[x]
    l[is.na(l)] <- m

    if (!is.null(sigma)) {
        l <- l * rnorm(length(l), mean = 1, sd = sigma)
    }

    list(
        encoded = l,
        encode_new = function(new_x) {
            new_l <- l[new_x]
            new_l[is.na(new_l)] <- m
            if (!is.null(sigma)) {
                new_l <- new_l * rnorm(length(new_l), mean = 1, sd = sigma)
            }
            return(new_l)
        }
    )
}

# ============================================================================
# 2. DOMAIN-SPECIFIC FEATURE FUNCTIONS
# ============================================================================

#' Create time-based features
#' @param df Data frame with Time column
#' @param time_col Name of time column
create_time_features <- function(df, time_col = "Time") {
    df$Time <- strptime(df[[time_col]], format = "%m/%d/%Y %H:%M")
    df$hour <- as.numeric(format(df$Time, format = "%H"))
    df$day_of_week <- as.integer(format(df$Time, format = "%w"))
    df$day_of_week <- ifelse(df$day_of_week == 0, "Sun",
        ifelse(df$day_of_week == 6, "Sat", "week")
    )
    df$Time <- NULL
    return(df)
}

#' Create load-based hour indicator
#' @param hour Hour value (0-23)
create_load_hour <- function(hour) {
    ifelse(hour %in% c(3:7), 0, 1)
}

#' Create tree-based energy category (Factor version)
#' @param nb_cell Number of cells
#' @param load Load value
#' @param anten_cat Antenna category
create_tree_energy_F <- function(nb_cell, load, anten_cat) {
    case_when(
        nb_cell == 1 & load <= 0.08 ~ "VL",
        (nb_cell == 1 & load >= 0.6) |
            (nb_cell == 2 & load <= 0.37 & anten_cat != 4) ~ "M",
        (nb_cell == 2 & load > 0.37 & anten_cat != 4) |
            (nb_cell == 2 & load <= 0.37 & anten_cat == 4) ~ "H",
        nb_cell == 2 & load > 0.37 & anten_cat == 4 ~ "VH",
        TRUE ~ "L"
    )
}

#' Create tree-based energy category (Numeric version)
#' @param nb_cell Number of cells
#' @param load Load value
#' @param anten_cat Antenna category
create_tree_energy_N <- function(nb_cell, load, anten_cat) {
    case_when(
        nb_cell == 1 & load <= 0.08 ~ 2,
        (nb_cell == 1 & load >= 0.6) |
            (nb_cell == 2 & load <= 0.37 & anten_cat != 4) ~ 3,
        (nb_cell == 2 & load > 0.37 & anten_cat != 4) |
            (nb_cell == 2 & load <= 0.37 & anten_cat == 4) ~ 4,
        nb_cell == 2 & load > 0.37 & anten_cat == 4 ~ 5,
        TRUE ~ 1
    )
}

#' Create ES1 low load indicator
create_es1_load_low <- function(ESMode1, load, anten_cat, nb_cell) {
    case_when(
        (ESMode1 >= 0.47 & load <= 0.044 & anten_cat %in% c(8, 32, 64)) |
            (nb_cell == 1 & load <= 0.5) |
            (nb_cell == 2 & ESMode1 >= 0.46 & load <= 0.044) ~ 1,
        TRUE ~ 0
    )
}

#' Create ES3 high load indicator
create_es3_load_high <- function(ESMode3, load) {
    ifelse(ESMode3 == 0 & load >= 0.055, 1, 0)
}

#' Create ES6 high load indicator
create_es6_load_high <- function(ESMode6, load) {
    ifelse(ESMode6 <= 0.13 & load >= 0.055, 1, 0)
}

#' Create frequency-cell-antenna based indicators
create_freq_cell_anten_low <- function(freq_max, nb_cell, anten_cat) {
    case_when(
        (freq_max == "426.98" & nb_cell == 1 & anten_cat == "1") |
            (freq_max == "697.002" & nb_cell == 2) ~ 1,
        TRUE ~ 0
    )
}

create_freq_cell_anten_high <- function(freq_max, nb_cell, anten_cat) {
    ifelse(freq_max == "426.98" & nb_cell == 2 & anten_cat == "4", 1, 0)
}

#' Create high/low use indicators
create_high_use <- function(RUType, anten_cat, load, load_hour, hour, es3_load_high) {
    case_when(
        (RUType == "Type1" & anten_cat == 4) &
            ((load <= 0.4 & load_hour == 1) |
                (hour %in% c(13, 14, 15, 16, 17, 18, 22)) |
                (es3_load_high == 1)) ~ 1,
        TRUE ~ 0
    )
}

create_low_use <- function(ESMode1, load, freq_max, nb_cell, anten_cat) {
    case_when(
        (ESMode1 >= 0.7 & load >= 0.03) |
            (freq_max == "697.002" & nb_cell == 2) |
            (freq_max == "426.98" & nb_cell == 1 & anten_cat == "1") ~ 1,
        TRUE ~ 0
    )
}

#' Create type categorizations
create_type_anten <- function(RUType) {
    case_when(
        RUType %in% c("Type9", "Type10", "Type11", "Type12") ~ "anten_unique",
        RUType %in% c("Type1", "Type2", "Type3") ~ "anten_mode2",
        RUType %in% c("Type4", "Type5", "Type6", "Type7", "Type8") ~ "anten_two"
    )
}

create_type_use <- function(RUType) {
    case_when(
        RUType %in% c("Type1", "Type10", "Type11") ~ "high",
        RUType %in% c("Type12", "Type7", "Type8") ~ "medium",
        RUType %in% c("Type2", "Type3", "Type4", "Type5", "Type6", "Type9") ~ "low"
    )
}

#' Create hour category
create_hour_ch <- function(hour) {
    case_when(
        hour %in% c(2:6) ~ "L",
        hour %in% c(0, 1, 7:11) ~ "M",
        TRUE ~ "H"
    )
}

#' Apply Box-Cox transformation safely
apply_boxcox <- function(x, y_for_fitting) {
    tryCatch(
        {
            model <- boxcox(y_for_fitting ~ x, plotit = FALSE)
            lambda <- model$x[which.max(model$y)]

            if (abs(lambda) < .Machine$double.eps^0.25) {
                result <- log(x)
            } else {
                result <- (x^lambda - 1) / lambda
            }
            result[is.infinite(result) & result < 0] <- 0
            return(result)
        },
        error = function(e) {
            return(x)
        }
    )
}

# ============================================================================
# 3. FACTOR RECODING FUNCTIONS
# ============================================================================

#' Recode antenna categories
recode_anten_cat <- function(anten_cat) {
    anten_cat <- fct_recode(anten_cat, "2" = "8")
    anten_cat <- fct_recode(anten_cat, "32" = "64")
    return(anten_cat)
}

#' Recode RUType categories (merge similar types)
recode_rutype <- function(RUType) {
    RUType <- fct_recode(RUType, "Type6" = "Type9")
    RUType <- fct_recode(RUType, "Type7" = "Type8")
    RUType <- fct_recode(RUType, "Type10" = "Type11")
    RUType <- fct_recode(RUType, "Type10" = "Type12")
    RUType <- fct_recode(RUType, "Type4" = "Type2")
    RUType <- fct_recode(RUType, "Type4" = "Type6")
    return(RUType)
}

#' Recode frequency categories
recode_freq <- function(freq) {
    freq <- fct_recode(freq, "697.002" = "979.998")
    freq <- fct_recode(freq, "697.002" = "715.998")
    freq <- fct_recode(freq, "532" = "189")
    return(freq)
}

#' Recode bandwidth categories
recode_band <- function(band) {
    fct_recode(band, "10" = "8")
}

cat("Feature engineering functions loaded.\n")
