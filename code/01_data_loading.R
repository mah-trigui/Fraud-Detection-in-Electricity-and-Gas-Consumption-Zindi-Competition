# =============================================================================
# 01_data_loading.R - Load and Join Client + Invoice Data
# STEG Fraud Detection
# =============================================================================

source("00_config.R")
print_section("Step 1: Data Loading & Initial Cleaning")

# -----------------------------------------------------------------------------
# 1. LOAD
# -----------------------------------------------------------------------------

train_cl <- fread("client.csv")
train_inv <- fread("invoice.csv")
test_cl <- fread("client_test.csv")
test_inv <- fread("invoice_test.csv")

test_cl$target <- NA
cl_tot <- rbind(train_cl, test_cl)
cl_tot$fraud <- as.factor(cl_tot$target)
inv_tot <- rbind(train_inv, test_inv)

cat(sprintf("  Train clients: %d | Test clients: %d\n", nrow(train_cl), nrow(test_cl)))
cat(sprintf("  Total invoices: %d\n", nrow(inv_tot)))
cat(sprintf("  Fraud rate: %.2f%%\n", 100 * mean(train_cl$target, na.rm = TRUE)))

# -----------------------------------------------------------------------------
# 2. DATE PARSING & JOIN
# -----------------------------------------------------------------------------

cl_tot$creation_date <- as.Date(cl_tot$creation_date, "%d/%m/%Y")
inv_tot$invoice_date <- as.Date(inv_tot$invoice_date)

tot <- inv_tot %>%
    left_join(cl_tot, by = "client_id")

tot$consom <- tot$consommation_level_1 + tot$consommation_level_2 +
    tot$consommation_level_3 + tot$consommation_level_4
tot$Year <- format(tot$invoice_date, "%Y")
tot$Month <- format(tot$invoice_date, "%m")
tot$Anc <- as.numeric(difftime(as.Date("2019-10-01"), tot$creation_date, units = "days")) / 30

# -----------------------------------------------------------------------------
# 3. LEVEL HARMONIZATION (train/test alignment)
# -----------------------------------------------------------------------------

print_step("Harmonizing factor levels across train/test...")

# Merge rare/mismatched levels to avoid unseen-level issues at prediction
tot$tarif_type[tot$tarif_type == "18"] <- "45"
tot$counter_statue[tot$counter_statue %in% c("269375", "420", "46", "618", "769", "A")] <- "2"
tot$counter_code[tot$counter_code %in% c("0", "1", "367")] <- "65"
tot$reading_remarque[tot$reading_remarque %in% c("5", "203", "207", "413")] <- "7"
tot$counter_coefficient[tot$counter_coefficient %in% c("6", "8", "9", "11", "20", "30", "33", "40", "50")] <- "3"
tot$counter_coefficient[tot$counter_coefficient == "5"] <- "10"
tot$counter_coefficient[tot$counter_coefficient == "21"] <- "3"
tot$region[tot$region == "199"] <- "399"

# Further minimization of levels based on fraud rate analysis
tot$tarif_type[tot$tarif_type == "9"] <- "11"
tot$tarif_type[tot$tarif_type %in% c("12", "14")] <- "13"
tot$tarif_type[tot$tarif_type == "21"] <- "29"
tot$counter_statue[tot$counter_statue == "3"] <- "1"

# Binary counter code grouping (based on fraud association)
tot$counter_code_B <- "2"
tot$counter_code_B[tot$counter_code %in% c("10", "25", "40", "204", "210", "214", "407", "433", "453", "506", "600")] <- "1"
tot$counter_code_B[tot$counter_code %in% c("65", "101", "102", "222", "305", "307", "310", "317", "532", "565")] <- "0"

cols <- c(
    "tarif_type", "counter_statue", "counter_code", "reading_remarque",
    "counter_coefficient", "counter_type", "disrict", "client_catg", "region"
)
tot <- tot %>% mutate(across(all_of(cols), as.factor))

cat(sprintf("  Final joined dataset: %d rows × %d columns\n", nrow(tot), ncol(tot)))

# -----------------------------------------------------------------------------
# 4. SAVE
# -----------------------------------------------------------------------------

saveRDS(tot, file.path(OUTPUT_DIR, "tot.rds"))
saveRDS(cl_tot, file.path(OUTPUT_DIR, "cl_tot.rds"))
saveRDS(train_cl, file.path(OUTPUT_DIR, "train_cl.rds"))

print_step("Data loading complete.")
# =============================================================================
# 01_data_loading.R - Load and Prepare Raw Data
# Sea Turtle Rescue Forecast Challenge
# =============================================================================

source("00_config.R")
print_section("Step 1: Data Loading")

# -----------------------------------------------------------------------------
# 1. LOAD
# -----------------------------------------------------------------------------

capt <- fread("CaptureSite_category.csv")
capt <- as.data.frame(unclass(capt))

df_orig <- fread("train.csv")
df_orig$year <- year(df_orig$Date_TimeCaught)
df_orig$month <- month(df_orig$Date_TimeCaught)
df_orig$LandingSite <- substring(df_orig$LandingSite, 13)
names(df_orig)[8] <- "CaptureSiteCategory"
df_orig <- df_orig[, -c(3, 7, 16, 17, 18, 19, 21)]
df_orig <- as.data.frame(unclass(df_orig))
df_orig$Date <- as.POSIXct(df_orig$Date_TimeCaught, "%Y-%m-%d", tz = "UTC")
df_orig$week <- cut(df_orig$Date, "week", start.on.monday = TRUE)
df_orig$yday <- yday(df_orig$Date)
df_orig <- df_orig[df_orig$Date >= TRAIN_START, ]

cat(sprintf("  Total rescue records: %d\n", nrow(df_orig)))
cat(sprintf("  Capture sites: %d\n", length(unique(df_orig$CaptureSite))))
cat(sprintf("  Date range: %s to %s\n", min(df_orig$Date), max(df_orig$Date)))
cat(sprintf("  Years: %s\n", paste(sort(unique(df_orig$year)), collapse = ", ")))

# Reference list of all capture sites
Capture_ref <- data.table(
    CaptureSite = sort(unique(df_orig$CaptureSite))
)

# Exclude problematic year
aux_df <- df_orig[df_orig$year != YEAR_EXCLUDE, ]

cat(sprintf("  Records after excluding %d: %d\n", YEAR_EXCLUDE, nrow(aux_df)))

# -----------------------------------------------------------------------------
# 2. BUILD DATE GRIDS (TRAIN + TEST)
# -----------------------------------------------------------------------------

print_step("Building date grids...")

# Train grid: all site × week combinations
dates_train <- seq(
    from = as.POSIXct(TRAIN_START, "%Y-%m-%d", tz = "UTC"),
    to   = as.POSIXct("2018-12-31", "%Y-%m-%d", tz = "UTC"),
    by   = "week"
)
aux_date <- data.table(Date = dates_train)
org <- data.table(sqldf("SELECT CaptureSite, Date FROM Capture_ref a CROSS JOIN aux_date b"))
org$year <- year(org$Date)
org$month <- month(org$Date)
org$week <- cut(org$Date, "week", start.on.monday = TRUE)

# Test grid: 2019 weeks
dates_test <- seq(
    from = as.POSIXct(TEST_START, "%Y-%m-%d", tz = "UTC"),
    to   = as.POSIXct(TEST_END, "%Y-%m-%d", tz = "UTC"),
    by   = "week"
)
aux_date <- data.table(Date = dates_test)
test <- data.table(sqldf("SELECT CaptureSite, Date FROM Capture_ref a CROSS JOIN aux_date b"))
test$year <- year(test$Date)
test$month <- month(test$Date)
test$week <- cut(test$Date, "week", start.on.monday = TRUE)

cat(sprintf("  Train grid: %d rows | Test grid: %d rows\n", nrow(org), nrow(test)))

# -----------------------------------------------------------------------------
# 3. SAVE
# -----------------------------------------------------------------------------

saveRDS(df_orig, file.path(OUTPUT_DIR, "df_orig.rds"))
saveRDS(aux_df, file.path(OUTPUT_DIR, "aux_df.rds"))
saveRDS(capt, file.path(OUTPUT_DIR, "capt.rds"))
saveRDS(Capture_ref, file.path(OUTPUT_DIR, "capture_ref.rds"))
saveRDS(org, file.path(OUTPUT_DIR, "grid_train.rds"))
saveRDS(test, file.path(OUTPUT_DIR, "grid_test.rds"))

print_step("Data loading complete.")
# =============================================================================
# 01_data_loading.R - Load All Data Sources
# IEEE Escalations in Customer Support
# =============================================================================

source("00_config.R")
print_section("Step 1: Data Loading")

# -----------------------------------------------------------------------------
# 1. LOAD ALL FILES
# -----------------------------------------------------------------------------

summary_dt <- fread(FILE_SUMMARY, header = TRUE, sep = ",", quote = "'")
history <- fread(FILE_HISTORY, header = TRUE, sep = ",", quote = "'")
milestones <- fread(FILE_MILESTONES, header = TRUE, sep = ",", quote = "'")
comments <- fread(FILE_COMMENTS, header = TRUE, sep = ",", quote = "'")
test_cases <- fread(FILE_TEST, header = TRUE, sep = ",", quote = "'")
dictionary <- fread(FILE_DICTIONARY, header = TRUE, sep = ",", quote = "'")
lemma <- fread(FILE_LEMMA, header = TRUE, sep = ",", quote = "")

# Set keys for efficient joins
setkey(summary_dt, REFERENCEID)
setkey(history, REFERENCEID, SECONDS_SINCE_CASE_START)
setkey(milestones, REFERENCEID, SECONDS_SINCE_CASE_START)
setkey(comments, REFERENCEID, SECONDS_SINCE_CASE_START)

cat(sprintf("  Cases (metadata): %d\n", nrow(summary_dt)))
cat(sprintf("  History events: %d\n", nrow(history)))
cat(sprintf("  Milestones: %d\n", nrow(milestones)))
cat(sprintf("  Comments: %d\n", nrow(comments)))
cat(sprintf("  Test cases: %d\n", nrow(test_cases)))
cat(sprintf(
    "  Dictionary terms: %d | Lemma mappings: %d\n",
    nrow(dictionary), nrow(lemma)
))

# -----------------------------------------------------------------------------
# 2. IDENTIFY KEY REFERENCE SETS
# -----------------------------------------------------------------------------

test_ids <- test_cases$REFERENCEID
ref_ids_escalated <- history[INV_TIME_TO_NEXT_ESCALATION > 0, unique(REFERENCEID)]

cat(sprintf(
    "  Escalated cases: %d / %d (%.1f%%)\n",
    length(ref_ids_escalated),
    uniqueN(summary_dt$REFERENCEID),
    100 * length(ref_ids_escalated) / uniqueN(summary_dt$REFERENCEID)
))

# -----------------------------------------------------------------------------
# 3. SAVE
# -----------------------------------------------------------------------------

saveRDS(summary_dt, file.path(OUTPUT_DIR, "summary.rds"))
saveRDS(history, file.path(OUTPUT_DIR, "history.rds"))
saveRDS(milestones, file.path(OUTPUT_DIR, "milestones.rds"))
saveRDS(comments, file.path(OUTPUT_DIR, "comments.rds"))
saveRDS(test_cases, file.path(OUTPUT_DIR, "test_cases.rds"))
saveRDS(dictionary, file.path(OUTPUT_DIR, "dictionary.rds"))
saveRDS(lemma, file.path(OUTPUT_DIR, "lemma.rds"))
saveRDS(test_ids, file.path(OUTPUT_DIR, "test_ids.rds"))
saveRDS(ref_ids_escalated, file.path(OUTPUT_DIR, "escalated_ids.rds"))

print_step("Data loading complete.")
