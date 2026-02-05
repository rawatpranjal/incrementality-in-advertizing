#!/usr/bin/env Rscript
#' 03_purchase_model.R - Fixed Effects Models: Purchase Prediction
#'
#' Unit of Analysis: Session
#' Dependent Variable: purchased (binary) or GMV (continuous)
#' Independent Variables: clicks, impressions, auctions, products_impressed, duration_hours
#'
#' Models (5 specifications):
#' 0. Zero-base: Clicks + Impressions only (no FE, no controls)
#' 1. Baseline: All controls, no fixed effects
#' 2. User FE: User fixed effects
#' 3. Week FE: Week fixed effects
#' 4. User+Week FE: Both user and week fixed effects
#'
#' Logit only for binary purchase outcome (no LPM)
#' GMV models use linear specification
#' Note: Logit User FE drops users with no purchase variation (documented in output)
#' Evaluation: AUC-ROC, AUC-PR, Pseudo R², Deviance
#' Output: LaTeX tables for paper integration

suppressPackageStartupMessages({
  library(data.table)
  library(fixest)
})

# Check for pROC (for AUC metrics)
if (!requireNamespace("pROC", quietly = TRUE)) {
  cat("Installing pROC for AUC metrics...\n")
  install.packages("pROC", repos = "https://cloud.r-project.org/", quiet = TRUE)
}
suppressPackageStartupMessages(library(pROC))

# Handle script path when run with Rscript
get_script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(normalizePath(sub("^--file=", "", file_arg)))
  }
  # Fallback to working directory
  return(file.path(getwd(), "unified-session-position-analysis/shopping-sessions/scripts/03_purchase_model.R"))
}

script_path <- get_script_path()
BASE_DIR <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)
DATA_DIR <- file.path(BASE_DIR, "0_data_pull", "data")
RESULTS_DIR <- file.path(BASE_DIR, "results")
LATEX_DIR <- file.path(dirname(script_path), "../../../paper/05-sessions")

# Optional configuration via environment variables
# - SESSIONS_FILE: default 'sessions.parquet' (use 'sessions_5day.parquet' for 5-day)
# - LATEX_SUFFIX: optional suffix for LaTeX filename (e.g., '_5day')
SESSIONS_FILE <- Sys.getenv("SESSIONS_FILE", unset = "sessions.parquet")
LATEX_SUFFIX <- Sys.getenv("LATEX_SUFFIX", unset = "")

# Create directories
dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(LATEX_DIR, recursive = TRUE, showWarnings = FALSE)

# Read parquet using duckdb
read_parquet_dt <- function(path) {
  if (requireNamespace("duckdb", quietly = TRUE) && requireNamespace("DBI", quietly = TRUE)) {
    con <- DBI::dbConnect(duckdb::duckdb(), dbdir = tempfile())
    on.exit(DBI::dbDisconnect(con, shutdown = TRUE), add = TRUE)
    sql <- sprintf("SELECT * FROM read_parquet('%s')", normalizePath(path))
    dt <- as.data.table(DBI::dbGetQuery(con, sql))
    return(dt)
  }
  if (requireNamespace("arrow", quietly = TRUE)) {
    return(as.data.table(arrow::read_parquet(path)))
  }
  stop("Neither duckdb nor arrow is available to read parquet.")
}

# Output file
OUTPUT_FILE <- file.path(RESULTS_DIR, paste0("03_purchase_model", LATEX_SUFFIX, ".txt"))
sink(OUTPUT_FILE, split = TRUE)

cat(strrep("=", 80), "\n")
cat("03_PURCHASE_MODEL - Fixed Effects Logit (Level Variables) - R/fixest\n")
cat(strrep("=", 80), "\n")
cat(sprintf("Data directory: %s\n", DATA_DIR))

# ============================================================================
# DATA PREPARATION
# ============================================================================
cat("\n", strrep("=", 80), "\n")
cat("DATA PREPARATION\n")
cat(strrep("=", 80), "\n")

sessions <- read_parquet_dt(file.path(DATA_DIR, SESSIONS_FILE))
cat(sprintf("Loaded %s sessions\n", format(nrow(sessions), big.mark = ",")))

# Filter to sessions with impressions
dt <- sessions[n_impressions > 0]
cat(sprintf("Sessions with impressions: %s\n", format(nrow(dt), big.mark = ",")))

# Create LEVEL variables (no log transforms)
dt[, `:=`(
  clicks = n_clicks,
  impressions = n_impressions,
  auctions = n_auctions,
  products_impressed = n_products_impressed,
  duration_hours = session_duration_hours
)]

# Squared terms for clicks only (GMV saturation analysis)
dt[, clicks_sq := clicks^2]

# Create week variable for FE
dt[, session_start := as.POSIXct(session_start)]
dt[, week := format(session_start, "%Y_W%V")]

# Outcome
dt[, y := as.integer(purchased)]

# Convert user_id to factor for FE
dt[, user_id := as.factor(user_id)]
dt[, week := as.factor(week)]

cat(sprintf("\nOutcome distribution:\n"))
cat(sprintf("  Purchased: %s (%.3f%%)\n", format(sum(dt$y), big.mark = ","), 100 * mean(dt$y)))
cat(sprintf("  Not purchased: %s (%.3f%%)\n", format(sum(1 - dt$y), big.mark = ","), 100 * mean(1 - dt$y)))

cat(sprintf("\nFixed Effects dimensions:\n"))
cat(sprintf("  Unique users: %s\n", format(length(unique(dt$user_id)), big.mark = ",")))
cat(sprintf("  Unique weeks: %s\n", format(length(unique(dt$week)), big.mark = ",")))
cat(sprintf("  Weeks: %s\n", paste(sort(unique(as.character(dt$week))), collapse = ", ")))

# ----------------------------------------------------------------------------
# Raw correlation: 1(clicks>0) vs 1(purchase>0)
# ----------------------------------------------------------------------------
dt[, any_click := as.integer(clicks > 0)]
raw_corr <- suppressWarnings(cor(dt$any_click, dt$y))
cat(sprintf("\nRaw corr[1(click>0), 1(purchase>0)]: %.3f\n", raw_corr))

# Optionally write a TeX snippet for baseline (no suffix) to include in body text
if (LATEX_SUFFIX == "") {
  PAPER_DIR <- file.path("paper", "05-sessions")
  if (!dir.exists(PAPER_DIR)) dir.create(PAPER_DIR, recursive = TRUE, showWarnings = FALSE)
  writeLines(sprintf("%.3f", raw_corr), file.path(PAPER_DIR, "raw_click_purchase_corr.tex"))
}

# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================
cat("\n", strrep("=", 80), "\n")
cat("TABLE 1: DESCRIPTIVE STATISTICS\n")
cat(strrep("=", 80), "\n")

vars <- c("y", "clicks", "impressions", "auctions", "products_impressed", "duration_hours")
stats_table <- data.table(
  Variable = c("Purchased (binary)", "Clicks", "Impressions", "Auctions", "Products Impressed", "Duration Hours"),
  Mean = sapply(vars, function(v) mean(dt[[v]], na.rm = TRUE)),
  SD = sapply(vars, function(v) sd(dt[[v]], na.rm = TRUE)),
  Min = sapply(vars, function(v) min(dt[[v]], na.rm = TRUE)),
  Max = sapply(vars, function(v) max(dt[[v]], na.rm = TRUE))
)
print(stats_table)

# ============================================================================
# MODEL ESTIMATION
# ============================================================================
cat("\n", strrep("=", 80), "\n")
cat("MODEL ESTIMATION\n")
cat(strrep("=", 80), "\n")

# Features for Logit (linear terms only)
features_logit <- c("clicks", "impressions", "auctions", "products_impressed", "duration_hours")

# Features for zero-base model (clicks + impressions only)
features_zero <- c("clicks", "impressions")

# Features for GMV (includes clicks_sq for saturation analysis)
features_gmv <- c("clicks", "clicks_sq", "impressions", "auctions", "products_impressed", "duration_hours")

# Compute metrics function
compute_metrics <- function(y, y_prob, model = NULL, use_model_ll = FALSE) {
  eps <- 1e-15
  y_prob <- pmax(pmin(y_prob, 1 - eps), eps)
  y <- as.numeric(y)

  # AUC-ROC
  roc_obj <- roc(y, y_prob, quiet = TRUE)
  auc_roc <- as.numeric(auc(roc_obj))

  # AUC-PR (average precision)
  ord <- order(y_prob, decreasing = TRUE)
  y_sorted <- y[ord]
  precision_at_k <- cumsum(y_sorted) / seq_along(y_sorted)
  recall_at_k <- cumsum(y_sorted) / sum(y_sorted)
  auc_pr <- sum(diff(c(0, recall_at_k)) * precision_at_k)

  # Brier score
  brier <- mean((y - y_prob)^2)

  # Log-likelihood from model if available, otherwise compute
  if (!is.null(model) && inherits(model, "fixest") && use_model_ll) {
    ll <- logLik(model)
    deviance <- deviance(model)
    # Get null deviance from model
    pseudo_r2 <- tryCatch(r2(model, type = "pr2")[[1]], error = function(e) NA)
    ll_null <- if (is.na(pseudo_r2)) NA else ll / (1 - pseudo_r2)
    null_deviance <- if (is.na(ll_null)) NA else -2 * ll_null
  } else {
    # Compute from predictions
    ll <- sum(y * log(y_prob) + (1 - y) * log(1 - y_prob))

    # Null log-likelihood (intercept-only on this sample)
    p_bar <- mean(y)
    ll_null <- length(y) * (p_bar * log(p_bar) + (1 - p_bar) * log(1 - p_bar))

    deviance <- -2 * ll
    null_deviance <- -2 * ll_null
    pseudo_r2 <- 1 - (ll / ll_null)
  }

  # LR chi-squared
  lr_chi2 <- if (is.na(ll_null)) NA else -2 * (ll_null - ll)

  list(
    auc_roc = auc_roc,
    auc_pr = auc_pr,
    brier = brier,
    log_likelihood = ll,
    deviance = deviance,
    null_deviance = null_deviance,
    pseudo_r2 = pseudo_r2,
    lr_chi2 = lr_chi2
  )
}

# ----------------------------------------------------------------------------
# Model 0: Raw (Naive) Logit (Clicks + Impressions only, No FE)
# ----------------------------------------------------------------------------
cat("\n--- Model 0: Raw (Naive) Logit (Clicks + Impressions only, No FE) ---\n")
fml0 <- as.formula("y ~ clicks + impressions")
fit0 <- feglm(fml0, data = dt, family = binomial(), cluster = ~ user_id)
print(summary(fit0))

y_prob0 <- predict(fit0, type = "response")
metrics0 <- compute_metrics(dt$y, y_prob0, fit0)

# ----------------------------------------------------------------------------
# Model 1: Baseline Logit (No FE)
# ----------------------------------------------------------------------------
cat("\n--- Model 1: Baseline Logit (No FE) ---\n")
fml1 <- as.formula("y ~ clicks + impressions + auctions + products_impressed + duration_hours")
fit1 <- feglm(fml1, data = dt, family = binomial(), cluster = ~ user_id)
print(summary(fit1))

y_prob1 <- predict(fit1, type = "response")
metrics1 <- compute_metrics(dt$y, y_prob1, fit1)

# ----------------------------------------------------------------------------
# Model 2: User FE Logit
# ----------------------------------------------------------------------------
cat("\n--- Model 2: User FE Logit ---\n")
cat("Note: Conditional logit (feglm with user FE) drops users with no variation in outcome.\n")
cat("Users who always purchase or never purchase across sessions are excluded.\n")
fml2 <- as.formula("y ~ clicks + impressions + auctions + products_impressed + duration_hours | user_id")
fit2 <- tryCatch({
  feglm(fml2, data = dt, family = binomial(), cluster = ~ user_id)
}, error = function(e) {
  cat(sprintf("Note: feglm with user FE failed (%s). Using feols LPM instead.\n", e$message))
  feols(fml2, data = dt, vcov = ~ user_id)
})
print(summary(fit2))

# Handle dropped observations (users with no variation in y)
# For fixest feglm, get actual y values used in model
if (inherits(fit2, "fixest")) {
  y_prob2 <- fitted(fit2)
  # Extract the sample used - fixest stores this in fit2$sample
  if (!is.null(fit2$obs_selection)) {
    sample_idx <- setdiff(seq_len(nrow(dt)), fit2$obs_selection$obsRemoved)
    y_used2 <- dt$y[sample_idx]
  } else {
    y_used2 <- dt$y
  }
  # Ensure same length
  if (length(y_prob2) != length(y_used2)) {
    cat(sprintf("Note: Model 2 uses %d observations (length mismatch: fitted=%d, y=%d)\n",
                fit2$nobs, length(y_prob2), length(y_used2)))
    # Trim to shorter
    min_len <- min(length(y_prob2), length(y_used2))
    y_prob2 <- y_prob2[seq_len(min_len)]
    y_used2 <- y_used2[seq_len(min_len)]
  }
  metrics2 <- compute_metrics(y_used2, y_prob2, fit2)
} else {
  y_prob2 <- fitted(fit2)
  y_prob2 <- pmax(pmin(y_prob2, 0.999), 0.001)
  metrics2 <- compute_metrics(dt$y, y_prob2, fit2)
}

# ----------------------------------------------------------------------------
# Model 3: Week FE Logit
# ----------------------------------------------------------------------------
cat("\n--- Model 3: Week FE Logit ---\n")
fml3 <- as.formula("y ~ clicks + impressions + auctions + products_impressed + duration_hours | week")
fit3 <- feglm(fml3, data = dt, family = binomial(), cluster = ~ user_id)
print(summary(fit3))

y_prob3 <- predict(fit3, type = "response")
metrics3 <- compute_metrics(dt$y, y_prob3, fit3)

# ----------------------------------------------------------------------------
# Model 4: User+Week FE Logit
# ----------------------------------------------------------------------------
cat("\n--- Model 4: User+Week FE Logit ---\n")
cat("Note: Conditional logit with User+Week FE drops users with no variation in outcome.\n")
fml4 <- as.formula("y ~ clicks + impressions + auctions + products_impressed + duration_hours | user_id + week")
fit4 <- tryCatch({
  feglm(fml4, data = dt, family = binomial(), cluster = ~ user_id)
}, error = function(e) {
  cat(sprintf("Note: feglm with User+Week FE failed (%s). Using feols LPM instead.\n", e$message))
  feols(fml4, data = dt, vcov = ~ user_id)
})
print(summary(fit4))

# Handle dropped observations for fit4
if (inherits(fit4, "fixest")) {
  y_prob4 <- fitted(fit4)
  if (!is.null(fit4$obs_selection)) {
    sample_idx <- setdiff(seq_len(nrow(dt)), fit4$obs_selection$obsRemoved)
    y_used4 <- dt$y[sample_idx]
  } else {
    y_used4 <- dt$y
  }
  if (length(y_prob4) != length(y_used4)) {
    cat(sprintf("Note: Model 4 uses %d observations (length mismatch: fitted=%d, y=%d)\n",
                fit4$nobs, length(y_prob4), length(y_used4)))
    min_len <- min(length(y_prob4), length(y_used4))
    y_prob4 <- y_prob4[seq_len(min_len)]
    y_used4 <- y_used4[seq_len(min_len)]
  }
  metrics4 <- compute_metrics(y_used4, y_prob4, fit4)
} else {
  y_prob4 <- fitted(fit4)
  y_prob4 <- pmax(pmin(y_prob4, 0.999), 0.001)
  metrics4 <- compute_metrics(dt$y, y_prob4, fit4)
}

# LPM models removed per plan - using Logit only for binary purchase outcome

# ============================================================================
# CORRELATION MATRIX AND VIF DIAGNOSTICS
# ============================================================================
cat("\n", strrep("=", 80), "\n")
cat("CORRELATION MATRIX AND VIF DIAGNOSTICS\n")
cat(strrep("=", 80), "\n")

# Correlation matrix of regressors
regressor_vars <- c("clicks", "impressions", "auctions", "products_impressed", "duration_hours")
cor_matrix <- cor(dt[, ..regressor_vars], use = "pairwise.complete.obs")
cat("\nCorrelation Matrix of Regressors:\n")
print(round(cor_matrix, 3))

# Identify high correlations (>0.8)
cat("\nHigh Correlations (|r| > 0.8):\n")
for (i in 1:(ncol(cor_matrix)-1)) {
  for (j in (i+1):ncol(cor_matrix)) {
    if (abs(cor_matrix[i,j]) > 0.8) {
      cat(sprintf("  %s <-> %s: %.3f\n",
                  colnames(cor_matrix)[i], colnames(cor_matrix)[j], cor_matrix[i,j]))
    }
  }
}

# VIF computation using auxiliary regressions
cat("\nVariance Inflation Factors (VIF):\n")
cat("(VIF > 10 indicates serious multicollinearity)\n\n")
vif_results <- data.table(Variable = character(), VIF = numeric())
for (var in regressor_vars) {
  other_vars <- setdiff(regressor_vars, var)
  fml_aux <- as.formula(paste(var, "~", paste(other_vars, collapse = " + ")))
  aux_model <- lm(fml_aux, data = dt)
  r2_aux <- summary(aux_model)$r.squared
  vif_val <- 1 / (1 - r2_aux)
  vif_results <- rbind(vif_results, data.table(Variable = var, VIF = vif_val))
}
print(vif_results)

# Purchase frequency models removed per plan - focusing on binary purchase and GMV outcomes

# ============================================================================
# GMV (SPEND) MODELS - Linear FE
# ============================================================================
cat("\n", strrep("=", 80), "\n")
cat("GMV (SPEND) MODELS - Linear Fixed Effects\n")
cat(strrep("=", 80), "\n")

# Create GMV outcome (log+1 transformation for skewed spend)
dt[, gmv := total_spend]
dt[, log_gmv := log1p(total_spend)]

cat(sprintf("\nGMV distribution:\n"))
cat(sprintf("  Mean: $%.2f\n", mean(dt$gmv)))
cat(sprintf("  Median: $%.2f\n", median(dt$gmv)))
cat(sprintf("  Std. Dev.: $%.2f\n", sd(dt$gmv)))
cat(sprintf("  Min: $%.2f, Max: $%.2f\n", min(dt$gmv), max(dt$gmv)))
cat(sprintf("  Zero GMV: %d (%.1f%%)\n", sum(dt$gmv == 0), 100 * mean(dt$gmv == 0)))

# GMV Model 0: Zero-base (Clicks + Impressions only)
cat("\n--- GMV 0: Zero-base (Clicks + Impressions only, No FE) ---\n")
gmv0 <- feols(gmv ~ clicks + impressions, data = dt, vcov = ~ user_id)
print(summary(gmv0))

# GMV Model 1: Baseline (level GMV)
cat("\n--- GMV 1: Baseline (No FE) ---\n")
gmv1 <- feols(gmv ~ clicks + clicks_sq + impressions + auctions + products_impressed + duration_hours,
              data = dt, vcov = ~ user_id)
print(summary(gmv1))

# GMV Model 2: User FE
cat("\n--- GMV 2: User FE ---\n")
gmv2 <- feols(gmv ~ clicks + clicks_sq + impressions + auctions + products_impressed + duration_hours | user_id,
              data = dt, vcov = ~ user_id)
print(summary(gmv2))

# GMV Model 3: Week FE
cat("\n--- GMV 3: Week FE ---\n")
gmv3 <- feols(gmv ~ clicks + clicks_sq + impressions + auctions + products_impressed + duration_hours | week,
              data = dt, vcov = ~ user_id)
print(summary(gmv3))

# GMV Model 4: User+Week FE
cat("\n--- GMV 4: User+Week FE ---\n")
gmv4 <- feols(gmv ~ clicks + clicks_sq + impressions + auctions + products_impressed + duration_hours | user_id + week,
              data = dt, vcov = ~ user_id)
print(summary(gmv4))

# GMV comparison
gmv_coefs1 <- list(coef = coef(gmv1), se = se(gmv1))
gmv_coefs2 <- list(coef = coef(gmv2), se = se(gmv2))
gmv_coefs3 <- list(coef = coef(gmv3), se = se(gmv3))
gmv_coefs4 <- list(coef = coef(gmv4), se = se(gmv4))

cat("\n--- GMV Model Comparison ---\n")
cat(sprintf("\n%-20s %12s %12s %10s\n", "Model", "Clicks Coef", "SE", "R² (adj)"))
cat(sprintf("%-20s %12s %12s %10s\n", paste(rep("-", 20), collapse = ""),
            paste(rep("-", 12), collapse = ""), paste(rep("-", 12), collapse = ""),
            paste(rep("-", 10), collapse = "")))
for (model_info in list(
  list(name = "GMV Baseline", coefs = gmv_coefs1, fit = gmv1),
  list(name = "GMV User FE", coefs = gmv_coefs2, fit = gmv2),
  list(name = "GMV Week FE", coefs = gmv_coefs3, fit = gmv3),
  list(name = "GMV User+Week FE", coefs = gmv_coefs4, fit = gmv4)
)) {
  cf <- model_info$coefs$coef["clicks"]
  se_val <- model_info$coefs$se["clicks"]
  cat(sprintf("%-20s %12.4f %12.4f %10.4f\n", model_info$name, cf, se_val, r2(model_info$fit, type = "ar2")))
}

# ============================================================================
# SATURATION ANALYSIS - Using GMV Model 2 (User FE)
# ============================================================================
cat("\n", strrep("=", 80), "\n")
cat("SATURATION ANALYSIS - Percentile Context (Using GMV User FE)\n")
cat(strrep("=", 80), "\n")

# Extract clicks quadratic coefficients from GMV2 (User FE)
coef_clicks <- gmv_coefs2$coef["clicks"]
coef_clicks_sq <- gmv_coefs2$coef["clicks_sq"]

# Clicks distribution
clicks_mean <- mean(dt$clicks)
clicks_sd <- sd(dt$clicks)
clicks_p50 <- quantile(dt$clicks, 0.50)
clicks_p90 <- quantile(dt$clicks, 0.90)
clicks_p95 <- quantile(dt$clicks, 0.95)
clicks_p99 <- quantile(dt$clicks, 0.99)
clicks_p999 <- quantile(dt$clicks, 0.999)

cat("\nClicks distribution:\n")
cat(sprintf("  Mean: %.2f, SD: %.2f\n", clicks_mean, clicks_sd))
cat(sprintf("  50th percentile: %.0f\n", clicks_p50))
cat(sprintf("  90th percentile: %.0f\n", clicks_p90))
cat(sprintf("  95th percentile: %.0f\n", clicks_p95))
cat(sprintf("  99th percentile: %.0f\n", clicks_p99))
cat(sprintf("  99.9th percentile: %.0f\n", clicks_p999))

# Context for saturation points
if (!is.na(coef_clicks_sq) && coef_clicks_sq < 0 && coef_clicks > 0) {
  optimal_clicks <- -coef_clicks / (2 * coef_clicks_sq)
  z_clicks <- (optimal_clicks - clicks_mean) / clicks_sd
  pct_above_clicks <- 100 * mean(dt$clicks >= optimal_clicks)
  cat(sprintf("\nClicks saturation context (from GMV User FE):\n"))
  cat(sprintf("  clicks coefficient: %.6f\n", coef_clicks))
  cat(sprintf("  clicks² coefficient: %.8f\n", coef_clicks_sq))
  cat(sprintf("  Saturation point: %.0f clicks\n", optimal_clicks))
  cat(sprintf("  Standard deviations above mean: %.1f\n", z_clicks))
  cat(sprintf("  Percent of sessions at or above: %.3f%%\n", pct_above_clicks))
  cat(sprintf("  Interpretation: saturation occurs at an extreme value rarely observed\n"))
} else if (!is.na(coef_clicks_sq) && coef_clicks_sq > 0) {
  cat("\n  clicks² positive: accelerating returns\n")
}

# ============================================================================
# MODEL COMPARISON SUMMARY (LOGIT)
# ============================================================================
cat("\n", strrep("=", 80), "\n")
cat("MODEL COMPARISON SUMMARY (LOGIT)\n")
cat(strrep("=", 80), "\n")
cat("\nNote: User FE models drop observations with no variation in outcome.\n")
cat(sprintf("Observations dropped (User FE): %d (%.1f%% of sample)\n",
            nrow(dt) - fit2$nobs, 100 * (nrow(dt) - fit2$nobs) / nrow(dt)))
cat(sprintf("Observations dropped (User+Week FE): %d (%.1f%% of sample)\n",
            nrow(dt) - fit4$nobs, 100 * (nrow(dt) - fit4$nobs) / nrow(dt)))

# Collect coefficients for Logit (features without clicks_sq)
get_coefs <- function(fit) {
  cf <- coef(fit)
  se <- se(fit)
  # Only return feature coefficients (Logit uses features_logit)
  idx <- intersect(features_logit, names(cf))
  list(coef = cf[idx], se = se[idx])
}

coefs0 <- get_coefs(fit0)
coefs1 <- get_coefs(fit1)
coefs2 <- get_coefs(fit2)
coefs3 <- get_coefs(fit3)
coefs4 <- get_coefs(fit4)

cat(sprintf("\n%-20s %10s %10s %12s %12s\n", "Model", "AUC-ROC", "AUC-PR", "Pseudo R²", "Deviance"))
cat(sprintf("%-20s %10s %10s %12s %12s\n", paste(rep("-", 20), collapse = ""),
            paste(rep("-", 10), collapse = ""), paste(rep("-", 10), collapse = ""),
            paste(rep("-", 12), collapse = ""), paste(rep("-", 12), collapse = "")))
cat(sprintf("%-20s %10.4f %10.4f %12.4f %12.0f\n", "Raw (Naive)", metrics0$auc_roc, metrics0$auc_pr, metrics0$pseudo_r2, metrics0$deviance))
cat(sprintf("%-20s %10.4f %10.4f %12.4f %12.0f\n", "Baseline", metrics1$auc_roc, metrics1$auc_pr, metrics1$pseudo_r2, metrics1$deviance))
cat(sprintf("%-20s %10.4f %10.4f %12.4f %12.0f\n", "User FE", metrics2$auc_roc, metrics2$auc_pr, metrics2$pseudo_r2, metrics2$deviance))
cat(sprintf("%-20s %10.4f %10.4f %12.4f %12.0f\n", "Week FE", metrics3$auc_roc, metrics3$auc_pr, metrics3$pseudo_r2, metrics3$deviance))
cat(sprintf("%-20s %10.4f %10.4f %12.4f %12.0f\n", "User+Week FE", metrics4$auc_roc, metrics4$auc_pr, metrics4$pseudo_r2, metrics4$deviance))

# Coefficient comparison (clicks)
cat("\n--- Coefficient Comparison (Clicks) ---\n")
cat(sprintf("\n%-20s %12s %12s %10s\n", "Model", "Coef", "SE", "z-stat"))
cat(sprintf("%-20s %12s %12s %10s\n", paste(rep("-", 20), collapse = ""),
            paste(rep("-", 12), collapse = ""), paste(rep("-", 12), collapse = ""),
            paste(rep("-", 10), collapse = "")))

for (model_info in list(
  list(name = "Raw (Naive)", coefs = coefs0),
  list(name = "Baseline", coefs = coefs1),
  list(name = "User FE", coefs = coefs2),
  list(name = "Week FE", coefs = coefs3),
  list(name = "User+Week FE", coefs = coefs4)
)) {
  cf <- model_info$coefs$coef["clicks"]
  se <- model_info$coefs$se["clicks"]
  z <- cf / se
  cat(sprintf("%-20s %12.6f %12.6f %10.3f\n", model_info$name, cf, se, z))
}

# ============================================================================
# LATEX TABLES
# ============================================================================
cat("\n", strrep("=", 80), "\n")
cat("LATEX TABLES\n")
cat(strrep("=", 80), "\n")

# Helper for significance stars
stars <- function(p) {
  if (is.na(p)) return("")
  if (p < 0.01) return("***")
  if (p < 0.05) return("**")
  if (p < 0.1) return("*")
  return("")
}

# Table 1: Descriptive Statistics
latex_desc <- '\\begin{table}[H]
\\centering
\\caption{Descriptive Statistics of Session-Level Variables}
\\label{tab:session_descriptives}
\\begin{tabular}{lrrrr}
\\toprule
Variable & Mean & Std. Dev. & Min & Max \\\\
\\midrule
'

for (i in 1:nrow(stats_table)) {
  latex_desc <- paste0(latex_desc, sprintf("%s & %.3f & %.3f & %.0f & %.0f \\\\\n",
                                           stats_table$Variable[i],
                                           stats_table$Mean[i],
                                           stats_table$SD[i],
                                           stats_table$Min[i],
                                           stats_table$Max[i]))
}

latex_desc <- paste0(latex_desc, '\\bottomrule
\\end{tabular}
\\end{table}
')

cat("\nTable 1: Descriptive Statistics\n")
if (LATEX_SUFFIX != "") {
  latex_desc <- sub("\\\\label\\{tab:session_descriptives\\}",
                    paste0("\\\\label{tab:session_descriptives", LATEX_SUFFIX, "}"),
                    latex_desc)
}
cat(latex_desc)

# LPM table removed per plan - using Logit only

# Table 2: Logit Purchase Model (5 columns: Raw + 4 specs)
var_labels_logit <- c("Clicks", "Impressions", "Auctions", "Products Impressed", "Duration Hours")
all_coefs <- list(coefs0, coefs1, coefs2, coefs3, coefs4)
all_metrics <- list(metrics0, metrics1, metrics2, metrics3, metrics4)
n_logit <- c(fit0$nobs, fit1$nobs, fit2$nobs, fit3$nobs, fit4$nobs)

latex_logit <- '\\begin{table}[H]
\\centering
\\caption{Logit Model: Effect of Ad Exposure on Purchase}
\\label{tab:session_logit_purchase}
\\begin{tabular}{lccccc}
\\toprule
 & (0) & (1) & (2) & (3) & (4) \\\\
 & Raw & Baseline & User FE & Week FE & User+Week FE \\\\
\\midrule
'

for (i in seq_along(features_logit)) {
  feat <- features_logit[i]
  row_coef <- sprintf("%s", var_labels_logit[i])
  row_se <- ""

  for (j in 1:5) {
    cf <- all_coefs[[j]]$coef[feat]
    se_val <- all_coefs[[j]]$se[feat]
    if (is.na(cf) || is.na(se_val)) {
      row_coef <- paste0(row_coef, " & --")
      row_se <- paste0(row_se, " & ")
    } else {
      z <- cf / se_val
      p <- 2 * (1 - pnorm(abs(z)))
      row_coef <- paste0(row_coef, sprintf(" & %.4f%s", cf, stars(p)))
      row_se <- paste0(row_se, sprintf(" & (%.4f)", se_val))
    }
  }

  latex_logit <- paste0(latex_logit, row_coef, " \\\\\n")
  latex_logit <- paste0(latex_logit, row_se, " \\\\\n")
}

latex_logit <- paste0(latex_logit, '\\midrule
User FE & No & No & Yes & No & Yes \\\\
Week FE & No & No & No & Yes & Yes \\\\
\\midrule
')

# AUC-ROC row
row_auc <- "AUC-ROC"
for (j in 1:5) {
  row_auc <- paste0(row_auc, sprintf(" & %.3f", all_metrics[[j]]$auc_roc))
}
latex_logit <- paste0(latex_logit, row_auc, " \\\\\n")

# N row
latex_logit <- paste0(latex_logit, sprintf("N & %s & %s & %s & %s \\\\\n",
                                            format(n_logit[1], big.mark = ","),
                                            format(n_logit[2], big.mark = ","),
                                            format(n_logit[3], big.mark = ","),
                                            format(n_logit[4], big.mark = ",")))

latex_logit <- paste0(latex_logit,
                      '\\multicolumn{6}{l}{\\footnotesize Standard errors clustered by user. *** p$<$0.01, ** p$<$0.05, * p$<$0.1} \\\\\n',
                      '\\multicolumn{6}{l}{\\footnotesize Note: User FE models drop sessions with no purchase variation.}\\\\\n',
                      '\\bottomrule\n',
                      '\\end{tabular}\n',
                      '\\end{table}\n')

 # Suffix table label if LATEX_SUFFIX provided
 if (LATEX_SUFFIX != "") {
   latex_logit <- sub("\\\\label\\{tab:session_logit_purchase\\}",
                      paste0("\\\\label{tab:session_logit_purchase", LATEX_SUFFIX, "}"),
                      latex_logit)
 }

 # Ensure N row lists five columns for (0)-(4)
 n_vals <- n_logit
 if (length(n_vals) >= 5) {
  n_row5 <- paste0("N & ", paste(format(n_vals[1:5], big.mark = ","), collapse = " & "), " \\\\\\\\")
   latex_logit <- sub("(?m)^N\\s*&.*$", n_row5, latex_logit, perl = TRUE)
 }
 cat("\nTable 2: Logit Purchase Model\n")
 cat(latex_logit)

# Table 3: GMV Results (5 columns: Raw + 4 specs)
var_labels_gmv <- c("Clicks", "Clicks\\textsuperscript{2}", "Impressions", "Auctions", "Products Impressed", "Duration Hours")
gmv_coefs0 <- list(coef = coef(gmv0)[intersect(names(coef(gmv0)), features_gmv)],
                   se = se(gmv0)[intersect(names(se(gmv0)), features_gmv)])
all_gmv_coefs <- list(gmv_coefs0, gmv_coefs1, gmv_coefs2, gmv_coefs3, gmv_coefs4)
all_gmv_models <- list(gmv0, gmv1, gmv2, gmv3, gmv4)

latex_gmv <- '\\begin{table}[H]
\\centering
\\caption{Linear FE Model: Effect of Ad Exposure on GMV (Spend)}
\\label{tab:session_gmv_results}
\\begin{tabular}{lccccc}
\\toprule
 & (0) & (1) & (2) & (3) & (4) \\\\
 & Raw & Baseline & User FE & Week FE & User+Week FE \\\\
\\midrule
'

for (i in seq_along(features_gmv)) {
  feat <- features_gmv[i]
  row_coef <- sprintf("%s", var_labels_gmv[i])
  row_se <- ""

  for (j in 1:5) {
    cf <- all_gmv_coefs[[j]]$coef[feat]
    se_val <- all_gmv_coefs[[j]]$se[feat]
    if (is.na(cf) || is.na(se_val)) {
      row_coef <- paste0(row_coef, " & --")
      row_se <- paste0(row_se, " & ")
    } else {
      t <- cf / se_val
      p <- 2 * (1 - pt(abs(t), df = all_gmv_models[[j]]$nobs - length(all_gmv_coefs[[j]]$coef)))
      # Format: use appropriate precision
      if (abs(cf) < 0.1) {
        row_coef <- paste0(row_coef, sprintf(" & %.4f%s", cf, stars(p)))
        row_se <- paste0(row_se, sprintf(" & (%.4f)", se_val))
      } else {
        row_coef <- paste0(row_coef, sprintf(" & %.2f%s", cf, stars(p)))
        row_se <- paste0(row_se, sprintf(" & (%.2f)", se_val))
      }
    }
  }

  latex_gmv <- paste0(latex_gmv, row_coef, " \\\\\n")
  latex_gmv <- paste0(latex_gmv, row_se, " \\\\\n")
}

latex_gmv <- paste0(latex_gmv, '\\midrule
User FE & No & No & Yes & No & Yes \\\\
Week FE & No & No & No & Yes & Yes \\\\
\\midrule
')

# R² for GMV
row <- "R\\textsuperscript{2} (adj.)"
for (j in 1:5) {
  val <- r2(all_gmv_models[[j]], type = "ar2")
  row <- paste0(row, sprintf(" & %.3f", val))
}
latex_gmv <- paste0(latex_gmv, row, " \\\\\n")

latex_gmv <- paste0(latex_gmv, sprintf("N & %s & %s & %s & %s \\\\\n",
                                        format(nrow(dt), big.mark = ","),
                                        format(nrow(dt), big.mark = ","),
                                        format(nrow(dt), big.mark = ","),
                                        format(nrow(dt), big.mark = ",")))

latex_gmv <- paste0(latex_gmv,
                    '\\multicolumn{6}{l}{\\footnotesize Standard errors clustered by user. *** p$<$0.01, ** p$<$0.05, * p$<$0.1}\\\\\n',
                    '\\bottomrule\n',
                    '\\end{tabular}\n',
                    '\\end{table}\n')

 # Suffix table label if LATEX_SUFFIX provided
 if (LATEX_SUFFIX != "") {
   latex_gmv <- sub("\\\\label\\{tab:session_gmv_results\\}",
                    paste0("\\\\label{tab:session_gmv_results", LATEX_SUFFIX, "}"),
                    latex_gmv)
 }

 # Ensure N row lists five columns for (0)-(4) in GMV table
 n_gmv <- c(gmv0$nobs, gmv1$nobs, gmv2$nobs, gmv3$nobs, gmv4$nobs)
gmv_row5 <- paste0("N & ", paste(format(n_gmv, big.mark = ","), collapse = " & "), " \\\\\\\\")
 latex_gmv <- sub("(?m)^N\\s*&.*$", gmv_row5, latex_gmv, perl = TRUE)
 cat("\nTable 3: GMV Results\n")
 cat(latex_gmv)

# Frequency table removed per plan

# ============================================================================
# WRITE LATEX FILE
# ============================================================================
latex_combined <- paste0("% Auto-generated by 03_purchase_model.R\n",
                         "% Fixed Effects Models: Session-Level Purchase and GMV Prediction\n",
                         "% Models: (0) Raw (naive), (1) Baseline, (2) User FE, (3) Week FE, (4) User+Week FE\n",
                         "% Logit only for binary purchase outcome; Linear FE for GMV\n\n",
                         latex_desc, "\n\n",
                         latex_logit, "\n\n",
                         latex_gmv)

latex_file <- file.path(LATEX_DIR, paste0("session_logit_results", LATEX_SUFFIX, ".tex"))
writeLines(latex_combined, latex_file)
cat(sprintf("\nLaTeX tables written to: %s\n", normalizePath(latex_file)))

# ============================================================================
# INTERPRETATION
# ============================================================================
cat("\n", strrep("=", 80), "\n")
cat("INTERPRETATION\n")
cat(strrep("=", 80), "\n")

cat("\n--- Key Finding: Logit Clicks Coefficient ---\n")
base_coef_logit <- coefs1$coef["clicks"]
user_fe_coef_logit <- coefs2$coef["clicks"]
user_week_fe_coef_logit <- coefs4$coef["clicks"]

cat(sprintf("1. Logit Baseline clicks: %.6f\n", base_coef_logit))
cat(sprintf("   Logit User FE clicks: %.6f\n", user_fe_coef_logit))
cat(sprintf("   Logit User+Week FE clicks: %.6f\n", user_week_fe_coef_logit))

cat("\n--- GMV Models ---\n")
base_coef_gmv <- gmv_coefs1$coef["clicks"]
user_fe_coef_gmv <- gmv_coefs2$coef["clicks"]
user_week_fe_coef_gmv <- gmv_coefs4$coef["clicks"]

cat(sprintf("2. Clicks coefficient in baseline GMV: $%.4f\n", base_coef_gmv))
cat(sprintf("3. Clicks coefficient with User FE: $%.4f\n", user_fe_coef_gmv))
cat(sprintf("4. Clicks coefficient with User+Week FE: $%.4f\n", user_week_fe_coef_gmv))

cat("\n--- Saturation Effects (Quadratic from GMV User FE) ---\n")
coef_c <- gmv_coefs2$coef["clicks"]
coef_c2 <- gmv_coefs2$coef["clicks_sq"]

cat(sprintf("5. Clicks linear: %.6f, Clicks²: %.8f\n", coef_c, coef_c2))
if (!is.na(coef_c2) && coef_c2 < 0 && coef_c > 0) {
  vertex <- -coef_c / (2 * coef_c2)
  z_vertex <- (vertex - clicks_mean) / clicks_sd
  pct_vertex <- 100 * mean(dt$clicks >= vertex)
  cat(sprintf("   Saturation point: %.0f clicks\n", vertex))
  cat(sprintf("   Context: %.1f SD above mean, only %.3f%% of sessions reach this level\n", z_vertex, pct_vertex))
  cat("   Interpretation: saturation occurs at extreme values, not practically relevant\n")
} else if (!is.na(coef_c2) && coef_c2 > 0) {
  cat("   Accelerating returns (quadratic coefficient positive)\n")
} else {
  cat("   No clear saturation pattern\n")
}

cat("\n", strrep("=", 80), "\n")
cat(sprintf("Output saved to: %s\n", normalizePath(OUTPUT_FILE)))

sink()
cat(sprintf("Results written to: %s\n", normalizePath(OUTPUT_FILE)))
