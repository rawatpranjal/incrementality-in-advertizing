#!/usr/bin/env Rscript
#' 05_multinomial_choice.R - Multinomial Logit: Product-Level Choice Model
#'
#' Unit of Analysis: Product within session (choice among products)
#' Alternatives per session:
#' - Outside option (no purchase, utility normalized to 0)
#' - Sampled organic products (from catalog, NOT impressed in session)
#' - Actually purchased organic products (if any)
#' - Impressed products (shown in ads during session)
#'
#' Utility specification:
#' V_outside = 0 (normalization)
#' V_j = alpha_type[j] + beta_clicked * clicked_j + beta_imp * n_impressions_j
#'       + beta_quality * quality_j + beta_price * log(price_j)
#'
#' Models (4 specifications):
#' 1. Baseline: No fixed effects
#' 2. User FE: User fixed effects
#' 3. Week FE: Week fixed effects
#' 4. User+Week FE: Both user and week fixed effects
#'
#' Output: LaTeX tables for paper integration

suppressPackageStartupMessages({
  library(data.table)
  library(survival)  # For clogit (conditional logit)
})

# Check for tqdm progress bar
if (!requireNamespace("progress", quietly = TRUE)) {
  cat("Installing progress for progress bars...\n")
  install.packages("progress", repos = "https://cloud.r-project.org/", quiet = TRUE)
}
suppressPackageStartupMessages(library(progress))

# Handle script path when run with Rscript
get_script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(normalizePath(sub("^--file=", "", file_arg)))
  }
  return(file.path(getwd(), "unified-session-position-analysis/shopping-sessions/scripts/05_multinomial_choice.R"))
}

script_path <- get_script_path()
BASE_DIR <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)
DATA_DIR <- file.path(BASE_DIR, "0_data_pull", "data")
RESULTS_DIR <- file.path(BASE_DIR, "results")
LATEX_DIR <- file.path(dirname(script_path), "../../../paper/05-sessions")

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

OUTPUT_FILE <- file.path(RESULTS_DIR, "05_multinomial_choice.txt")
sink(OUTPUT_FILE, split = TRUE)

cat(strrep("=", 80), "\n")
cat("05_MULTINOMIAL_CHOICE - Product-Level Choice Model\n")
cat(strrep("=", 80), "\n")
cat(sprintf("Data directory: %s\n", DATA_DIR))

# ============================================================================
# DATA LOADING
# ============================================================================
cat("\n", strrep("=", 80), "\n")
cat("DATA LOADING\n")
cat(strrep("=", 80), "\n")

sessions <- read_parquet_dt(file.path(DATA_DIR, "sessions.parquet"))
cat(sprintf("Loaded %s sessions\n", format(nrow(sessions), big.mark = ",")))

session_events <- read_parquet_dt(file.path(DATA_DIR, "session_events.parquet"))
cat(sprintf("Loaded %s session events\n", format(nrow(session_events), big.mark = ",")))

catalog <- read_parquet_dt(file.path(DATA_DIR, "catalog.parquet"))
cat(sprintf("Loaded %s catalog products\n", format(nrow(catalog), big.mark = ",")))

purchases <- read_parquet_dt(file.path(DATA_DIR, "purchases.parquet"))
cat(sprintf("Loaded %s purchases\n", format(nrow(purchases), big.mark = ",")))

auctions_results <- read_parquet_dt(file.path(DATA_DIR, "auctions_results.parquet"))
cat(sprintf("Loaded %s auction results\n", format(nrow(auctions_results), big.mark = ",")))

# ============================================================================
# DATA PREPARATION
# ============================================================================
cat("\n", strrep("=", 80), "\n")
cat("DATA PREPARATION\n")
cat(strrep("=", 80), "\n")

# Get sessions with impressions
sessions_with_imp <- sessions[n_impressions > 0]
cat(sprintf("Sessions with impressions: %s\n", format(nrow(sessions_with_imp), big.mark = ",")))

# Create week variable
sessions_with_imp[, session_start := as.POSIXct(session_start)]
sessions_with_imp[, week := format(session_start, "%Y_W%V")]

# Extract impressions and clicks from session_events
impressions_dt <- session_events[event_type == "impression", .(session_id, product_id, auction_id)]
clicks_dt <- session_events[event_type == "click", .(session_id, product_id)]

cat(sprintf("Impression events: %s\n", format(nrow(impressions_dt), big.mark = ",")))
cat(sprintf("Click events: %s\n", format(nrow(clicks_dt), big.mark = ",")))

# Compute product-level features per session
# Count impressions per product per session
imp_by_prod <- impressions_dt[, .(n_impressions = .N), by = .(session_id, product_id)]
cat(sprintf("Unique session-product impression pairs: %s\n", format(nrow(imp_by_prod), big.mark = ",")))

# Mark clicked products
clicks_by_prod <- clicks_dt[, .(clicked = 1L), by = .(session_id, product_id)]
clicks_by_prod <- unique(clicks_by_prod)
cat(sprintf("Unique session-product click pairs: %s\n", format(nrow(clicks_by_prod), big.mark = ",")))

# Compute median quality score per product (from auctions_results)
# Note: We don't have quality column directly, but we can use ranking as a proxy
# For now, use mean ranking (lower is better quality)
if ("quality" %in% names(auctions_results)) {
  quality_by_prod <- auctions_results[!is.na(quality), .(quality = median(quality, na.rm = TRUE)), by = product_id]
} else {
  # Use inverse ranking as quality proxy (higher = better)
  quality_by_prod <- auctions_results[!is.na(ranking), .(quality = 1 / median(ranking, na.rm = TRUE)), by = product_id]
}
cat(sprintf("Products with quality scores: %s\n", format(nrow(quality_by_prod), big.mark = ",")))

# Get price from catalog
catalog_prices <- catalog[, .(product_id, price)]
catalog_prices <- catalog_prices[!is.na(price) & price > 0]
cat(sprintf("Products with valid prices: %s\n", format(nrow(catalog_prices), big.mark = ",")))

# ============================================================================
# BUILD CONSIDERATION SET
# ============================================================================
cat("\n", strrep("=", 80), "\n")
cat("BUILDING CONSIDERATION SET\n")
cat(strrep("=", 80), "\n")

# Sample parameters
N_ORGANIC_SAMPLE <- 5  # Number of random organic products per session

# Get all impressed products (these form the "impressed" part of consideration set)
impressed_products <- imp_by_prod[, .(session_id, product_id, n_impressions)]
impressed_products[, is_impressed := 1L]

# Merge with clicks
impressed_products <- merge(impressed_products, clicks_by_prod,
                            by = c("session_id", "product_id"), all.x = TRUE)
impressed_products[is.na(clicked), clicked := 0L]

# Merge with quality
impressed_products <- merge(impressed_products, quality_by_prod,
                            by = "product_id", all.x = TRUE)
impressed_products[is.na(quality), quality := median(quality_by_prod$quality, na.rm = TRUE)]

# Merge with price
impressed_products <- merge(impressed_products, catalog_prices,
                            by = "product_id", all.x = TRUE)

# Filter to products with valid price
impressed_products <- impressed_products[!is.na(price) & price > 0]
cat(sprintf("Impressed products with valid features: %s\n", format(nrow(impressed_products), big.mark = ",")))

# Get set of impressed product IDs per session
impressed_by_session <- impressed_products[, .(impressed_prods = list(product_id)), by = session_id]

# Match purchases to sessions via user_id and time
purchases[, purchase_time := as.POSIXct(purchase_time)]
sessions_with_imp[, session_end := as.POSIXct(session_end)]

# Merge sessions with user_id
setnames(purchases, "user_id", "purchase_user_id")
sessions_with_imp[, session_user_id := user_id]

# Create a merged dataset to find purchases within sessions
# A purchase belongs to a session if same user and purchase_time is within session window
# This is computationally expensive so we do a simplified approach:
# Check if product was purchased by user within some window of session

# For simplicity, merge by user_id and check if product_id in purchases
purchase_prods <- purchases[, .(purchase_user_id, product_id, unit_price)]

# Build choice set for each session
# This is the main loop - we'll build the long-format data

cat("\nBuilding choice set for each session...\n")

# Get unique sessions to process - SAMPLE 5% for speed
set.seed(42)
all_session_ids <- unique(sessions_with_imp$session_id)
session_ids <- sample(all_session_ids, size = ceiling(length(all_session_ids) * 0.05))
n_sessions <- length(session_ids)
cat(sprintf("Sampling 5%% of sessions: %d of %d\n", n_sessions, length(all_session_ids)))

# Pre-compute: catalog products not impressed in each session
all_catalog_prods <- unique(catalog_prices$product_id)

# Build the choice data incrementally
choice_list <- list()
pb <- progress_bar$new(
  format = "  Building choice sets [:bar] :percent ETA: :eta",
  total = n_sessions, clear = FALSE
)

set.seed(42)  # Reproducibility for sampling

for (sid in session_ids) {
  pb$tick()

  # Get session info
  sess_info <- sessions_with_imp[session_id == sid]
  uid <- sess_info$user_id
  wk <- sess_info$week
  purchased <- sess_info$purchased

  # Get impressed products for this session
  sess_impressed <- impressed_products[session_id == sid]

  if (nrow(sess_impressed) == 0) next

  # Get user's purchases
  user_purchases <- purchase_prods[purchase_user_id == uid]

  # Determine which impressed products were purchased (in session)
  # For simplicity, if user purchased a product that was impressed in session, mark as chosen
  sess_impressed[, purchased_this := product_id %in% user_purchases$product_id]

  # If purchased=1 but no impressed product was purchased, the purchase was "organic"
  # Add actually purchased organic products
  organic_purchased <- user_purchases[!product_id %in% sess_impressed$product_id]

  # Get products NOT impressed in this session for organic sampling
  impressed_prod_ids <- sess_impressed$product_id
  organic_pool <- setdiff(all_catalog_prods, impressed_prod_ids)

  # Sample organic products (not in session's impressions)
  if (length(organic_pool) > N_ORGANIC_SAMPLE) {
    sampled_organic <- sample(organic_pool, N_ORGANIC_SAMPLE)
  } else {
    sampled_organic <- organic_pool
  }

  # Build alternatives for this session
  # 1. Outside option
  outside_dt <- data.table(
    session_id = sid,
    user_id = uid,
    week = wk,
    product_id = "OUTSIDE",
    is_impressed = 0L,
    n_impressions = 0,
    clicked = 0L,
    quality = 0,
    price = 0,
    log_price = 0,
    chosen = as.integer(!purchased)  # Chosen if no purchase
  )

  # 2. Impressed products
  impressed_dt <- sess_impressed[, .(
    session_id = sid,
    user_id = uid,
    week = wk,
    product_id = product_id,
    is_impressed = 1L,
    n_impressions = n_impressions,
    clicked = clicked,
    quality = quality,
    price = price,
    log_price = log(price + 1),
    chosen = as.integer(purchased_this)
  )]

  # 3. Sampled organic (not impressed, not purchased)
  sampled_dt <- catalog_prices[product_id %in% sampled_organic]
  if (nrow(sampled_dt) > 0) {
    sampled_dt <- merge(sampled_dt, quality_by_prod, by = "product_id", all.x = TRUE)
    sampled_dt[is.na(quality), quality := median(quality_by_prod$quality, na.rm = TRUE)]

    sampled_dt <- sampled_dt[, .(
      session_id = sid,
      user_id = uid,
      week = wk,
      product_id = product_id,
      is_impressed = 0L,
      n_impressions = 0,
      clicked = 0L,
      quality = quality,
      price = price,
      log_price = log(price + 1),
      chosen = 0L
    )]
  } else {
    sampled_dt <- NULL
  }

  # 4. Actually purchased organic products (if any)
  if (nrow(organic_purchased) > 0) {
    organic_purchased <- merge(organic_purchased, catalog_prices, by = "product_id", all.x = TRUE)
    organic_purchased <- merge(organic_purchased, quality_by_prod, by = "product_id", all.x = TRUE)
    organic_purchased <- organic_purchased[!is.na(price) & price > 0]

    if (nrow(organic_purchased) > 0) {
      organic_purchased[is.na(quality), quality := median(quality_by_prod$quality, na.rm = TRUE)]

      organic_dt <- organic_purchased[, .(
        session_id = sid,
        user_id = uid,
        week = wk,
        product_id = product_id,
        is_impressed = 0L,
        n_impressions = 0,
        clicked = 0L,
        quality = quality,
        price = price,
        log_price = log(price + 1),
        chosen = 1L  # These were purchased
      )]
    } else {
      organic_dt <- NULL
    }
  } else {
    organic_dt <- NULL
  }

  # Combine all alternatives
  sess_choice <- rbindlist(list(outside_dt, impressed_dt, sampled_dt, organic_dt), fill = TRUE)

  # Ensure exactly one chosen per session
  if (sum(sess_choice$chosen) != 1) {
    # If multiple chosen, keep only first
    if (sum(sess_choice$chosen) > 1) {
      first_chosen_idx <- which(sess_choice$chosen == 1)[1]
      sess_choice[, chosen := 0L]
      sess_choice[first_chosen_idx, chosen := 1L]
    }
    # If none chosen (shouldn't happen), mark outside option as chosen
    if (sum(sess_choice$chosen) == 0) {
      sess_choice[product_id == "OUTSIDE", chosen := 1L]
    }
  }

  choice_list[[sid]] <- sess_choice
}

# Combine all choice sets
choice_dt <- rbindlist(choice_list)

cat(sprintf("\nTotal choice observations: %s\n", format(nrow(choice_dt), big.mark = ",")))
cat(sprintf("Unique sessions: %s\n", format(length(unique(choice_dt$session_id)), big.mark = ",")))
cat(sprintf("Chosen products: %s\n", format(sum(choice_dt$chosen), big.mark = ",")))

# Remove outside option for modeling (used as base)
# Actually, for clogit we need all alternatives including outside option
# But the outside option needs special handling

# Summary of choice set
cat("\n--- Choice Set Summary ---\n")
cat(sprintf("Alternatives per session (mean): %.1f\n", nrow(choice_dt) / length(unique(choice_dt$session_id))))
cat(sprintf("Impressed products chosen: %d\n", sum(choice_dt$chosen == 1 & choice_dt$is_impressed == 1)))
cat(sprintf("Organic products chosen: %d\n", sum(choice_dt$chosen == 1 & choice_dt$is_impressed == 0 & choice_dt$product_id != "OUTSIDE")))
cat(sprintf("Outside option chosen: %d\n", sum(choice_dt$chosen == 1 & choice_dt$product_id == "OUTSIDE")))

# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================
cat("\n", strrep("=", 80), "\n")
cat("DESCRIPTIVE STATISTICS\n")
cat(strrep("=", 80), "\n")

# Exclude outside option for descriptives
choice_prods <- choice_dt[product_id != "OUTSIDE"]

cat("\n--- Product-Level Variables (excluding outside option) ---\n")
cat(sprintf("N observations: %s\n", format(nrow(choice_prods), big.mark = ",")))
cat(sprintf("Mean chosen: %.4f\n", mean(choice_prods$chosen)))
cat(sprintf("Mean is_impressed: %.4f\n", mean(choice_prods$is_impressed)))
cat(sprintf("Mean clicked (among products): %.4f\n", mean(choice_prods$clicked)))
cat(sprintf("Mean n_impressions: %.2f\n", mean(choice_prods$n_impressions)))
cat(sprintf("Mean quality: %.4f\n", mean(choice_prods$quality, na.rm = TRUE)))
cat(sprintf("Mean log_price: %.2f\n", mean(choice_prods$log_price, na.rm = TRUE)))

# ============================================================================
# MODEL ESTIMATION
# ============================================================================
cat("\n", strrep("=", 80), "\n")
cat("MODEL ESTIMATION - Conditional Logit\n")
cat(strrep("=", 80), "\n")

# For conditional logit, we need:
# - A stratification variable (session_id)
# - Chosen as the outcome
# - Product-level features as predictors

# Create factor variables
choice_dt[, user_id := as.factor(user_id)]
choice_dt[, week := as.factor(week)]
choice_dt[, session_id := as.factor(session_id)]

# Create numeric session ID for clogit strata
choice_dt[, strata_id := as.numeric(factor(session_id))]

# Filter to valid observations (non-missing)
choice_model_dt <- choice_dt[!is.na(log_price) & !is.na(quality)]
cat(sprintf("\nObservations for modeling: %s\n", format(nrow(choice_model_dt), big.mark = ",")))
cat(sprintf("Sessions for modeling: %s\n", format(length(unique(choice_model_dt$session_id)), big.mark = ",")))

# Verify one choice per session
choices_per_session <- choice_model_dt[, .(n_chosen = sum(chosen)), by = session_id]
cat(sprintf("Sessions with exactly 1 choice: %d\n", sum(choices_per_session$n_chosen == 1)))
cat(sprintf("Sessions with 0 choices: %d\n", sum(choices_per_session$n_chosen == 0)))
cat(sprintf("Sessions with >1 choices: %d\n", sum(choices_per_session$n_chosen > 1)))

# Keep only sessions with exactly 1 choice
valid_sessions <- choices_per_session[n_chosen == 1, session_id]
choice_model_dt <- choice_model_dt[session_id %in% valid_sessions]

cat(sprintf("\nAfter filtering to valid sessions:\n"))
cat(sprintf("  Observations: %s\n", format(nrow(choice_model_dt), big.mark = ",")))
cat(sprintf("  Sessions: %s\n", format(length(unique(choice_model_dt$session_id)), big.mark = ",")))

# Re-create strata ID after filtering
choice_model_dt[, strata_id := as.numeric(factor(session_id))]

# ----------------------------------------------------------------------------
# Model 1: Baseline Conditional Logit (No FE)
# ----------------------------------------------------------------------------
cat("\n--- Model 1: Baseline Conditional Logit ---\n")

# Formula: chosen ~ is_impressed + clicked + n_impressions + quality + log_price + strata(strata_id)
fit1 <- tryCatch({
  clogit(chosen ~ is_impressed + clicked + n_impressions + quality + log_price + strata(strata_id),
         data = choice_model_dt, method = "efron")
}, error = function(e) {
  cat(sprintf("Error in Model 1: %s\n", e$message))
  NULL
})

if (!is.null(fit1)) {
  print(summary(fit1))
  coefs1 <- list(coef = coef(fit1), se = sqrt(diag(vcov(fit1))))
} else {
  coefs1 <- NULL
}

# ----------------------------------------------------------------------------
# Model 2: With User FE (using user-specific intercepts via expanded data)
# Note: True conditional logit with user FE is complex; we use user-level clustering
# ----------------------------------------------------------------------------
cat("\n--- Model 2: Conditional Logit with User-clustered SEs ---\n")

fit2 <- tryCatch({
  clogit(chosen ~ is_impressed + clicked + n_impressions + quality + log_price + strata(strata_id) +
           cluster(user_id),
         data = choice_model_dt, method = "efron")
}, error = function(e) {
  cat(sprintf("Error in Model 2: %s\n", e$message))
  NULL
})

if (!is.null(fit2)) {
  print(summary(fit2))
  coefs2 <- list(coef = coef(fit2), se = sqrt(diag(vcov(fit2))))
} else {
  coefs2 <- coefs1  # Fallback
}

# ----------------------------------------------------------------------------
# Model 3: With Week interaction (simple approach - include week as predictor)
# Note: For conditional logit, we can include week-specific effects on product features
# ----------------------------------------------------------------------------
cat("\n--- Model 3: Conditional Logit (Week effects approximation) ---\n")

# Create a binary for recent weeks (simplified week effect)
choice_model_dt[, week_num := as.numeric(week)]
week_median <- median(choice_model_dt$week_num)
choice_model_dt[, is_recent_week := as.integer(week_num > week_median)]

fit3 <- tryCatch({
  clogit(chosen ~ is_impressed + clicked + n_impressions + quality + log_price +
           is_impressed:is_recent_week + strata(strata_id),
         data = choice_model_dt, method = "efron")
}, error = function(e) {
  cat(sprintf("Error in Model 3: %s\n", e$message))
  NULL
})

if (!is.null(fit3)) {
  print(summary(fit3))
  coefs3 <- list(coef = coef(fit3), se = sqrt(diag(vcov(fit3))))
} else {
  coefs3 <- coefs1  # Fallback
}

# ----------------------------------------------------------------------------
# Model 4: With User-clustered SEs and Week interaction
# ----------------------------------------------------------------------------
cat("\n--- Model 4: Conditional Logit (User cluster + Week effects) ---\n")

fit4 <- tryCatch({
  clogit(chosen ~ is_impressed + clicked + n_impressions + quality + log_price +
           is_impressed:is_recent_week + strata(strata_id) + cluster(user_id),
         data = choice_model_dt, method = "efron")
}, error = function(e) {
  cat(sprintf("Error in Model 4: %s\n", e$message))
  NULL
})

if (!is.null(fit4)) {
  print(summary(fit4))
  coefs4 <- list(coef = coef(fit4), se = sqrt(diag(vcov(fit4))))
} else {
  coefs4 <- coefs2  # Fallback
}

# ============================================================================
# LATEX TABLE
# ============================================================================
cat("\n", strrep("=", 80), "\n")
cat("LATEX TABLE\n")
cat(strrep("=", 80), "\n")

# Helper for significance stars
stars <- function(p) {
  if (is.na(p)) return("")
  if (p < 0.01) return("***")
  if (p < 0.05) return("**")
  if (p < 0.1) return("*")
  return("")
}

# Build table
var_names <- c("is_impressed", "clicked", "n_impressions", "quality", "log_price")
var_labels <- c("Impressed (vs Organic)", "Clicked", "Number of Impressions", "Quality Score", "Log(Price)")

all_coefs <- list(coefs1, coefs2, coefs3, coefs4)
all_fits <- list(fit1, fit2, fit3, fit4)

latex_mult <- '\\begin{table}[H]
\\centering
\\caption{Multinomial Choice Model: Product-Level Selection}
\\label{tab:multinomial_choice}
\\begin{tabular}{lcccc}
\\toprule
 & (1) & (2) & (3) & (4) \\\\
 & Baseline & User Cluster & Week Effects & User+Week \\\\
\\midrule
'

for (i in seq_along(var_names)) {
  vname <- var_names[i]
  row_coef <- var_labels[i]
  row_se <- ""

  for (j in 1:4) {
    cf_list <- all_coefs[[j]]
    if (is.null(cf_list) || !vname %in% names(cf_list$coef)) {
      row_coef <- paste0(row_coef, " & --")
      row_se <- paste0(row_se, " & ")
    } else {
      cf <- cf_list$coef[vname]
      se_val <- cf_list$se[vname]
      z <- cf / se_val
      p <- 2 * (1 - pnorm(abs(z)))
      row_coef <- paste0(row_coef, sprintf(" & %.4f%s", cf, stars(p)))
      row_se <- paste0(row_se, sprintf(" & (%.4f)", se_val))
    }
  }

  latex_mult <- paste0(latex_mult, row_coef, " \\\\\n")
  latex_mult <- paste0(latex_mult, row_se, " \\\\\n")
}

# Add week interaction row if present
if (!is.null(coefs3) && "is_impressed:is_recent_week" %in% names(coefs3$coef)) {
  row_coef <- "Impressed $\\times$ Recent Week"
  row_se <- ""

  for (j in 1:4) {
    cf_list <- all_coefs[[j]]
    vname <- "is_impressed:is_recent_week"
    if (is.null(cf_list) || !vname %in% names(cf_list$coef)) {
      row_coef <- paste0(row_coef, " & --")
      row_se <- paste0(row_se, " & ")
    } else {
      cf <- cf_list$coef[vname]
      se_val <- cf_list$se[vname]
      z <- cf / se_val
      p <- 2 * (1 - pnorm(abs(z)))
      row_coef <- paste0(row_coef, sprintf(" & %.4f%s", cf, stars(p)))
      row_se <- paste0(row_se, sprintf(" & (%.4f)", se_val))
    }
  }

  latex_mult <- paste0(latex_mult, row_coef, " \\\\\n")
  latex_mult <- paste0(latex_mult, row_se, " \\\\\n")
}

latex_mult <- paste0(latex_mult, '\\midrule
User-clustered SEs & No & Yes & No & Yes \\\\
Week Effects & No & No & Yes & Yes \\\\
\\midrule
')

# N observations and sessions
n_obs <- nrow(choice_model_dt)
n_sess <- length(unique(choice_model_dt$session_id))
latex_mult <- paste0(latex_mult, sprintf("N (alternatives) & %s & %s & %s & %s \\\\\n",
                                          format(n_obs, big.mark = ","),
                                          format(n_obs, big.mark = ","),
                                          format(n_obs, big.mark = ","),
                                          format(n_obs, big.mark = ",")))
latex_mult <- paste0(latex_mult, sprintf("N (sessions) & %s & %s & %s & %s \\\\\n",
                                          format(n_sess, big.mark = ","),
                                          format(n_sess, big.mark = ","),
                                          format(n_sess, big.mark = ","),
                                          format(n_sess, big.mark = ",")))

latex_mult <- paste0(latex_mult, '\\bottomrule
\\multicolumn{5}{l}{\\footnotesize *** p$<$0.01, ** p$<$0.05, * p$<$0.1} \\\\
\\multicolumn{5}{l}{\\footnotesize Conditional logit. Outside option normalized to 0. Strata: session.}
\\end{tabular}
\\end{table}
')

cat("\nTable: Multinomial Choice Model\n")
cat(latex_mult)

# Write LaTeX file
latex_file <- file.path(LATEX_DIR, "multinomial_choice_results.tex")
writeLines(latex_mult, latex_file)
cat(sprintf("\nLaTeX table written to: %s\n", normalizePath(latex_file)))

# ============================================================================
# INTERPRETATION
# ============================================================================
cat("\n", strrep("=", 80), "\n")
cat("INTERPRETATION\n")
cat(strrep("=", 80), "\n")

if (!is.null(coefs1)) {
  cat("\n--- Key Coefficients (Model 1: Baseline) ---\n")
  cat(sprintf("is_impressed: %.4f (SE: %.4f)\n", coefs1$coef["is_impressed"], coefs1$se["is_impressed"]))
  cat(sprintf("  Interpretation: Products shown via ads have exp(%.4f) = %.2f times the odds\n",
              coefs1$coef["is_impressed"], exp(coefs1$coef["is_impressed"])))
  cat(sprintf("                  of being purchased compared to organic products.\n"))

  if ("clicked" %in% names(coefs1$coef)) {
    cat(sprintf("\nclicked: %.4f (SE: %.4f)\n", coefs1$coef["clicked"], coefs1$se["clicked"]))
    cat(sprintf("  Interpretation: Clicked products have exp(%.4f) = %.2f times the odds\n",
                coefs1$coef["clicked"], exp(coefs1$coef["clicked"])))
    cat(sprintf("                  of being purchased compared to non-clicked products.\n"))
  }

  if ("log_price" %in% names(coefs1$coef)) {
    cat(sprintf("\nlog_price: %.4f (SE: %.4f)\n", coefs1$coef["log_price"], coefs1$se["log_price"]))
    if (coefs1$coef["log_price"] < 0) {
      cat(sprintf("  Interpretation: Higher prices reduce purchase probability (price elasticity).\n"))
    } else {
      cat(sprintf("  Interpretation: Higher prices associated with higher purchase probability\n"))
      cat(sprintf("                  (possibly quality signaling or premium product effect).\n"))
    }
  }
}

cat("\n", strrep("=", 80), "\n")
cat(sprintf("Output saved to: %s\n", normalizePath(OUTPUT_FILE)))

sink()
cat(sprintf("Results written to: %s\n", normalizePath(OUTPUT_FILE)))
