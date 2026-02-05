#!/usr/bin/env Rscript
# Near-Tie RD Model: Pair-clustered LPM with local linear trend
# Model: clicked ~ lucky + z | pair_id, cluster = AUCTION_ID
suppressPackageStartupMessages({
  library(data.table)
  library(fixest)
})

# ==============================================================================
# Utilities
# ==============================================================================
read_parquet_dt <- function(path, cols = NULL) {
  if (requireNamespace("duckdb", quietly = TRUE) && requireNamespace("DBI", quietly = TRUE)) {
    con <- DBI::dbConnect(duckdb::duckdb(), dbdir = tempfile())
    on.exit(DBI::dbDisconnect(con, shutdown = TRUE), add = TRUE)
    tbl <- sprintf("read_parquet('%s')", normalizePath(path))
    sql <- if (is.null(cols)) sprintf("SELECT * FROM %s", tbl)
           else sprintf("SELECT %s FROM %s", paste(cols, collapse = ","), tbl)
    return(as.data.table(DBI::dbGetQuery(con, sql)))
  }
  if (requireNamespace("arrow", quietly = TRUE)) {
    return(as.data.table(arrow::read_parquet(path, as_data_frame = TRUE, col_select = cols)))
  }
  stop("Neither duckdb nor arrow is available.")
}

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  hit <- which(args == flag)
  if (length(hit) == 0) return(default)
  if (hit == length(args)) return(TRUE)
  val <- args[hit + 1]
  if (startsWith(val, "--")) return(TRUE) else return(val)
}

round_name <- get_arg("--round", "round2")
placements <- as.character(strsplit(get_arg("--placements", "1,2,3,5"), ",")[[1]])
boundaries <- as.integer(strsplit(get_arg("--boundaries", "2,4,6"), ",")[[1]])
tau_windows <- as.numeric(strsplit(get_arg("--tau_windows", "0.01,0.02,0.05"), ",")[[1]])

BASE <- normalizePath("analysis/position-effects/0_data", mustWork = TRUE)
paths <- list(
  auctions_results = file.path(BASE, if (round_name == "round1") "round1/auctions_results_all.parquet" else "round2/auctions_results_r2.parquet"),
  impressions      = file.path(BASE, if (round_name == "round1") "round1/impressions_all.parquet" else "round2/impressions_r2.parquet"),
  clicks           = file.path(BASE, if (round_name == "round1") "round1/clicks_all.parquet" else "round2/clicks_r2.parquet"),
  auctions_users   = file.path(BASE, if (round_name == "round1") "round1/auctions_users_all.parquet" else "round2/auctions_users_r2.parquet")
)

out_dir <- file.path("analysis", "position-effects-analysis-R", "results")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
outfile <- file.path(out_dir, sprintf("uber_near_tie_rdd_model_%s.txt", round_name))

sink(outfile, split = TRUE)

cat("Near-Tie RD Model (Uber-Inspired)\n")
cat(sprintf("Round: %s\n", round_name))
cat(sprintf("Placements: %s\n", paste(placements, collapse = ", ")))
cat(sprintf("Boundaries: %s\n", paste(boundaries, collapse = ", ")))
cat(sprintf("Tau windows: %s\n\n", paste(tau_windows, collapse = ", ")))

# ==============================================================================
# Load Data
# ==============================================================================
cat("Loading data...\n")
ar <- read_parquet_dt(paths$auctions_results, c("AUCTION_ID", "PRODUCT_ID", "VENDOR_ID", "RANKING", "QUALITY", "FINAL_BID"))
au <- read_parquet_dt(paths$auctions_users, c("AUCTION_ID", "PLACEMENT", "CREATED_AT"))
imps <- read_parquet_dt(paths$impressions, c("AUCTION_ID", "PRODUCT_ID", "VENDOR_ID"))
clks <- read_parquet_dt(paths$clicks, c("AUCTION_ID", "PRODUCT_ID", "VENDOR_ID"))

au <- unique(au, by = "AUCTION_ID")
au[, PLACEMENT := as.character(PLACEMENT)]

ar <- ar[!is.na(QUALITY) & !is.na(FINAL_BID) & !is.na(RANKING)]
ar <- ar[(QUALITY > 0) & (FINAL_BID > 0) & (RANKING >= 1)]
ar[, score := as.numeric(QUALITY) * as.numeric(FINAL_BID)]
ar <- merge(ar, au[, .(AUCTION_ID, PLACEMENT)], by = "AUCTION_ID", all.x = FALSE)

# First impression flag
first_imp <- unique(imps[, .(AUCTION_ID, PRODUCT_ID, VENDOR_ID, impressed = 1L)])
# Click flag
click_flag <- unique(clks[, .(AUCTION_ID, PRODUCT_ID, VENDOR_ID, clicked = 1L)])

# ==============================================================================
# Build Pairs Function
# ==============================================================================
build_pairs <- function(ar_sub, boundaries) {
  setorder(ar_sub, AUCTION_ID, -score)
  ar_sub[, pos_by_score := frank(-score, ties.method = "first"), by = AUCTION_ID]

  keep_pos <- unique(sort(c(boundaries, boundaries + 1)))
  ar_small <- ar_sub[pos_by_score %in% keep_pos]

  pairs_list <- list()
  for (b in boundaries) {
    tmp <- ar_small[pos_by_score %in% c(b, b + 1)]
    tmp[, boundary := as.integer(b)]
    tmp <- tmp[, .SD[.N >= 2][1:2], by = .(AUCTION_ID, boundary)]
    tmp <- tmp[, if (.N == 2) .SD, by = .(AUCTION_ID, boundary)]
    pairs_list[[as.character(b)]] <- tmp
  }
  pairs_dt <- rbindlist(pairs_list, use.names = TRUE, fill = TRUE)
  if (nrow(pairs_dt) == 0) return(NULL)

  setorder(pairs_dt, AUCTION_ID, boundary, -score)
  pairs_dt[, pair_id := .GRP, by = .(AUCTION_ID, boundary)]
  pairs_dt[, score_hi := max(score), by = pair_id]
  pairs_dt[, score_lo := min(score), by = pair_id]
  pairs_dt[, rel_gap := (score_hi - score_lo) / score_hi]
  pairs_dt[, z := score - score_lo]  # running variable (distance from threshold)
  pairs_dt[, lucky := as.integer(score == score_hi)]
  pairs_dt
}

# ==============================================================================
# Model Fitting
# ==============================================================================
results_table <- data.table()

for (pl in placements) {
  cat(sprintf("\n========== PLACEMENT %s ==========\n", pl))

  ar_pl <- ar[PLACEMENT == pl]
  if (nrow(ar_pl) == 0) {
    cat("No data for this placement.\n")
    next
  }

  pairs_dt <- build_pairs(ar_pl, boundaries)
  if (is.null(pairs_dt) || nrow(pairs_dt) == 0) {
    cat("No pairs constructed.\n")
    next
  }

  # Attach impressions and clicks
  setkey(pairs_dt, AUCTION_ID, PRODUCT_ID, VENDOR_ID)
  setkey(first_imp, AUCTION_ID, PRODUCT_ID, VENDOR_ID)
  setkey(click_flag, AUCTION_ID, PRODUCT_ID, VENDOR_ID)
  pairs_dt <- first_imp[pairs_dt]
  pairs_dt <- click_flag[pairs_dt]
  pairs_dt[is.na(clicked), clicked := 0L]
  pairs_dt[is.na(impressed), impressed := 0L]

  for (tau in tau_windows) {
    for (b in boundaries) {
      sub <- pairs_dt[boundary == b & rel_gap <= tau]
      if (nrow(sub) < 10) {
        cat(sprintf("  tau=%.3f boundary=%d: insufficient data (n=%d)\n", tau, b, nrow(sub)))
        next
      }

      # Model 1: LPM for clicked ~ lucky | pair_id (cluster=AUCTION_ID)
      # Only among both-impressed pairs
      both_imp <- sub[, .(n_imp = sum(impressed)), by = pair_id][n_imp == 2, pair_id]
      sub_both <- sub[pair_id %in% both_imp]

      if (nrow(sub_both) < 10) {
        cat(sprintf("  tau=%.3f boundary=%d: insufficient both-impressed pairs (n=%d)\n", tau, b, nrow(sub_both)))
        next
      }

      # Model: clicked ~ lucky | pair_id
      fit <- tryCatch({
        feols(clicked ~ lucky | pair_id, data = sub_both, cluster = ~ AUCTION_ID)
      }, error = function(e) NULL)

      if (!is.null(fit)) {
        coef_lucky <- coef(fit)["lucky"]
        se_lucky <- sqrt(vcov(fit)["lucky", "lucky"])
        pval <- 2 * pnorm(-abs(coef_lucky / se_lucky))

        results_table <- rbind(results_table, data.table(
          placement = pl, boundary = b, tau = tau,
          N_pairs = length(both_imp), N_obs = nrow(sub_both),
          beta_lucky = round(coef_lucky, 5), SE = round(se_lucky, 5),
          p_value = round(pval, 4)
        ))

        cat(sprintf("  tau=%.3f boundary=%d: beta=%.4f (SE=%.4f) p=%.4f n=%d pairs\n",
                    tau, b, coef_lucky, se_lucky, pval, length(both_imp)))
      }

      # Fuzzy IV decomposition: First stage (lucky -> impressed), Reduced form (lucky -> clicked)
      # IV estimate: LATE = reduced_form / first_stage
      sub_full <- pairs_dt[boundary == b & rel_gap <= tau]
      if (nrow(sub_full) >= 20) {
        # First stage: P(impressed | lucky) - P(impressed | unlucky)
        first_stage <- mean(sub_full[lucky == 1, impressed]) - mean(sub_full[lucky == 0, impressed])
        # Reduced form: P(clicked | lucky) - P(clicked | unlucky)
        reduced_form <- mean(sub_full[lucky == 1, clicked]) - mean(sub_full[lucky == 0, clicked])
        # LATE
        late <- if (abs(first_stage) > 0.01) reduced_form / first_stage else NA_real_

        cat(sprintf("    Fuzzy IV: first_stage=%.4f reduced_form=%.4f LATE=%.4f\n",
                    first_stage, reduced_form, if (is.na(late)) NA_real_ else late))
      }
    }
  }
}

# ==============================================================================
# Summary Table
# ==============================================================================
cat("\n", paste(rep("=", 70), collapse = ""), "\n", sep = "")
cat("SUMMARY TABLE: Pair-clustered LPM Results\n")
cat(paste(rep("=", 70), collapse = ""), "\n\n", sep = "")

if (nrow(results_table) > 0) {
  print(results_table)
} else {
  cat("No models fit successfully.\n")
}

cat("\nModel: clicked ~ lucky | pair_id, cluster = AUCTION_ID\n")
cat("Interpretation: beta_lucky is the effect of being the higher-score bid on click probability,\n")
cat("conditional on both items in the pair being impressed.\n")

cat(sprintf("\nOutput saved to: %s\n", outfile))

sink()

cat("Done.\n")
