#!/usr/bin/env Rscript
# Near-Tie RD EDA (Uber-Inspired Design)
# Running variable: z = score - threshold_b where score = QUALITY * FINAL_BID
# Outcome: impressed, clicked
suppressPackageStartupMessages({
  library(data.table)
})

# ==============================================================================
# Module 0: CLI and Utilities
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
boundaries <- as.integer(strsplit(get_arg("--boundaries", "2,4,6,7"), ",")[[1]])
tau_windows <- as.numeric(strsplit(get_arg("--tau_windows", "0.005,0.01,0.02,0.05"), ",")[[1]])
time_slices <- as.integer(get_arg("--time_slices", "3"))

BASE <- normalizePath("analysis/position-effects/0_data", mustWork = TRUE)
paths <- list(
  auctions_results = file.path(BASE, if (round_name == "round1") "round1/auctions_results_all.parquet" else "round2/auctions_results_r2.parquet"),
  impressions      = file.path(BASE, if (round_name == "round1") "round1/impressions_all.parquet" else "round2/impressions_r2.parquet"),
  clicks           = file.path(BASE, if (round_name == "round1") "round1/clicks_all.parquet" else "round2/clicks_r2.parquet"),
  auctions_users   = file.path(BASE, if (round_name == "round1") "round1/auctions_users_all.parquet" else "round2/auctions_users_r2.parquet")
)

out_dir <- file.path("analysis", "position-effects-analysis-R", "results")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

summarize_col <- function(x) {
  x <- as.numeric(x)
  x <- x[!is.na(x)]
  if (length(x) == 0) return(data.table(n = 0, mean = NA_real_, sd = NA_real_, min = NA_real_, p25 = NA_real_, median = NA_real_, p75 = NA_real_, max = NA_real_))
  data.table(
    n = length(x), mean = mean(x), sd = sd(x), min = min(x),
    p25 = quantile(x, 0.25), median = median(x), p75 = quantile(x, 0.75), max = max(x)
  )
}

# ==============================================================================
# Module 1: Data Loading & Score Computation
# ==============================================================================
cat("Loading data...\n")
ar <- read_parquet_dt(paths$auctions_results, c("AUCTION_ID", "PRODUCT_ID", "VENDOR_ID", "RANKING", "QUALITY", "FINAL_BID"))
au <- read_parquet_dt(paths$auctions_users, c("AUCTION_ID", "PLACEMENT", "CREATED_AT"))
imps <- read_parquet_dt(paths$impressions, c("AUCTION_ID", "PRODUCT_ID", "VENDOR_ID", "USER_ID", "OCCURRED_AT"))
clks <- read_parquet_dt(paths$clicks, c("AUCTION_ID", "PRODUCT_ID", "VENDOR_ID", "USER_ID", "OCCURRED_AT"))

au <- unique(au, by = "AUCTION_ID")
au[, PLACEMENT := as.character(PLACEMENT)]
au[, CREATED_AT := as.POSIXct(CREATED_AT, tz = "UTC")]

ar <- ar[!is.na(QUALITY) & !is.na(FINAL_BID) & !is.na(RANKING)]
ar <- ar[(QUALITY > 0) & (FINAL_BID > 0) & (RANKING >= 1)]
ar[, score := as.numeric(QUALITY) * as.numeric(FINAL_BID)]
ar <- merge(ar, au[, .(AUCTION_ID, PLACEMENT, CREATED_AT)], by = "AUCTION_ID", all.x = FALSE)

imps[, OCCURRED_AT := as.POSIXct(OCCURRED_AT, tz = "UTC")]
clks[, OCCURRED_AT := as.POSIXct(OCCURRED_AT, tz = "UTC")]

# First impression per bid
first_imp <- imps[, .(FIRST_IMP_AT = min(OCCURRED_AT, na.rm = TRUE)), by = .(AUCTION_ID, PRODUCT_ID, VENDOR_ID)]
# Click flag
click_flag <- unique(clks[, .(AUCTION_ID, PRODUCT_ID, VENDOR_ID, clicked = 1L)])

# ==============================================================================
# Module 2: Running Variable & Pair Construction
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
  pairs_dt[, z := score - score_lo]  # running variable
  pairs_dt[, lucky := as.integer(score == score_hi)]
  pairs_dt
}

# ==============================================================================
# Main Loop Over Placements
# ==============================================================================
for (pl in placements) {
  cat(sprintf("\n========== PLACEMENT %s ==========\n", pl))
  outfile <- file.path(out_dir, sprintf("uber_near_tie_rdd_eda_%s_pl%s.txt", round_name, pl))
  sink(outfile, split = TRUE)

  cat(sprintf("Near-Tie RD EDA (Uber-Inspired) - %s PLACEMENT=%s\n", round_name, pl))
  cat(sprintf("Boundaries: %s\n", paste(boundaries, collapse = ", ")))
  cat(sprintf("Tau windows: %s\n", paste(tau_windows, collapse = ", ")))
  cat(sprintf("Time slices: %d\n\n", time_slices))

  ar_pl <- ar[PLACEMENT == pl]
  if (nrow(ar_pl) == 0) {
    cat("No data for this placement.\n")
    sink()
    next
  }

  pairs_dt <- build_pairs(ar_pl, boundaries)
  if (is.null(pairs_dt) || nrow(pairs_dt) == 0) {
    cat("No pairs constructed.\n")
    sink()
    next
  }

  # Attach impressions and clicks
  setkey(pairs_dt, AUCTION_ID, PRODUCT_ID, VENDOR_ID)
  setkey(first_imp, AUCTION_ID, PRODUCT_ID, VENDOR_ID)
  setkey(click_flag, AUCTION_ID, PRODUCT_ID, VENDOR_ID)
  pairs_dt <- first_imp[pairs_dt]
  pairs_dt <- click_flag[pairs_dt]
  pairs_dt[is.na(clicked), clicked := 0L]
  pairs_dt[, impressed := as.integer(!is.na(FIRST_IMP_AT))]

  # ==============================================================================
  # Module 3: Bandwidth Filter Tables
  # ==============================================================================
  cat(paste0(rep("=", 62), collapse = ""), "\n")
  cat("MODULE 3: BANDWIDTH FILTER TABLES\n")
  cat(paste0(rep("=", 62), collapse = ""), "\n\n")

  bw_table <- data.table()
  for (tau in tau_windows) {
    for (b in boundaries) {
      sub <- pairs_dt[boundary == b & rel_gap <= tau]
      n_pairs <- length(unique(sub$pair_id))
      # Both impressed pairs
      both_imp <- sub[, .(n_imp = sum(impressed)), by = pair_id][n_imp == 2, pair_id]
      n_both <- length(both_imp)
      pct_both <- if (n_pairs > 0) n_both / n_pairs else NA_real_
      # Clicks among both-impressed
      n_clicks <- sub[pair_id %in% both_imp, sum(clicked)]
      bw_table <- rbind(bw_table, data.table(
        tau = tau, boundary = b, N_pairs = n_pairs, N_both_imp = n_both,
        pct_both = round(pct_both, 3), N_clicks = n_clicks
      ))
    }
  }
  print(bw_table)
  cat("\n")

  # ==============================================================================
  # Module 4: Balance Checks (Local Randomization Validity)
  # ==============================================================================
  cat(paste0(rep("=", 62), collapse = ""), "\n")
  cat("MODULE 4: BALANCE CHECKS\n")
  cat(paste0(rep("=", 62), collapse = ""), "\n\n")

  balance_table <- data.table()
  for (tau in tau_windows) {
    for (b in boundaries) {
      sub <- pairs_dt[boundary == b & rel_gap <= tau]
      if (nrow(sub) < 4) next
      lucky_q <- sub[lucky == 1, QUALITY]
      unlucky_q <- sub[lucky == 0, QUALITY]
      lucky_b <- sub[lucky == 1, FINAL_BID]
      unlucky_b <- sub[lucky == 0, FINAL_BID]

      mean_q_l <- mean(lucky_q, na.rm = TRUE)
      mean_q_u <- mean(unlucky_q, na.rm = TRUE)
      var_q_l <- var(lucky_q, na.rm = TRUE)
      var_q_u <- var(unlucky_q, na.rm = TRUE)
      std_diff_q <- (mean_q_l - mean_q_u) / sqrt((var_q_l + var_q_u) / 2)

      mean_b_l <- mean(lucky_b, na.rm = TRUE)
      mean_b_u <- mean(unlucky_b, na.rm = TRUE)
      var_b_l <- var(lucky_b, na.rm = TRUE)
      var_b_u <- var(unlucky_b, na.rm = TRUE)
      std_diff_b <- (mean_b_l - mean_b_u) / sqrt((var_b_l + var_b_u) / 2)

      balance_table <- rbind(balance_table, data.table(
        tau = tau, boundary = b,
        mean_Q_lucky = round(mean_q_l, 6), mean_Q_unlucky = round(mean_q_u, 6), std_diff_Q = round(std_diff_q, 4),
        mean_B_lucky = round(mean_b_l, 2), mean_B_unlucky = round(mean_b_u, 2), std_diff_B = round(std_diff_b, 4)
      ))
    }
  }
  # Add pass/fail flag
  balance_table[, valid := fifelse(abs(std_diff_Q) < 0.1 & abs(std_diff_B) < 0.1, "PASS", "FAIL")]
  cat(capture.output(print(balance_table)), sep = "\n")
  cat("\nValidity criterion: |std_diff| < 0.1\n")
  cat("\nSummary: Windows passing balance (|std_diff| < 0.1 for both Q and B):\n")
  passing <- balance_table[valid == "PASS", .(tau, boundary)]
  if (nrow(passing) > 0) {
    print(passing)
  } else {
    cat("None\n")
  }
  cat("\n")

  # ==============================================================================
  # Module 5: Density/Bunching Analysis (McCrary-style bin counts)
  # ==============================================================================
  cat(paste0(rep("=", 62), collapse = ""), "\n")
  cat("MODULE 5: DENSITY/BUNCHING ANALYSIS\n")
  cat(paste0(rep("=", 62), collapse = ""), "\n\n")

  # Bin edges for rel_gap histogram
  bins <- c(0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, Inf)
  bin_labels <- c("(0,0.001]", "(0.001,0.002]", "(0.002,0.005]", "(0.005,0.01]",
                  "(0.01,0.02]", "(0.02,0.05]", "(0.05,0.10]", ">0.10")

  cat("Pair counts by rel_gap bin (checking for bunching/manipulation):\n")
  cat("Smooth, monotonically decreasing counts suggest no manipulation.\n\n")

  density_table <- data.table()
  for (b in boundaries) {
    sub <- pairs_dt[boundary == b]
    # Count unique pairs in each bin
    pair_gaps <- unique(sub[, .(pair_id, rel_gap)])
    for (i in 1:(length(bins) - 1)) {
      n_pairs <- nrow(pair_gaps[rel_gap > bins[i] & rel_gap <= bins[i + 1]])
      density_table <- rbind(density_table, data.table(
        boundary = b, bin = bin_labels[i], N_pairs = n_pairs
      ))
    }
  }

  # Reshape to wide format for readability
  density_wide <- dcast(density_table, bin ~ boundary, value.var = "N_pairs")
  setcolorder(density_wide, c("bin", sort(as.character(boundaries))))
  print(density_wide)

  # Summary: ratio of adjacent bins (smoothness check)
  cat("\nBin-to-bin ratios (boundary 2, first 5 bins):\n")
  if (2 %in% boundaries) {
    b2_counts <- density_table[boundary == 2, N_pairs]
    if (length(b2_counts) >= 5) {
      ratios <- b2_counts[2:5] / b2_counts[1:4]
      ratios[!is.finite(ratios)] <- NA_real_
      cat(sprintf("  Bins 1->2: %.2f, 2->3: %.2f, 3->4: %.2f, 4->5: %.2f\n",
                  ratios[1], ratios[2], ratios[3], ratios[4]))
      cat("  Smooth pattern: ratios stable or gradually declining.\n")
      cat("  Bunching concern: ratio[1] << 1 (excess mass in smallest bin).\n")
    }
  }
  cat("\n")

  # ==============================================================================
  # Module 6: Rank-Score Alignment Audit
  # ==============================================================================
  cat(paste0(rep("=", 62), collapse = ""), "\n")
  cat("MODULE 6: RANK-SCORE ALIGNMENT AUDIT\n")
  cat(paste0(rep("=", 62), collapse = ""), "\n\n")

  align_table <- data.table()
  for (tau in tau_windows) {
    for (b in boundaries) {
      sub <- pairs_dt[boundary == b & rel_gap <= tau]
      if (nrow(sub) < 2) next
      # Within each pair: does lucky (higher score) have lower RANKING?
      pair_align <- sub[, .(
        rank_lucky = RANKING[lucky == 1][1],
        rank_unlucky = RANKING[lucky == 0][1]
      ), by = pair_id]
      pair_align[, aligned := as.integer(rank_lucky < rank_unlucky)]
      align_pct <- mean(pair_align$aligned, na.rm = TRUE)
      align_table <- rbind(align_table, data.table(
        tau = tau, boundary = b, N_pairs = nrow(pair_align),
        alignment_pct = round(align_pct, 4)
      ))
    }
  }
  print(align_table)
  cat("\nLower alignment at tight ties may indicate tie-breaking rules.\n\n")

  # ==============================================================================
  # Module 7: Exposure vs Click Decomposition
  # ==============================================================================
  cat(paste0(rep("=", 62), collapse = ""), "\n")
  cat("MODULE 7: EXPOSURE VS CLICK DECOMPOSITION\n")
  cat(paste0(rep("=", 62), collapse = ""), "\n\n")

  decomp_table <- data.table()
  for (tau in tau_windows) {
    for (b in boundaries) {
      sub <- pairs_dt[boundary == b & rel_gap <= tau]
      if (nrow(sub) < 4) next

      # Stage A: Exposure
      imp_rate_lucky <- mean(sub[lucky == 1, impressed], na.rm = TRUE)
      imp_rate_unlucky <- mean(sub[lucky == 0, impressed], na.rm = TRUE)
      delta_exposure <- imp_rate_lucky - imp_rate_unlucky

      # Stage B: Conditional click (among both-impressed pairs)
      both_imp <- sub[, .(n_imp = sum(impressed)), by = pair_id][n_imp == 2, pair_id]
      sub_both <- sub[pair_id %in% both_imp]
      ctr_lucky <- mean(sub_both[lucky == 1, clicked], na.rm = TRUE)
      ctr_unlucky <- mean(sub_both[lucky == 0, clicked], na.rm = TRUE)
      delta_ctr <- ctr_lucky - ctr_unlucky

      decomp_table <- rbind(decomp_table, data.table(
        tau = tau, boundary = b,
        imp_lucky = round(imp_rate_lucky, 4), imp_unlucky = round(imp_rate_unlucky, 4),
        delta_exp = round(delta_exposure, 4),
        ctr_lucky = round(ctr_lucky, 4), ctr_unlucky = round(ctr_unlucky, 4),
        delta_ctr = round(delta_ctr, 4),
        N_both_imp = length(both_imp)
      ))
    }
  }
  print(decomp_table)
  cat("\n")

  # ==============================================================================
  # Module 8: Heterogeneity Maps
  # ==============================================================================
  cat(paste0(rep("=", 62), collapse = ""), "\n")
  cat("MODULE 8: HETEROGENEITY MAPS\n")
  cat(paste0(rep("=", 62), collapse = ""), "\n\n")

  # Device proxy from impressions (modal burst size)
  cat("8a. Device proxy (from impression burst sizes)...\n")
  imps_pl <- imps[AUCTION_ID %in% unique(pairs_dt$AUCTION_ID)]
  imps_pl[, occ_second := floor(as.numeric(OCCURRED_AT))]
  burst_sizes <- imps_pl[, .(burst_size = .N), by = .(USER_ID, occ_second)]
  user_modal_burst <- burst_sizes[, .(modal_burst = as.integer(names(which.max(table(burst_size))))), by = USER_ID]
  user_modal_burst[, device := fifelse(modal_burst <= 2, "mobile", "desktop")]

  # Auction size deciles
  cat("8b. Auction size (bidder density) deciles...\n")
  auction_size <- ar_pl[, .(n_bids = .N), by = AUCTION_ID]
  breaks_size <- unique(quantile(auction_size$n_bids, probs = seq(0, 1, 0.1), na.rm = TRUE))
  if (length(breaks_size) > 1) {
    auction_size[, size_decile := cut(n_bids, breaks = breaks_size, include.lowest = TRUE, labels = FALSE)]
  } else {
    auction_size[, size_decile := 1L]
  }

  # Quality deciles
  cat("8c. Quality deciles...\n")
  breaks_q <- unique(quantile(pairs_dt$QUALITY, probs = seq(0, 1, 0.1), na.rm = TRUE))
  if (length(breaks_q) > 1) {
    pairs_dt[, quality_decile := cut(QUALITY, breaks = breaks_q, include.lowest = TRUE, labels = FALSE)]
  } else {
    pairs_dt[, quality_decile := 1L]
  }

  # Price deciles
  cat("8d. Price (FINAL_BID) deciles...\n")
  breaks_p <- unique(quantile(pairs_dt$FINAL_BID, probs = seq(0, 1, 0.1), na.rm = TRUE))
  if (length(breaks_p) > 1) {
    pairs_dt[, price_decile := cut(FINAL_BID, breaks = breaks_p, include.lowest = TRUE, labels = FALSE)]
  } else {
    pairs_dt[, price_decile := 1L]
  }

  # Heterogeneity by boundary (main)
  tau_het <- 0.02  # use 2% bandwidth for heterogeneity
  cat(sprintf("\nHeterogeneity at tau=%.3f by boundary:\n", tau_het))
  het_boundary <- data.table()
  for (b in boundaries) {
    sub <- pairs_dt[boundary == b & rel_gap <= tau_het]
    if (nrow(sub) < 4) next
    delta_exp <- mean(sub[lucky == 1, impressed]) - mean(sub[lucky == 0, impressed])
    both_imp <- sub[, .(n_imp = sum(impressed)), by = pair_id][n_imp == 2, pair_id]
    sub_both <- sub[pair_id %in% both_imp]
    delta_ctr <- if (nrow(sub_both) > 0) mean(sub_both[lucky == 1, clicked]) - mean(sub_both[lucky == 0, clicked]) else NA_real_
    het_boundary <- rbind(het_boundary, data.table(boundary = b, delta_exp = round(delta_exp, 4), delta_ctr = round(delta_ctr, 4), N = nrow(sub) / 2))
  }
  print(het_boundary)

  # Heterogeneity by quality decile
  cat("\nHeterogeneity by quality decile (boundary 2):\n")
  het_quality <- data.table()
  sub_b2 <- pairs_dt[boundary == 2 & rel_gap <= tau_het]
  for (q in 1:10) {
    sub_q <- sub_b2[quality_decile == q]
    if (nrow(sub_q) < 4) next
    delta_exp <- mean(sub_q[lucky == 1, impressed], na.rm = TRUE) - mean(sub_q[lucky == 0, impressed], na.rm = TRUE)
    het_quality <- rbind(het_quality, data.table(quality_decile = q, delta_exp = round(delta_exp, 4), N = nrow(sub_q) / 2))
  }
  if (nrow(het_quality) > 0) print(het_quality)

  # Heterogeneity by device (boundary 2)
  cat("\nHeterogeneity by device (boundary 2):\n")
  # Create auction->device lookup (modal user device per auction)
  auction_user <- unique(imps_pl[, .(AUCTION_ID, USER_ID)])
  auction_user <- merge(auction_user, user_modal_burst[, .(USER_ID, device)], by = "USER_ID", all.x = TRUE)
  auction_device <- auction_user[, .(device = names(which.max(table(device)))), by = AUCTION_ID]

  het_device <- data.table()
  for (dev in c("mobile", "desktop")) {
    auctions_dev <- auction_device[device == dev, AUCTION_ID]
    sub_dev <- pairs_dt[boundary == 2 & rel_gap <= tau_het & AUCTION_ID %in% auctions_dev]
    if (nrow(sub_dev) < 4) next
    delta_exp <- mean(sub_dev[lucky == 1, impressed], na.rm = TRUE) - mean(sub_dev[lucky == 0, impressed], na.rm = TRUE)
    both_imp <- sub_dev[, .(n_imp = sum(impressed)), by = pair_id][n_imp == 2, pair_id]
    sub_both <- sub_dev[pair_id %in% both_imp]
    delta_ctr <- if (nrow(sub_both) > 0) mean(sub_both[lucky == 1, clicked], na.rm = TRUE) - mean(sub_both[lucky == 0, clicked], na.rm = TRUE) else NA_real_
    het_device <- rbind(het_device, data.table(device = dev, delta_exp = round(delta_exp, 4), delta_ctr = round(delta_ctr, 4), N = nrow(sub_dev) / 2))
  }
  if (nrow(het_device) > 0) {
    print(het_device)
  } else {
    cat("Insufficient data for device heterogeneity.\n")
  }

  cat("\n")

  # ==============================================================================
  # Module 9: Sensitivity: Time Windows
  # ==============================================================================
  cat(paste0(rep("=", 62), collapse = ""), "\n")
  cat("MODULE 9: SENSITIVITY - TIME SLICES\n")
  cat(paste0(rep("=", 62), collapse = ""), "\n\n")

  # Split data into time slices
  time_range <- range(pairs_dt$CREATED_AT, na.rm = TRUE)
  slice_breaks <- seq(time_range[1], time_range[2], length.out = time_slices + 1)
  pairs_dt[, time_slice := cut(CREATED_AT, breaks = slice_breaks, labels = FALSE, include.lowest = TRUE)]

  time_sens <- data.table()
  tau_sens <- 0.02
  for (b in boundaries[1:min(2, length(boundaries))]) {  # just first 2 boundaries for time
    for (ts in 1:time_slices) {
      sub <- pairs_dt[boundary == b & rel_gap <= tau_sens & time_slice == ts]
      if (nrow(sub) < 4) next
      delta_exp <- mean(sub[lucky == 1, impressed]) - mean(sub[lucky == 0, impressed])
      both_imp <- sub[, .(n_imp = sum(impressed)), by = pair_id][n_imp == 2, pair_id]
      sub_both <- sub[pair_id %in% both_imp]
      delta_ctr <- if (nrow(sub_both) > 0) mean(sub_both[lucky == 1, clicked]) - mean(sub_both[lucky == 0, clicked]) else NA_real_
      time_sens <- rbind(time_sens, data.table(
        boundary = b, time_slice = ts, N_pairs = nrow(sub) / 2,
        delta_exp = round(delta_exp, 4), delta_ctr = round(delta_ctr, 4)
      ))
    }
  }
  if (nrow(time_sens) > 0) print(time_sens)
  cat("\n")

  # ==============================================================================
  # Module 10: Latency as Mechanism
  # ==============================================================================
  cat(paste0(rep("=", 62), collapse = ""), "\n")
  cat("MODULE 10: LATENCY AS MECHANISM\n")
  cat(paste0(rep("=", 62), collapse = ""), "\n\n")

  pairs_dt[, time_to_imp := as.numeric(difftime(FIRST_IMP_AT, CREATED_AT, units = "secs"))]

  latency_table <- data.table()
  for (tau in tau_windows) {
    for (b in boundaries[1:min(2, length(boundaries))]) {
      sub <- pairs_dt[boundary == b & rel_gap <= tau & impressed == 1]
      if (nrow(sub) < 4) next
      # Compare time_to_imp between lucky and unlucky
      med_lucky <- median(sub[lucky == 1, time_to_imp], na.rm = TRUE)
      med_unlucky <- median(sub[lucky == 0, time_to_imp], na.rm = TRUE)
      dwell_delta <- med_unlucky - med_lucky
      latency_table <- rbind(latency_table, data.table(
        tau = tau, boundary = b,
        med_time_lucky = round(med_lucky, 3), med_time_unlucky = round(med_unlucky, 3),
        dwell_delta_s = round(dwell_delta, 3)
      ))
    }
  }
  if (nrow(latency_table) > 0) print(latency_table)
  cat("\nPositive dwell_delta means unlucky items seen later (exposure timing channel).\n\n")

  # ==============================================================================
  # Module 11: Falsification/Placebo (7v8 Boundary)
  # ==============================================================================
  cat(paste0(rep("=", 62), collapse = ""), "\n")
  cat("MODULE 11: FALSIFICATION/PLACEBO (7v8 BOUNDARY)\n")
  cat(paste0(rep("=", 62), collapse = ""), "\n\n")

  if (7 %in% boundaries) {
    sub_placebo <- pairs_dt[boundary == 7]
    cat("Placebo analysis at boundary 7v8 (below any reasonable fold):\n")
    placebo_table <- data.table()
    for (tau in tau_windows) {
      sub <- sub_placebo[rel_gap <= tau]
      if (nrow(sub) < 4) next
      delta_exp <- mean(sub[lucky == 1, impressed]) - mean(sub[lucky == 0, impressed])
      both_imp <- sub[, .(n_imp = sum(impressed)), by = pair_id][n_imp == 2, pair_id]
      sub_both <- sub[pair_id %in% both_imp]
      delta_ctr <- if (nrow(sub_both) > 0) mean(sub_both[lucky == 1, clicked]) - mean(sub_both[lucky == 0, clicked]) else NA_real_
      placebo_table <- rbind(placebo_table, data.table(
        tau = tau, N_pairs = nrow(sub) / 2, delta_exp = round(delta_exp, 4), delta_ctr = round(delta_ctr, 4)
      ))
    }
    if (nrow(placebo_table) > 0) print(placebo_table)
    cat("\nExpect: delta_exposure ~ 0, delta_ctr ~ 0 (no effect below fold).\n\n")
  } else {
    cat("Boundary 7 not included in analysis.\n\n")
  }

  # ==============================================================================
  # Module 12: Discretization Robustness
  # ==============================================================================
  cat(paste0(rep("=", 62), collapse = ""), "\n")
  cat("MODULE 12: DISCRETIZATION ROBUSTNESS\n")
  cat(paste0(rep("=", 62), collapse = ""), "\n\n")

  cat("FINAL_BID unique values (first 50):\n")
  unique_bids <- sort(unique(pairs_dt$FINAL_BID))[1:min(50, length(unique(pairs_dt$FINAL_BID)))]
  cat(paste(unique_bids, collapse = ", "), "\n")
  cat(sprintf("Total unique FINAL_BID values: %d\n\n", length(unique(pairs_dt$FINAL_BID))))

  cat("QUALITY unique values (first 50):\n")
  unique_q <- sort(unique(pairs_dt$QUALITY))[1:min(50, length(unique(pairs_dt$QUALITY)))]
  cat(paste(round(unique_q, 6), collapse = ", "), "\n")
  cat(sprintf("Total unique QUALITY values: %d\n\n", length(unique(pairs_dt$QUALITY))))

  cat("Score distribution:\n")
  print(summarize_col(pairs_dt$score))

  cat("\n")
  cat(sprintf("Output saved to: %s\n", outfile))

  sink()
}

cat("\nDone. Check results in:", out_dir, "\n")
