#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(data.table)
  library(fixest)
})

read_parquet_dt <- function(path, cols = NULL) {
  if (requireNamespace("duckdb", quietly = TRUE) && requireNamespace("DBI", quietly = TRUE)) {
    con <- DBI::dbConnect(duckdb::duckdb(), dbdir = tempfile())
    on.exit(DBI::dbDisconnect(con, shutdown = TRUE), add = TRUE)
    tbl <- sprintf("read_parquet('%s')", normalizePath(path))
    sql <- if (is.null(cols)) sprintf("SELECT * FROM %s", tbl) else sprintf("SELECT %s FROM %s", paste(cols, collapse=","), tbl)
    dt <- as.data.table(DBI::dbGetQuery(con, sql))
    return(dt)
  }
  if (requireNamespace("arrow", quietly = TRUE)) {
    if (!nzchar(Sys.getenv("OMP_NUM_THREADS"))) Sys.setenv(OMP_NUM_THREADS = 1)
    if (!nzchar(Sys.getenv("ARROW_NUM_THREADS"))) Sys.setenv(ARROW_NUM_THREADS = 1)
    dt <- as.data.table(arrow::read_parquet(path, as_data_frame = TRUE, col_select = cols))
    return(dt)
  }
  stop("Neither duckdb nor arrow is available to read parquet.")
}

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  hit <- which(args == flag)
  if (length(hit) == 0) return(default)
  if (hit == length(args)) return(TRUE)
  val <- args[hit + 1]
  if (startsWith(val, "--")) return(TRUE) else return(val)
}

round_name <- get_arg("--round")
placements_arg <- get_arg("--placements", default = "1,2,3,5")
if (is.null(round_name)) stop("--round is required")
placements <- as.character(unlist(strsplit(placements_arg, ",")))

BASE <- normalizePath("analysis/position-effects/0_data", mustWork = TRUE)
paths <- list(
  auctions_results = if (round_name == "round1") file.path(BASE, "round1/auctions_results_all.parquet") else file.path(BASE, "round2/auctions_results_r2.parquet"),
  auctions_users   = if (round_name == "round1") file.path(BASE, "round1/auctions_users_all.parquet")  else file.path(BASE, "round2/auctions_users_r2.parquet")
)

ar <- read_parquet_dt(paths$auctions_results, c("AUCTION_ID","VENDOR_ID","PRODUCT_ID","RANKING","QUALITY","FINAL_BID"))
au <- read_parquet_dt(paths$auctions_users, c("AUCTION_ID","PLACEMENT"))
au <- unique(as.data.table(au), by = "AUCTION_ID")

setDT(ar); setDT(au)
ar <- ar[!is.na(QUALITY) & !is.na(FINAL_BID) & QUALITY > 0 & FINAL_BID > 0]
ar[, PLACEMENT := au[.SD, PLACEMENT, on = .(AUCTION_ID)]]
ar <- ar[!is.na(PLACEMENT)]
ar[, score := QUALITY * FINAL_BID]

auction_stats <- ar[, {
  bidder_count <- .N
  scores_sorted <- sort(score, decreasing = TRUE)
  quality_sorted <- QUALITY[order(-score)]
  Q1 <- quality_sorted[1]
  S1 <- scores_sorted[1]
  S2 <- if (length(scores_sorted) > 1) scores_sorted[2] else NA_real_
  threshold_bid <- if (!is.na(S2) && Q1 > 0) S2 / Q1 else NA_real_
  list(
    bidder_count = bidder_count,
    Q1 = Q1,
    S1 = S1,
    S2 = S2,
    threshold_bid = threshold_bid,
    PLACEMENT = PLACEMENT[1]
  )
}, by = AUCTION_ID]

auction_stats <- auction_stats[!is.na(Q1) & !is.na(S2) & !is.na(threshold_bid)]
auction_stats <- auction_stats[bidder_count >= 2]
auction_stats[, `:=`(
  log_count = log(bidder_count),
  log_Q1 = log(Q1),
  log_S2 = log(S2),
  log_thr_bid = log(threshold_bid)
)]
auction_stats <- auction_stats[is.finite(log_Q1) & is.finite(log_S2) & is.finite(log_thr_bid)]

auction_stats[, depth_bin := cut(bidder_count,
  breaks = c(2, 5, 10, 20, 50, Inf),
  labels = c("(2,5]","(5,10]","(10,20]","(20,50]","(50,Inf)"),
  include.lowest = TRUE)]

out_dir <- file.path("analysis","position-effects-analysis-R","results")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
outfile <- file.path(out_dir, sprintf("congestion_thresholds_%s.txt", round_name))
con <- file(outfile, open = "wt")
on.exit(close(con), add = TRUE)

cat(sprintf("Congestion Thresholds Analysis - %s\n", round_name), file = con)
cat(paste(rep("=", 60), collapse=""), "\n\n", file = con, append = TRUE)

cat("Unit of analysis: Auction-level (one row per AUCTION_ID)\n\n", file = con, append = TRUE)
cat("Key variables:\n", file = con, append = TRUE)
cat("  bidder_count: number of valid bids per auction\n", file = con, append = TRUE)
cat("  Q1: QUALITY of winner (top score)\n", file = con, append = TRUE)
cat("  S2: score of runner-up (score = QUALITY * FINAL_BID)\n", file = con, append = TRUE)
cat("  threshold_bid = S2 / Q1: bid required for winner to beat runner-up\n\n", file = con, append = TRUE)

cat(sprintf("Total auctions with 2+ bidders: %s\n", format(nrow(auction_stats), big.mark=",")), file = con, append = TRUE)
cat(sprintf("Placements analyzed: %s\n\n", paste(placements, collapse = ", ")), file = con, append = TRUE)

cat(paste(rep("=", 60), collapse=""), "\n", file = con, append = TRUE)
cat("ELASTICITY ESTIMATES (log-log OLS by placement)\n", file = con, append = TRUE)
cat(paste(rep("=", 60), collapse=""), "\n\n", file = con, append = TRUE)

for (pl in placements) {
  dt_pl <- auction_stats[as.character(PLACEMENT) == pl]
  if (nrow(dt_pl) < 100) {
    cat(sprintf("PLACEMENT %s: Insufficient observations (N=%d)\n\n", pl, nrow(dt_pl)), file = con, append = TRUE)
    next
  }

  cat(sprintf("PLACEMENT %s (N=%s)\n", pl, format(nrow(dt_pl), big.mark=",")), file = con, append = TRUE)
  cat(paste(rep("-", 40), collapse=""), "\n", file = con, append = TRUE)

  fit_Q1 <- feols(log_Q1 ~ log_count, data = dt_pl, vcov = "hetero")
  fit_S2 <- feols(log_S2 ~ log_count, data = dt_pl, vcov = "hetero")
  fit_thr <- feols(log_thr_bid ~ log_count, data = dt_pl, vcov = "hetero")

  cat("\nlog(Q1) ~ log(bidder_count):\n", file = con, append = TRUE)
  cat(sprintf("  beta = %.4f (SE = %.4f), R2 = %.4f\n",
    coef(fit_Q1)["log_count"], se(fit_Q1)["log_count"], r2(fit_Q1, "r2")), file = con, append = TRUE)

  cat("\nlog(S2) ~ log(bidder_count):\n", file = con, append = TRUE)
  cat(sprintf("  beta = %.4f (SE = %.4f), R2 = %.4f\n",
    coef(fit_S2)["log_count"], se(fit_S2)["log_count"], r2(fit_S2, "r2")), file = con, append = TRUE)

  cat("\nlog(threshold_bid) ~ log(bidder_count):\n", file = con, append = TRUE)
  cat(sprintf("  beta = %.4f (SE = %.4f), R2 = %.4f\n\n",
    coef(fit_thr)["log_count"], se(fit_thr)["log_count"], r2(fit_thr, "r2")), file = con, append = TRUE)
}

cat(paste(rep("=", 60), collapse=""), "\n", file = con, append = TRUE)
cat("DEPTH-BIN MEDIAN TABLES BY PLACEMENT\n", file = con, append = TRUE)
cat(paste(rep("=", 60), collapse=""), "\n\n", file = con, append = TRUE)

for (pl in placements) {
  dt_pl <- auction_stats[as.character(PLACEMENT) == pl]
  if (nrow(dt_pl) < 100) next

  cat(sprintf("PLACEMENT %s\n", pl), file = con, append = TRUE)
  cat(paste(rep("-", 40), collapse=""), "\n", file = con, append = TRUE)

  bin_stats <- dt_pl[, .(
    N = .N,
    median_Q1 = median(Q1),
    median_S2 = median(S2),
    median_threshold_bid = median(threshold_bid)
  ), by = depth_bin]
  bin_stats <- bin_stats[order(depth_bin)]

  cat(sprintf("%-12s %8s %12s %12s %18s\n", "Depth_Bin", "N", "median_Q1", "median_S2", "median_thr_bid"), file = con, append = TRUE)
  for (i in seq_len(nrow(bin_stats))) {
    cat(sprintf("%-12s %8d %12.6f %12.4f %18.4f\n",
      as.character(bin_stats$depth_bin[i]),
      bin_stats$N[i],
      bin_stats$median_Q1[i],
      bin_stats$median_S2[i],
      bin_stats$median_threshold_bid[i]), file = con, append = TRUE)
  }
  cat("\n", file = con, append = TRUE)
}

cat(paste(rep("=", 60), collapse=""), "\n", file = con, append = TRUE)
cat("WINSORIZED VARIANTS (99th percentile)\n", file = con, append = TRUE)
cat(paste(rep("=", 60), collapse=""), "\n\n", file = con, append = TRUE)

winsorize <- function(x, prob = 0.99) {
  q <- quantile(x, prob, na.rm = TRUE)
  x[x > q] <- q
  x
}

for (pl in placements) {
  dt_pl <- auction_stats[as.character(PLACEMENT) == pl]
  if (nrow(dt_pl) < 100) next

  dt_win <- copy(dt_pl)
  dt_win[, `:=`(
    Q1_win = winsorize(Q1),
    S2_win = winsorize(S2),
    threshold_bid_win = winsorize(threshold_bid)
  )]
  dt_win[, `:=`(
    log_Q1_win = log(Q1_win),
    log_S2_win = log(S2_win),
    log_thr_bid_win = log(threshold_bid_win)
  )]
  dt_win <- dt_win[is.finite(log_Q1_win) & is.finite(log_S2_win) & is.finite(log_thr_bid_win)]

  cat(sprintf("PLACEMENT %s (winsorized, N=%s)\n", pl, format(nrow(dt_win), big.mark=",")), file = con, append = TRUE)
  cat(paste(rep("-", 40), collapse=""), "\n", file = con, append = TRUE)

  fit_Q1 <- feols(log_Q1_win ~ log_count, data = dt_win, vcov = "hetero")
  fit_S2 <- feols(log_S2_win ~ log_count, data = dt_win, vcov = "hetero")
  fit_thr <- feols(log_thr_bid_win ~ log_count, data = dt_win, vcov = "hetero")

  cat(sprintf("  log(Q1_win) beta = %.4f (SE = %.4f)\n", coef(fit_Q1)["log_count"], se(fit_Q1)["log_count"]), file = con, append = TRUE)
  cat(sprintf("  log(S2_win) beta = %.4f (SE = %.4f)\n", coef(fit_S2)["log_count"], se(fit_S2)["log_count"]), file = con, append = TRUE)
  cat(sprintf("  log(thr_bid_win) beta = %.4f (SE = %.4f)\n\n", coef(fit_thr)["log_count"], se(fit_thr)["log_count"]), file = con, append = TRUE)
}

cat(paste(rep("=", 60), collapse=""), "\n", file = con, append = TRUE)
cat("SUMMARY STATISTICS\n", file = con, append = TRUE)
cat(paste(rep("=", 60), collapse=""), "\n\n", file = con, append = TRUE)

for (pl in placements) {
  dt_pl <- auction_stats[as.character(PLACEMENT) == pl]
  if (nrow(dt_pl) < 100) next

  cat(sprintf("PLACEMENT %s\n", pl), file = con, append = TRUE)
  cat(sprintf("  bidder_count: mean=%.1f, median=%.0f, sd=%.1f, min=%d, max=%d\n",
    mean(dt_pl$bidder_count), median(dt_pl$bidder_count), sd(dt_pl$bidder_count),
    min(dt_pl$bidder_count), max(dt_pl$bidder_count)), file = con, append = TRUE)
  cat(sprintf("  Q1 (winner quality): mean=%.6f, median=%.6f, sd=%.6f\n",
    mean(dt_pl$Q1), median(dt_pl$Q1), sd(dt_pl$Q1)), file = con, append = TRUE)
  cat(sprintf("  S2 (runner-up score): mean=%.4f, median=%.4f, sd=%.4f\n",
    mean(dt_pl$S2), median(dt_pl$S2), sd(dt_pl$S2)), file = con, append = TRUE)
  cat(sprintf("  threshold_bid: mean=%.4f, median=%.4f, sd=%.4f\n\n",
    mean(dt_pl$threshold_bid), median(dt_pl$threshold_bid), sd(dt_pl$threshold_bid)), file = con, append = TRUE)
}

cat(sprintf("\nOutput saved to: %s\n", outfile), file = con, append = TRUE)
message(sprintf("Output saved to: %s", outfile))
