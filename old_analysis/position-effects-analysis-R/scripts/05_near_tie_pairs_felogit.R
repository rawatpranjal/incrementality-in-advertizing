#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(data.table)
  library(fixest)
})

# Read parquet via duckdb/DBI if available; fallback to arrow
read_parquet_dt <- function(path, cols = NULL) {
  if (requireNamespace("duckdb", quietly = TRUE) && requireNamespace("DBI", quietly = TRUE)) {
    con <- DBI::dbConnect(duckdb::duckdb(), dbdir = tempfile())
    on.exit(DBI::dbDisconnect(con, shutdown = TRUE), add = TRUE)
    tbl <- sprintf("read_parquet('%s')", normalizePath(path))
    sql <- if (is.null(cols)) sprintf("SELECT * FROM %s", tbl) else sprintf("SELECT %s FROM %s", paste(cols, collapse=","), tbl)
    return(as.data.table(DBI::dbGetQuery(con, sql)))
  }
  if (requireNamespace("arrow", quietly = TRUE)) {
    if (!is.null(Sys.getenv("OMP_NUM_THREADS")) && Sys.getenv("OMP_NUM_THREADS") == "") Sys.setenv(OMP_NUM_THREADS = 1)
    if (!is.null(Sys.getenv("ARROW_NUM_THREADS")) && Sys.getenv("ARROW_NUM_THREADS") == "") Sys.setenv(ARROW_NUM_THREADS = 1)
    return(as.data.table(arrow::read_parquet(path, as_data_frame = TRUE, col_select = cols)))
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
window_min <- as.integer(get_arg("--window_minutes", default = 600))
tau1 <- as.numeric(get_arg("--tau1", default = 0.01))
tau2 <- as.numeric(get_arg("--tau2", default = 0.02))
max_boundary <- as.integer(get_arg("--max_boundary", default = 10))
if (is.null(round_name)) stop("--round is required")

BASE <- normalizePath("analysis/position-effects/0_data", mustWork = TRUE)
paths <- list(
  auctions_results = if (round_name == "round1") file.path(BASE, "round1/auctions_results_all.parquet") else file.path(BASE, "round2/auctions_results_r2.parquet"),
  impressions      = if (round_name == "round1") file.path(BASE, "round1/impressions_all.parquet")    else file.path(BASE, "round2/impressions_r2.parquet"),
  clicks           = if (round_name == "round1") file.path(BASE, "round1/clicks_all.parquet")         else file.path(BASE, "round2/clicks_r2.parquet"),
  auctions_users   = if (round_name == "round1") file.path(BASE, "round1/auctions_users_all.parquet")  else file.path(BASE, "round2/auctions_users_r2.parquet")
)

# Load minimal columns
imps <- read_parquet_dt(paths$impressions, c("AUCTION_ID","PRODUCT_ID","USER_ID","VENDOR_ID","OCCURRED_AT"))
clks <- read_parquet_dt(paths$clicks,      c("AUCTION_ID","PRODUCT_ID","USER_ID","VENDOR_ID","OCCURRED_AT"))
ar   <- read_parquet_dt(paths$auctions_results, c("AUCTION_ID","PRODUCT_ID","VENDOR_ID","RANKING","QUALITY","FINAL_BID"))
au   <- read_parquet_dt(paths$auctions_users, c("AUCTION_ID","PLACEMENT","CREATED_AT"))
au   <- unique(as.data.table(au), by = "AUCTION_ID")

# Filter placement=1 and time window (last window_min)
au_dt <- as.data.table(au)
au_dt[, PLACEMENT := as.character(PLACEMENT)]
au_dt <- au_dt[PLACEMENT == "1"]
au_dt[, CREATED_AT := as.POSIXct(CREATED_AT, tz = "UTC")]
end_time <- max(au_dt$CREATED_AT, na.rm = TRUE)
start_time <- end_time - window_min*60
au_dt <- au_dt[CREATED_AT >= start_time & CREATED_AT <= end_time]

# Prepare AR with score and rank-by-score positions per auction
ar <- as.data.table(ar)
ar <- ar[!is.na(QUALITY) & !is.na(FINAL_BID) & !is.na(RANKING)]
ar <- ar[(QUALITY > 0) & (FINAL_BID > 0) & (RANKING >= 1)]
ar[, score := as.numeric(QUALITY) * as.numeric(FINAL_BID)]
ar <- ar[AUCTION_ID %in% au_dt$AUCTION_ID]
setorder(ar, AUCTION_ID, -score)
ar[, pos_by_score := frank(-score, ties.method = "first"), by = AUCTION_ID]

# Keep positions up to max_boundary+1 (to form b vs b+1 for b in 2,4,6,...)
keep_pos <- unique(sort(c(seq(2, max_boundary, by = 2), seq(3, max_boundary+1, by = 2))))
ar_small <- ar[pos_by_score %in% keep_pos]

# Build pairs for each boundary b: (b vs b+1) by score
boundaries <- seq(2, max_boundary, by = 2)
pairs_list <- list()
for (b in boundaries) {
  tmp <- ar_small[pos_by_score %in% c(b, b+1)]
  tmp[, boundary := as.integer(b)]
  # pairs per auction+boundary should have exactly 2 rows
  tmp <- tmp[, .SD[.N >= 2][1:2], by = .(AUCTION_ID, boundary)]
  # ensure two rows
  tmp <- tmp[, if (.N == 2) .SD, by = .(AUCTION_ID, boundary)]
  pairs_list[[as.character(b)]] <- tmp
}
pairs_dt <- rbindlist(pairs_list, use.names = TRUE, fill = TRUE)

if (nrow(pairs_dt) == 0) {
  out_dir <- file.path("analysis","position-effects-analysis-R","results")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  outfile <- file.path(out_dir, sprintf("near_tie_pairs_felogit_%s.txt", round_name))
  writeLines("No pairs constructed.", con = outfile)
  quit(save = "no", status = 0)
}

# Compute pair-level identifiers and near-tie gaps using score by boundary
setorder(pairs_dt, AUCTION_ID, boundary, -score)
pairs_dt[, pair_id := .GRP, by = .(AUCTION_ID, boundary)]

# score_hi = score at pos b, score_lo = score at pos b+1 within pair
pairs_dt[, score_rank := 1:.N, by = pair_id]  # 1=hi, 2=lo under score order
pairs_dt[, score_hi := max(score), by = pair_id]
pairs_dt[, score_lo := min(score), by = pair_id]
pairs_dt[, rel_gap := (score_hi - score_lo) / score_hi]

# Attach first impression and click flags
imps <- as.data.table(imps)
imps[, FIRST_IMP_AT := as.POSIXct(OCCURRED_AT, tz = "UTC")]
imps <- imps[, .(FIRST_IMP_AT = min(FIRST_IMP_AT, na.rm = TRUE)), by = .(AUCTION_ID, PRODUCT_ID, VENDOR_ID)]
clks <- as.data.table(clks)
clks[, CLICK_AT := as.POSIXct(OCCURRED_AT, tz = "UTC")]
clks <- unique(clks[, .(clicked = 1L), by = .(AUCTION_ID, PRODUCT_ID, VENDOR_ID)])

setkey(pairs_dt, AUCTION_ID, PRODUCT_ID, VENDOR_ID)
setkey(imps, AUCTION_ID, PRODUCT_ID, VENDOR_ID)
setkey(clks, AUCTION_ID, PRODUCT_ID, VENDOR_ID)
pairs_dt <- imps[pairs_dt]
pairs_dt <- clks[pairs_dt]
pairs_dt[is.na(clicked), clicked := 0L]

# Keep only pairs where both members were impressed
keep_pairs <- pairs_dt[, .(n_imp = sum(!is.na(FIRST_IMP_AT))), by = pair_id][n_imp == 2, pair_id]
pairs_dt <- pairs_dt[pair_id %in% keep_pairs]

# Define 'lucky' as the one with better provided RANKING within pair (lower is better)
pairs_dt[, min_rank := min(as.numeric(RANKING)), by = pair_id]
pairs_dt[, lucky := as.integer(as.numeric(RANKING) == min_rank)]

# Controls
pairs_dt[, quality := as.numeric(QUALITY)]
pairs_dt[, price := as.numeric(FINAL_BID)]  # Use FINAL_BID or PRICE? request focuses on score mechanics; keep FINAL_BID control
# Optional: include item price too (not always present here); skip to keep minimal

# Filter near-ties by thresholds
sub_by_tau <- function(thr) pairs_dt[rel_gap <= thr]
dt_tau1 <- sub_by_tau(tau1)
dt_tau2 <- sub_by_tau(tau2)

fit_and_summarize <- function(dt, label) {
  if (nrow(dt) == 0) return(list(text = sprintf("%s: no rows", label)))
  # Two rows per pair_id by construction; use pair fixed effects
  fml <- as.formula("clicked ~ lucky + quality + price | pair_id")
  fit <- tryCatch(feglm(fml, data = dt, family = binomial(), cluster = ~ AUCTION_ID), error = function(e) e)
  return(list(fit = fit, label = label))
}

fit_tau1 <- fit_and_summarize(dt_tau1, sprintf("rel_gap <= %.3f", tau1))
fit_tau2 <- fit_and_summarize(dt_tau2, sprintf("rel_gap <= %.3f", tau2))

# Also boundary-specific fits (2v3, 4v5, ...)
fits_by_boundary <- list()
for (b in boundaries) {
  d <- dt_tau1[boundary == b]
  fits_by_boundary[[as.character(b)]] <- fit_and_summarize(d, sprintf("boundary %d vs %d (rel_gap <= %.3f)", b, b+1, tau1))
}

# Write output
out_dir <- file.path("analysis","position-effects-analysis-R","results")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
outfile <- file.path(out_dir, sprintf("near_tie_pairs_felogit_%s.txt", round_name))
con <- file(outfile, open = "wt"); on.exit(close(con))
cat(sprintf("Near-tie Matched Pairs FE-logit — %s\n", round_name), file = con)
cat(sprintf("Placement: 1 only | Window: last %d minutes\n", window_min), file = con, append = TRUE)
cat("Pairs constructed by score order across mobile fold boundaries: (2 vs 3), (4 vs 5), (6 vs 7), ... up to max_boundary.\n", file = con, append = TRUE)
cat("Both items must be impressed; outcome is clicked (ever clicked in auction).\n\n", file = con, append = TRUE)

cat(sprintf("Universe pairs (before near-tie filter): %s rows (2 rows per pair), %s pairs\n", format(nrow(pairs_dt), big.mark=","), format(length(unique(pairs_dt$pair_id)), big.mark=",")), file = con, append = TRUE)
cat(sprintf("rel_gap <= %.3f: %s rows, %s pairs\n", tau1, format(nrow(dt_tau1), big.mark=","), format(length(unique(dt_tau1$pair_id)), big.mark=",")), file = con, append = TRUE)
cat(sprintf("rel_gap <= %.3f: %s rows, %s pairs\n\n", tau2, format(nrow(dt_tau2), big.mark=","), format(length(unique(dt_tau2$pair_id)), big.mark=",")), file = con, append = TRUE)

write_fit <- function(obj) {
  if (inherits(obj$fit, "error")) {
    cat(sprintf("%s\n  ERROR: %s\n\n", obj$label, obj$fit$message), file = con, append = TRUE)
    return()
  }
  if (is.null(obj$fit)) {
    cat(sprintf("%s\n  (no model)\n\n", obj$label), file = con, append = TRUE)
    return()
  }
  cat(sprintf("%s\n", obj$label), file = con, append = TRUE)
  capture.output(print(summary(obj$fit)), file = con, append = TRUE)
  # Odds ratio for 'lucky'
  co <- tryCatch(coef(obj$fit)["lucky"], error = function(e) NA_real_)
  if (!is.na(co)) {
    or <- exp(co)
    cat(sprintf("Odds ratio(lucky): %.4f\n", or), file = con, append = TRUE)
  }
  cat("\n", file = con, append = TRUE)
}

cat("=== Main fits ===\n", file = con, append = TRUE)
write_fit(fit_tau1)
write_fit(fit_tau2)

cat("=== Boundary-specific fits (rel_gap <= tau1) ===\n", file = con, append = TRUE)
for (nm in names(fits_by_boundary)) write_fit(fits_by_boundary[[nm]])

cat(sprintf("\nOutput saved to: %s\n", outfile), file = con, append = TRUE)

