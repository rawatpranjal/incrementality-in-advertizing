#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(data.table)
  library(fixest)
})

# Prefer duckdb to read parquet without Arrow/OpenMP; fallback to arrow if available
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
    if (!is.null(Sys.getenv("OMP_NUM_THREADS")) && Sys.getenv("OMP_NUM_THREADS") == "") Sys.setenv(OMP_NUM_THREADS = 1)
    if (!is.null(Sys.getenv("ARROW_NUM_THREADS")) && Sys.getenv("ARROW_NUM_THREADS") == "") Sys.setenv(ARROW_NUM_THREADS = 1)
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
placement_filter <- get_arg("--placement", default = NULL)
window_min <- as.integer(get_arg("--window_minutes", default = 15))
if (is.null(round_name)) stop("--round is required")

BASE <- normalizePath("analysis/position-effects/0_data", mustWork = TRUE)
paths <- list(
  auctions_results = if (round_name == "round1") file.path(BASE, "round1/auctions_results_all.parquet") else file.path(BASE, "round2/auctions_results_r2.parquet"),
  impressions      = if (round_name == "round1") file.path(BASE, "round1/impressions_all.parquet")    else file.path(BASE, "round2/impressions_r2.parquet"),
  clicks           = if (round_name == "round1") file.path(BASE, "round1/clicks_all.parquet")         else file.path(BASE, "round2/clicks_r2.parquet"),
  auctions_users   = if (round_name == "round1") file.path(BASE, "round1/auctions_users_all.parquet")  else file.path(BASE, "round2/auctions_users_r2.parquet")
)

# Load minimal columns
imps <- read_parquet_dt(paths$impressions, c("AUCTION_ID","PRODUCT_ID","USER_ID","VENDOR_ID"))
clks <- read_parquet_dt(paths$clicks,      c("AUCTION_ID","PRODUCT_ID","USER_ID","VENDOR_ID"))
ar   <- read_parquet_dt(paths$auctions_results, c("AUCTION_ID","PRODUCT_ID","VENDOR_ID","RANKING","QUALITY","PRICE","CONVERSION_RATE"))
au   <- read_parquet_dt(paths$auctions_users, c("AUCTION_ID","PLACEMENT","CREATED_AT"))
au   <- unique(as.data.table(au), by = "AUCTION_ID")

# 15-minute slice based on CREATED_AT window (last window_min minutes)
au_dt <- as.data.table(au)
au_dt[, CREATED_AT := as.POSIXct(CREATED_AT, tz = "UTC")]
end_time <- max(au_dt$CREATED_AT, na.rm = TRUE)
start_time <- end_time - window_min*60
au_dt <- au_dt[CREATED_AT >= start_time & CREATED_AT <= end_time]

# Build impression-level join
imps <- as.data.table(imps)
clks <- as.data.table(clks)
ar   <- as.data.table(ar)
setkey(imps, AUCTION_ID, PRODUCT_ID, USER_ID, VENDOR_ID)
setkey(clks, AUCTION_ID, PRODUCT_ID, USER_ID, VENDOR_ID)
# Create clicked indicator: 1 if impression was clicked, 0 otherwise
clks[, clicked := 1L]
dt <- clks[imps, on = .(AUCTION_ID, PRODUCT_ID, USER_ID, VENDOR_ID)]
dt[is.na(clicked), clicked := 0L]
dt <- au_dt[dt, on = .(AUCTION_ID)]
ar_small <- unique(ar, by = c("AUCTION_ID","PRODUCT_ID","VENDOR_ID"))
dt <- ar_small[dt, on = .(AUCTION_ID, PRODUCT_ID, VENDOR_ID)]

# Variable prep and filters
setnames(dt, c("RANKING","QUALITY","PRICE","CONVERSION_RATE"), c("rank","quality","price","cvr"))
dt <- dt[!is.na(rank) & !is.na(quality) & !is.na(price) & !is.na(cvr) & !is.na(PLACEMENT)]
dt <- dt[rank > 0 & quality > 0 & price > 0 & cvr > 0]
if (!is.null(placement_filter)) dt <- dt[as.character(PLACEMENT) == as.character(placement_filter)]
dt[, `:=`(clicked = as.integer(clicked), VENDOR_ID = as.character(VENDOR_ID), PLACEMENT = as.character(PLACEMENT))]

# FE-logit
fml <- as.formula("clicked ~ quality + rank + price + cvr | VENDOR_ID + PLACEMENT")
fit <- feglm(fml, data = dt, family = binomial(), cluster = ~ AUCTION_ID)

# 1 SD odds changes for numeric vars
sdv <- dt[, .(sd_q = sd(quality), sd_r = sd(rank), sd_p = sd(price), sd_c = sd(cvr))]
co <- coef(fit)
beta_quality <- as.numeric(co["quality"])
beta_rank <- as.numeric(co["rank"])
beta_price <- as.numeric(co["price"])
beta_cvr <- as.numeric(co["cvr"])
odds_1sd <- c(
  quality = exp(beta_quality * sdv$sd_q) - 1,
  rank    = exp(beta_rank    * sdv$sd_r) - 1,
  price   = exp(beta_price   * sdv$sd_p) - 1,
  cvr     = exp(beta_cvr     * sdv$sd_c) - 1
)

# Placement FE odds multiples (relative to baseline absorbed level)
plc_levels <- sort(unique(dt$PLACEMENT))
pl_odds <- NULL
if (length(plc_levels) > 1) {
  # fixest stores FE via i() expansions if used; here we used fixed part, so we compute means by placement using predict with FE shifts
  # Simpler: compute placement-specific intercept differences via fe fixef
  fe_list <- fixef(fit)
  if (!is.null(fe_list$PLACEMENT)) {
    base <- mean(fe_list$PLACEMENT)
    pl_odds <- exp(fe_list$PLACEMENT - base)
  }
}

# Write out
out_dir <- file.path("analysis","position-effects-analysis-R","results")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
outfile <- file.path(out_dir, sprintf("position_bias_felogit_fixest_%s%s.txt", round_name, if (!is.null(placement_filter)) paste0("_pl", placement_filter) else ""))
con <- file(outfile, open = "wt"); on.exit(close(con))
cat(sprintf("Position Bias FE-logit (fixest) — %s\n", round_name), file = con)
if (!is.null(placement_filter)) cat(sprintf("Restricted to PLACEMENT=%s\n", placement_filter), file = con, append = TRUE)
cat(sprintf("Window: last %d minutes based on AUCTIONS_USERS.CREATED_AT\n\n", window_min), file = con, append = TRUE)
cat("Variables (precise):\n", file = con, append = TRUE)
cat("  clicked: impression click indicator (0/1)\n", file = con, append = TRUE)
cat("  quality: AUCTIONS_RESULTS.QUALITY (pCTR proxy)\n", file = con, append = TRUE)
cat("  rank: AUCTIONS_RESULTS.RANKING (1 = best)\n", file = con, append = TRUE)
cat("  price: AUCTIONS_RESULTS.PRICE\n", file = con, append = TRUE)
cat("  cvr: AUCTIONS_RESULTS.CONVERSION_RATE (pCVR proxy)\n", file = con, append = TRUE)
cat("  Fixed effects: VENDOR_ID, PLACEMENT; cluster by AUCTION_ID\n\n", file = con, append = TRUE)
cat(sprintf("Rows=%s CTR=%.3f%% Vendors=%s Placements=%s\n\n", format(nrow(dt), big.mark=","), 100*mean(dt$clicked), length(unique(dt$VENDOR_ID)), length(unique(dt$PLACEMENT))), file = con, append = TRUE)
capture.output(print(summary(fit)), file = con, append = TRUE)
cat("\n1 SD odds changes (exp(beta*sd) - 1):\n", file = con, append = TRUE)
cat(sprintf("  quality: %.4f\n  rank: %.4f\n  price: %.4f\n  cvr: %.4f\n", odds_1sd["quality"], odds_1sd["rank"], odds_1sd["price"], odds_1sd["cvr"]), file = con, append = TRUE)
if (!is.null(pl_odds)) {
  cat("\nPlacement FE odds multipliers (relative to baseline mean FE):\n", file = con, append = TRUE)
  for (nm in names(pl_odds)) cat(sprintf("  %s: %.3f\n", nm, pl_odds[[nm]]), file = con, append = TRUE)
}
cat(sprintf("\nOutput saved to: %s\n", outfile), file = con, append = TRUE)

