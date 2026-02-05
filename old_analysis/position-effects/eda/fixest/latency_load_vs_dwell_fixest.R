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
placement_filter <- get_arg("--placement", default = "1")
if (is.null(round_name)) stop("--round is required")

BASE <- normalizePath("analysis/position-effects/0_data", mustWork = TRUE)
paths <- list(
  auctions_results = if (round_name == "round1") file.path(BASE, "round1/auctions_results_all.parquet") else file.path(BASE, "round2/auctions_results_r2.parquet"),
  impressions      = if (round_name == "round1") file.path(BASE, "round1/impressions_all.parquet")    else file.path(BASE, "round2/impressions_r2.parquet"),
  clicks           = if (round_name == "round1") file.path(BASE, "round1/clicks_all.parquet")         else file.path(BASE, "round2/clicks_r2.parquet"),
  auctions_users   = if (round_name == "round1") file.path(BASE, "round1/auctions_users_all.parquet")  else file.path(BASE, "round2/auctions_users_r2.parquet")
)

imps <- read_parquet_dt(paths$impressions, c("AUCTION_ID","PRODUCT_ID","USER_ID","VENDOR_ID","OCCURRED_AT"))
clks <- read_parquet_dt(paths$clicks,      c("AUCTION_ID","PRODUCT_ID","USER_ID","VENDOR_ID"))
ar   <- read_parquet_dt(paths$auctions_results, c("AUCTION_ID","PRODUCT_ID","VENDOR_ID","RANKING","QUALITY","PRICE"))
au   <- read_parquet_dt(paths$auctions_users, c("AUCTION_ID","PLACEMENT","CREATED_AT"))
au   <- unique(as.data.table(au), by = "AUCTION_ID")

setDT(imps); setDT(clks); setDT(ar); setDT(au)
imps[, OCCURRED_AT := as.POSIXct(OCCURRED_AT, tz = "UTC")]
au[, CREATED_AT := as.POSIXct(CREATED_AT, tz = "UTC")]
clks[, clicked := 1L]
setkey(imps, AUCTION_ID, PRODUCT_ID, USER_ID, VENDOR_ID)
setkey(clks, AUCTION_ID, PRODUCT_ID, USER_ID, VENDOR_ID)
dt <- clks[imps, on = .(AUCTION_ID, PRODUCT_ID, USER_ID, VENDOR_ID)]
dt[is.na(clicked), clicked := 0L]
dt <- au[dt, on = .(AUCTION_ID)]
ar_small <- unique(ar, by = c("AUCTION_ID","PRODUCT_ID","VENDOR_ID"))
dt <- ar_small[dt, on = .(AUCTION_ID, PRODUCT_ID, VENDOR_ID)]

setnames(dt, c("RANKING","QUALITY","PRICE"), c("rank","quality","price"))
dt <- dt[!is.na(rank) & !is.na(quality) & !is.na(price) & !is.na(PLACEMENT) & !is.na(OCCURRED_AT) & !is.na(CREATED_AT)]
dt <- dt[rank > 0 & quality > 0 & price > 0]
dt[, `:=`(clicked = as.integer(clicked), VENDOR_ID = as.character(VENDOR_ID), PLACEMENT = as.character(PLACEMENT))]
if (!is.null(placement_filter)) dt <- dt[PLACEMENT == placement_filter]

# Load/dwell construction
first_occ <- dt[, .(first_occ = min(OCCURRED_AT)), by = AUCTION_ID]
dt <- first_occ[dt, on = .(AUCTION_ID)]
dt[, load_s := as.numeric(difftime(first_occ, CREATED_AT, units = "secs"))]
dt[, dwell_s := as.numeric(difftime(OCCURRED_AT, first_occ, units = "secs"))]
dt[load_s <= 0, load_s := NA_real_]
dt[dwell_s < 0, dwell_s := NA_real_]
dt[, log_load := log(load_s)]
dt[is.na(dwell_s) | dwell_s == 0, log_dwell := 0]  # first impression gets 0
dt[dwell_s > 0, log_dwell := log(dwell_s)]

# Normalize dwell by placement×rank baseline (cap rank)
dt[, rank_capped := pmin(rank, 50L)]
base_dwell <- dt[dwell_s > 0, .(baseline_log_dwell = mean(log_dwell)), by = .(PLACEMENT, rank_capped)]
dt <- base_dwell[dt, on = .(PLACEMENT, rank_capped)]
dt[is.na(baseline_log_dwell), baseline_log_dwell := 0]
dt[, dwell_resid := log_dwell - baseline_log_dwell]

# Quadratics
dt[, `:=`(log_load_sq = log_load^2, dwell_resid_sq = dwell_resid^2)]

# FE-logit (vendor FE); placement is restricted to 1 here, so not included as FE
fml <- as.formula("clicked ~ quality + rank + price + log_load + log_load_sq + dwell_resid + dwell_resid_sq | VENDOR_ID")
fit <- feglm(fml, data = dt, family = binomial(), cluster = ~ AUCTION_ID)

# APEs (delta on probability) for +10% changes
invlogit <- function(x) 1.0/(1.0 + exp(-x))
eta <- as.numeric(fit$linear.predictors)
p  <- invlogit(eta)
beta <- coef(fit)
dl <- log(1.1)
# Use model data (after FE removal) for APE calculation
dt_used <- model.matrix(fit, type = "lhs", as.matrix = FALSE)
n_used <- length(eta)
# Get log_load and dwell_resid for used observations
log_load_used <- dt$log_load[1:n_used]  # fallback: use first n_used
dwell_resid_used <- dt$dwell_resid[1:n_used]
slope_load  <- as.numeric(beta["log_load"])  + 2*as.numeric(beta["log_load_sq"])  * log_load_used
slope_dwell <- as.numeric(beta["dwell_resid"]) + 2*as.numeric(beta["dwell_resid_sq"]) * dwell_resid_used
ape_load  <- mean(slope_load  * p * (1 - p), na.rm = TRUE)
ape_dwell <- mean(slope_dwell * p * (1 - p), na.rm = TRUE)

out_dir <- file.path("analysis","position-effects","eda","results")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
outfile <- file.path(out_dir, sprintf("latency_load_vs_dwell_fixest_%s_pl%s.txt", round_name, placement_filter))
con <- file(outfile, open = "wt"); on.exit(close(con))
cat(sprintf("Latency (load vs dwell) FE-logit (fixest) — %s, PLACEMENT=%s\n\n", round_name, placement_filter), file = con)
cat("Variables:\n  clicked: 0/1\n  quality: AUCTIONS_RESULTS.QUALITY\n  rank: AUCTIONS_RESULTS.RANKING\n  price: AUCTIONS_RESULTS.PRICE\n  load_s: first impression time minus request time\n  log_load, log_load_sq: log(load_s) and its square\n  dwell_s: impression time minus first impression time\n  log_dwell: log(dwell_s) for non-first rows, 0 for first\n  dwell_resid: log_dwell minus mean at placement×rank (rank capped 50)\n  dwell_resid_sq: squared residual\n  FE: VENDOR_ID; cluster by AUCTION_ID\n\n", file = con, append = TRUE)
cat(sprintf("Rows=%s CTR=%.3f%% Vendors=%s\n\n", format(nrow(dt), big.mark=","), 100*mean(dt$clicked), length(unique(dt$VENDOR_ID))), file = con, append = TRUE)
capture.output(print(summary(fit)), file = con, append = TRUE)
cat(sprintf("\nAPE for log_load: %.6f per unit log; +10%% effect: %.6f\n", ape_load, ape_load*dl), file = con, append = TRUE)
cat(sprintf("APE for dwell_resid: %.6f per unit log; +10%% effect: %.6f\n", ape_dwell, ape_dwell*dl), file = con, append = TRUE)
cat(sprintf("\nOutput saved to: %s\n", outfile), file = con, append = TRUE)

