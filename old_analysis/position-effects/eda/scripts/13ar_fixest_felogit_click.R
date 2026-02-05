#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(arrow)
  library(data.table)
  library(fixest)
  library(pROC)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("Usage: Rscript 13ar_fixest_felogit_click.R --round round1|round2 [--placement 1|2|3|5]", call. = FALSE)
}

# Simple arg parse
get_arg <- function(flag, default = NULL) {
  hit <- which(args == flag)
  if (length(hit) == 0) return(default)
  if (hit == length(args)) return(TRUE)
  val <- args[hit + 1]
  if (startsWith(val, "--")) return(TRUE) else return(val)
}

round_name <- get_arg("--round")
placement_filter <- get_arg("--placement", default = NULL)
if (is.null(round_name)) stop("--round is required", call. = FALSE)

BASE <- normalizePath(file.path(dirname(dirname(dirname(sys.frame(1)$ofile %||% ""))), "0_data"), mustWork = FALSE)
if (!dir.exists(BASE)) {
  BASE <- normalizePath("analysis/position-effects/0_data", mustWork = TRUE)
}

paths <- list(
  auctions_results = if (round_name == "round1") file.path(BASE, "round1/auctions_results_all.parquet") else file.path(BASE, "round2/auctions_results_r2.parquet"),
  impressions      = if (round_name == "round1") file.path(BASE, "round1/impressions_all.parquet")    else file.path(BASE, "round2/impressions_r2.parquet"),
  clicks           = if (round_name == "round1") file.path(BASE, "round1/clicks_all.parquet")         else file.path(BASE, "round2/clicks_r2.parquet"),
  auctions_users   = if (round_name == "round1") file.path(BASE, "round1/auctions_users_all.parquet")  else file.path(BASE, "round2/auctions_users_r2.parquet")
)

# Load data
imps <- as.data.table(read_parquet(paths$impressions, as_data_frame = TRUE, col_select = c("AUCTION_ID","PRODUCT_ID","USER_ID","VENDOR_ID")))
clks <- as.data.table(read_parquet(paths$clicks,      as_data_frame = TRUE, col_select = c("AUCTION_ID","PRODUCT_ID","USER_ID","VENDOR_ID")))
ar   <- as.data.table(read_parquet(paths$auctions_results, as_data_frame = TRUE,
                col_select = c("AUCTION_ID","PRODUCT_ID","VENDOR_ID","RANKING","QUALITY","PRICE","CONVERSION_RATE")))
au   <- as.data.table(read_parquet(paths$auctions_users, as_data_frame = TRUE, col_select = c("AUCTION_ID","PLACEMENT")))
au   <- unique(au, by = "AUCTION_ID")

# Build impression-level frame
clks[, clicked := 1L]
setkey(imps, AUCTION_ID, PRODUCT_ID, USER_ID, VENDOR_ID)
setkey(clks, AUCTION_ID, PRODUCT_ID, USER_ID, VENDOR_ID)
dt <- clks[imps, on = .(AUCTION_ID, PRODUCT_ID, USER_ID, VENDOR_ID)]
dt[is.na(clicked), clicked := 0L]
dt <- au[dt, on = .(AUCTION_ID)]
ar_small <- unique(ar, by = c("AUCTION_ID","PRODUCT_ID","VENDOR_ID"))
dt <- ar_small[dt, on = .(AUCTION_ID, PRODUCT_ID, VENDOR_ID)]

# Rename variables and filter
setnames(dt, c("RANKING","QUALITY","PRICE","CONVERSION_RATE"), c("rank","quality","price","cvr"))
dt <- dt[!is.na(rank) & !is.na(quality) & !is.na(price) & !is.na(cvr) & !is.na(PLACEMENT)]
dt <- dt[rank > 0 & quality > 0 & price > 0 & cvr > 0]
if (!is.null(placement_filter)) dt <- dt[as.character(PLACEMENT) == as.character(placement_filter)]
dt[, `:=`(clicked = as.integer(clicked), VENDOR_ID = as.character(VENDOR_ID), PLACEMENT = as.character(PLACEMENT))]

# Fit FE-logit with fixest
fml <- as.formula("clicked ~ quality + rank + price + cvr | VENDOR_ID + PLACEMENT")
fit <- feglm(fml, data = dt, family = binomial(), cluster = ~ AUCTION_ID)

# Predictions and AUC
dt[, p := as.numeric(predict(fit, type = "response"))]
auc <- as.numeric(pROC::auc(dt$clicked, dt$p))

# Write results
out_dir <- file.path(dirname(dirname(paths$impressions)), "..", "eda", "results")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
outfile <- file.path(out_dir, sprintf("13ar_fixest_felogit_click_%s%s.txt", round_name, if (!is.null(placement_filter)) paste0("_pl", placement_filter) else ""))
con <- file(outfile, open = "wt"); on.exit(close(con))
cat(sprintf("FE-logit (fixest) — %s\n", round_name), file = con)
if (!is.null(placement_filter)) cat(sprintf("Restricted to PLACEMENT=%s\n", placement_filter), file = con, append = TRUE)
cat("Variables (precise definitions):\n", file = con, append = TRUE)
cat("  clicked: binary indicator for whether the impression received a click\n", file = con, append = TRUE)
cat("  quality: QUALITY score from AUCTIONS_RESULTS (platform pCTR proxy)\n", file = con, append = TRUE)
cat("  rank: bid rank (1 = best) from AUCTIONS_RESULTS\n", file = con, append = TRUE)
cat("  price: PRICE from AUCTIONS_RESULTS (AOV proxy)\n", file = con, append = TRUE)
cat("  cvr: CONVERSION_RATE from AUCTIONS_RESULTS (platform pCVR proxy)\n", file = con, append = TRUE)
cat("  PLACEMENT: surface identifier from AUCTIONS_USERS (fixed effect)\n", file = con, append = TRUE)
cat("  VENDOR_ID: advertiser identifier from AUCTIONS_RESULTS (fixed effect)\n\n", file = con, append = TRUE)

cat(sprintf("Rows=%s CTR=%.3f%% Vendors=%s Placements=%s\n", format(nrow(dt), big.mark=","), 100*mean(dt$clicked), length(unique(dt$VENDOR_ID)), length(unique(dt$PLACEMENT))), file = con, append = TRUE)
cat(sprintf("ROC_AUC=%.4f\n\n", auc), file = con, append = TRUE)
cat("Model summary (fixest):\n", file = con, append = TRUE)
capture.output(print(summary(fit)), file = con, append = TRUE)
cat(sprintf("\nOutput saved to: %s\n", outfile), file = con, append = TRUE)

