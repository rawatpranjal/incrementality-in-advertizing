#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(data.table)
  library(fixest)
})

if (!requireNamespace("rdrobust", quietly = TRUE)) {
  stop("Package 'rdrobust' is required. Install with: install.packages('rdrobust')")
}
if (!requireNamespace("rddensity", quietly = TRUE)) {
  stop("Package 'rddensity' is required. Install with: install.packages('rddensity')")
}

library(rdrobust)
library(rddensity)

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
placements_arg <- get_arg("--placements", default = "1")
devices_arg <- get_arg("--devices", default = "mobile,desktop")
if (is.null(round_name)) stop("--round is required")
placements <- as.character(unlist(strsplit(placements_arg, ",")))
devices <- as.character(unlist(strsplit(devices_arg, ",")))

BASE <- normalizePath("analysis/position-effects/0_data", mustWork = TRUE)
paths <- list(
  auctions_results = if (round_name == "round1") file.path(BASE, "round1/auctions_results_all.parquet") else file.path(BASE, "round2/auctions_results_r2.parquet"),
  impressions      = if (round_name == "round1") file.path(BASE, "round1/impressions_all.parquet")    else file.path(BASE, "round2/impressions_r2.parquet"),
  clicks           = if (round_name == "round1") file.path(BASE, "round1/clicks_all.parquet")         else file.path(BASE, "round2/clicks_r2.parquet"),
  auctions_users   = if (round_name == "round1") file.path(BASE, "round1/auctions_users_all.parquet")  else file.path(BASE, "round2/auctions_users_r2.parquet")
)

imps <- read_parquet_dt(paths$impressions, c("AUCTION_ID","PRODUCT_ID","USER_ID","VENDOR_ID","OCCURRED_AT"))
clks <- read_parquet_dt(paths$clicks, c("AUCTION_ID","PRODUCT_ID","USER_ID","VENDOR_ID"))
ar <- read_parquet_dt(paths$auctions_results, c("AUCTION_ID","PRODUCT_ID","VENDOR_ID","RANKING","QUALITY","PRICE","CONVERSION_RATE"))
au <- read_parquet_dt(paths$auctions_users, c("AUCTION_ID","PLACEMENT"))
au <- unique(as.data.table(au), by = "AUCTION_ID")

setDT(imps); setDT(clks); setDT(ar); setDT(au)
imps[, OCCURRED_AT := as.POSIXct(OCCURRED_AT, tz = "UTC")]
imps[, occ_second := floor(as.numeric(OCCURRED_AT))]

clks[, clicked := 1L]
setkey(imps, AUCTION_ID, PRODUCT_ID, USER_ID, VENDOR_ID)
setkey(clks, AUCTION_ID, PRODUCT_ID, USER_ID, VENDOR_ID)
dt <- clks[imps, on = .(AUCTION_ID, PRODUCT_ID, USER_ID, VENDOR_ID)]
dt[is.na(clicked), clicked := 0L]
dt <- au[dt, on = .(AUCTION_ID)]
ar_small <- unique(ar, by = c("AUCTION_ID","PRODUCT_ID","VENDOR_ID"))
dt <- ar_small[dt, on = .(AUCTION_ID, PRODUCT_ID, VENDOR_ID)]

setnames(dt, c("RANKING","QUALITY","PRICE","CONVERSION_RATE"), c("rank","quality","price","cvr"))
dt <- dt[!is.na(rank) & !is.na(quality) & !is.na(price) & !is.na(PLACEMENT)]
dt <- dt[rank > 0 & quality > 0 & price > 0]
dt[, PLACEMENT := as.character(PLACEMENT)]

burst_sizes <- dt[, .(burst_size = .N), by = .(USER_ID, PLACEMENT, occ_second)]
user_modal_burst <- burst_sizes[, {
  tbl <- table(burst_size)
  modal_burst <- as.integer(names(tbl)[which.max(tbl)])
  list(modal_burst = modal_burst)
}, by = USER_ID]

user_modal_burst[, device := fifelse(modal_burst <= 2, "mobile", "desktop")]
dt <- user_modal_burst[, .(USER_ID, device)][dt, on = .(USER_ID)]
dt <- dt[!is.na(device)]

out_dir <- file.path("analysis","position-effects-analysis-R","results")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

run_rdd_analysis <- function(dt_sub, pl, dev, con) {
  if (dev == "mobile") {
    cutoff <- 2.5
    fold_label <- "rank <= 2 (above fold)"
  } else {
    cutoff <- 4.5
    fold_label <- "rank <= 4 (above fold)"
  }

  dt_sub[, z := rank - cutoff]
  dt_sub[, above_fold := as.integer(rank <= floor(cutoff))]

  cat(sprintf("\n%s\n", paste(rep("=", 60), collapse="")), file = con, append = TRUE)
  cat(sprintf("PLACEMENT %s - %s (cutoff = %.1f)\n", pl, toupper(dev), cutoff), file = con, append = TRUE)
  cat(sprintf("%s\n\n", paste(rep("=", 60), collapse="")), file = con, append = TRUE)

  cat(sprintf("N observations: %s\n", format(nrow(dt_sub), big.mark=",")), file = con, append = TRUE)
  cat(sprintf("N above fold (rank <= %.0f): %s (%.1f%%)\n", floor(cutoff), format(sum(dt_sub$above_fold), big.mark=","), 100*mean(dt_sub$above_fold)), file = con, append = TRUE)
  cat(sprintf("N below fold: %s (%.1f%%)\n", format(sum(1 - dt_sub$above_fold), big.mark=","), 100*mean(1 - dt_sub$above_fold)), file = con, append = TRUE)
  cat(sprintf("Overall CTR: %.3f%%\n", 100*mean(dt_sub$clicked)), file = con, append = TRUE)
  cat(sprintf("CTR above fold: %.3f%%\n", 100*mean(dt_sub[above_fold == 1]$clicked)), file = con, append = TRUE)
  cat(sprintf("CTR below fold: %.3f%%\n\n", 100*mean(dt_sub[above_fold == 0]$clicked)), file = con, append = TRUE)

  cat(sprintf("%s\n", paste(rep("-", 40), collapse="")), file = con, append = TRUE)
  cat("McCrary Density Test (rddensity)\n", file = con, append = TRUE)
  cat(sprintf("%s\n", paste(rep("-", 40), collapse="")), file = con, append = TRUE)

  tryCatch({
    dens_test <- rddensity(dt_sub$z, c = 0)
    cat(sprintf("  Test statistic: %.4f\n", dens_test$test$t_jk), file = con, append = TRUE)
    cat(sprintf("  p-value: %.4f\n", dens_test$test$p_jk), file = con, append = TRUE)
    cat(sprintf("  Interpretation: %s\n\n", ifelse(dens_test$test$p_jk < 0.05, "Manipulation concern (reject null)", "No evidence of manipulation")), file = con, append = TRUE)
  }, error = function(e) {
    cat(sprintf("  Error in density test: %s\n\n", e$message), file = con, append = TRUE)
  })

  cat(sprintf("%s\n", paste(rep("-", 40), collapse="")), file = con, append = TRUE)
  cat("RD Estimates (rdrobust)\n", file = con, append = TRUE)
  cat(sprintf("%s\n", paste(rep("-", 40), collapse="")), file = con, append = TRUE)

  rd_success <- FALSE
  tryCatch({
    rd_fit <- rdrobust(dt_sub$clicked, dt_sub$z, c = 0, kernel = "triangular")
    rd_success <- TRUE
    cat("\nConventional RD estimate:\n", file = con, append = TRUE)
    cat(sprintf("  LATE: %.6f (SE = %.6f)\n", rd_fit$coef["Conventional"], rd_fit$se["Conventional"]), file = con, append = TRUE)
    cat(sprintf("  95%% CI: [%.6f, %.6f]\n", rd_fit$ci["Conventional", "CI Lower"], rd_fit$ci["Conventional", "CI Upper"]), file = con, append = TRUE)
    cat(sprintf("  p-value: %.4f\n", rd_fit$pv["Conventional"]), file = con, append = TRUE)

    cat("\nBias-corrected RD estimate:\n", file = con, append = TRUE)
    cat(sprintf("  LATE: %.6f (SE = %.6f)\n", rd_fit$coef["Bias-Corrected"], rd_fit$se["Bias-Corrected"]), file = con, append = TRUE)
    cat(sprintf("  95%% CI: [%.6f, %.6f]\n", rd_fit$ci["Bias-Corrected", "CI Lower"], rd_fit$ci["Bias-Corrected", "CI Upper"]), file = con, append = TRUE)
    cat(sprintf("  p-value: %.4f\n", rd_fit$pv["Bias-Corrected"]), file = con, append = TRUE)

    cat("\nRobust RD estimate:\n", file = con, append = TRUE)
    cat(sprintf("  LATE: %.6f (SE = %.6f)\n", rd_fit$coef["Robust"], rd_fit$se["Robust"]), file = con, append = TRUE)
    cat(sprintf("  95%% CI: [%.6f, %.6f]\n", rd_fit$ci["Robust", "CI Lower"], rd_fit$ci["Robust", "CI Upper"]), file = con, append = TRUE)
    cat(sprintf("  p-value: %.4f\n", rd_fit$pv["Robust"]), file = con, append = TRUE)

    cat(sprintf("\nBandwidth (h): %.4f\n", rd_fit$bws["h", "left"]), file = con, append = TRUE)
    cat(sprintf("N effective left: %d\n", rd_fit$N_h["left"]), file = con, append = TRUE)
    cat(sprintf("N effective right: %d\n\n", rd_fit$N_h["right"]), file = con, append = TRUE)
  }, error = function(e) {
    cat(sprintf("  rdrobust failed (likely due to discrete running variable): %s\n", e$message), file = con, append = TRUE)
  })

  if (!rd_success) {
    cat("\nFallback: Local Linear Regression (feols)\n", file = con, append = TRUE)
    cat("  Model: clicked ~ above_fold * z (within +/- 3 ranks of cutoff)\n", file = con, append = TRUE)
    dt_local <- dt_sub[abs(z) <= 3]
    if (nrow(dt_local) >= 100) {
      fit_lm <- feols(clicked ~ above_fold * z, data = dt_local, vcov = "hetero")
      beta_fold <- coef(fit_lm)["above_fold"]
      se_fold <- se(fit_lm)["above_fold"]
      pv_fold <- pvalue(fit_lm)["above_fold"]
      cat(sprintf("  Discontinuity (above_fold): %.6f (SE = %.6f)\n", beta_fold, se_fold), file = con, append = TRUE)
      cat(sprintf("  p-value: %.4f\n", pv_fold), file = con, append = TRUE)
      cat(sprintf("  N within bandwidth: %d\n", nrow(dt_local)), file = con, append = TRUE)
      cat(sprintf("  Interpretation: CTR %s by %.3f pp at fold\n\n",
        ifelse(beta_fold > 0, "increases", "decreases"), abs(beta_fold) * 100), file = con, append = TRUE)
    } else {
      cat(sprintf("  Insufficient observations within bandwidth (N=%d)\n\n", nrow(dt_local)), file = con, append = TRUE)
    }
  }

  cat(sprintf("%s\n", paste(rep("-", 40), collapse="")), file = con, append = TRUE)
  cat("Covariate Balance Tests\n", file = con, append = TRUE)
  cat(sprintf("%s\n", paste(rep("-", 40), collapse="")), file = con, append = TRUE)

  for (covar in c("quality", "price", "cvr")) {
    if (!covar %in% names(dt_sub)) next
    dt_covar <- dt_sub[!is.na(get(covar)) & is.finite(get(covar))]
    if (nrow(dt_covar) < 100) next

    tryCatch({
      rd_covar <- rdrobust(dt_covar[[covar]], dt_covar$z, c = 0, kernel = "triangular")
      cat(sprintf("\n%s:\n", covar), file = con, append = TRUE)
      cat(sprintf("  Discontinuity: %.6f (SE = %.6f)\n", rd_covar$coef["Conventional"], rd_covar$se["Conventional"]), file = con, append = TRUE)
      cat(sprintf("  p-value: %.4f\n", rd_covar$pv["Conventional"]), file = con, append = TRUE)
      cat(sprintf("  Interpretation: %s\n", ifelse(rd_covar$pv["Conventional"] < 0.05, "Imbalance detected", "Balanced")), file = con, append = TRUE)
    }, error = function(e) {
      cat(sprintf("\n%s: Error - %s\n", covar, e$message), file = con, append = TRUE)
    })
  }

  cat(sprintf("\n%s\n", paste(rep("-", 40), collapse="")), file = con, append = TRUE)
  cat("Sensitivity: Manual Bandwidth Windows\n", file = con, append = TRUE)
  cat(sprintf("%s\n", paste(rep("-", 40), collapse="")), file = con, append = TRUE)

  for (bw in c(1, 2, 3)) {
    dt_window <- dt_sub[abs(z) <= bw]
    if (nrow(dt_window) < 50) {
      cat(sprintf("\n+/- %d ranks: Insufficient observations (N=%d)\n", bw, nrow(dt_window)), file = con, append = TRUE)
      next
    }

    tryCatch({
      rd_manual <- rdrobust(dt_window$clicked, dt_window$z, c = 0, kernel = "uniform", h = bw)
      cat(sprintf("\n+/- %d ranks (N=%s):\n", bw, format(nrow(dt_window), big.mark=",")), file = con, append = TRUE)
      cat(sprintf("  LATE: %.6f (SE = %.6f), p = %.4f\n", rd_manual$coef["Conventional"], rd_manual$se["Conventional"], rd_manual$pv["Conventional"]), file = con, append = TRUE)
    }, error = function(e) {
      naive_above <- mean(dt_window[z < 0]$clicked)
      naive_below <- mean(dt_window[z >= 0]$clicked)
      cat(sprintf("\n+/- %d ranks (N=%s):\n", bw, format(nrow(dt_window), big.mark=",")), file = con, append = TRUE)
      cat(sprintf("  Naive diff: %.6f (above CTR: %.4f, below CTR: %.4f)\n", naive_above - naive_below, naive_above, naive_below), file = con, append = TRUE)
    })
  }

  cat("\n", file = con, append = TRUE)
}

for (pl in placements) {
  for (dev in devices) {
    dt_sub <- dt[PLACEMENT == pl & device == dev]
    if (nrow(dt_sub) < 1000) {
      message(sprintf("Skipping PLACEMENT %s, device %s: insufficient observations (N=%d)", pl, dev, nrow(dt_sub)))
      next
    }

    outfile <- file.path(out_dir, sprintf("viewport_fold_rdd_%s_pl%s_%s.txt", round_name, pl, dev))
    con <- file(outfile, open = "wt")

    cat(sprintf("Viewport Fold RDD Analysis - %s\n", round_name), file = con)
    cat(sprintf("Placement: %s, Device: %s\n", pl, dev), file = con, append = TRUE)
    cat(sprintf("Date: %s\n", Sys.time()), file = con, append = TRUE)
    cat("\n", file = con, append = TRUE)

    cat("Design:\n", file = con, append = TRUE)
    cat("  Running variable: z = rank - cutoff\n", file = con, append = TRUE)
    if (dev == "mobile") {
      cat("  Mobile cutoff: c = 2.5 (above fold: rank <= 2)\n", file = con, append = TRUE)
    } else {
      cat("  Desktop cutoff: c = 4.5 (above fold: rank <= 4)\n", file = con, append = TRUE)
    }
    cat("  Treatment: above_fold indicator\n", file = con, append = TRUE)
    cat("  Outcome: clicked (0/1)\n", file = con, append = TRUE)
    cat("\n", file = con, append = TRUE)

    cat("Device proxy:\n", file = con, append = TRUE)
    cat("  Inferred from viewport burst size (impressions grouped by user/placement/second)\n", file = con, append = TRUE)
    cat("  Modal burst <= 2 -> mobile; modal burst > 2 -> desktop\n", file = con, append = TRUE)

    run_rdd_analysis(dt_sub, pl, dev, con)

    close(con)
    message(sprintf("Output saved to: %s", outfile))
  }
}

message("RDD analysis complete.")
