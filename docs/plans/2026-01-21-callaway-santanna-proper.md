# Callaway-Sant'Anna Proper Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run Callaway-Sant'Anna (2021) using the official R `did` package via rpy2, add segmentation analysis, and consolidate all findings into a single master document.

**Architecture:** Export panel data to R, run `att_gt()` and `aggte()` from the `did` package, return results to Python for segmentation analysis and final document creation.

**Tech Stack:** R 4.4.2 with `did` package v2.2.1.910, Python via rpy2, pandas, numpy.

---

## Background

The Callaway-Sant'Anna (2021) estimator computes group-time average treatment effects ATT(g,t) for each cohort g at each time t, then aggregates them into event-study coefficients θ(e). The R `did` package is the canonical implementation by the paper authors.

**Current State:**
- Panel exists: `staggered-adoption/data/panel_total_gmv.parquet` (846,430 obs)
- Manual implementation completed: `results/CALLAWAY_SANTANNA_FINAL.txt`
- TWFE descriptive results: `results/DESCRIPTIVE_FINDINGS.txt`

**Target State:**
- Run `did::att_gt()` with proper standard errors
- Event study aggregation via `did::aggte()`
- Segmentation by vendor activity/price tier
- Single master document: `results/MASTER_RESULTS.txt`

---

### Task 1: Prepare Panel Data for R

**Files:**
- Read: `staggered-adoption/data/panel_total_gmv.parquet`
- Create: `staggered-adoption/data/panel_for_r.csv`

**Step 1: Export panel to CSV with R-compatible format**

```python
import pandas as pd
import numpy as np

panel = pd.read_parquet('staggered-adoption/data/panel_total_gmv.parquet')

# Create integer IDs for R
panel['vendor_id_int'] = panel['VENDOR_ID'].astype('category').cat.codes + 1

# Convert week to integer (period number)
panel['period'] = (pd.to_datetime(panel['week']) - pd.to_datetime('2025-03-24')).dt.days // 7 + 1

# Cohort as period number (0 for never-treated per did package convention)
first_period = pd.to_datetime('2025-03-24')
panel['cohort_period'] = panel['cohort'].apply(
    lambda x: 0 if pd.isna(x) else int((pd.to_datetime(x) - first_period).days // 7 + 1)
)

# Export columns needed for did package
r_panel = panel[['vendor_id_int', 'period', 'cohort_period',
                 'impressions', 'clicks', 'total_gmv', 'log_gmv']].copy()
r_panel.to_csv('staggered-adoption/data/panel_for_r.csv', index=False)
print(f"Exported {len(r_panel)} rows")
```

**Step 2: Verify export**

Run: `head -5 staggered-adoption/data/panel_for_r.csv`
Expected: CSV with vendor_id_int, period, cohort_period, outcomes

---

### Task 2: Run Callaway-Sant'Anna in R via rpy2

**Files:**
- Read: `staggered-adoption/data/panel_for_r.csv`
- Create: `staggered-adoption/results/CS_RESULTS_R.txt`

**Step 1: Run did::att_gt() for each outcome**

```python
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import sys

pandas2ri.activate()

# Import R packages
did = importr('did')
base = importr('base')

# Load data in R
ro.r('''
panel <- read.csv("staggered-adoption/data/panel_for_r.csv")
panel$cohort_period[panel$cohort_period == 0] <- 0  # Never-treated = 0 in did package
''')

# Run att_gt for impressions
ro.r('''
out_imp <- att_gt(
    yname = "impressions",
    tname = "period",
    idname = "vendor_id_int",
    gname = "cohort_period",
    data = panel,
    control_group = "nevertreated",
    est_method = "reg",
    base_period = "varying"
)
summary(out_imp)
''')

# Run att_gt for clicks
ro.r('''
out_clicks <- att_gt(
    yname = "clicks",
    tname = "period",
    idname = "vendor_id_int",
    gname = "cohort_period",
    data = panel,
    control_group = "nevertreated",
    est_method = "reg",
    base_period = "varying"
)
summary(out_clicks)
''')

# Run att_gt for total_gmv
ro.r('''
out_gmv <- att_gt(
    yname = "total_gmv",
    tname = "period",
    idname = "vendor_id_int",
    gname = "cohort_period",
    data = panel,
    control_group = "nevertreated",
    est_method = "reg",
    base_period = "varying"
)
summary(out_gmv)
''')
```

**Step 2: Aggregate to event study**

```python
ro.r('''
# Event study aggregation
es_imp <- aggte(out_imp, type = "dynamic", na.rm = TRUE)
es_clicks <- aggte(out_clicks, type = "dynamic", na.rm = TRUE)
es_gmv <- aggte(out_gmv, type = "dynamic", na.rm = TRUE)

# Simple (overall) aggregation
simple_imp <- aggte(out_imp, type = "simple", na.rm = TRUE)
simple_clicks <- aggte(out_clicks, type = "simple", na.rm = TRUE)
simple_gmv <- aggte(out_gmv, type = "simple", na.rm = TRUE)

# Print results
cat("\n================================================================================\n")
cat("IMPRESSIONS - Overall ATT\n")
cat("================================================================================\n")
summary(simple_imp)

cat("\n================================================================================\n")
cat("CLICKS - Overall ATT\n")
cat("================================================================================\n")
summary(simple_clicks)

cat("\n================================================================================\n")
cat("TOTAL GMV - Overall ATT\n")
cat("================================================================================\n")
summary(simple_gmv)

cat("\n================================================================================\n")
cat("IMPRESSIONS - Event Study\n")
cat("================================================================================\n")
summary(es_imp)

cat("\n================================================================================\n")
cat("CLICKS - Event Study\n")
cat("================================================================================\n")
summary(es_clicks)

cat("\n================================================================================\n")
cat("TOTAL GMV - Event Study\n")
cat("================================================================================\n")
summary(es_gmv)
''')
```

**Step 3: Extract numeric results to Python**

```python
# Extract event study coefficients
es_imp_ests = ro.r('es_imp$egt')
es_imp_se = ro.r('es_imp$se.egt')
es_imp_e = ro.r('es_imp$egt')

# Overall ATT
att_imp = ro.r('simple_imp$overall.att')[0]
se_imp = ro.r('simple_imp$overall.se')[0]

att_clicks = ro.r('simple_clicks$overall.att')[0]
se_clicks = ro.r('simple_clicks$overall.se')[0]

att_gmv = ro.r('simple_gmv$overall.att')[0]
se_gmv = ro.r('simple_gmv$overall.se')[0]
```

**Expected Output:** Three overall ATT estimates with clustered standard errors, plus event study coefficients for pre-trends testing.

---

### Task 3: Pre-Trends Test

**Step 1: Extract pre-treatment event study coefficients**

```python
ro.r('''
# Pre-trends test (e < 0)
pre_imp <- es_imp$egt[es_imp$egt < 0]
pre_imp_se <- es_imp$se.egt[es_imp$egt < 0]

# Joint test: H0: all pre-treatment effects = 0
cat("\nPre-trends test for impressions:\n")
cat("Pre-period coefficients:\n")
print(data.frame(
    e = es_imp$egt[es_imp$egt < 0],
    att = es_imp$att.egt[es_imp$egt < 0],
    se = es_imp$se.egt[es_imp$egt < 0]
))
''')
```

**Expected:** Pre-treatment θ(e) should be statistically indistinguishable from zero.

---

### Task 4: Segmentation Analysis

**Files:**
- Read: `staggered-adoption/data/panel_total_gmv.parquet`
- Create: Segment-specific results in master document

**Step 1: Create vendor segments based on pre-treatment activity**

```python
import pandas as pd
import numpy as np

panel = pd.read_parquet('staggered-adoption/data/panel_total_gmv.parquet')

# Pre-treatment characteristics (week 1 only for simplicity)
first_week = panel[panel['week'] == '2025-03-24'].copy()

# Segment by auction participation
first_week['activity_quartile'] = pd.qcut(
    first_week['bids'].clip(lower=0),
    q=4,
    labels=['Q1_Low', 'Q2_MedLow', 'Q3_MedHigh', 'Q4_High'],
    duplicates='drop'
)

# Merge back to panel
segments = first_week[['VENDOR_ID', 'activity_quartile']].drop_duplicates()
panel = panel.merge(segments, on='VENDOR_ID', how='left')
```

**Step 2: Run C-S by segment (simplified TWFE for speed)**

```python
import statsmodels.formula.api as smf

results_by_segment = {}
for segment in panel['activity_quartile'].dropna().unique():
    seg_data = panel[panel['activity_quartile'] == segment]

    # TWFE within segment
    model = smf.ols('total_gmv ~ treated + C(VENDOR_ID) + C(week)', data=seg_data).fit(
        cov_type='cluster', cov_kwds={'groups': seg_data['VENDOR_ID']}
    )

    results_by_segment[segment] = {
        'beta': model.params['treated'],
        'se': model.bse['treated'],
        'pval': model.pvalues['treated'],
        'n': len(seg_data)
    }
```

---

### Task 5: Create Master Results Document

**Files:**
- Create: `staggered-adoption/results/MASTER_RESULTS.txt`

**Content Structure:**

```
================================================================================
CALLAWAY-SANT'ANNA (2021) ANALYSIS: MASTER RESULTS
================================================================================

Reference: Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences
           with multiple time periods. Journal of Econometrics, 225(2), 200-230.

Estimation: R `did` package v2.2.1.910

================================================================================
1. DATA
================================================================================

Panel: Vendor × Week
  Observations: 846,430
  Vendors: 142,920
  Weeks: 26 (2025-03-24 to 2025-09-15)

Treatment: G_i = first week vendor wins any auction
  Ever-treated: 139,356 (97.5%)
  Never-treated: 3,564 (2.5%)

================================================================================
2. MAIN RESULTS
================================================================================

Outcome         Overall ATT    SE        95% CI              p-value
------------------------------------------------------------------------
Impressions     +X.XX          X.XX      [X.XX, X.XX]        <0.001
Clicks          +X.XX          X.XX      [X.XX, X.XX]        <0.001
Total GMV ($)   +X.XX          X.XX      [X.XX, X.XX]        X.XXX

================================================================================
3. EVENT STUDY
================================================================================

[Event study coefficients θ(e) for e = -5 to e = 20]

================================================================================
4. PRE-TRENDS TEST
================================================================================

H0: θ(e) = 0 for all e < 0
Result: [PASS/FAIL]
Joint p-value: X.XX

================================================================================
5. SEGMENTATION (HTE)
================================================================================

By Activity Quartile:
Segment         ATT         SE          p-value     N
----------------------------------------------------------------
Q1_Low          +X.XX       X.XX        X.XX        XXX,XXX
Q2_MedLow       +X.XX       X.XX        X.XX        XXX,XXX
Q3_MedHigh      +X.XX       X.XX        X.XX        XXX,XXX
Q4_High         +X.XX       X.XX        X.XX        XXX,XXX

================================================================================
6. INTERPRETATION
================================================================================

[Summary of findings]

================================================================================
```

---

### Task 6: Verify and Compile

**Step 1: Run full script**

Run: `python staggered-adoption/run_cs_proper.py`
Expected: `results/MASTER_RESULTS.txt` created

**Step 2: Verify results match manual implementation**

Compare overall ATT from R `did` package with manual implementation in `CALLAWAY_SANTANNA_FINAL.txt`.

---

## Execution Notes

1. The R `did` package computes proper clustered standard errors at the unit level by default
2. For large panels, `att_gt()` may take several minutes per outcome
3. Never-treated control group (G=0) is the recommended comparison group per the paper
4. Event study aggregation uses cohort-size weights by default

## Success Criteria

- [ ] R `did` package runs successfully via rpy2
- [ ] Overall ATT estimates match manual implementation (approximately)
- [ ] Pre-trends test completed
- [ ] Segmentation results produced
- [ ] Single MASTER_RESULTS.txt created with all findings
