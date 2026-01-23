# Data Construction Clarification Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Clarify and reconcile the different panel data constructions used across the paper.

**Architecture:** Document the four distinct panel constructions, their unit of analysis, and why results differ.

**Tech Stack:** LaTeX documentation, Python data exploration

---

## The Confusion

The codebase uses **multiple different data constructions** with inconsistent terminology:

| Panel | Unit of Analysis | Time Aggregation | Gap Definition | N |
|-------|------------------|------------------|----------------|---|
| Vendor-Week | Vendor × Week | Weekly | Calendar week | 979K |
| User-Week | User × Week | Weekly | Calendar week | 31M |
| User-Vendor-Week | User × Vendor × Week | Weekly | Calendar week | 3.5M |
| Macro-Session-Product | User × Session × Product | 3-day gap | **3-day inactivity** | 1.8M |

**Problem 1: Diagram Mismatch**
- `session_diagram.py` shows: 2-hour sessions + 48-hour journeys
- `funnel_analysis_v2.tex` uses: 3-day macro-sessions
- The diagram in the presentation doesn't match the analysis

**Problem 2: Conflicting Results**

| Panel | Click → Revenue Effect | Interpretation |
|-------|------------------------|----------------|
| Vendor-Week | **+0.64** elasticity | Vendors with more clicks earn more |
| User-Week | **-0.29** elasticity (mixed effects) | Users who click more spend less |
| User-Vendor-Week | **+1.66** log-odds | Clicking vendor's ad → purchase from vendor |
| Macro-Session-Product | **+5.75%** revenue | Clicking product → purchase product |

---

## Task 1: Audit All Session/Panel Definitions

**Files to check:**
- `latex/funnel_analysis.tex` (2-hour journey)
- `latex/funnel_analysis_v2.tex` (3-day macro-session)
- `latex/session_diagram.py` (2-hour session + 48-hour journey)
- `latex/macro_session_diagram.py` (3-day macro-session)
- `latex/user_week.tex` (weekly)
- `latex/vendor_week.tex` (weekly)
- `latex/user_vendor_week.tex` (weekly)

**Step 1: Read each file and extract:**
- Exact gap/window definition
- Unit of observation
- Sample size
- Key result

**Step 2: Create reconciliation table**

---

## Task 2: Fix Diagram-Paper Mismatch

**Option A:** Update diagram to 3-day macro-sessions (match funnel_analysis_v2.tex)
- Use `macro_session_diagram.py` instead of `session_diagram.py`

**Option B:** Update paper to 2-hour sessions (match current diagram)
- Rewrite funnel_analysis_v2.tex

**Step 1: Decide which definition is correct for the analysis**

**Step 2: Update either diagram or paper to match**

---

## Task 3: Explain Why Results Differ (Add to Paper)

The sign flip between user-week (-0.29) and session-product (+5.75%) is **not a bug**—it's Simpson's paradox:

**User-Week (negative):** "Do users who click a lot spend more overall?"
- No—heavy clickers are browsers/comparison shoppers
- Selection: high-intent buyers click once and purchase; low-intent browsers click many times

**Session-Product (positive):** "Does clicking THIS product increase purchase of THIS product?"
- Yes—clicking signals interest in that specific product
- Within-session, within-product comparison

**Step 1: Add reconciliation section to paper explaining this**

---

## Task 4: Update Presentation Slide

Current slide uses `session_construction.png` which shows 2-hour sessions.

**If analysis uses 3-day macro-sessions:**
- Replace with output from `macro_session_diagram.py`
- Or regenerate `session_construction.png` with correct parameters

**Step 1: Verify which diagram matches the actual analysis**

**Step 2: Update `presentation.tex` to use correct diagram

---

## Summary of Issues

1. **Terminology inconsistency:** "session" vs "journey" vs "macro-session"
2. **Diagram mismatch:** Diagram shows 2hr/48hr, paper uses 3-day
3. **Missing reconciliation:** Paper doesn't explain why user-level is negative but session-level is positive

## Recommended Resolution

1. Standardize terminology: Use "macro-session" for 3-day gap throughout
2. Use `macro_session_diagram.py` output for all presentations/papers
3. Add reconciliation section explaining the Simpson's paradox
4. Update presentation slide to use correct diagram
