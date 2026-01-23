# Daily Summaries Data

This directory contains daily aggregated datasets for ad-platform incrementality analysis, covering the period from **2025-03-14 to 2025-09-07** (178 days).

## Directory Structure

```
daily_summaries/
├── data/                           # Core datasets (178 daily files each)
│   ├── product_daily_purchases_dataset/     # Purchase transactions
│   ├── product_daily_clicks_dataset/        # Promoted product clicks
│   ├── product_daily_impressions_dataset/   # Promoted product impressions
│   └── product_catalog_processed/           # Product catalog snapshots
├── eda/                           # Exploratory data analysis
├── extract_*.ipynb               # Data extraction notebooks
└── product_vendor_day_panel.ipynb # Panel data construction
```

## Dataset Specifications

### 1. Product Daily Purchases Dataset
**File Pattern:** `data_YYYY-MM-DD.parquet`
**Unit of Analysis:** Product-day aggregations of all purchases (promoted + organic)
**Sample Size:** ~132K products per day

**Schema:**
- `PRODUCT_ID` (string): Unique product identifier
- `DATE` (date): Transaction date
- `PURCHASES` (int): Total purchase transactions
- `LINES_SOLD` (int): Total line items sold
- `UNITS_SOLD` (int): Total units purchased
- `REVENUE_CENTS` (int): Total revenue in cents
- `AVG_UNIT_PRICE_CENTS` (float): Average unit price
- `MIN_UNIT_PRICE_CENTS` (int): Minimum unit price
- `MAX_UNIT_PRICE_CENTS` (int): Maximum unit price
- `STDDEV_UNIT_PRICE_CENTS` (float): Price standard deviation
- `DISTINCT_USERS_PURCHASED` (int): Unique users who purchased

**Coverage:** All products with purchase activity (organic + promoted)

### 2. Product Daily Clicks Dataset
**File Pattern:** `data_YYYY-MM-DD.parquet`
**Unit of Analysis:** Product-vendor-campaign-day aggregations of promoted clicks
**Sample Size:** ~482K records per day

**Schema:**
- `PRODUCT_ID` (string): Product identifier
- `DATE` (date): Click date
- `VENDOR_ID` (string): Advertiser identifier
- `CAMPAIGN_ID` (string): Campaign identifier
- `TOTAL_CLICKS` (int): Total clicks for product-day
- `CLICKS` (int): Clicks for specific vendor-campaign-product-day
- `DISTINCT_USERS_CLICKED` (int): Unique users who clicked

**Coverage:** Only promoted/advertised products (~35K vendors, ~37K campaigns)

### 3. Product Daily Impressions Dataset
**File Pattern:** `data_YYYY-MM-DD.parquet`
**Unit of Analysis:** Product-vendor-campaign-day aggregations of promoted impressions
**Sample Size:** ~3.5M records per day

**Schema:**
- `PRODUCT_ID` (string): Product identifier
- `DATE` (date): Impression date
- `VENDOR_ID` (string): Advertiser identifier
- `CAMPAIGN_ID` (string): Campaign identifier
- `TOTAL_IMPRESSIONS` (int): Total impressions for product-day
- `IMPRESSIONS` (int): Impressions for specific vendor-campaign-product-day
- `DISTINCT_USERS_IMPRESSED` (int): Unique users who saw impression

**Coverage:** Only promoted/advertised products (~35K vendors, ~38K campaigns)

### 4. Product Catalog Processed
**File Pattern:** `catalog_processed_YYYY-MM-DD.parquet`
**Unit of Analysis:** Product-level catalog snapshots
**Sample Size:** ~72K products per day

**Schema:**
- `PRODUCT_ID` (string): Product identifier
- `NAME` (string): Product name
- `PRICE` (float): Product price
- `ACTIVE` (boolean): Product active status
- `IS_DELETED` (boolean): Product deletion flag
- `DESCRIPTION` (string): Product description
- `BRAND` (string): Product brand
- `DEPARTMENT_ID`, `CATEGORY_ID`, `CATEGORY_FEATURE_ID`: Category hierarchy
- `COLORS`, `SIZE_INFO`, `STYLE_TAGS`: Product attributes
- `DOMAIN` (string): Product domain/source
- `PRODUCT_CREATED_AT` (timestamp): Product creation time
- `TOTAL_CATEGORIES`, `COLOR_COUNT`, `STYLE_TAG_COUNT`: Attribute counts
- `PRICE_OUTLIER` (boolean): Price outlier flag
- `SHORT_NAME`, `POOR_DESCRIPTION` (boolean): Data quality flags
- `VENDORS` (array): Associated vendor IDs
- `EXTRACTION_DATE`, `PROCESSED_AT`: Processing timestamps

**Coverage:** Complete product catalog (~57K active, ~15K deleted, ~11K brands)

## Key Data Relationships

1. **Purchase Data**: All products (promoted + organic)
2. **Click/Impression Data**: Only promoted products with advertising activity
3. **Catalog Data**: Complete product universe with attributes and vendor relationships

## Data Quality Notes

- **Consistent Coverage**: All datasets maintain 178 daily files with identical date ranges
- **Product Overlap**: Purchase data covers broader product set than click/impression data
- **Vendor-Campaign Granularity**: Click/impression data provides detailed advertising attribution
- **Temporal Consistency**: Daily snapshots enable time-series analysis and incrementality measurement

## Analysis Applications

This data structure supports:
- **Incrementality Analysis**: Comparing promoted vs organic product performance
- **Attribution Modeling**: Vendor-campaign level performance measurement
- **Time Series Analysis**: Daily trend analysis across 178-day period
- **Panel Data Construction**: Product-vendor-day level observations
- **Causal Inference**: Treatment (promoted) vs control (organic) product analysis

## File Sizes

- **Purchases**: ~23MB per daily file
- **Clicks**: ~141MB per daily file
- **Impressions**: ~1GB per daily file
- **Catalog**: ~99MB per daily file
- **Total Dataset**: ~230GB across all files