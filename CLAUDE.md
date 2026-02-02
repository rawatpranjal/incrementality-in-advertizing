# Ad-platform Incrementality Analysis

## Notebook vs Script Convention

- **Notebooks (.ipynb)**: Data pull ONLY. No analysis code.
- **Scripts (.py)**: All analysis code. One .py creates one .txt output.
- Run .py scripts via CLI, not in notebooks.
- EDA: Single comprehensive file, all tables printed to stdout, no opinions.
- Output: Raw dump of all data processing, models, results. No interpretation.

## Snowflake Data Pull Pattern

When pulling data from Snowflake, always use CTE-based deterministic user sampling. Each query must include the CTE so the same users are sampled consistently across all tables.

```python
# CONFIG
SAMPLE_FRACTION = 0.01  # 1% of users
TOTAL_BUCKETS = 10000
SELECTION_THRESHOLD = int(TOTAL_BUCKETS * SAMPLE_FRACTION)

# CTE - include in EVERY query
CTE_SQL = f"""
WITH SAMPLED_USERS AS (
    SELECT OPAQUE_USER_ID FROM (
        SELECT OPAQUE_USER_ID, MOD(ABS(HASH(OPAQUE_USER_ID)), {TOTAL_BUCKETS}) AS bucket
        FROM (SELECT DISTINCT OPAQUE_USER_ID FROM AUCTIONS_USERS
              WHERE CREATED_AT BETWEEN '{start_date}' AND '{end_date}')
    ) WHERE bucket < {SELECTION_THRESHOLD}
)
"""

# Each query joins to SAMPLED_USERS
auctions_users = pd.read_sql(CTE_SQL + """
SELECT ... FROM AUCTIONS_USERS au
JOIN SAMPLED_USERS s ON au.OPAQUE_USER_ID = s.OPAQUE_USER_ID
WHERE ...
""", conn)

auctions_results = pd.read_sql(CTE_SQL + """
SELECT ... FROM AUCTIONS_RESULTS ar
JOIN AUCTIONS_USERS au ON ar.AUCTION_ID = au.AUCTION_ID
JOIN SAMPLED_USERS s ON au.OPAQUE_USER_ID = s.OPAQUE_USER_ID
WHERE ...
""", conn)

impressions = pd.read_sql(CTE_SQL + """
SELECT ... FROM IMPRESSIONS i
JOIN SAMPLED_USERS s ON i.USER_ID = s.OPAQUE_USER_ID
WHERE ...
""", conn)

clicks = pd.read_sql(CTE_SQL + """
SELECT ... FROM CLICKS c
JOIN SAMPLED_USERS s ON c.USER_ID = s.OPAQUE_USER_ID
WHERE ...
""", conn)
```

Key points:
- CTE is included in each query (not stored as temp table)
- Hash-based sampling is deterministic (same users every time)
- AUCTIONS_RESULTS joins via AUCTIONS_USERS to get user filter
- IMPRESSIONS and CLICKS join directly on USER_ID = OPAQUE_USER_ID
- Adjust SAMPLE_FRACTION to control data volume (0.01 = 1%, 0.001 = 0.1%)

## Guidelines

Claude prompt: 
- Do not create .md files
- Do not try to create "summaries" or "be enthusiastic" and cheerful. Just state the facts. 
- Always use tqdm, with verbosity so we see progress
- Never create extra cells for notebook, just give me code in line. 
- When checking something via python, do not create a new file — just run python in CLI and get results
- When asked to code: We do not want so many code files, generally one modular .py files which leads to a SINGLE .txt results. One for one. 
- We need to do more, all types of experiments with less, so we need to make it highly modular. 
- Results in .txt need to be ultra verbose — basically the entire stdout, everything needs to be captured in it. But no opinions, just a raw dump of all the data processing, models, results, so on. 
- Do not need to add heavy comments, your own opinions or interpretation. Let the facts be dumped. 
- Another AI will obejctively work on the facts, so just present them as is.
- Tone should be academic like a statistician, economist or machine learning engineer. Highly objective and balanced. The audience is a PhD Economist / Machine Learning Engineer. 
- No graphs, csv, pkl. Each .py creates one .txt. Unless otherwise stated. 
- Do not delete old code, move to archive folder. 
- Work only on 1 script at a time till I tell you we are done.
- When doing Statistical Model analysis: highlight unit of analysis, model equation, indep and indep vars why this equation? what is the purpose? interpretatino of coefficients? what is error expected to capture?
- Always give a TLDR in paragraph form when done with a task. In full sentences, like an adult senior economist / data scientist: the context, what you tried, what you found, issues encountered, takeaways.
- No shorthand anywhere in slides or writing. Explain properly with full words (e.g., "Clicks" not "C", "Spend" not "Y").
- Always show full file paths when reporting outputs or results (e.g., "Output saved to: /path/to/file.txt").

## Data Dictionary

### AUCTIONS_USERS (AUCTIONS/QUERIES)
- `AUCTION_ID` (Binary): Unique auction identifier
- `OPAQUE_USER_ID` (Varchar): Anonymized user identifier
- `CREATED_AT` (Timestamp_NTZ): Auction creation time
- `PLACEMENT` (Varchar): Placement/position context of the auction

### AUCTIONS_RESULTS (BIDS)
- `AUCTION_ID` (Binary): Links to AUCTIONS_USERS
- `VENDOR_ID` (Binary): Advertiser/vendor identifier
- `CAMPAIGN_ID` (Binary): Campaign identifier
- `PRODUCT_ID` (Varchar): Product being advertised
- `RANKING` (Number): Bid rank (1=highest)
- `IS_WINNER` (Boolean): Whether bid won impression slot
- `CREATED_AT` (Timestamp_NTZ): Bid creation time
- `QUALITY` (Float): Quality score metric for the bid
- `FINAL_BID` (Number): Final bid amount submitted
- `PRICE` (Number): Product or bid price
- `CONVERSION_RATE` (Float): Conversion rate metric
- `PACING` (Float): Pacing multiplier/factor for bid adjustment

### IMPRESSIONS (PROMOTED ONLY)
- `INTERACTION_ID` (Varchar): Unique impression identifier
- `AUCTION_ID` (Varchar): Links to auction
- `PRODUCT_ID` (Varchar): Product shown
- `USER_ID` (Varchar): User who saw impression
- `CAMPAIGN_ID` (Varchar): Campaign identifier
- `VENDOR_ID` (Varchar): Vendor identifier
- `OCCURRED_AT` (Timestamp_NTZ): Impression time

### CLICKS (PROMOTED ONLY)
- `INTERACTION_ID` (Varchar): Unique click identifier
- `AUCTION_ID` (Varchar): Links to auction
- `PRODUCT_ID` (Varchar): Product clicked
- `USER_ID` (Varchar): User who clicked
- `CAMPAIGN_ID` (Varchar): Campaign identifier
- `VENDOR_ID` (Varchar): Vendor identifier
- `OCCURRED_AT` (Timestamp_NTZ): Click time

### PURCHASES (ANY)
- `PURCHASE_ID` (Varchar): Unique purchase identifier
- `PURCHASED_AT` (Timestamp_NTZ): Purchase time
- `PRODUCT_ID` (Varchar): Product purchased
- `QUANTITY` (Number): Units purchased
- `UNIT_PRICE` (Number): Price per unit
- `USER_ID` (Varchar): Purchaser
- `PURCHASE_LINE` (Number): Line item number

### CATALOG
- `PRODUCT_ID` (Varchar): Unique product identifier
- `NAME` (Varchar): Product name
- `ACTIVE` (Boolean): Product active status
- `CATEGORIES` (Array): Product categories
- `DESCRIPTION` (Varchar): Product description
- `PRICE` (Float): Product price
- `VENDORS` (Array): Associated vendors
- `IS_DELETED` (Boolean): Product deletion status


Sample data: 

AUCTIONS_RESULTS
AUCTION_ID	VENDOR_ID	CAMPAIGN_ID	PRODUCT_ID	RANKING	IS_WINNER	CREATED_AT
0680d40b45075b699e04eab0c20d5619	064f8a4f58d277a7b02c46524d7809ab4d	01986697acb57812ade5506399b9b40f	648516875e485e2a3a41988c	23	FALSE	2025-09-02T04:31:07.289+0000
0680d40d69e8274f9d4e04ea6c20d5619	018f5d3cd235ff7d5a7d70eb9e7a8c2bf	01985ed34064179d0afff8b8c4a5a18e	6579119b17b40b1bb517b20a	41	FALSE	2025-09-02T04:31:07.981+0000
0680d40c147e47b204ea6c20d5619	018fbce38987d372ca3c472454790e244	019a5b896fe317f629f34e3e26fdf2677	61d25a2b9b8c9d16d0b705b	7	TRUE	2025-09-02T04:31:08.128+0000
0680d40d c2177583804ea6c20d5619	0187242e52d17b71565b1f0d10c1f0d3	019886e534ef7d23d4bca6e2055f4af4	68929dfb8d78941 c6143b63	3	TRUE	2025-09-02T04:31:08.140+0000
0680d40dc2177583804ea6c20d5619	0187242e52d17b71565b1f0d10c1f0d3	019886e534ef7d23d4bca6e2055f4af4	6852308b216d807f34787e881b	13	TRUE	2025-09-02T04:31:08.140+0000

AUCTIONS_USERS
AUCTION_ID	OPAQUE_USER_ID	CREATED_AT
0670d37d45d0c7642d0147e63e1e2d86	3e3b0a4e-8d9b-4922-841d-8c03c54a1fa0	2025-03-14T00:33:09.865+0000
0670d3786028720a104f87bfc1de45f	e9b042d8-51b1-4a49-8709-005887087f53	2025-03-14T00:33:40.003+0000
0670d376a184e748def400f3e1e2d86	ae3d245e-3115-4600-aa7e-fceb3ad4149c	2025-03-14T00:17:37.758+0000
0670d378f98870f08850414e044ca3e98d	9d63960e-29d4-4633-bfb7-f8bba3de1cc31	2025-03-14T00:27:30.584+0000
0670d37d44108789820407594ca3e98d	0d08ff44-98f6-4802-a8d9-eea75c67dba4d	2025-03-14T00:33:08.060+0000

CATALOG
PRODUCT_ID	NAME	ACTIVE	CATEGORIES	DESCRIPTION	PRICE	VENDORS
f1de74d1a04d2ab4c585d13b5	Audrey London Leather Ankle Boots Sz 37	TRUE	["bionic-retail","brand-audrey london","department-003a97b-5e5d3-ae...	Audrey London Leather Ankle Boots Sz 37 Audrey London Smooth Leather Sued...	18	["ten1:8d0954e4-a0d7-44de-92aa-9f2233f2cbe...
67fe72bf47d87289f3c03868	100%Linen black cargo anke pants/capris size L	TRUE	["bionic-retail","brand-white stag","department-003a97b-5e5d3-ae...	These very light weight 100% linen pants are in great condition. Measurements...	25	["ten1:4d06d0f4-3d04-450a-8352-3d1b038c82b...
67b64a2c996a47cda80d4911e	Samantha Thavasa Boston Duffle small shoulder bag mini purse crossbody w wh...	TRUE	["bionic-retail","brand-samantha thavasa","department-003a97b-5e5d3-ae...	Samantha Thavasa Boston Duffle small shoulder bag mini purse crossbody w wh...	78	["ten1:89fcf41c-b21f-4cd4-8db3-0c1b01b31c0...
647a3d3004a45b248c8b4b140	B DARLIN Open Back Fit N Flare Dress Juniors 7/8 Royal Blue Square Neck Pock...	TRUE	["bionic-retail","brand-b. darlin","department-003a97b-5e5d3-ae...	B DARLIN Open Back Fit Flare Dress Juniors 7/8 Royal Blue Square Neck Pocket...	28	["ten1:02cea404-f47b-40b0-807c-3f156698f0...
6bada6eaf0b579fac21c0086	EUC pinstripe-striped tee super soft size L	TRUE	["bionic-retail","brand-rd style","department-003a97b-5e5d3-ae...	EUC short sleeve t-shirt from RD Style in size Large. ✨ Sought to wear to ca...	16	["ten1:036d1a8-16f7-47ba-8493-c3b46908b5...

IMPRESSIONS
INTERACTION_ID	AUCTION_ID	PRODUCT_ID	USER_ID	CAMPAIGN_ID	VENDOR_ID	OCCURRED_AT
067aea309-abcb-73c8-aa0d-7b0e1d88bbd1	067aea30-7c1b-70e-1aa4-d400382de41d	67ab8b5b5759c5886ac7388	042e-ca9e-400b-a53b-f9620227a0ca	0165e069-4ee8-7701-b1e-f6143f17bd	01b1cae-ecbc-f1f0-b7f0-b43778d	2025-03-01T09:47:00.000-0000
067aea30c-d703-7e72-a0e-bb1d-045a11e4	067aea30e-0cbe-74ee-a-004-e0451a8174	64969e624027bea17011b	66d-dd67-40fc-a4a-000845184a	0165a2cc-c4d-7702-e2a-0cae0170a41d	010356bb-9801-7001-bdd-f1173ca8a13	2025-03-01T09:59:15.000-0000
067aea701-d7c9-7220-ac18-32ee069d9ba9	067aeca68-e35e-7f35-a1c4-aa0c1ab0e51d	67b20ee4c0c700e47e8b	76a-446-4dc-a21b-107825a2e2f	0165e34a-9e73-bda-dd-59b-6f8b24a9e	0104eab-da21-f431-b00-a6e96de14fd	2025-03-01T09:59:05.000-0000
067aec633-a16e-7b0b-8219-08b5451a8174	067aec63b-a61-75fb-a04-5451a8174	6649b56ee222ca4d08	a08-1668-47c-878-3f1932187	0165ad-2e7c-7d-31e3-bf-3e0e2417-000	010aa-2aef-7142-b69-8a-de5de-45f0d	2025-03-01T09:24:17.000-0000
067aed1d-d19c-7d4a-923f-4e05b9b66ba3	067aec95-0ff-7bee-a04-b9b66ba3	6712395d10b7f84d2443	3d56-a38-4f8-8f82-96582d1c1	0165ae1d-8762-74f-b1b-9657b761	010d3da-9fd9-e3d1-ae-391d-ca9d556812	2025-03-01T09:51:46.000-0000

CLICKS
INTERACTION_ID	AUCTION_ID	PRODUCT_ID	USER_ID	CAMPAIGN_ID	VENDOR_ID	OCCURRED_AT
067adefe8-0eb9-706f-a00d-f5f43ea2edb8	067adefe8-0f0a-71b-a00d-f5f43ea2edb8	67a6d022d89f3a694104	04-f011-4ae-871d-dae58022e78d	0165b261-0a9e-711-b00-c23cde89ec61	01048a9b-e263-706f-b00-f691f07b2940	2025-03-14T03:10:59.000-0000
067adf17-aab-712d-a00d-43fa01b8a5b0	067adf17-aab-712d-a00d-43fa01b8a5b0	666e417e5b3b15cb239a	0b8-4dc-4a-add-459801002a	01657bd-717-b6f0-986b-1083502283b	010899e-a0e-7fc8-b5a-de32e0d19b	2025-03-14T03:10:30.000-0000
067ad6b07-9327-7c1e-b00a-6c3b0c63c861	067ad6b07-91f6-7b79-b00a-f9523c94a0b	6328c3866b7066e9940b	06922e8-0102-447-817-da7f607d1f7	0165d216-3e4f-7c-b6f0-3b8c2203165	0654b178-f9b-7c-9121-0b2f4c020861	2025-03-14T03:10:44.000-0000
067ad3a97-c80d-700d-a00d-d9b9f7c14eab	067ad3a97-c15-7d0-a00d-fa9aef	63707b1c1d682d121076	087780b-70b-4de-9e-2d12cc46da	0165d292-7de-7c-b4f0-ce-e4653b8e0	01040b8-2e-b-b-248521f6d5	2025-03-14T04:49:13.000-0000
067ad54a1-8d-7171-b00a-f775c04ecaa6	067ad54a1-7d6e-71c-a00a-fa04ecaa6	67b6cccae817ecce0ab4	02244e2-1a2-40b2-807-ee5fc9094d	0165b26b-d791-76f-b4e-3da9d41d1d	0102636c-48b-70-bde-f0a0712d063	2025-03-14T03:11:04.000-0000

PURCHASES
PURCHASE_ID	PURCHASED_AT	PRODUCT_ID	QUANTITY	UNIT_PRICE	USER_ID	PURCHASE_LINE
17d13b70282b0b545f8de808	2025-03-14T08:28:42.000-0000	674cb06ca39b8b0f7d4e	1	800	0625586-1f01-4f0f-8f0a-7f002ec1085	1
17d13c93a4c3571ae1178df5	2025-03-14T08:27:52.000-0000	67262b14778d1aa6ba6d	1	1100	0499d24e-e9f2-4370-b27b-34f0d673acb9	1
17d13c93a4c3571ae1178df5	2025-03-14T08:27:52.000-0000	67d26ad5214432de8a0c	1	800	0b958f1-cae-422f-9023-b1b0ca8d4a9	11
17d137b266024a51604d4d03	2025-03-14T08:27:32.000-0000	67d35f5c88b9338b97de	1	500	50f6622-2d99-4bbe-bee-1e6a6e5cea5	8
17d13d76baeea4d963744a22	2025-03-14T08:28:50.000-0000	67c2f57f685388f65eec1	1	8000	2d497fe-411-43e-b64-5d5cadf6a9	1


More about the data: 

1. Ad funnel events (Auctions, Impressions, Clicks) can only be joined reliably using a standardized, multi-column composite key (e.g., AUCTION_ID, PRODUCT_ID), not single IDs.
2. The PRODUCT_ID from the PURCHASES table is only a valid key for joining to the CATALOG if the purchase was part of a promoted journey (i.e., can be linked to a click/impression). Organic purchases cannot be reliably joined to the catalog.
3. Very important: all these impressions and clicks are for sponsered products only. They do not cover organic products. Further, the catalog also only covers sponsered products. The purchases table however has ALL purchases regarless of promoted or not. This means we do not see the bulk of what the user impressed or clicked on. 