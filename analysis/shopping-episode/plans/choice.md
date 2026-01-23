This model moves beyond simple "Did they buy?" to "Which vendor did they choose, and why?" It explicitly models the **zero-sum competition** for the user's wallet.

### 1. The Statistical Tool: Conditional Logistic Regression (McFadden's Choice Model)

Unlike standard regression, this model requires a **Long Format** dataset. For every shopping session, we create multiple rows representing the **Menu of Options** available to the user.

**The Choice Set ($J$) for Session $i$:**
1.  **Option 0 (Outside Option):** The user buys nothing.
2.  **Option 1 (Organic):** The user buys from a vendor who *did not* advertise in this session.
3.  **Options $k \in \{2..N\}$ (Promoted Vendors):** The specific vendors who served ads to the user in this session.

---

### 2. The Dataset: `df_choice_long`

**Unit of Analysis (Row):** A specific **(Session, Candidate Option)** pair.
*   If a user sees ads from Vendor A and Vendor B, this creates **4 rows** for that session (None, Organic, A, B).

| Column Name | Level | Definition | Value for "None/Organic" |
| :--- | :--- | :--- | :--- |
| `EPISODE_ID` | Group | Unique Session ID. | Same for all options. |
| `OPTION_ID` | ID | `None`, `Organic`, or `Vendor_Hash`. | |
| `IS_CHOSEN` | **Target** | **1** if the user purchased this option, **0** otherwise. | Only one row per session is 1 (assuming single purchase focus). |
| `HAS_ADS` | Feature | 1 if this is a Promoted Vendor option. | 0 for None/Organic. |
| `IMP_COUNT` | Feature | Number of impressions shown by this vendor. | 0 for None/Organic. |
| `CLICK_COUNT` | Feature | Number of clicks on this vendor. | 0 for None/Organic. |
| `SOV_RANK` | Feature | Vendor's Share of Voice Rank (1st, 2nd..). | 999 for None/Organic. |
| `AVG_PRICE` | Feature | Average price of this vendor's shown ads. | 0 (or Global Avg) for None. |
| `SESSION_DURATION` | Context | Duration of the session in hours. | Same for all options. |

---

### 3. The Mathematical Model

We define the **Utility ($U_{ij}$)** that User $i$ derives from choosing Option $j$ as a linear combination of the option's attributes and the user's context.

$$ U_{ij} = V_{ij} + \epsilon_{ij} $$

**The Utility Function ($V_{ij}$):**
$$ V_{ij} = \alpha_j + \beta_1(\text{IMPRESSIONS}_{ij}) + \beta_2(\text{CLICKS}_{ij}) + \beta_3(\text{PRICE}_{ij}) + \beta_4(\text{HAS\_ADS}_j) $$

*   $\alpha_j$: The "Alternative Specific Constant" (ASC). This captures the baseline preference for "Doing Nothing" vs "Buying Organic."
*   $\beta_1$: The marginal utility of being seen (Brand Awareness).
*   $\beta_2$: The marginal utility of being clicked (Engagement).
*   $\beta_4$: The "Ad Label" effect. Do users inherently trust/distrust advertised items?

**The Probability (Softmax):**
The probability that User $i$ chooses Vendor $k$ depends on Vendor $k$'s utility **relative to the sum of all other competitors' utilities**:

$$ P(Choice = k) = \frac{e^{V_{ik}}}{\sum_{j \in \text{ChoiceSet}} e^{V_{ij}}} $$

---

### 4. Interpretation & Hypothesis

This model provides the most nuanced view of competition:

1.  **Cannibalization vs. Conquest:**
    *   The denominator ($\sum e^{V}$) captures competition. If Vendor A increases their Ad Impressions ($\text{IMPRESSIONS}_{iA} \uparrow$), their Utility $V_{iA}$ rises.
    *   Mathematically, this **automatically decreases** the probability $P(Choice=B)$ for Vendor B, even if Vendor B did nothing.
    *   It also decreases $P(Choice=\text{None})$, measuring market expansion.

2.  **The "Click" Coefficient ($\beta_2$):**
    *   This measures the **conversion efficiency** of a click.
    *   If $\beta_2$ is high, it means the click is the decisive factor in the choice, overriding price or brand.

3.  **The "Impression" Coefficient ($\beta_1$):**
    *   This measures the **passive value** of exposure.
    *   If $\beta_1 > 0$ but $\beta_2 \approx 0$, it implies "Billboard Effect": seeing the ad helps, but the click itself isn't the cause.

### Summary of Implementation in `05_incrementality_modeling.py`

I will implement this as the third module in the script:
1.  **Construct Long Data:** Iterate through sessions, identifying the set of unique `VENDOR_IDs` shown.
2.  **Append Counterfactuals:** Add "None" and "Organic" rows for every session.
3.  **Feature Engineering:** Fill 0s for non-ad options.
4.  **Estimation:** Since standard `sklearn` does not support Conditional Logit easily (it assumes fixed classes), I will use a **stratified sampling approach** with Logistic Regression (with `EPISODE_ID` as a group) or a simplified Multinomial Logit if the vendor set is too large, grouping competitors into "Competitor Strong" vs "Competitor Weak".

*Refinement for Code:* Given the high cardinality of vendors, I will group all "Other Promoted Vendors" into a single competitor option if a specific vendor is not the focus, OR I will run the model on the **Top 50 Vendors** only to ensure convergence.

**Decision:** I will proceed with a simplified **"Selected vs. Best Competitor vs. Outside"** formulation to make the matrix computable in the text output.

1.  **Option A:** The Vendor we are analyzing.
2.  **Option B:** The "Best" Alternative Vendor (highest SOV).
3.  **Option C:** The Outside Option (No Buy/Organic).

This creates a manageable 3-choice system per session.

To implement a **Multinomial Logit (MNL)** or **Conditional Logit** model, we must restructure the data from "Event Level" to "Choice Level."

This transforms the problem from "Did a click happen?" to "Given a menu of options, which one did the user pick?"

### 1. Eligibility: Who gets to be on the "Menu"?

For every Shopping Session $i$, we define a specific **Choice Set** $C_i$. A vendor is eligible to be in this set **only if** they were present in the user's consideration set during that session.

**The Four Types of Options:**

1.  **The "No Buy" Option (Outside Option):**
    *   *Definition:* The user leaves the session without purchasing anything.
    *   *Eligibility:* **Always** present in every session.
    *   *Attributes:* Clicks = 0, Impressions = 0, Price = 0.

2.  **The "Organic" Option:**
    *   *Definition:* The user purchases an item that was **not** promoted to them in this session (e.g., they searched for it directly or found it via navigation).
    *   *Eligibility:* **Always** present in every session.
    *   *Attributes:* Clicks = 0, Impressions = 0. (This serves as the baseline for "buying without ad pressure").

3.  **The "Promoted Candidates" (The Shown Vendors):**
    *   *Definition:* Vendors who successfully won an auction and displayed an impression to the user.
    *   *Eligibility:* Vendor $j$ is included if `COUNT(IMPRESSIONS_ij) > 0`.
    *   *Optimization Rule:* To prevent data explosion (some sessions have 50+ vendors), we include:
        *   The **Purchased Vendor** (if any).
        *   The **Top 5 Vendors** by Share of Voice (Impressions) or Rank.
        *   *Note:* Vendors shown at Rank 50 are statistically irrelevant and are excluded to save memory.

---

### 2. Data Structure: The Long Format Matrix

We will generate a dataset `df_choice` where the unique identifier is `(EPISODE_ID, OPTION_ID)`.

**Columns (Features):**

| Column | Type | Description |
| :--- | :--- | :--- |
| `EPISODE_ID` | Group ID | Links the rows together into one "Choice Occasion". |
| `OPTION_ID` | Identifier | `No_Buy`, `Organic`, or `Vendor_Hash`. |
| `IS_CHOSEN` | **Target (Y)** | **1** if this option was purchased, **0** otherwise. (Sum per episode = 0 or 1). |
| `HAS_ADS` | Binary | 1 if this is a Promoted Candidate. 0 for No_Buy/Organic. |
| `IMP_COUNT` | Count | Number of times this vendor was shown in this session. |
| `CLICK_COUNT` | Count | Number of times this vendor was clicked. |
| `MEAN_RANK` | Continuous | Average position (1=Top). 0 or 999 for non-promoted options. |
| `MEAN_PRICE` | Continuous | Average price of the items shown. |
| `SESSION_DURATION` | Context | **Invariant** across options. Controls for session intent. |

**Example of ONE Session (User saw Nike & Adidas, bought Nike):**

| Episode | Option | Is_Chosen | Has_Ads | Imps | Clicks | Rank |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 101 | No_Buy | 0 | 0 | 0 | 0 | 0 |
| 101 | Organic | 0 | 0 | 0 | 0 | 0 |
| 101 | Nike | **1** | 1 | 5 | 1 | 2.5 |
| 101 | Adidas | 0 | 1 | 3 | 0 | 4.0 |

---

### 3. The Mathematical Model (Conditional Logit)

We model the **Utility** ($U_{ij}$) that User $i$ gets from Option $j$. The user chooses the option that maximizes this utility.

$$ U_{ij} = V_{ij} + \epsilon_{ij} $$

**The Systematic Utility Function ($V_{ij}$):**

$$ V_{ij} = \alpha_j + \beta_{imp} (\text{IMP}_{ij}) + \beta_{click} (\text{CLICK}_{ij}) + \beta_{price} (\text{PRICE}_{ij}) + \beta_{rank} (\text{RANK}_{ij}) $$

*   **$\alpha_j$ (Alternative Specific Constants):**
    *   We cannot have one for every vendor. We will simplify to three intercepts:
    *   $\alpha_{nobuy}$ (Reference, usually 0).
    *   $\alpha_{organic}$ (Baseline preference for buying intrinsic items).
    *   $\alpha_{promoted}$ (Baseline preference for advertised items, captured via `HAS_ADS`).

**The Probability Function (Softmax):**

$$ P(Y_{ij} = 1) = \frac{ \exp(V_{ij}) }{ \sum_{k \in \text{ChoiceSet}_i} \exp(V_{ik}) } $$

### 4. Interpretation of Coefficients

This model isolates the **Value of the Click** ($\beta_{click}$) from the **Value of Visibility** ($\beta_{imp}$) relative to the **Outside Options**.

1.  **If $\beta_{click} > 0$:**
    *   Clicking significantly increases the probability of selection *within* the set of options.
    *   It quantifies the "Persuasion" or "Selection" effect.

2.  **If $\beta_{imp} > 0$:**
    *   Mere exposure increases preference, even without a click. (The "Billboard Effect").

3.  **The Denominator Effect (Competition):**
    *   Because probabilities sum to 1, if Vendor A increases their ad intensity ($V_{iA} \uparrow$), the probability of choosing Vendor B **must decrease**.
    *   This explicitly models **Cannibalization**.

### 5. Implementation Strategy for Script

Since we don't want heavy dependencies (like `pylogit` or `biogeme`), we will use `statsmodels.genmod` or a manual Likelihood implementation if needed, but the most robust standard way in Python is:

**Stratified Logistic Regression (Approximation):**
We can approximate the Conditional Logit by running a standard Logistic Regression on the "Long Format" data, provided we include **Session Fixed Effects**.
*   *However*, with millions of sessions, Fixed Effects are computationally impossible.
*   *Alternative:* We use **`statsmodels.discrete.discrete_model.MNLogit`** if we fix the classes (NoBuy, Organic, Promoted_Winner, Promoted_Loser).

**The "Generic Competitor" Approach (Simplest & Best):**
To make this computationally feasible, we define fixed classes for the MNL:
1.  **Class 0:** No Purchase.
2.  **Class 1:** Organic Purchase.
3.  **Class 2:** The "Focal" Promoted Vendor (The one with the most impressions).
4.  **Class 3:** The "Competitor" Promoted Vendor (The second most impressions).

We define the features for Class 2 and 3 dynamically. This allows us to run a standard Multinomial Logit to see how Ad Intensity shifts probability from Class 0/1 to Class 2.