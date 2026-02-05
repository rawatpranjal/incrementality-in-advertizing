# Summary of Papers on Conversion Attribution and Recommendation Systems

## 2507.15113v1 (1).txt: Click A, Buy B: Rethinking Conversion Attribution in ECommerce Recommendations

This paper introduces a novel approach to conversion attribution in e-commerce recommendations, specifically addressing the "Click A, Buy B" (CABB) phenomenon, where users click on one product (A) but purchase a different one (B). Traditional models that assume a one-to-one relationship between clicks and purchases lead to biased learning and suboptimal recommendations.

The authors reframe conversion prediction as a multi-task learning problem with separate heads for "Click A, Buy A" (CABA) and CABB. To distinguish meaningful CABB conversions (e.g., substitutions or complements) from unrelated ones, they propose a taxonomy-aware collaborative filtering weighting scheme. This scheme maps products to a hierarchical taxonomy and learns category-to-category similarity from large-scale co-engagement logs. This weighting amplifies genuinely related product pairs while down-weighting coincidental cross-category purchases.

Offline experiments showed a 13.9% reduction in normalized entropy compared to a last-click attribution baseline. Online A/B tests demonstrated a +0.25% gain in the primary business metric and a 1.27% increase in the CABA rate, indicating more personalized recommendations.

The limitations include reliance on the quality of the product taxonomy, potential noise in co-engagement signals, and the need for careful tuning of the hyperparameter λ, which balances CABA and CABB loss contributions. Future work suggests incorporating Large Language Models for better CABB event labeling and dynamic weighting mechanisms.

## 2507.15113v1.txt: Duplicate of 2507.15113v1 (1).txt

This file appears to be a duplicate of the previously summarized document, "2507.15113v1 (1).txt".

## Article_V2_updated.txt: Between Click and Purchase: Predicting Purchase Decisions Using Clickstream Data

This paper proposes an innovative approach to predict online purchases by analyzing customers' clickstream data, specifically focusing on "viewing behaviors" rather than just page categories. Traditional methods often overlook the detailed actions customers take within a page, leading to less accurate purchase predictions.

The authors categorize browsing actions into four "Viewing Behaviors": repeated-viewing (R), filtering (F), searching for a different date (DD), and searching for a different route (DR). They demonstrate that sequences of these viewing behaviors provide more detailed and accurate insights into purchase tendencies than sequences of page categories. For instance, "repeated-viewing" strongly correlates with purchase intent, while "search-for-a-different-date" suggests weaker intent.

A structural equation modeling approach, combining multinomial logit for viewing behavior choice and Poisson regression for observed options, is used to predict the next viewing behavior. This predicted behavior is then used to forecast purchase decisions. The model achieved 65% in-sample and 64% out-of-sample prediction accuracy for viewing behaviors, and 58% in-sample and 57% out-of-sample for purchase predictions, outperforming existing methods.

The research contributes by offering a more granular conceptualization of customer navigation paths and a robust modeling approach that improves purchase prediction accuracy with reduced computational load.

## BNT_ECMA_rev.txt: Consumer Heterogeneity and Paid Search Effectiveness: A Large Scale Field Experiment

This paper investigates the causal effectiveness of paid search ads through a series of large-scale field experiments conducted at eBay. The study addresses the challenge of measuring the true impact of advertising, particularly in the context of internet advertising where endogeneity (correlation between search clicks and purchase intent) can lead to inflated non-experimental estimates.

Key findings include:
*   **Brand Keyword Ads**: For brand-keyword ads (e.g., searching "eBay"), there are no measurable short-term benefits. Users typing brand names are primarily using search as navigation and would likely reach the company's website through organic search even without the paid ad. Substitution between paid and unpaid traffic is nearly complete.
*   **Non-Brand Keyword Ads**: For non-brand keywords, the overall effect of paid search on sales is very small and statistically insignificant.
*   **Consumer Heterogeneity**: Paid search ads are effective in acquiring new users and influencing purchases by infrequent and less recent users. More frequent users, who account for the majority of advertising expenses, are not significantly influenced by these ads. This supports the "informative view" of advertising, where ads are beneficial for informing consumers who are unfamiliar with a product or platform.
*   **Negative ROI**: The short-term return on investment (ROI) for paid search on eBay was found to be negative (-63%, with a 95% confidence interval of [-124%, -3%]), primarily because most advertising spend targets frequent buyers who would have purchased anyway. Non-experimental methods (OLS) drastically overestimate ROI (e.g., over 4000%).

The research emphasizes the critical importance of controlled experiments to accurately measure advertising effectiveness, especially for well-known brands where informational gaps among consumers are smaller. It suggests that much internet advertising expenditure by large, established companies might be inefficient.

## BuyingRep.txt: Buying Reputation as a Signal of Quality: Evidence from an Online Marketplace

This paper examines the role of a "reward-for-feedback" (RFF) mechanism on Taobao, a large online marketplace, as a signal of product quality and a solution to the "cold-start" problem for new products. Sellers can offer rebates to buyers who leave informative feedback, with Taobao guaranteeing the rebate based on feedback informativeness (not sentiment).

Key findings and contributions:
*   **Signaling Mechanism**: RFF acts as a credible signal of high product quality. Sellers are more likely to adopt RFF for high-quality products, especially those without established feedback ("cold-start" products). As sellers gain reputation, their need for RFF diminishes.
*   **Buyer Response and Sales Increase**: Items with RFF adoption see approximately 36% higher sales. A conservative lower bound suggests that over 27% of this sales increase is attributable to the signaling effect, not just the price discount from the rebate.
*   **Feedback Quality**: RFF incentivizes buyers to leave more informative and longer feedback. Critically, RFF does not bias feedback towards positive sentiment, ensuring the reliability of the generated information.
*   **Dynamic Reputation Building**: RFF creates a "flywheel effect," leading to more sales and feedback, which rapidly builds both product and seller reputation, thus alleviating the cold-start problem.
*   **Market Design Implications**: The study suggests that online marketplaces can effectively reduce asymmetric information problems by allowing sellers to self-select into RFF mechanisms, leveraging strategic seller and buyer behavior to enhance market quality. Differences in marketplace structures (e.g., product-seller vs. product-level reviews) are discussed as reasons why such mechanisms might not be universally adopted (e.g., on eBay or Amazon).

## ec15_causal_impact_recommendations.txt: Estimating the causal impact of recommendation systems from observational data

This paper presents a method for estimating the *causal impact* of recommendation systems from purely observational data, addressing the challenge that naive click-through counts often overestimate their true effect due to correlated demand. The authors propose using natural experiments where a product experiences an instantaneous shock in direct traffic, while products recommended alongside it do not. This scenario allows for causal identification through an instrumental variable approach.

Key aspects of their methodology and findings:
*   **Problem Statement**: Traditional methods of measuring recommendation system impact (e.g., counting clicks) are biased because user interest in a focal product is often correlated with interest in its recommended products. Randomized experiments are ideal but costly.
*   **Causal Identification**: The method identifies "shocks" to a focal product's direct traffic (e.g., a sudden surge in pageviews) that act as an instrumental variable. By ensuring that direct traffic to recommended products remains constant during these shocks, the approach decouples the causal effect of recommendations from confounding factors like correlated demand.
*   **Data and Application**: The method is applied to browsing logs of 2.1 million Amazon.com users over nine months, analyzing over 4,000 products that experienced such shocks.
*   **Results**: The study finds that while recommendation click-throughs account for a large fraction of traffic for these products, at least **75% of this activity would likely occur even in the absence of recommendations**. This implies that only about a quarter of observed recommendation-driven traffic is truly *causal*.
*   **Implications**: The findings suggest that recommendation systems primarily serve as a "convenience" for users to find products they would have otherwise discovered, rather than genuinely increasing overall demand or exposure to entirely new items. The causal click-through rate is estimated to be around 3%.

The paper discusses limitations, such as the specific nature of identified shocks (often popular products, particular categories like ebooks), and acknowledges that the local average treatment effect (LATE) estimated may not generalize to all recommendations or user behaviors on Amazon. However, the methodology is presented as generalizable for practitioners with access to granular log data.

## FlashSales_R1_web.txt: Using Clickstream Data to Improve Flash Sales Effectiveness

This paper introduces a hierarchical demand model that utilizes clickstream data to improve the effectiveness of flash sales, where products are sold for a short period at deep discounts. Flash sales are characterized by high demand uncertainty and limited inventory flexibility, making accurate demand forecasting and responsive pricing crucial.

Key aspects of their methodology and findings:
*   **Hierarchical Demand Model (AEIO)**: The authors develop a four-layered model to predict sequential customer decisions within the shopping funnel:
    *   **A (Arrivals)**: Total customer clicks to the platform.
    *   **E (Entries)**: Clicks into specific campaigns.
    *   **I (Interest)**: Visits to product information pages or add-to-cart actions.
    *   **O (Orders)**: Actual product purchases.
    This hierarchical approach efficiently decomposes sources of variation and provides faster learning, especially for new products without sales history, by aggregating information across higher layers of the funnel.
*   **Data and Validation**: The model is validated using a large dataset from a leading European flash sales retailer, comprising hourly events over three years. It incorporates various covariates, including time-dependent properties (seasonality, campaign lifecycle), campaign properties (brand strength, number of products), and product properties (category, price, number of sizes).
*   **Key Findings**:
    *   **Life-cycle Dynamics**: Strong variations exist across the campaign lifecycle. Browsing behavior is intense at the beginning of a campaign, while purchasing behavior is stronger towards the end.
    *   **Heterogeneity**: Significant heterogeneity is observed across campaigns and products, making models with individual fixed effects perform much better in terms of goodness-of-fit. Early clickstream data is highly informative for identifying product attractiveness and predicting sell-out rates.
    *   **Forecasting Accuracy**: The hierarchical model (especially the saturated version with fixed effects) significantly outperforms direct machine learning techniques like random forests in out-of-sample prediction accuracy by leveraging information from different stages of the funnel and learning faster from new products.
*   **Price Optimization**: The model enables dynamic price optimization. Simulations show that re-optimizing prices based on early clicks can significantly increase expected profits, especially for high-demand items predicted to sell out. Optimal prices generally increase with product attractiveness and decrease with inventory.

The paper emphasizes that understanding customer behavior within the shopping funnel, improving forecast accuracy, and allowing retailers to learn about demand faster are key benefits. Future research directions include examining individual responses to various operational aspects and developing analytical models for customer browsing and purchasing.

## spillovers.txt: Modeling Cross-Category Purchases in Sponsored Search Advertising

This paper investigates cross-category purchase behavior in sponsored search advertising, using a unique dataset from a large nationwide retailer. The research focuses on understanding how consumers' initial search queries influence not only purchases within the searched category but also purchases across different, potentially related, product categories within the same shopping session ("spillovers").

Key aspects of their methodology and findings:
*   **Modeling Framework**: A Hierarchical Bayesian modeling framework is developed to analyze multi-category choice, explicitly separating the *intrinsic utility* (preference for a product category itself) from *extrinsic utility* (interdependence between categories in joint purchases). It accounts for unobserved heterogeneity across keywords.
*   **Data**: The study uses a 6-month panel dataset of several hundred keywords from a retailer advertising on Google, covering four product categories: bath, bedding, kitchen, and home decor. It highlights that 12.78% of observed conversions involved joint purchases from two different categories.
*   **Intrinsic Utility Findings**:
    *   Significant spillovers between initial search and final purchase across categories are observed. Consumers often search for a product in one category but purchase from another, in addition to the original.
    *   These search-purchase spillover effects are not necessarily symmetric between any two given product categories.
*   **Extrinsic Utility Findings (Cross-Category Interdependence)**:
    *   Evidence of positive cross-category interdependence is found for retailer-specific keywords.
    *   Brand-specific and generic keywords are *less likely* to induce cross-category purchases compared to retailer keywords.
    *   Unlike prior offline studies, which often found universal positive correlations, this study identifies both positive and negative interdependence (e.g., between bedding and kitchen items) in online purchase behavior, especially when focusing on conversions.
*   **Impact of Other Covariates**:
    *   Price is negatively associated with purchase intention, with varying sensitivities across categories.
    *   Higher ad rank generally leads to more purchases.
    *   Latency (time between click and purchase) does not consistently affect purchase incidence across all categories, but individual keyword-level estimates show variability.
*   **Managerial Implications (Counterfactual Experiments)**:
    *   **Customized Price Discounts**: Uniform price discounts across all keywords for a category are not always profitable; tailored discounts based on keyword type (retailer-specific, brand-specific, product-specific) and category can significantly increase profits.
    *   **Recommendation Systems**: Retailers can design explicit recommendation systems leveraging these insights to promote cross-selling (horizontal spillovers) by pairing products whose sales are known to be associated with specific keywords.
    *   **Keyword Optimization**: SEM firms can develop systems to identify commercially valuable keywords that induce cross-category purchases and suggest them to advertisers, improving the efficiency of the sponsored search market.

The paper emphasizes that understanding these complex interdependencies is crucial for optimizing online advertising strategies and improving profitability. Limitations include a lack of precise competitor data and comprehensive information on other ad content.

## ssrn-2621304 (1).txt: Attributing Conversions in a Multichannel Online Marketing Environment: An Empirical Model and a Field Experiment

This paper proposes a three-level measurement model to attribute conversions in a multichannel online marketing environment, addressing the limitations of traditional aggregate metrics like "last-click" attribution. These traditional metrics often misrepresent the true incremental value of each marketing channel (display, paid search, referral, email, etc.) by ignoring prior customer "touches" and non-converting paths.

Key aspects of their methodology and findings:
*   **Three-Level Model**: The model analyzes:
    1.  **Consideration Stage**: Accounts for customer heterogeneity in considering different online channels (customer-initiated vs. firm-initiated).
    2.  **Visit Stage**: Models visits through channels over time, incorporating carryover (impact of prior visits in the *same* channel) and spillover (impact of prior visits in *different* channels) effects on visit costs.
    3.  **Purchase Stage**: Models subsequent purchases, integrating the cumulative informational stock gathered from prior visits, which influences purchase utility.
*   **Data and Validation**: The model is estimated using individual-level path data from a hospitality firm. A field study, where paid search was paused for a week, validated the model's ability to estimate the incremental impact of a channel on conversions, with predictions closely matching observed outcomes.
*   **Key Findings**:
    *   **Significant Carryover & Spillover Effects**: The study finds substantial carryover and spillover effects at both visit and purchase stages, varying significantly across channels. For instance, email and display ads can trigger visits through search and referral, and email can lead to purchases via search.
    *   **Misleading Traditional Metrics**: Traditional metrics significantly undervalue channels like email, display, and referral, while overvaluing search channels. Organic search, for example, is often used as a navigational tool by customers who were influenced by other channels earlier in their journey.
    *   **Informational Stock Decay**: The informational impact of past channel visits decays over time, with search and email having longer-lasting effects than display ads.
*   **Managerial Implications**:
    *   **Budget Allocation**: The model provides a more accurate basis for allocating marketing budgets by revealing the true incremental contribution of each channel.
    *   **Targeting Strategies**: It can inform customized targeting strategies. For example, email retargeting can increase conversion probabilities in some path scenarios but decrease them in others, highlighting the need for path-aware intervention.
    *   **Paid Search Effectiveness**: The study suggests that for strong brands, much of the "lost" paid search conversions when the channel is paused are recaptured by organic search, indicating a lower incremental value for paid search than commonly perceived.

The research emphasizes the necessity of integrated models that consider the full customer journey, including non-converting paths, to accurately measure channel effectiveness and optimize marketing investments.