# Marketing ROI & Budget Reallocation Analysis

**Author:** Vaishnavi Bedre  
**Project Type:** Marketing Analytics Case Study  
**Tools Used:** Python, Pandas, NumPy, Scikit-Learn, Power BI  
**Focus Areas:** Marketing Analytics | Attribution Modeling | Regression | Budget Optimization | Dashboarding


Project Type: Portfolio Project
Domain: Marketing Analytics

---

## Project Overview

This project builds an **end-to-end marketing analytics pipeline** to evaluate multi-channel marketing performance and recommend an optimized marketing budget allocation.

The analysis integrates **data engineering, marketing analytics, statistical modeling, and dashboard visualization** to diagnose marketing efficiency and identify opportunities to improve return on marketing investment.

The project includes:

• A Python ETL pipeline to prepare marketing datasets  
• Analytical modeling to evaluate channel and campaign performance  
• Attribution analysis to understand channel contribution  
• Regression modeling to estimate incremental revenue impact  
• A Power BI dashboard to present executive insights  

---

## Business Objective

Optimize marketing budget allocation across channels to maximize marketing ROI under a fixed monthly marketing budget.

Key questions answered:

• Which channels generate the highest ROI?  
• Which campaigns show inefficient spend?  
• How does marketing spend impact revenue?  
• What budget allocation should be recommended?  
• What business impact can be expected?  

---

## North Star Metric

Primary metric:

**ROAS (Return on Ad Spend)**

ROAS measures marketing efficiency:

ROAS = Revenue / Marketing Spend

Higher ROAS indicates better revenue generation per ₹1 spent.

Supporting metrics:

• CAC Proxy  
• Conversion Rate  
• Revenue per Session  
• Contribution Margin Proxy  
• Spend vs Revenue Share  

---

## Project Architecture

The project follows a typical analytics engineering workflow:

**Raw Data → ETL Pipeline → Feature Engineering → Analysis → Modeling → Dashboard → Business Recommendations**

---

## Repository Structure

```
marketing-roi-budget-optimization/

README.md
requirements.txt

data/
└── processed/
    ├── fact_sessions.csv
    ├── fact_campaign_daily.csv
    └── fact_channel_daily.csv

etl/
└── marketing_etl_pipeline.py

analysis/
└── marketing_analysis.py

dashboard/
├── executive_summary.png
├── channel_campaign.png
├── attribution_vs_regression.png
└── segments_opportunities.png

reports/
├── marketing_project_report.pdf
├── problem_framing.pdf
└── analysis_output.pdf

docs/
└── data_dictionary.md
```

---

## Data Engineering (ETL Pipeline)

The ETL pipeline performs:

### Data Cleaning

• Duplicate removal  
• Channel normalization  
• Missing value handling  
• Revenue outlier treatment  

### Feature Engineering

Three analytical datasets were created:

### fact_sessions

Session-level attribution dataset including:

• Purchase flags  
• Revenue metrics  
• New vs returning users  
• Session to order time  

### fact_campaign_daily

Daily campaign performance metrics:

• Spend  
• Revenue  
• Orders  
• ROAS  
• CAC proxy  
• CTR  
• CVR  
• CPC  

### fact_channel_daily

Channel level daily metrics:

• Channel spend  
• Revenue  
• Orders  
• Promotion flags  
• Weekly trends  

---

## Analytical Components

### Marketing Performance Diagnosis

Evaluates:

• ROAS by channel  
• CAC efficiency  
• Campaign revenue contribution  
• Marketing efficiency  

---

### Attribution Analysis

Last-touch attribution assigns revenue to the final channel interaction before purchase.

This identifies channels responsible for closing conversions.

Limitation:
Attribution does not measure incremental impact.

---

### Regression Impact Modeling

Regression modeling estimates the marginal revenue impact of marketing spend.

Model includes:

• Log transformed spend  
• Lag variables  
• Seasonality  
• Time trend  
• Day-of-week effects  

Model performance:

R² ≈ 0.58  
MAPE ≈ 10%

---

### Budget Optimization Strategy

Budget allocation based on:

• Channel efficiency  
• Regression marginal impact  
• Allocation constraints  

Constraints:

Minimum channel allocation: **5%**  
Maximum channel allocation: **40%**

Goal:

Improve efficiency without increasing total marketing spend.

---

## Key Insights

• Organic delivers highest efficiency  
• Email shows strong conversion economics  
• Search drives highest revenue scale  
• Paid Social shows weakest ROI  
• Returning customers generate majority revenue  
• Marketing explains meaningful revenue variation  

---

## Business Recommendations

Increase investment:

• Organic
• High efficiency campaigns

Maintain:

• Email
• Referral

Reduce:

• Paid Social inefficient campaigns

Expected impact:

• Improved marketing ROI  
• Reduced inefficient spend  
• Higher revenue productivity  
• Better allocation efficiency  

---

## Dashboard

The Power BI dashboard presents insights across four views:

### Executive Summary

• Marketing KPIs  
• Forecasted performance  
• Revenue trends  

### Channel & Campaign Performance

• Channel efficiency comparison  
• Campaign ROAS  
• CAC analysis  

### Attribution vs Regression

• Attributed revenue comparison  
• Incremental impact estimation  

### Segments & Opportunities

• Customer analysis  
• Product performance  
• Channel × device insights  

---

## How to Run the Project

### Install dependencies

```
pip install -r requirements.txt
```

### Run ETL pipeline

```
python etl/marketing_etl_pipeline.py
```

### Run analysis

```
python analysis/marketing_analysis.py
```

---

## Tools & Technologies

Python  
Pandas  
NumPy  
Scikit-Learn  
Statsmodels  
Matplotlib  
Power BI  

Techniques:

• Data Engineering  
• Marketing Analytics  
• Attribution Modeling  
• Regression Modeling  
• Forecasting  
• Budget Optimization  
• Dashboarding  

---

## Skills Demonstrated

Marketing Analytics  
Data Engineering  
Attribution Modeling  
Regression Modeling  
Forecasting  
Power BI  
Business Strategy  
Budget Optimization  

---

## Author

**Vaishnavi Bedre**

Marketing Analytics | Data Analytics | Business Analytics

---

## Future Improvements

Add marketing mix modeling  
Add causal inference techniques  
Automate ETL pipeline  
Deploy dashboard to cloud  

---

If you found this project interesting, feel free to connect.
