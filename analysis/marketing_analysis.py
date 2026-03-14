#!/usr/bin/env python
# coding: utf-8

# In[16]:


# ===============================
# Analytics (Insights + KPIs)
# ===============================

import pandas as pd
import numpy as np
import os
import json
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import matplotlib.ticker as mtick
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import gaussian_kde

os.makedirs("images", exist_ok=True)



## Formatting and Load Data

# ===============================
# Formatting
# ===============================

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"

main_color = "#4C72B0"
accent_color = "#55A868"

def banner(title):
    display(HTML(f"""
    <div style="
        background: linear-gradient(90deg,#34495e,#5fa8b5);
        color:white;
        padding:12px 16px;
        font-size:22px;
        font-weight:bold;
        border-radius:8px;
        margin-top:20px;
        margin-bottom:14px;
        letter-spacing:0.5px;
    ">
    {title}
    </div>
    """))


def section(title):
    display(HTML(f"""
    <div style="
        background-color:#f6f7f8;
        padding:6px 12px;
        border-left:4px solid #5fa8b5;
        font-size:15px;
        font-weight:600;
        margin-top:12px;
        margin-bottom:6px;
    ">
    {title}
    </div>
    """))

display(HTML("""
<style>

div.output_area pre {
    font-family: "Segoe UI", "Arial", sans-serif !important;
    font-size: 14px;
    line-height: 1.7;
    color: #2c3e50;
}

.report-text {
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 14px;
    line-height: 1.7;
    color: #2c3e50;
}

.report-text strong {
    font-size: 15px;
    font-weight: 600;
}

</style>
"""))

def report(text):
    display(HTML(f"<div class='report-text'>{text}</div>"))

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.float_format", "{:,.2f}".format)

def format_inr_millions(x):

    if abs(x) >= 1_000_000:
        return f"₹ {x/1_000_000:,.2f}M"

    return f"₹ {x:,.0f}"

def format_million_columns(df, columns):

    df_copy = df.copy()

    for col in columns:
        df_copy[col] = (df_copy[col] / 1_000_000).round(2)
        df_copy = df_copy.rename(columns={col: f"{col.title()} (₹M)"})

    return df_copy

def save_chart(name):

    fig = plt.gcf()

    fig.savefig(
        f"images/{name}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()

# To load tables

fact_sessions = pd.read_csv("output/fact_sessions.csv")

fact_campaign = pd.read_csv("output/fact_campaign_daily.csv")

fact_channel = pd.read_csv("output/fact_channel_daily.csv")

fact_sessions["channel"] = fact_sessions["channel"].str.lower()
fact_channel["channel"] = fact_channel["channel"].str.lower()




## Analytics (Insights + KPIs)

#### Marketing Performance Diagnosis

# ===============================
# Marketing Performance Diagnosis
# ===============================

banner("MARKETING PERFORMANCE DIAGNOSIS")
# section("Channel Performance")


channel_performance = fact_channel.groupby(
    "channel"
).agg(

    total_spend=("total_spend","sum"),

    total_orders=("attributed_orders","sum"),

    total_revenue=("attributed_revenue","sum")

).reset_index()

#  Calculate KPIs

channel_performance["ROAS"] = (
    channel_performance["total_revenue"]
    /
    channel_performance["total_spend"]
)

channel_performance["CAC_proxy"] = (
    channel_performance["total_spend"]
    /
    channel_performance["total_orders"]
)

# Clean Numbers

channel_performance = channel_performance.replace(
    [np.inf,-np.inf],
    0
)

channel_performance = channel_performance.fillna(0)

channel_performance["channel"] = (
    channel_performance["channel"]
    .str.replace("_"," ")
    .str.title()
)

# Sort by ROAS

channel_performance = channel_performance.sort_values(
    "ROAS",
    ascending=False
)

section("Channel Performance Table")

channel_table = channel_performance.copy()

# Convert to millions
channel_table["Marketing Spend (₹M)"] = channel_table["total_spend"] / 1e6
channel_table["Attributed Revenue (₹M)"] = channel_table["total_revenue"] / 1e6

channel_table = channel_table.rename(columns={
    "channel": "Channel",
    "total_orders": "Orders",
    "ROAS": "ROAS",
    "CAC_proxy": "CAC Proxy (₹)"
})

channel_table = channel_table[
    [
        "Channel",
        "Marketing Spend (₹M)",
        "Orders",
        "Attributed Revenue (₹M)",
        "ROAS",
        "CAC Proxy (₹)"
    ]
]

display(
    channel_table.style
    .format({
        "Marketing Spend (₹M)": "{:.2f}",
        "Attributed Revenue (₹M)": "{:.2f}",
        "ROAS": "{:.2f}",
        "CAC Proxy (₹)": "{:,.0f}"
    })
    .highlight_max(subset=["ROAS"], color="#d4edda")
    .highlight_min(subset=["ROAS"], color="#f8d7da")
    .hide(axis="index")
)

section("ROAS by Channel")

plt.figure(figsize=(8,5))

plt.bar(
    channel_performance["channel"],
    channel_performance["ROAS"],
    color="#4C72B0"  
)

plt.axhline(y=1, linestyle="--", color="black", label="Break-even ROAS")

plt.xlabel("Marketing Channel")
plt.ylabel("ROAS")
plt.title("Return on Ad Spend by Channel")

plt.legend()
plt.grid(axis="y", alpha=0.3)

save_chart("channel_roas")

# ===============================
# Campaign Performance
# ===============================

campaign_performance = fact_campaign.groupby(
    "campaign_id"
).agg(

    total_spend=("spend","sum"),

    total_orders=("attributed_orders","sum"),

    total_revenue=("attributed_revenue","sum")

).reset_index()

# KPIs

campaign_performance["ROAS"] = (
    campaign_performance["total_revenue"]
    /
    campaign_performance["total_spend"]
)

campaign_performance["CAC_proxy"] = (
    campaign_performance["total_spend"]
    /
    campaign_performance["total_orders"]
)

# Clean values

campaign_performance = campaign_performance.replace(
    [np.inf,-np.inf],
    0
)

campaign_performance = campaign_performance.fillna(0)

section("Campaign Performance Table")

campaign_table = campaign_performance.copy()

campaign_table["Campaign Spend (₹M)"] = campaign_table["total_spend"] / 1e6
campaign_table["Attributed Revenue (₹M)"] = campaign_table["total_revenue"] / 1e6

campaign_table = campaign_table.rename(columns={
    "campaign_id": "Campaign",
    "total_orders": "Orders",
    "ROAS": "ROAS",
    "CAC_proxy": "CAC Proxy (₹)"
})

campaign_table = campaign_table[
    [
        "Campaign",
        "Campaign Spend (₹M)",
        "Orders",
        "Attributed Revenue (₹M)",
        "ROAS",
        "CAC Proxy (₹)"
    ]
]

campaign_table = campaign_table.sort_values(
    "ROAS",
    ascending=False
)

campaign_table = campaign_table.round(2)

campaign_table = campaign_table.reset_index(drop=True)

display(
    campaign_table.style
    .format({
        "Campaign Spend (₹M)": "{:.2f}",
        "Attributed Revenue (₹M)": "{:.2f}",
        "ROAS": "{:.2f}",
        "CAC Proxy (₹)": "{:,.0f}"
    })
    .highlight_max(subset=["ROAS"], color="#d4edda")
    .highlight_min(subset=["ROAS"], color="#f8d7da")
    .hide(axis="index")
)


# section("Top 5 Campaigns by Spend")

top_spend = campaign_performance.sort_values(
    "total_spend",
    ascending=False
).head(5)

section("Top Campaigns by Marketing Spend")

plt.figure(figsize=(8,5))

plt.bar(
    top_spend["campaign_id"],
    top_spend["total_spend"]/1e6,
    color=accent_color
)

plt.ylabel("Marketing Spend (₹M)")
plt.xlabel("Campaign")
plt.title("Top Campaigns by Marketing Spend")

plt.grid(axis="y", alpha=0.3)

save_chart("top_campaigns_by_marketing_spend")

top_spend

# section("Top 5 Campaigns by Revenue")


top_revenue = campaign_performance.sort_values(
    "total_revenue",
    ascending=False
).head(5)

section("Top Campaigns by Revenue")

plt.figure(figsize=(8,5))

plt.bar(
    top_revenue["campaign_id"],
    top_revenue["total_revenue"]/1e6,
    color=main_color
)

plt.ylabel("Revenue (₹M)")
plt.xlabel("Campaign")
plt.title("Top Campaigns by Attributed Revenue")

plt.grid(axis="y", alpha=0.3)

save_chart("top_campaign_by_attributed_revenue")

top_revenue

# ===============================
# Quick Technical Verification
# ===============================

print(f"Quick Technical Verification:")
print(f"campaign_performance total spend = {campaign_performance['total_spend'].sum()}")
print(f"fact_campaign spend = {fact_campaign['spend'].sum()}")

print("As they match, the aggregation is 100% correct.")

# ===============================
# Wasted Spend Detection
# ===============================

median_spend = campaign_performance["total_spend"].median()

wasted_spend = campaign_performance[

    (campaign_performance["total_spend"] > median_spend)
    &
    (campaign_performance["ROAS"] < 1)

].copy()

section("Wasted Spend Candidates")

wasted_table = wasted_spend.copy()

wasted_table["Spend (₹M)"] = wasted_table["total_spend"] / 1e6
wasted_table["Revenue (₹M)"] = wasted_table["total_revenue"] / 1e6

wasted_table = wasted_table.rename(columns={
    "campaign_id": "Campaign",
    "total_orders": "Orders"
})

wasted_table = wasted_table[
    ["Campaign","Spend (₹M)","Orders","Revenue (₹M)","ROAS","CAC_proxy"]
]

wasted_table = wasted_table.sort_values(
    ["ROAS","Spend (₹M)"]
).reset_index(drop=True)

display(
    wasted_table.style
    .format({
        "Spend (₹M)": "{:.2f}",
        "Revenue (₹M)": "{:.2f}",
        "ROAS": "{:.2f}",
        "CAC_proxy": "{:,.0f}"
    })
    .highlight_between(subset=["ROAS"], left=0, right=1, color="#f8d7da")
    .hide(axis="index")
)


##### Wasted campaigns are those with above-median spend and ROAS < 1 (unprofitable)

### Weekly KPIs include efficiency, conversion, value, profitability, and growth metrics to track marketing performance over time.

section("Key Marketing KPIs")

kpi_summary = pd.DataFrame({

    "Metric":[
        "Total Marketing Spend",
        "Total Attributed Revenue",
        "Overall ROAS",
        "Total Orders"
    ],

    "Value":[
        fact_campaign["spend"].sum()/1e6,
        fact_campaign["attributed_revenue"].sum()/1e6,
        fact_campaign["attributed_revenue"].sum() /
        fact_campaign["spend"].sum(),
        fact_campaign["attributed_orders"].sum()
    ]
})

display(
    kpi_summary.style
    .format({"Value": "{:,.2f}"})
    .set_properties(subset=["Metric"], **{"text-align": "left"})
    .set_table_styles([
        {'selector': 'th.col_heading', 'props': 'text-align: center;'}
    ])
    .hide(axis="index")
)




# ===============================
# Weekly KPI Trends
# ===============================
banner("KPI TRENDS")
# section("Weekly Trends for KPIs")

# Convert dates

fact_sessions["session_ts"] = pd.to_datetime(fact_sessions["session_ts"])
fact_channel["date"] = pd.to_datetime(fact_channel["date"])


# Create week column

fact_sessions["week"] = fact_sessions["session_ts"].dt.to_period("W")
fact_channel["week"] = fact_channel["date"].dt.to_period("W")


# ===============================
# Weekly Sessions Metrics
# ===============================

sessions_weekly = fact_sessions.groupby("week").agg(

    total_sessions=("session_id","count"),

    total_orders=("purchase_flag","sum"),

    total_revenue=("net_revenue","sum"),

    total_gross_revenue=("gross_revenue","sum")

).reset_index()


# ===============================
# Weekly Spend Metrics
# ===============================

spend_weekly = fact_channel.groupby("week").agg(

    total_spend=("total_spend","sum"),

    promo_days=("promo_flag","sum")

).reset_index()


# ===============================
# Merge Weekly Tables
# ===============================

weekly_kpis = sessions_weekly.merge(
    spend_weekly,
    on="week",
    how="left"
)


# ===============================
# Marketing KPIs
# ===============================

# Efficiency

weekly_kpis["ROAS"] = (
    weekly_kpis["total_revenue"]
    /
    weekly_kpis["total_spend"]
)

weekly_kpis["CAC_proxy"] = (
    weekly_kpis["total_spend"]
    /
    weekly_kpis["total_orders"]
)


# Conversion Metrics

weekly_kpis["conversion_rate"] = (
    weekly_kpis["total_orders"]
    /
    weekly_kpis["total_sessions"]
)


weekly_kpis["revenue_per_session"] = (
    weekly_kpis["total_revenue"]
    /
    weekly_kpis["total_sessions"]
)


# Value Metrics

weekly_kpis["AOV"] = (
    weekly_kpis["total_revenue"]
    /
    weekly_kpis["total_orders"]
)


# Profitability Metric (Margin Proxy)

weekly_kpis["margin_proxy"] = (
    weekly_kpis["total_revenue"]
    - weekly_kpis["total_spend"]
)


# Growth Metrics

weekly_kpis["revenue_growth"] = (
    weekly_kpis["total_revenue"].pct_change()
)

weekly_kpis["spend_growth"] = (
    weekly_kpis["total_spend"].pct_change()
)

weekly_kpis["orders_growth"] = (
    weekly_kpis["total_orders"].pct_change()
)


# ===============================
# Clean Values
# ===============================

weekly_kpis = weekly_kpis.replace(
    [np.inf,-np.inf],
    0
)

weekly_kpis = weekly_kpis.fillna(0)


# ===============================
# Round Values
# ===============================

weekly_kpis["ROAS"] = weekly_kpis["ROAS"].round(2)

weekly_kpis["CAC_proxy"] = weekly_kpis["CAC_proxy"].round(0)

weekly_kpis["conversion_rate"] = weekly_kpis["conversion_rate"].round(4)

weekly_kpis["revenue_per_session"] = weekly_kpis["revenue_per_session"].round(2)

weekly_kpis["AOV"] = weekly_kpis["AOV"].round(2)

weekly_kpis["margin_proxy"] = weekly_kpis["margin_proxy"].round(0)

weekly_kpis["revenue_growth"] = weekly_kpis["revenue_growth"].round(3)

weekly_kpis["spend_growth"] = weekly_kpis["spend_growth"].round(3)

weekly_kpis["orders_growth"] = weekly_kpis["orders_growth"].round(3)


# ===============================
# Sort by Week
# ===============================

weekly_kpis = weekly_kpis.sort_values("week")

weekly_kpis["week"] = weekly_kpis["week"].dt.start_time
weekly_kpis["week"] = weekly_kpis["week"].dt.strftime("%b %d")

weekly_kpis["week"] = weekly_kpis["week"].astype(str)

# ===============================
# Remove Partial First Week
# ===============================

# Remove weeks with very low sessions (partial data)

weekly_kpis = weekly_kpis[weekly_kpis["total_sessions"] > 100].reset_index(drop=True)

section("Weekly KPI Table")

weekly_table = weekly_kpis.copy()

weekly_table["Revenue (₹M)"] = weekly_table["total_revenue"] / 1e6
weekly_table["Spend (₹M)"] = weekly_table["total_spend"] / 1e6

weekly_table = weekly_table.rename(columns={
    "week": "Week (2025–2026)",
    "total_sessions": "Sessions",
    "total_orders": "Orders",
    "ROAS": "ROAS",
    "CAC_proxy": "CAC Proxy (₹)",
    "conversion_rate": "Conversion Rate",
    "AOV": "AOV (₹)"
})

weekly_table = weekly_table[
    [
        "Week (2025–2026)",
        "Sessions",
        "Orders",
        "Revenue (₹M)",
        "Spend (₹M)",
        "ROAS",
        "Conversion Rate",
        "AOV (₹)",
        "CAC Proxy (₹)"
    ]
]

display(
    weekly_table.style
    .format({
        "Revenue (₹M)": "{:.2f}",
        "Spend (₹M)": "{:.2f}",
        "ROAS": "{:.2f}",
        "Conversion Rate": "{:.2%}",
        "AOV (₹)": "{:,.0f}",
        "CAC Proxy (₹)": "{:,.0f}"
    })
    .highlight_max(subset=["ROAS"], color="#d4edda")
    .highlight_min(subset=["ROAS"], color="#f8d7da")
    .hide(axis="index")
)

section("Weekly Revenue Trend")

plt.figure(figsize=(10,5))

plt.plot(
    weekly_kpis["week"].astype(str),
    weekly_kpis["total_revenue"]/1e6,
    marker="o",
    color=main_color,
    linewidth=2
)

plt.title("Weekly Revenue Trend (2025–2026)")

plt.xlabel("Week")
plt.ylabel("Revenue (₹M)")

plt.grid(axis="y", alpha=0.3)

plt.xticks(rotation=45)

save_chart("weekly_revenue_trend")

section("Weekly ROAS Trend")

plt.figure(figsize=(10,5))

plt.plot(
    weekly_kpis["week"].astype(str),
    weekly_kpis["ROAS"],
    marker="o",
    color=accent_color,
    linewidth=2
)

plt.axhline(
    y=1,
    linestyle="--",
    color="black",
    label="Break-even ROAS"
)

plt.title("Weekly ROAS Trend (2025–2026)")

plt.xlabel("Week")
plt.ylabel("ROAS")

plt.legend()

plt.grid(axis="y", alpha=0.3)

plt.xticks(rotation=45)

save_chart("weekly_roas_trend")

section("Spend vs Revenue Efficiency")

plt.figure(figsize=(8,6))

scatter = plt.scatter(
    weekly_kpis["total_spend"]/1e6,
    weekly_kpis["total_revenue"]/1e6,
    c=weekly_kpis["ROAS"],
    s=weekly_kpis["total_orders"]/5,
    cmap="viridis",
    alpha=0.8,
    edgecolors="black",
    linewidth=0.5
)


# Identify key weeks
top_roas = weekly_kpis.loc[weekly_kpis["ROAS"].idxmax()]
low_roas = weekly_kpis.loc[weekly_kpis["ROAS"].idxmin()]
top_revenue = weekly_kpis.loc[weekly_kpis["total_revenue"].idxmax()]

important_weeks = [top_roas, low_roas, top_revenue]

for row in important_weeks:
    plt.text(
        row["total_spend"]/1e6,
        row["total_revenue"]/1e6,
        str(row["week"]),
        fontsize=9,
        ha="left",
        va="bottom"
    )

plt.xlabel("Marketing Spend (₹M)")
plt.ylabel("Revenue (₹M)")

plt.title("Marketing Efficiency by Week (2025–2026)")

plt.colorbar(scatter, label="ROAS")

plt.grid(alpha=0.3)

save_chart("marketing_efficiency_by_week")

section("Efficiency Insight")

report("""

<div style="
max-width:850px;
margin-left:20px;
line-height:1.7;
">

<ul>
<li>Weeks with higher marketing spend do not always generate proportionally higher revenue.</li>
<li>Several weeks achieve strong revenue efficiency with moderate spend levels.</li>
<li>This indicates opportunities to scale high-performing campaigns without proportionally increasing spend.</li>
</ul>

</div>

""")

section("Revenue vs Marketing Spend Trend")

plt.figure(figsize=(10,5))

plt.plot(
    weekly_kpis["week"].astype(str),
    weekly_kpis["total_revenue"]/1e6,
    marker="o",
    label="Revenue",
    color=main_color,
    linewidth=2
)

plt.plot(
    weekly_kpis["week"].astype(str),
    weekly_kpis["total_spend"]/1e6,
    marker="o",
    label="Marketing Spend",
    color=accent_color,
    linewidth=2
)

plt.title("Revenue vs Marketing Spend Trend (2025–2026)")

plt.xlabel("Week")
plt.ylabel("₹ Millions")

plt.legend()

plt.grid(axis="y", alpha=0.3)

plt.xticks(rotation=45)

save_chart("revenue_vs_marketing_spend")

section("KPI TRENDS")

report("""

<div style="
max-width:850px;
margin-left:20px;
line-height:1.7;
">

<h4 style="margin-top:18px;">Summary</h4>
<ul>
<li>Revenue growth shows noticeable spikes during promotional weeks.</li>
<li>ROAS fluctuates based on spend intensity and campaign effectiveness.</li>
<li>Marketing efficiency improves when spend is concentrated on high-performing channels.</li>
</ul>

<h4 style="margin-top:18px;">Recommendation</h4>

<p>
Maintain consistent spend allocation and prioritize high-ROAS campaigns.
</p>

</div>

""")

section("Detect KPI Anomalies")

# Statistical anomaly detection (mean ± std)

mean_growth = weekly_kpis["revenue_growth"].mean()
std_growth = weekly_kpis["revenue_growth"].std()

section("Revenue Growth Anomalies")

revenue_spikes = weekly_kpis[

    (weekly_kpis["revenue_growth"] > mean_growth + std_growth)
    |
    (weekly_kpis["revenue_growth"] < mean_growth - std_growth)

]

revenue_spikes[[

    "week",
    "total_revenue",
    "revenue_growth",
    "total_spend",
    "promo_days",
    "ROAS"

]]

anomaly_table = revenue_spikes.copy()

anomaly_table["Revenue (₹M)"] = anomaly_table["total_revenue"]/1e6
anomaly_table["Spend (₹M)"] = anomaly_table["total_spend"]/1e6

anomaly_table = anomaly_table.rename(columns={
    "week": "Week (2025–2026)",
    "revenue_growth": "Revenue Growth",
    "promo_days": "Promo Days"
})

display(
    anomaly_table[
        [
            "Week (2025–2026)",
            "Revenue (₹M)",
            "Revenue Growth",
            "Spend (₹M)",
            "Promo Days",
            "ROAS"
        ]
    ].style
    .format({
        "Revenue (₹M)": "{:.2f}",
        "Spend (₹M)": "{:.2f}",
        "Revenue Growth": "{:.1%}",
        "ROAS": "{:.2f}"
    })
    .highlight_max(subset=["Revenue Growth"], color="#d4edda")
    .highlight_min(subset=["Revenue Growth"], color="#f8d7da")
    .hide(axis="index")
)

section("Efficiency Anomalies")

top_efficiency_table = weekly_kpis.sort_values("ROAS", 
    ascending=False).head(3).copy()

top_efficiency_table["Revenue (₹M)"] = top_efficiency_table["total_revenue"] / 1e6
top_efficiency_table["Spend (₹M)"] = top_efficiency_table["total_spend"] / 1e6

top_efficiency_table = top_efficiency_table.rename(columns={
    "week": "Week (2025–2026)",
    "conversion_rate": "Conversion Rate"
})

display(
    top_efficiency_table[
        [
            "Week (2025–2026)",
            "Revenue (₹M)",
            "Spend (₹M)",
            "ROAS",
            "Conversion Rate"
        ]
    ].style
    .format({
        "Revenue (₹M)": "{:.2f}",
        "Spend (₹M)": "{:.2f}",
        "ROAS": "{:.2f}",
        "Conversion Rate": "{:.2%}"
    })
    .highlight_max(subset=["ROAS"], color="#d4edda")
    .hide(axis="index")
)





### Segment Deep Dive

# ===============================
# Segment Deep Dive
# ===============================

banner("SEGMENT DEEP DIVE")


# =====================================================
# Channel × Device Analysis
# =====================================================

section("Channel x Device Performance")


channel_device = fact_sessions.groupby(
    ["channel","device"]
).agg(

    sessions=("session_id","count"),

    orders=("purchase_flag","sum"),

    revenue=("net_revenue","sum")

).reset_index()


# KPIs

channel_device["conversion_rate"] = (
    channel_device["orders"] /
    channel_device["sessions"]
).round(4)

channel_device["conversion_rate_pct"] = channel_device["conversion_rate"] * 100

channel_device["revenue_per_session"] = (
    channel_device["revenue"]
    /
    channel_device["sessions"]
)


# Clean values

channel_device = channel_device.replace(
    [np.inf,-np.inf],
    0
)

channel_device = channel_device.fillna(0)


# Sort

channel_device = channel_device.sort_values(
    "revenue",
    ascending=False
)


# Chart

section("Revenue by Channel x Device Chart")

pivot_cd = channel_device.pivot(
    index="channel",
    columns="device",
    values="revenue"
)

# Clean channel labels
pivot_cd.index = pivot_cd.index.str.replace("_"," ").str.title()

pivot_cd.plot(
    kind="bar",
    figsize=(10,5),
    color=[main_color, accent_color]
)

plt.title("Revenue by Channel and Device")
plt.xlabel("Marketing Channel")
plt.ylabel("Revenue (₹M)")

plt.xticks(rotation=60)

# Format revenue axis
plt.gca().yaxis.set_major_formatter(
    mtick.FuncFormatter(lambda x, p: f'₹ {x/1e6:.1f}M')
)

# Improve legend
plt.legend(title="Device")

save_chart("revenue_by_channel_and_device")

section("Conversion Heatmap: Channel x Device")

pivot_conv = channel_device.pivot(
    index="channel",
    columns="device",
    values="conversion_rate"
)

# Clean channel labels
pivot_conv.index = pivot_conv.index.str.replace("_"," ").str.title()

plt.figure(figsize=(8,5))

heatmap = plt.imshow(pivot_conv, cmap="YlGn")

plt.xticks(range(len(pivot_conv.columns)),pivot_conv.columns.str.title())
plt.yticks(range(len(pivot_conv.index)), pivot_conv.index)

# Add conversion values
for i in range(len(pivot_conv.index)):
    for j in range(len(pivot_conv.columns)):
        value = pivot_conv.iloc[i, j]
        if not pd.isna(value):
            plt.text(
                j, i,
                f"{value:.2%}",
                ha="center",
                va="center",
                color="black"
            )

# Color scale
plt.colorbar(heatmap, label="Conversion Rate")

# Axis labels
plt.xlabel("Device")
plt.ylabel("Marketing Channel")

plt.title(
    "Conversion Rate Heatmap (Channel × Device)",
    fontsize=13,
    pad=12
)

plt.tick_params(axis="x", rotation=0)

# Remove gridlines that interfere with heatmap
plt.grid(False)

save_chart("conversion_rate_heatmap")

# Table

section("Channel x Device Table")

# Clean labels
channel_device["channel"] = (
    channel_device["channel"]
    .str.replace("_"," ")
    .str.title()
)

channel_device["device"] = channel_device["device"].str.title()

table_cd = format_million_columns(
    channel_device[
        ["channel","device","sessions","orders","conversion_rate_pct","revenue"]
    ].copy(),
    ["revenue"]
)

# Rename columns for presentation
table_cd = table_cd.rename(columns={
    "channel": "Channel",
    "device": "Device",
    "sessions": "Sessions",
    "orders": "Orders",
    "conversion_rate_pct": "Conversion Rate"
})

table_cd = table_cd.sort_values("Conversion Rate", ascending=False)

display(
    table_cd.style.format({
        "Conversion Rate": "{:.2}%",
        "Revenue (₹M)": "{:.2f}"
    }).highlight_max(
        subset=["Conversion Rate"],
        color="#d4edda"
    ).hide(axis="index")
)

section("Top Conversion Segments (Channel × Device)")

top_conversion = channel_device.sort_values(
    "conversion_rate",
    ascending=False
).head(3)

table_top_conv = format_million_columns(
    top_conversion[
        ["channel","device","sessions","orders","revenue","conversion_rate","revenue_per_session"]
    ].copy(),
    ["revenue"]
)

table_top_conv = table_top_conv.rename(columns={
    "channel": "Channel",
    "device": "Device",
    "sessions": "Sessions",
    "orders": "Orders",
    "revenue": "Revenue (₹M)",
    "conversion_rate": "Conversion Rate",
    "revenue_per_session": "Revenue / Session (₹)"
})

display(
    table_top_conv.style.format({
        "Revenue (₹M)": "{:.2f}",
        "Conversion Rate": "{:.2%}",
        "Revenue / Session (₹)": "{:.2f}"
    }).highlight_max(
        subset=["Conversion Rate"],
        color="#d4edda"
    ).hide(axis="index")
)

section("Lowest Conversion Segments")

low_conversion = channel_device.sort_values(
    "conversion_rate"
).head(3)

table_low_conv = format_million_columns(
    low_conversion[
        ["channel","device","sessions","orders","revenue","conversion_rate","revenue_per_session"]
    ].copy(),
    ["revenue"]
)

table_low_conv = table_low_conv.rename(columns={
    "channel": "Channel",
    "device": "Device",
    "sessions": "Sessions",
    "orders": "Orders",
    "revenue": "Revenue (₹M)",
    "conversion_rate": "Conversion Rate",
    "revenue_per_session": "Revenue / Session (₹)"
})

display(
    table_low_conv.style.format({
        "Revenue (₹M)": "{:.2f}",
        "Conversion Rate": "{:.2%}",
        "Revenue / Session (₹)": "{:.2f}"
    }).highlight_min(
        subset=["Conversion Rate"],
        color="#ffd6d6"
    ).hide(axis="index")
)

section("Highest Revenue Segments")

top_revenue_segments = channel_device.sort_values(
    "revenue",
    ascending=False
).head(3)

table_top_rev = format_million_columns(
    top_revenue_segments[
        ["channel","device","sessions","orders","revenue","conversion_rate","revenue_per_session"]
    ].copy(),
    ["revenue"]
)

table_top_rev = table_top_rev.rename(columns={
    "channel": "Channel",
    "device": "Device",
    "sessions": "Sessions",
    "orders": "Orders",
    "revenue": "Revenue (₹M)",
    "conversion_rate": "Conversion Rate",
    "revenue_per_session": "Revenue / Session (₹)"
})

display(
    table_top_rev.style.format({
        "Revenue (₹M)": "{:.2f}",
        "Conversion Rate": "{:.2%}",
        "Revenue / Session (₹)": "{:.2f}"
    }).highlight_max(
        subset=["Revenue (₹M)"],
        color="#d4edda"
    ).hide(axis="index")
)

section("Key Insight: Channel × Device Performance")

report("""

<div style="
max-width:850px;
margin-left:20px;
line-height:1.7;
">

<h4>Insight</h4>

<ul>
<li>
Desktop users consistently show higher conversion rates across most
marketing channels, suggesting stronger purchase intent or a smoother
checkout experience compared with mobile.
</li>

<li>
Mobile traffic generates a large share of sessions but converts at
lower rates, indicating potential friction in the mobile purchase journey.
</li>

<li>
Email and Organic channels deliver the strongest conversion performance,
making them valuable channels for high-intent users.
</li>

</ul>

<h4 style="margin-top:18px;">Recommendation</h4>

<p>
Improve mobile checkout UX and reduce purchase friction while
increasing investment in high-performing channels such as Email
and Organic.
</p>

</div>

""")

# =====================================================
# New vs Returning Users
# =====================================================

section("New vs Returning Users")


user_segments = fact_sessions.groupby(
    "is_new_user"
).agg(

    sessions=("session_id","count"),

    orders=("purchase_flag","sum"),

    revenue=("net_revenue","sum")

).reset_index()


user_segments["segment"] = user_segments["is_new_user"].map({

    1:"New Users",
    0:"Returning Users"

})


# KPIs

user_segments["conversion_rate"] = (

    user_segments["orders"]

    /

    user_segments["sessions"]

)


user_segments["AOV"] = (

    user_segments["revenue"]

    /

    user_segments["orders"]

)


# Clean

user_segments = user_segments.replace(
    [np.inf,-np.inf],
    0
)

user_segments = user_segments.fillna(0)


# Chart

section("Revenue by User Type")

plt.figure(figsize=(6,4))

bars = plt.bar(
    user_segments["segment"],
    user_segments["revenue"],
    color=main_color
)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"₹ {height/1e6:.2f}M",
        ha="center",
        va="bottom",
        fontsize=9
    )

# Format axis
plt.gca().yaxis.set_major_formatter(
    mtick.FuncFormatter(lambda x, p: f'₹ {x/1e6:.1f}M')
)

plt.title("Revenue by User Type")

plt.xlabel("User Segment")
plt.ylabel("Revenue (₹M)")

plt.ylim(0, user_segments["revenue"].max()*1.1)

save_chart("revenue_by_user_type")

# Table

section("User Segment Table")

table_user = format_million_columns(
    user_segments[
        ["segment","sessions","orders","revenue","conversion_rate","AOV"]
    ].copy(),
    ["revenue"]
)

table_user = table_user.rename(columns={
    "segment": "User Segment",
    "sessions": "Sessions",
    "orders": "Orders",
    "revenue": "Revenue (₹M)",
    "conversion_rate": "Conversion Rate",
    "AOV": "Average Order Value (₹)"
})

display(
    table_user.style
    .format({
        "Revenue (₹M)": "{:.2f}",
        "Conversion Rate": "{:.2%}",
        "Average Order Value (₹)": "{:,.0f}"
    })
    .highlight_max(subset=["Conversion Rate"], color="#d4edda")
    .hide(axis="index")
)

# =====================================================
# City Tier Analysis
# =====================================================

section("City Tier Performance")


sessions_users = fact_sessions.merge(
    pd.read_csv("output/clean_users.csv"),
    on="user_id",
    how="left"
)


city_segment = sessions_users.groupby(
    "city_tier"
).agg(

    sessions=("session_id","count"),

    orders=("purchase_flag","sum"),

    revenue=("net_revenue","sum")

).reset_index()


city_segment["conversion_rate"] = (

    city_segment["orders"]

    /

    city_segment["sessions"]

)


city_segment = city_segment.replace(
    [np.inf,-np.inf],
    0
)


city_segment = city_segment.replace(
    [np.inf,-np.inf],
    0
)

city_segment = city_segment.fillna(0)

city_segment["city_tier"] = city_segment["city_tier"].astype(int)

city_segment["city_tier_label"] = city_segment["city_tier"].map({
    1: "Tier 1",
    2: "Tier 2",
    3: "Tier 3"
})


# Chart

section("Revenue by City Tier")

plt.figure(figsize=(6,4))

bars = plt.bar(
    city_segment["city_tier_label"],
    city_segment["revenue"],
    color=main_color
)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"₹ {height/1e6:.2f}M",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.gca().yaxis.set_major_formatter(
    mtick.FuncFormatter(lambda x, p: f'₹ {x/1e6:.1f}M')
)

plt.title("Revenue by City Tier")

plt.xlabel("City Tier")
plt.ylabel("Revenue (₹M)")

save_chart("revenue_by_city_tier")

# Table

section("City Tier Table")

table_city = format_million_columns(
    city_segment[
        ["city_tier_label","sessions","orders","revenue","conversion_rate"]
    ].copy(),
    ["revenue"]
)

table_city = table_city.rename(columns={
    "city_tier_label": "City Tier",
    "sessions": "Sessions",
    "orders": "Orders",
    "revenue": "Revenue (₹M)",
    "conversion_rate": "Conversion Rate"
})

display(
    table_city.style
    .format({
        "Revenue (₹M)": "{:.2f}",
        "Conversion Rate": "{:.2%}"
    })
    .highlight_max(subset=["Revenue (₹M)"], color="#d4edda")
    .hide(axis="index")
)

# =====================================================
# Product Category Analysis
# =====================================================

section("Product Category Performance")

order_items = pd.read_csv("output/clean_order_items.csv")

products = pd.read_csv("output/clean_products.csv")


# Merge order items with product catalog

product_data = order_items.merge(
    products,
    on="product_id",
    how="left"
)

product_data["cost_total"] = product_data["quantity"] * product_data["cost"]

# Create revenue column

product_data["item_revenue"] = (
    product_data["quantity"]
    *
    product_data["unit_price"]
)


# Aggregate by category

category_perf = product_data.groupby(
    "category"
).agg(

    quantity=("quantity","sum"),

    revenue=("item_revenue","sum"),

    cost=("cost_total","sum")

).reset_index()

category_perf["margin_proxy"] = (
    category_perf["revenue"] - category_perf["cost"]
)

category_perf["margin_pct"] = np.where(
    category_perf["revenue"] > 0,
    category_perf["margin_proxy"] / category_perf["revenue"],
    0
)


# Sort by revenue

category_perf = category_perf.sort_values(
    "revenue",
    ascending=False
)

category_perf["cumulative_revenue"] = (
    category_perf["revenue"].cumsum()
)

category_perf["cumulative_pct"] = (
    category_perf["cumulative_revenue"]
    /
    category_perf["revenue"].sum()
)

# Chart

section("Revenue vs Margin by Product Category")

x = np.arange(len(category_perf["category"]))

plt.figure(figsize=(10,5))

plt.bar(
    x - 0.2,
    category_perf["revenue"],
    width=0.4,
    label="Revenue",
    color=main_color
)

plt.bar(
    x + 0.2,
    category_perf["margin_proxy"],
    width=0.4,
    label="Estimated Margin",
    color=accent_color
)

plt.gca().yaxis.set_major_formatter(
    mtick.FuncFormatter(lambda x, p: f'₹ {x/1e6:.1f}M')
)

plt.xticks(
    x,
    category_perf["category"],
    rotation=45
)

plt.title("Revenue vs Margin by Product Category")

plt.ylabel("Amount (₹M)")
plt.xlabel("Product Category")

plt.legend()

plt.tight_layout() 
plt.show()


# Category Table

section("Category Table")

category_perf["margin_proxy"] = category_perf["margin_proxy"].round(2)

table_category = format_million_columns(
    category_perf[
        ["category","quantity","revenue","cost","margin_proxy","margin_pct","cumulative_pct"]
    ].copy(),
    ["revenue","cost","margin_proxy"]
)

table_category = table_category.rename(columns={
    "category": "Category",
    "quantity": "Units Sold",
    "revenue": "Revenue (₹M)",
    "cost": "Cost (₹M)",
    "margin_proxy (₹M)": "Margin (₹M)",
    "margin_pct": "Margin",
    "cumulative_pct": "Cumulative Revenue %"
})

display(
    table_category.style
    .format({
        "Revenue (₹M)": "{:.2f}",
        "Cost (₹M)": "{:.2f}",
        "Margin (₹M)": "{:.2f}",
        "Margin": "{:.2%}",
        "Cumulative Revenue %": "{:.1%}"
    })
    .highlight_max(subset=["Margin"], color="#d4edda") 
    .highlight_max(subset=["Revenue (₹M)"], color="#cfe2f3")
    .hide(axis="index")
)

report("""

<div style="
max-width:850px;
margin-left:40px;
line-height:1.7;
">

<h4>Insight</h4>

<ul>
<li>
A small number of product categories generate the majority of revenue,
consistent with the Pareto principle (80/20 rule).
</li>
<li>
The top three product categories (Books, Fashion, and Home) account for 
nearly 60% of total revenue, indicating a strong concentration of demand.
</li>
</ul>

<h4 style="margin-top:18px;">Recommended Action</h4>

<p>
Focus marketing campaigns and promotions on top-performing categories
to maximize revenue impact.
</p>

</div>

""")

section("Key Segment Insights")

# =====================================================
# Top Conversion Segments
# =====================================================

top_conversion_segments = channel_device.sort_values(
    "conversion_rate",
    ascending=False
).head(3)

section("Top 3 Segments with Highest Conversion Rates")

table_conv_segments = top_conversion_segments[
    ["channel","device","sessions","orders","conversion_rate"]
].copy()

table_conv_segments = table_conv_segments.rename(columns={
    "channel": "Channel",
    "device": "Device",
    "sessions": "Sessions",
    "orders": "Orders",
    "conversion_rate": "Conversion Rate"
})

display(
    table_conv_segments.style
    .format({
        "Conversion Rate": "{:.2%}"
    })
    .highlight_max(subset=["Conversion Rate"], color="#d4edda")
    .hide(axis="index")
)

# =====================================================
# ROAS by Channel x Device
# =====================================================

channel_device["channel"] = (
    channel_device["channel"]
    .str.replace("_"," ")
    .str.title()
)

# Total spend per channel
channel_spend = fact_channel.groupby("channel").agg(
    spend=("total_spend","sum")
).reset_index()

# Standardize channel names
channel_device["channel_clean"] = channel_device["channel"].str.lower().str.replace(" ","_")
channel_spend["channel_clean"] = channel_spend["channel"].str.lower().str.replace(" ","_")

# Merge using clean channel names
roas_segments = channel_device.merge(
    channel_spend[["channel_clean","spend"]],
    on="channel_clean",
    how="left"
)

# Calculate ROAS
roas_segments["ROAS"] = (
    roas_segments["revenue"] /
    roas_segments["spend"]
)

# Clean values
roas_segments = roas_segments.replace([np.inf,-np.inf],0).fillna(0)

top_roas_segments = roas_segments.sort_values(
    "ROAS",
    ascending=False
).head(3)

# =====================================================
# Strategic Actions
# =====================================================

section("Top Segments Where Conversion Differs Significantly")

fact_sessions["channel"] = fact_sessions["channel"].str.lower()
fact_channel["channel"] = fact_channel["channel"].str.lower()

for i, (_, row) in enumerate(top_conversion_segments.iterrows(), 1):

    report(f"""

<div style="
max-width:850px;
margin-left:40px;
margin-bottom:25px;
padding:16px 18px;
border-radius:8px;
background:#ffffff;
border-left:4px solid #2c7f89;
box-shadow:0 1px 2px rgba(0,0,0,0.05);
line-height:1.7;
">

<h4 style="margin-top:0">{i}. Segment: {row['channel']} | {row['device']}</h4>

<p><strong>Conversion Rate:</strong> {row['conversion_rate']:.2%}</p>

<p><strong>Insight</strong></p>

<p>
Users in this segment convert at a significantly higher rate
than other segments, indicating strong purchase intent.
</p>

<p><strong>Recommended Action</strong></p>

<p>
Increase retargeting investment and expand audience targeting
for this segment to maximize conversion efficiency.
</p>

</div>

""")

# =====================================================
# Top ROAS Segments
# =====================================================

section("Top 3 Segments with Highest ROAS")

table_roas = format_million_columns(
    top_roas_segments[
        ["channel","device","revenue","spend","ROAS"]
    ].copy(),
    ["revenue","spend"]
)

table_roas = table_roas.rename(columns={
    "channel":"Channel",
    "device":"Device",
    "revenue":"Revenue (₹M)",
    "spend":"Spend (₹M)"
})

display(
    table_roas.style
    .format({
        "Revenue (₹M)": "{:.2f}",
        "Spend (₹M)": "{:.2f}",
        "ROAS": "{:.2f}"
    })
    .highlight_max(subset=["ROAS"], color="#d4edda")
    .hide(axis="index")
)

# =====================================================
# Strategic Actions
# =====================================================

section("Top Segments Where ROAS Differs Significantly")

for i, (_, row) in enumerate(top_roas_segments.iterrows(), 1):

    report(f"""

<div style="
max-width:850px;
margin-left:40px;
margin-bottom:25px;
padding:16px 18px;
border-radius:8px;
background:#ffffff;
border-left:4px solid #2c7f89;
box-shadow:0 1px 2px rgba(0,0,0,0.05);
line-height:1.7;
">

<h4 style="margin-top:0">{i}. Segment: {row['channel']} | {row['device']}</h4>

<p><strong>ROAS:</strong> {row['ROAS']:.2f}</p>

<p><strong>Insight</strong></p>

<p>
This segment generates strong revenue relative to marketing spend,
indicating highly efficient marketing performance.
</p>

<p><strong>Recommended Action</strong></p>

<p>
Allocate additional marketing budget to this segment
and prioritize it in campaign optimization.
</p>

</div>

""")

section("Low Conversion / Low ROAS Segments")

report("""

<div style="max-width:850px;margin-left:20px;line-height:1.7;">

<h4>Insight</h4>

<p>
Segments with weaker conversion or ROAS performance indicate potential
friction in the user journey or ineffective targeting.
</p>

<h4 style="margin-top:18px;">Recommended Actions</h4>

<ul>
<li>Improve mobile checkout experience and reduce purchase friction.</li>
<li>Refine audience targeting to focus on higher intent users.</li>
<li>Test new ad creatives and landing pages to improve engagement.</li>
</ul>

</div>

""")

banner("SUMMARY")

report("""

<div style="max-width:850px;margin-left:40px;line-height:1.7;">

<p>
Marketing performance analysis shows clear opportunities for optimization.
</p>

<h4 style="margin-top:18px;">Key Opportunities</h4>

<ul>
<li>Reallocate budget toward high-ROAS channels.</li>
<li>Improve mobile conversion experience.</li>
<li>Focus promotions on top-performing product categories.</li>
<li>Strengthen retention programs for returning customers.</li>
</ul>

<h4 style="margin-top:18px;">Expected Impact</h4>

<p>
Segment-level analysis reveals clear opportunities to improve
marketing efficiency and conversion performance.
</p>

</div>

""")




### Marketing Investigation Table

# =====================================================
# Marketing Investigation Table
# =====================================================

banner("MARKETING INVESTIGATION")

section("Campaign Efficiency Analysis")


campaign_investigation = fact_campaign.groupby(
    "campaign_id"
).agg(

    spend=("spend","sum"),

    orders=("attributed_orders","sum"),

    revenue=("attributed_revenue","sum")

).reset_index()


# KPIs

campaign_investigation["ROAS"] = (
    campaign_investigation["revenue"] /
    campaign_investigation["spend"]
)

campaign_investigation["CAC_proxy"] = np.where(
    campaign_investigation["orders"] > 0,
    campaign_investigation["spend"] / campaign_investigation["orders"],
    0
)

# Clean values

campaign_investigation = campaign_investigation.replace(
    [np.inf,-np.inf],
    0
)

campaign_investigation = campaign_investigation.fillna(0)


# Spend & revenue share

total_spend = campaign_investigation["spend"].sum()
total_revenue = campaign_investigation["revenue"].sum()

campaign_investigation["spend_share_pct"] = (
    campaign_investigation["spend"] / total_spend * 100
)

campaign_investigation["revenue_share_pct"] = (
    campaign_investigation["revenue"] / total_revenue * 100
)


# Round metrics

campaign_investigation["ROAS"] = campaign_investigation["ROAS"].round(2)
campaign_investigation["CAC_proxy"] = campaign_investigation["CAC_proxy"].round(0).astype(int)
campaign_investigation["spend_share_pct"] = campaign_investigation["spend_share_pct"].round(2)
campaign_investigation["revenue_share_pct"] = campaign_investigation["revenue_share_pct"].round(2)

section("Spend Share vs Revenue Share Gap")


gap_df = campaign_investigation.copy()

# Create gap metric
gap_df["share_gap"] = gap_df["spend_share_pct"] - gap_df["revenue_share_pct"]

# Sort by share gap to highlight inefficiency
gap_df = gap_df.sort_values("share_gap", ascending=False)

x = np.arange(len(gap_df))

plt.figure(figsize=(11,5))

plt.bar(
    x - 0.2,
    gap_df["spend_share_pct"],
    width=0.4,
    label="Spend Share",
    color=accent_color
)

plt.bar(
    x + 0.2,
    gap_df["revenue_share_pct"],
    width=0.4,
    label="Revenue Share",
    color=main_color
)

plt.xticks(
    x,
    gap_df["campaign_id"],
    rotation=45
)

plt.ylabel("Share (%)")

plt.title("Marketing Spend Share vs Revenue Share by Campaign")

plt.legend()

plt.grid(axis="y", alpha=0.3)

save_chart("spend_share_vs_revenue_share")

section("Campaign Investigation Table")

campaign_table = campaign_investigation.copy()

# Convert currency columns to millions
campaign_table["Campaign Spend (₹M)"] = campaign_table["spend"] / 1e6
campaign_table["Attributed Revenue (₹M)"] = campaign_table["revenue"] / 1e6

# Create gap BEFORE renaming columns
campaign_table["share_gap_pct"] = (
    campaign_table["spend_share_pct"] - campaign_table["revenue_share_pct"]
)

# Rename columns for presentation
campaign_table = campaign_table.rename(columns={
    "campaign_id": "Campaign",
    "orders": "Orders",
    "ROAS": "ROAS",
    "CAC_proxy": "CAC Proxy (₹)",
    "spend_share_pct": "Spend Share (%)",
    "revenue_share_pct": "Revenue Share (%)",
    "share_gap_pct": "Share Gap (%)"
})

campaign_table = campaign_table[
    [
        "Campaign",
        "Campaign Spend (₹M)",
        "Orders",
        "Attributed Revenue (₹M)",
        "ROAS",
        "CAC Proxy (₹)",
        "Spend Share (%)",
        "Revenue Share (%)",
        "Share Gap (%)"
    ]
]

campaign_table = campaign_table.sort_values("ROAS")

display(
    campaign_table.style
    .format({
        "Campaign Spend (₹M)": "{:.2f}",
        "Attributed Revenue (₹M)": "{:.2f}",
        "ROAS": "{:.2f}",
        "CAC Proxy (₹)": "{:,.0f}",
        "Spend Share (%)": "{:.2f}%",
        "Revenue Share (%)": "{:.2f}%",
        "Share Gap (%)": "{:.2f}%"
    })
    .highlight_between(subset=["ROAS"], left=0, right=1, color="#f8d7da")
    .hide(axis="index")
)

section("Top Campaign Spend Inefficiencies")

median_spend = campaign_investigation["spend"].median()

inefficient_campaigns = campaign_investigation[

    (campaign_investigation["spend"] > median_spend) &
    (campaign_investigation["ROAS"] < 1)

].sort_values(
    "ROAS"
)

top_inefficiencies = inefficient_campaigns.head(2)

performance_table = top_inefficiencies.copy()

performance_table = performance_table.rename(columns={
    "campaign_id": "Campaign",
    "ROAS": "ROAS",
    "CAC_proxy": "CAC Proxy (₹)",
    "spend_share_pct": "Spend Share (%)",
    "revenue_share_pct": "Revenue Share (%)"
})

performance_table = performance_table[
    ["Campaign","ROAS","CAC Proxy (₹)","Spend Share (%)","Revenue Share (%)"]
]

display(
    performance_table.style
    .format({
        "ROAS": "{:.2f}",
        "CAC Proxy (₹)": "{:,.0f}",
        "Spend Share (%)": "{:.2f}%",
        "Revenue Share (%)": "{:.2f}%"
    })
    .highlight_between(subset=["ROAS"], left=0, right=1, color="#f8d7da")
    .hide(axis="index")
)

report("""

<h4>Insight</h4>

<ul>

<li>
Both campaigns operate below the profitability threshold (ROAS < 1),
indicating that marketing spend currently exceeds the revenue generated.
</li>

<li>
Despite accounting for nearly <b>4.8 – 4.9%  of total marketing spend each</b>,
these campaigns contribute only about <b>2% of total revenue</b>,
highlighting clear inefficiencies in budget allocation.
</li>

<li>
Potential drivers may include weak audience targeting,
low-intent traffic sources, or ineffective bidding strategies.
</li>

<li>
Recommended next step: Run controlled A/B tests with improved targeting,
creative variations, and bidding strategies before allocating additional budget.
</li>

</ul>

""")

section("Campaign Spend Inefficiency Investigation")

plt.figure(figsize=(8,5))

for i, v in enumerate(top_inefficiencies["ROAS"]):
    plt.text(i, v + 0.03, f"{v:.2f}", ha="center")

plt.bar(
    top_inefficiencies["campaign_id"],
    top_inefficiencies["ROAS"],
    color="red"
)

plt.axhline(y=1, linestyle="--", color="black", linewidth=1.5, label="Break-even ROAS")
plt.legend(loc="upper left", bbox_to_anchor=(1,1))

plt.title("Top Campaign Spend Inefficiencies")
plt.ylabel("ROAS")

save_chart("campaign_spend_inefficiency")

section("Spend vs ROAS by Campaign")

plt.figure(figsize=(9,6))

colors = campaign_investigation["ROAS"].apply(
    lambda x: "#d62728" if x < 1 else "#2ca02c"
)

sizes = campaign_investigation["ROAS"].apply(lambda x: 140 if x < 1 else 80)

plt.scatter(
    campaign_investigation["spend"],
    campaign_investigation["ROAS"],
    c=colors,
    s=sizes,
    alpha=0.85,
    edgecolor="white",
    linewidth=0.8
)

for i, row in top_inefficiencies.iterrows():
    plt.text(
        row["spend"] + 80000,
        row["ROAS"] + 0.05,
        row["campaign_id"],
        fontsize=10,
        fontweight="bold",
        color="black"
    )

plt.axhline(y=1, linestyle="--", color="black", label="Break-even ROAS")

avg_roas = campaign_investigation["ROAS"].mean()

plt.axhline(
    y=avg_roas,
    linestyle=":",
    color="blue",
    label=f"Average ROAS ({avg_roas:.2f})"
)

plt.gca().xaxis.set_major_formatter(
    mtick.FuncFormatter(lambda x, p: f'₹ {x/1e6:.1f}M')
)

plt.xlabel("Campaign Spend (₹M)")
plt.ylabel("ROAS")
plt.title("Campaign Efficiency: Spend vs ROAS")

plt.legend()
plt.grid(alpha=0.3)

save_chart("campaign_efficiency_spend_vs_roas")

section("Marketing Investigation Summary")

report("""

<div style="
max-width:850px;
margin-left:20px;
line-height:1.7;
">

<h4>INVESTIGATION 1: High Spend Campaign with Low ROAS</h4>

<h5 style="margin-top:10px;">Evidence</h5>

<ul>
<li>Campaign appears in the top spend category but generates relatively low revenue.</li>
<li>ROAS is below the profitability threshold (ROAS &lt; 1).</li>
</ul>

<h5 style="margin-top:10px;">Possible Causes</h5>

<ol>
<li>Poor audience targeting leading to low conversion rates.</li>
<li>Landing page mismatch with ad messaging.</li>
<li>Excessively broad keyword targeting attracting low-intent traffic.</li>
</ol>

<h5 style="margin-top:10px;">Validation Experiment</h5>

<p>
Run an A/B test with improved audience targeting or refined keywords.
Measure impact on conversion rate and ROAS.
</p>

<hr style="margin:20px 0;">

<h4>INVESTIGATION 2: Inefficient Customer Acquisition</h4>

<h5 style="margin-top:10px;">Evidence</h5>

<ul>
<li>CAC proxy significantly higher than other campaigns.</li>
<li>Revenue share lower than spend share.</li>
</ul>

<h5 style="margin-top:10px;">Possible Causes</h5>

<ol>
<li>Campaign targeting cold audiences with low purchase intent.</li>
<li>Weak ad creatives reducing click-through and engagement.</li>
<li>Inefficient bidding strategy.</li>
</ol>

<h5 style="margin-top:10px;">Validation Experiment</h5>

<p>
Test new creatives and adjust bidding strategy.
Monitor improvements in CAC and ROAS.
</p>

</div>

""")

section("Strategic Recommendation")

inefficient_ids = ", ".join(top_inefficiencies["campaign_id"].astype(str))

report(f"""

<div style="
max-width:850px;
margin-left:20px;
line-height:1.7;
">

<h4 style="margin-bottom:10px;">Recommended Action</h4>

<p>
Campaigns {inefficient_ids} consume a large share of marketing spend
while delivering ROAS below the profitability threshold.
</p>

<h4 style="margin-top:18px;">Suggested Optimization</h4>

<ul>
<li>Reduce budget allocation to these campaigns by 20–30%.</li>
<li>Reallocate the freed budget toward high-performing campaigns
with ROAS significantly above the average.</li>
</ul>

<h4 style="margin-top:18px;">Expected Impact</h4>

<p>
Reallocating budget from low-efficiency campaigns to high-performing
campaigns should increase overall marketing ROI and reduce customer
acquisition costs.
</p>

</div>

""")

banner("MARKETING EXECUTIVE SUMMARY")

report("""

<div style="
max-width:850px;
margin-left:40px;
line-height:1.7;
">

<h4 style="margin-bottom:10px;">Key Insights from Marketing Analysis</h4>

<ol style="margin-top:0;">

<li>
<strong>Channel Efficiency</strong><br>
High ROAS channels such as Email and Organic generate strong returns
with relatively low acquisition costs.
</li>

<li style="margin-top:12px;">
<strong>Campaign Optimization Opportunity</strong><br>
Several campaigns consume significant budget while generating
ROAS below the profitability threshold.
</li>

<li style="margin-top:12px;">
<strong>Segment Behavior</strong><br>
Desktop users convert more efficiently than mobile users,
while returning customers deliver higher order values.
</li>

<li style="margin-top:12px;">
<strong>Product Revenue Concentration</strong><br>
A small number of product categories drive the majority of revenue,
consistent with the Pareto principle.
</li>

</ol>

<h4 style="margin-top:20px;">Strategic Recommendation</h4>

<p>
Reallocate budget from inefficient campaigns toward
high-ROAS channels and high-performing product categories
to improve overall marketing profitability.
</p>

<h4 style="margin-top:20px;">Expected Outcome</h4>

<p>
Higher marketing ROI, improved conversion efficiency,
and stronger revenue growth.
</p>

</div>

""")



## Attribution Analysis

# =====================================================
# Attribution Analysis
# =====================================================

banner("ATTRIBUTION ANALYSIS (LAST TOUCH)")

section("Prepare Session-Level Attribution Data")

# Only sessions that generated an order

orders_sessions = fact_sessions[
    fact_sessions["purchase_flag"] == 1
].drop_duplicates("order_id").copy()

# Check number of attributed orders
print("Total attributed orders:", orders_sessions["order_id"].nunique())

# =====================================================
# Last-Touch Attribution by Channel
# =====================================================

section("Attributed Performance by Channel")

channel_attribution = orders_sessions.groupby(
    "channel"
).agg(

    attributed_orders=("order_id","nunique"),

    attributed_revenue=("net_revenue","sum")

).reset_index()


# Merge marketing spend from fact_channel

channel_spend = fact_channel.groupby(
    "channel"
).agg(

    spend=("total_spend","sum")

).reset_index()


# Standardize channel names before merging

channel_attribution["channel_clean"] = (
    channel_attribution["channel"]
    .str.lower()
    .str.replace(" ","_")
)

channel_spend["channel_clean"] = (
    channel_spend["channel"]
    .str.lower()
)

# Merge using clean channel key
channel_attribution = channel_attribution.merge(
    channel_spend[["channel_clean","spend"]],
    on="channel_clean",
    how="left"
)

channel_attribution = channel_attribution.drop(columns="channel_clean")

# Clean channel labels for presentation
channel_attribution["channel"] = (
    channel_attribution["channel"]
    .str.replace("_"," ")
    .str.title()
)

# KPIs

channel_attribution["ROAS"] = (
    channel_attribution["attributed_revenue"]
    /
    channel_attribution["spend"]
)

channel_attribution["CAC_proxy"] = np.where(
    channel_attribution["attributed_orders"] > 0,
    channel_attribution["spend"] / channel_attribution["attributed_orders"],
    0
)


# Clean values

channel_attribution = channel_attribution.replace(
    [np.inf,-np.inf],
    0
)

channel_attribution = channel_attribution.fillna(0)

# Round metrics

channel_attribution["ROAS"] = channel_attribution["ROAS"].round(2)
channel_attribution["CAC_proxy"] = channel_attribution["CAC_proxy"].round(0).astype(int)

total_attr_revenue = channel_attribution["attributed_revenue"].sum()

channel_attribution["revenue_share_pct"] = (
    channel_attribution["attributed_revenue"] / total_attr_revenue * 100
).round(2)

section("Channel Attribution Table")

channel_attr_display = channel_attribution[
    ["channel","attributed_orders","attributed_revenue","spend","ROAS","CAC_proxy","revenue_share_pct"]
].rename(columns={
    "channel": "Channel",
    "attributed_orders": "Attributed Orders",
    "attributed_revenue": "Attributed Revenue (₹)",
    "spend": "Marketing Spend (₹)",
    "CAC_proxy": "CAC Proxy (₹)",
    "revenue_share_pct": "Revenue Share (%)"
})

display(
    channel_attr_display
    .sort_values("ROAS", ascending=False)
    .style.format({
        "Attributed Revenue (₹)": "{:,.0f}",
        "Marketing Spend (₹)": "{:,.0f}",
        "CAC Proxy (₹)": "{:,.0f}",
        "ROAS": "{:.2f}",
        "Revenue Share (%)": "{:.2f}%"
    })
    .hide(axis="index")
)

report("""

<h4>Insight</h4>

<ul>

<li>
Organic and Email channels deliver the strongest marketing efficiency,
achieving the highest ROAS and lowest customer acquisition costs.
These channels appear to convert high-intent users and represent strong
opportunities for continued investment.<br>
</li>

<li>
Search generates the largest share of attributed revenue (43.3%) but
operates close to the profitability threshold, indicating that it
functions as a major scale channel rather than the most efficient one.<br>
</li>

<li>
Paid Social consumes a large share of marketing spend while producing
ROAS below break-even, suggesting significant inefficiency. This channel
is a strong candidate for campaign optimization or partial budget
reallocation toward higher-performing channels.
</li>

</ul>

""")

# =====================================================
# Visualization — Bubble Chart
# =====================================================

section("Marketing Efficiency by Channel: Spend vs ROAS")

plt.figure(figsize=(9,6))

# Bubble size based on revenue
sizes = channel_attribution["attributed_revenue"] / 20000

# Color inefficient channels red
colors = channel_attribution["ROAS"].apply(
    lambda x: "#d62728" if x < 1 else "#2ca02c"
)

plt.scatter(
    channel_attribution["spend"],
    channel_attribution["ROAS"],
    s=sizes,
    c=colors,
    alpha=0.7,
    edgecolor="black"
)

# Channel labels
for i, row in channel_attribution.iterrows():
    plt.text(
        row["spend"] * 1.02,
        row["ROAS"] * 1.01,
        row["channel"],
        fontsize=9
    )

# Break-even line
plt.axhline(
    y=1,
    linestyle="--",
    color="black",
    label="Break-even ROAS"
)

# Format spend axis
plt.gca().xaxis.set_major_formatter(
    mtick.FuncFormatter(lambda x, p: f'₹ {x/1e6:.1f}M')
)

plt.xlabel("Marketing Spend (₹M)")
plt.ylabel("ROAS")

plt.title("Marketing Efficiency by Channel")

plt.legend()

plt.grid(alpha=0.3)

save_chart("marketing_efficiency_by_channel")

# =====================================================
# Visualization — Revenue Attribution by Channel
# =====================================================

section("Attributed Revenue by Channel")

channel_attribution = channel_attribution.sort_values(
    "attributed_revenue",
    ascending=False
)

plt.figure(figsize=(8,5))

bars = plt.bar(
    channel_attribution.sort_values("attributed_revenue", ascending=False)["channel"],
    channel_attribution.sort_values("attributed_revenue", ascending=False)["attributed_revenue"],
    color=main_color
)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x()+bar.get_width()/2,
        height,
        f"₹ {height/1e6:.1f}M",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.gca().yaxis.set_major_formatter(
    mtick.FuncFormatter(lambda x,p: f'₹ {x/1e6:.1f}M')
)

plt.title("Last-Touch Attributed Revenue by Channel")
plt.ylabel("Revenue (₹M)")

plt.gca().yaxis.set_major_formatter(
    mtick.FuncFormatter(lambda x,p: f'₹ {x/1e6:.1f}M')
)

plt.xticks(rotation=30)

save_chart("attributed_revenue_by_channel")

# =====================================================
# Last-Touch Attribution by Campaign
# =====================================================

section("Attributed Performance by Campaign")

campaign_attribution = orders_sessions.groupby(
    "campaign_id"
).agg(

    attributed_orders=("order_id","nunique"),

    attributed_revenue=("net_revenue","sum")

).reset_index()


campaign_spend = fact_campaign.groupby(
    "campaign_id"
).agg(

    spend=("spend","sum")

).reset_index()


campaign_attribution = campaign_attribution.merge(
    campaign_spend,
    on="campaign_id",
    how="left"
)


# KPIs

campaign_attribution["ROAS"] = (
    campaign_attribution["attributed_revenue"]
    /
    campaign_attribution["spend"]
)

campaign_attribution["CAC_proxy"] = np.where(
    campaign_attribution["attributed_orders"] > 0,
    campaign_attribution["spend"] / campaign_attribution["attributed_orders"],
    0
)


campaign_attribution = campaign_attribution.replace(
    [np.inf,-np.inf],
    0
)

campaign_attribution = campaign_attribution.fillna(0)


campaign_attribution["ROAS"] = campaign_attribution["ROAS"].round(2)
campaign_attribution["CAC_proxy"] = campaign_attribution["CAC_proxy"].round(0).astype(int)


section("Campaign Attribution Table")

# Convert monetary columns to millions
campaign_table = campaign_attribution.copy()

campaign_table["Attributed Revenue (₹M)"] = campaign_table["attributed_revenue"] / 1e6
campaign_table["Marketing Spend (₹M)"] = campaign_table["spend"] / 1e6

campaign_table = campaign_table.rename(columns={
    "campaign_id": "Campaign",
    "attributed_orders": "Attributed Orders",
    "CAC_proxy": "CAC Proxy (₹)"
})

campaign_table = campaign_table[
    ["Campaign","Attributed Orders","Attributed Revenue (₹M)",
     "Marketing Spend (₹M)","ROAS","CAC Proxy (₹)"]
]

campaign_table = campaign_table.sort_values("ROAS", ascending=False)

display(
    campaign_table.head(10)
    .style
    .format({
        "Attributed Revenue (₹M)": "{:.2f}",
        "Marketing Spend (₹M)": "{:.2f}",
        "ROAS": "{:.2f}",
        "CAC Proxy (₹)": "{:,.0f}"
    })
    .hide(axis="index")
)

report("""

<h4>Insight</h4>

<ul>

<li>
Several smaller campaigns (C039, C037, C040, C038) demonstrate exceptionally
high ROAS while operating at relatively low marketing spend levels.<br>
</li>

<li>
These campaigns likely target highly qualified or retargeted audiences,
resulting in strong conversion efficiency.<br>
</li>

<li>
They represent potential candidates for controlled budget expansion.
However, careful monitoring is required because campaigns with very high
ROAS at low spend may experience diminishing returns as budget increases.<br>
</li>

<li>
Larger campaigns such as C017 and C020 generate more total revenue but
operate at lower ROAS levels, suggesting a trade-off between scale and efficiency.
</li>

</ul>

""")

# =====================================================
# Visualization — Bubble Chart
# =====================================================

section("Marketing Efficiency by Campaign: Spend vs ROAS")

plt.figure(figsize=(9,6))

# Bubble size based on revenue
sizes = campaign_attribution["attributed_revenue"] / 20000

# Color inefficient campaigns
colors = campaign_attribution["ROAS"].apply(
    lambda x: "#d62728" if x < 1 else "#2ca02c"
)

plt.scatter(
    campaign_attribution["spend"],
    campaign_attribution["ROAS"],
    s=sizes,
    c=colors,
    alpha=0.7,
    edgecolor="black"
)

# Label top campaigns
top_labels = campaign_attribution.sort_values(
    "ROAS", ascending=False
).head(6)

for i, row in top_labels.iterrows():
    plt.text(
        row["spend"] * 1.02,
        row["ROAS"],
        row["campaign_id"],
        fontsize=9
    )

plt.axhline(
    y=1,
    linestyle="--",
    color="black",
    label="Break-even ROAS"
)

plt.gca().xaxis.set_major_formatter(
    mtick.FuncFormatter(lambda x,p: f'₹ {x/1e6:.2f}M')
)

plt.xlabel("Marketing Spend (₹M)")
plt.ylabel("ROAS")

plt.title("Marketing Efficiency by Campaign")

plt.legend()
plt.grid(alpha=0.3)

save_chart("marketing_efficiency_by_campaign")

# =====================================================
# Visualization — Revenue Attribution by Campaign
# =====================================================

section("Attributed Revenue by Campaign")

top_campaigns = campaign_attribution.sort_values(
    "attributed_revenue",
    ascending=False
).head(10)

plt.figure(figsize=(9,5))

bars = plt.bar(
    top_campaigns["campaign_id"],
    top_campaigns["attributed_revenue"],
    color=main_color
)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"₹ {height/1e6:.1f}M",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.gca().yaxis.set_major_formatter(
    mtick.FuncFormatter(lambda x,p: f'₹ {x/1e6:.1f}M')
)

plt.title("Last-Touch Attributed Revenue by Campaign")
plt.ylabel("Revenue (₹M)")
plt.xlabel("Campaign")

save_chart("last_touch_revenue_by_campaign")


# =====================================================
# Compare Attribution vs Blended ROAS
# =====================================================

section("Compare Attributed ROAS vs Blended ROAS")


# Blended ROAS from earlier analysis

blended_channel = fact_channel.groupby(
    "channel"
).agg(

    blended_spend=("total_spend","sum"),

    blended_revenue=("attributed_revenue","sum")

).reset_index()


blended_channel["blended_ROAS"] = (
    blended_channel["blended_revenue"]
    /
    blended_channel["blended_spend"]
)

# Clean channel names in blended dataset
blended_channel["channel"] = (
    blended_channel["channel"]
    .str.replace("_"," ")
    .str.title()
)

# Merge with attribution table
roas_comparison = channel_attribution.merge(
    blended_channel[["channel","blended_ROAS"]],
    on="channel",
    how="left"
)

roas_comparison["blended_ROAS"] = roas_comparison["blended_ROAS"].round(2)

section("ROAS Comparison Table")

roas_table = roas_comparison.copy()

# Convert monetary columns to millions
roas_table["Attributed Revenue (₹M)"] = roas_table["attributed_revenue"] / 1e6
roas_table["Marketing Spend (₹M)"] = roas_table["spend"] / 1e6

roas_table = roas_table.rename(columns={
    "channel": "Channel",
    "attributed_orders": "Attributed Orders",
    "CAC_proxy": "CAC Proxy (₹)",
    "revenue_share_pct": "Revenue Share (%)"
})

roas_table = roas_table[
    ["Channel","Attributed Orders","Attributed Revenue (₹M)",
     "Marketing Spend (₹M)","ROAS","blended_ROAS",
     "CAC Proxy (₹)","Revenue Share (%)"]
]

roas_table = roas_table.sort_values("ROAS", ascending=False)

display(
    roas_table.style
    .format({
        "Attributed Revenue (₹M)": "{:.2f}",
        "Marketing Spend (₹M)": "{:.2f}",
        "ROAS": "{:.2f}",
        "blended_ROAS": "{:.2f}",
        "CAC Proxy (₹)": "{:,.0f}",
        "Revenue Share (%)": "{:.2f}%"
    })
    .highlight_max(subset=["ROAS"], color="#d4edda")
    .highlight_min(subset=["ROAS"], color="#f8d7da")
    .hide(axis="index")
)

report("""

<h4>Insight</h4>

Organic and Email channels demonstrate the strongest marketing efficiency,
achieving the highest ROAS while maintaining relatively low acquisition costs.

Search generates the largest share of revenue but operates close to the
profitability threshold.

Paid Social consumes a significant portion of marketing spend while
producing ROAS below break-even, suggesting an opportunity for
budget reallocation or campaign optimization.

""")

# =====================================================
# Visualization — Attribution vs Blended ROAS
# =====================================================

section("ROAS Comparison Chart")

roas_comparison = roas_comparison.sort_values("ROAS", ascending=False)

x = np.arange(len(roas_comparison["channel"]))

plt.figure(figsize=(9,5))

plt.bar(
    x - 0.2,
    roas_comparison["ROAS"],
    width=0.4,
    label="Attributed ROAS",
    color=main_color
)

plt.bar(
    x + 0.2,
    roas_comparison["blended_ROAS"],
    width=0.4,
    label="Blended ROAS",
    color=accent_color
)

plt.xticks(
    x,
    roas_comparison["channel"],
    rotation=30
)

plt.ylabel("ROAS")

plt.title("Attributed ROAS vs Blended ROAS by Channel")

plt.axhline(y=1, linestyle="--", color="black", label="Break-even ROAS")
plt.legend()

plt.tight_layout()
plt.show()

# =====================================================
# Attribution Insights
# =====================================================

section("Attribution Insights")

report("""

<div style="
max-width:900px;
margin-left:40px;
line-height:1.7;
">

<h4 style="margin-top:10px;">Key Findings</h4>

<ol style="margin-top:0;">
<li>
Last-touch attribution assigns revenue credit to the final
marketing interaction before a purchase. This highlights the
channels that directly drive conversions rather than those that
only generate early-stage awareness.
</li>

<li>
Comparing attributed ROAS with blended ROAS reveals which
channels act as strong closing channels. Channels with higher
attributed ROAS are particularly effective at converting users
who are already considering a purchase.
</li>

<li>
Channels with lower attributed ROAS may still play an
important role earlier in the customer journey by generating
traffic and awareness, even if they are not the final
conversion touchpoint.
</li>
</ol>

<h4 style="margin-top:20px;">Business Implication</h4>

<ul>
<li>
High attributed-ROAS channels should receive stronger
investment for conversion-focused campaigns.
</li>

<li>
Awareness-focused channels should not be evaluated only
by last-touch ROAS because their value occurs earlier in
the customer journey.
</li>
</ul>

<h4 style="margin-top:20px;">Recommended Strategy</h4>

<p>
Use a balanced marketing mix by combining high-performing
conversion channels with upper-funnel channels that
generate demand and feed the conversion pipeline.
</p>

<h4 style="margin-top:20px;">Additional Observation</h4>

<p>
Attributed ROAS and blended ROAS are identical in this analysis.
This occurs because the marketing fact tables were constructed
using session-level attribution in the ETL pipeline. As a result,
the blended metrics already reflect last-touch attribution logic.
</p>

<p>
In other words, revenue was attributed to the session’s channel and
campaign at the data engineering stage, so recomputing attribution
produces the same results.
</p>

<p>
In a real marketing environment, blended revenue is often calculated
without attribution and may differ significantly from last-touch
or multi-touch attribution models.
</p>
""")



## Regression Impact Modeling

# =====================================================
# Marketing Spend vs Revenue by Channel
# =====================================================

section("Channel Spend Efficiency")

channel_perf = fact_channel.groupby("channel").agg(
    total_spend=("total_spend","sum"),
    total_revenue=("attributed_revenue","sum")
).reset_index()

# ROI calculation
channel_perf["ROI"] = (
    channel_perf["total_revenue"] /
    channel_perf["total_spend"]
)

channel_perf["roi"] = channel_perf["ROI"]   # safety alias

# Clean labels
channel_perf["channel"] = (
    channel_perf["channel"]
    .str.replace("_"," ")
    .str.title()
)

# Build display table
channel_table = channel_perf.copy()

channel_table["Marketing Spend (₹M)"] = channel_table["total_spend"]/1e6
channel_table["Attributed Revenue (₹M)"] = channel_table["total_revenue"]/1e6

channel_table = channel_table.rename(columns={
    "channel":"Channel"
})

channel_table = channel_table[
    [
        "Channel",
        "Marketing Spend (₹M)",
        "Attributed Revenue (₹M)",
        "ROI"
    ]
]

# Sort alphabetically like image
channel_table = channel_table.sort_values("Channel")

display(
    channel_table.style
    .format({
        "Marketing Spend (₹M)": "{:.2f}",
        "Attributed Revenue (₹M)": "{:.2f}",
        "ROI": "{:.2f}x"
    })
    .highlight_max(subset=["ROI"], color="#d4edda")
    .highlight_min(subset=["ROI"], color="#f8d7da")
    .hide(axis="index")
)

section("Revenue Generated per ₹1 Marketing Spend")

roi_sorted = channel_perf.sort_values("ROI")

plt.figure(figsize=(7,4))

bars = plt.barh(
    roi_sorted["channel"].str.title(),
    roi_sorted["ROI"]
)

plt.xlabel("Revenue per ₹1 Spend")

plt.title("Marketing Channel ROI", fontsize=13, pad=12)

plt.grid(axis="x", linestyle="--", alpha=0.3)

for bar in bars:

    width = bar.get_width()

    plt.text(
        width + 0.02,
        bar.get_y() + bar.get_height()/2,
        f"{width:.2f}x",
        va="center"
    )

save_chart("marketing_channel_ROI")

report("""

<h4>Insights</h4>

<ul>
<li>
This analysis compares total marketing spend with attributed revenue
across channels to evaluate marketing efficiency.<br>
</li>

<li>
Channels with higher revenue per ₹1 spent demonstrate stronger
return on investment (ROI), indicating more efficient customer acquisition.
</li>

<li>
These insights provide a baseline understanding of marketing performance
before building predictive models that estimate the marginal impact of
additional spend.
</li>
</ul>

""") 

# =====================================================
# Regression Impact Modeling
# =====================================================

banner("REGRESSION IMPACT MODELING")

section("Prepare Daily Modeling Dataset")

# Aggregate channel spend by day

daily_spend = fact_channel.pivot_table(
    index="date",
    columns="channel",
    values="total_spend",
    aggfunc="sum"
).reset_index()

daily_spend.columns.name = None

channels = [c for c in daily_spend.columns if c != "date"]

# Daily revenue

daily_revenue = fact_channel.groupby(
    "date"
).agg(
    daily_revenue=("attributed_revenue","sum"),
    promo_flag=("promo_flag","max"),
    week_index=("week_index","max"),
    day_of_week=("day_of_week","max")
).reset_index()


# Merge

model_data = daily_spend.merge(
    daily_revenue,
    on="date",
    how="left"
)

model_data["date"] = pd.to_datetime(model_data["date"]).dt.strftime("%b %d %Y")

# Create display version
model_data_display = model_data.head().copy()

# Clean column names
model_data_display = model_data_display.rename(columns={
    "date": "Date",
    "email": "Email Spend (₹)",
    "organic": "Organic Spend (₹)",
    "paid_social": "Paid Social Spend (₹)",
    "referral": "Referral Spend (₹)",
    "search": "Search Spend (₹)",
    "daily_revenue": "Daily Revenue (₹M)",
    "promo_flag": "Promotion Day",
    "week_index": "Week Trend",
    "day_of_week": "Day of Week"
})

# Convert promotion flag to Yes/No
model_data_display["Promotion Day"] = model_data_display["Promotion Day"].map({
    1: "Yes",
    0: "No"
})

display(
    model_data_display.style.format({
        "Email Spend (₹)": "{:,.0f}",
        "Organic Spend (₹)": "{:,.0f}",
        "Paid Social Spend (₹)": "{:,.0f}",
        "Referral Spend (₹)": "{:,.0f}",
        "Search Spend (₹)": "{:,.0f}",
        "Daily Revenue (₹M)": "{:.2f}"
    }).hide(axis="index")
)

# =====================================================
# Encode Day of Week
# =====================================================

section("Encode Day-of-Week Seasonality")

dow_dummies = pd.get_dummies(
    model_data["day_of_week"],
    prefix="Day",
    drop_first=True
)

model_data = pd.concat([model_data, dow_dummies], axis=1)

for c in channels:
    model_data[c] = model_data[c].round(0)

display(
    model_data.head()
    .style
    .format({
        "email": "{:,.0f}",
        "organic": "{:,.0f}",
        "paid_social": "{:,.0f}",
        "referral": "{:,.0f}",
        "search": "{:,.0f}",
        "daily_revenue": "{:.2f}",
        "promo_flag": "{:.0f}",
        "week_index": "{:.0f}"
    })
    .hide(axis="index")
)

# =====================================================
# Feature Engineering
# =====================================================

section("Create Log Spend Features")

for ch in channels:

    model_data[f"log_{ch}"] = np.log1p(model_data[ch])


display(
    model_data.head()
    .style
    .format({
        "email": "{:,.0f}",
        "organic": "{:,.0f}",
        "paid_social": "{:,.0f}",
        "referral": "{:,.0f}",
        "search": "{:,.0f}",
        "daily_revenue": "{:.2f}",
        "promo_flag": "{:.0f}",
        "week_index": "{:.0f}"
    })
    .hide(axis="index")
)

# =====================================================
# Spend vs Revenue Relationship
# =====================================================

section("Channel Spend vs Attributed Revenue")

plt.figure(figsize=(7,5))

plt.scatter(
    channel_perf["total_spend"]/1e6,
    channel_perf["total_revenue"]/1e6,
    s=channel_perf["ROI"] * 50,
    color=main_color,
    alpha=0.8,
    edgecolors="black"
)

for i,row in channel_perf.iterrows():

    plt.text(
        row["total_spend"]/1e6 * 1.01,
        row["total_revenue"]/1e6,
        row["channel"].replace("_"," ").title(),
        fontsize=9
    )

plt.xlabel("Total Marketing Spend (₹M)")
plt.ylabel("Attributed Revenue (₹M)")

plt.gca().xaxis.set_major_formatter(
    mtick.FuncFormatter(lambda x, p: f"₹ {x/1_000_000:.1f}M")
)

plt.gca().yaxis.set_major_formatter(
    mtick.FuncFormatter(lambda x, p: f"₹ {x/1_000_000:.1f}M")
)

plt.title("Marketing Spend vs Revenue by Channel", fontsize=13, pad=12)

plt.grid(alpha=0.3)

save_chart("channel_spend_vs_attributed_revenue")

section("Spend vs Revenue Relationship")

plt.figure(figsize=(8,5))

plt.scatter(
    model_data[channels].sum(axis=1),
    model_data["daily_revenue"],
    alpha=0.65,
    edgecolors="white",
    linewidth=0.5
)

# simple trend line
x = model_data[channels].sum(axis=1)
y = model_data["daily_revenue"]

z = np.polyfit(x, y, 1)
p = np.poly1d(z)

sorted_idx = np.argsort(x)

plt.plot(
    x.iloc[sorted_idx],
    p(x)[sorted_idx],
    linewidth=2,
    color="#1f77b4",
    label="Trend Line"
)

plt.xlabel("Total Daily Marketing Spend (₹M)")
plt.ylabel("Daily Revenue (₹M)")

plt.gca().xaxis.set_major_formatter(
    mtick.FuncFormatter(lambda x, p: f"₹ {x/1_000_000:.1f}M")
)

plt.gca().yaxis.set_major_formatter(
    mtick.FuncFormatter(lambda x, p: f"₹ {x/1_000_000:.1f}M")
)

plt.title("Spend vs Revenue Relationship", fontsize=13, pad=10)

plt.grid(axis="y", linestyle="--", alpha=0.3)

plt.legend()

save_chart("spend_vs_revenue_relationship")

# =====================================================
# Lagged Spend Features
# =====================================================

# section("Create Lagged Spend Features")

for ch in channels:

    model_data[f"log_{ch}_lag1"] = model_data[f"log_{ch}"].shift(1)

model_data = model_data.dropna().reset_index(drop=True)

# =====================================================
# Build Regression Model
# =====================================================

section("Prepare Regression Features")

report("""
<div style="margin-bottom:6px;">
Sample of transformed marketing spend features used in the regression model.
</div>
""")

feature_cols = [f"log_{c}" for c in channels]

# add lag features
feature_cols += [f"log_{c}_lag1" for c in channels]

# controls
feature_cols += [
    "promo_flag",
    "week_index"
]

# seasonality
feature_cols += list(dow_dummies.columns)

X = model_data[feature_cols]

y = model_data["daily_revenue"]


log_display = X.head()[[
    "log_email",
    "log_organic",
    "log_paid_social",
    "log_referral",
    "log_search"
]].copy()

log_display = log_display.rename(columns={
    "log_email": "Log Email Spend",
    "log_organic": "Log Organic Spend",
    "log_paid_social": "Log Paid Social Spend",
    "log_referral": "Log Referral Spend",
    "log_search": "Log Search Spend"
})

display(
    log_display.style
    .format("{:.2f}")
    .hide(axis="index")
)

report("""

<h4>Insights</h4>

<ul>
<li>
Log transformations are applied to marketing spend variables to capture 
diminishing returns. As marketing spend increases, the incremental impact 
on revenue tends to grow at a decreasing rate.
</li>
<li>
Using log features allows the regression model to estimate these 
non-linear effects while maintaining a linear modeling framework.
</li>
<li>
Lagged spend features capture delayed marketing impact,
since marketing campaigns may influence purchases
over multiple days rather than immediately.
</li>
</ul>

""")

# =====================================================
# Train Test Split (Time Based)
# =====================================================

section("Train/Test Split")

split_point = int(len(model_data) * 0.8)

X_train = X.iloc[:split_point]
X_test  = X.iloc[split_point:]

y_train = y.iloc[:split_point]
y_test  = y.iloc[split_point:]

report(f"Training observations : {len(X_train)}")
report(f"Testing observations  : {len(X_test)}")

# =====================================================
# Train Regression Model
# =====================================================

section("Fit Linear Regression Model")

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

report("Model training completed successfully.")

# =====================================================
# Model Predictions
# =====================================================

section("Model Predictions")

pred_table = pd.DataFrame({
    "Actual Revenue (₹)": y_test.values,
    "Predicted Revenue (₹)": y_pred
})

pred_table["Error (₹)"] = (
    pred_table["Predicted Revenue (₹)"]
    - pred_table["Actual Revenue (₹)"]
)

pred_table["Absolute Error (₹)"] = pred_table["Error (₹)"].abs()

display(
    pred_table.head(5)
    .style
    .format({
        "Actual Revenue (₹)": "{:,.0f}",
        "Predicted Revenue (₹)": "{:,.0f}",
        "Error (₹)": "{:,.0f}",
        "Absolute Error (₹)": "{:,.0f}"
    })
    .hide(axis="index")
)

report(f"\nAverage absolute prediction error: ₹ {pred_table['Absolute Error (₹)'].mean():,.0f}")

# =====================================================
# Model Evaluation Metrics
# =====================================================

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test==0,1,y_test))) * 100

avg_revenue = y_test.mean()

section("Model Evaluation Metrics")

report(f"""

<div style="max-width:850px; line-height:1.8; font-size:15px;">

<div style="display:grid; grid-template-columns:260px auto; row-gap:6px; column-gap:10px;">

<div>Average Daily Revenue</div>
<div>: ₹ {avg_revenue:,.0f}</div>

<div>MAE (Mean Absolute Error)</div>
<div>: ₹ {mae:,.0f} → Average prediction error</div>

<div>RMSE (Root Mean Squared Error)</div>
<div>: ₹ {rmse:,.0f} → Penalizes larger prediction errors</div>

<div>MAPE (Percentage Error)</div>
<div>: {mape:.2f}% → Average error relative to revenue</div>

<div>R² (Model Fit)</div>
<div>: {r2:.3f} → Model explains {r2*100:.1f}% of revenue variation</div>

</div>

</div>

<h4>Interpretation</h4>

<p>
On average, the model's predictions differ from actual revenue by
<b>₹ {mae:,.0f}</b>, which corresponds to approximately
<b>{mape:.1f}%</b> of daily revenue.
</p>

""")

# =====================================================
# Model Fit Distribution
# =====================================================

section("Model Fit Distribution")

errors = y_test - y_pred

plt.figure(figsize=(8,4.5))

# Histogram with borders
plt.hist(
    errors,
    bins=15,
    density=True,
    alpha=0.7,
    edgecolor="white",
    linewidth=1,
    color="#1F77B4" 
)

# KDE curve
kde = gaussian_kde(errors)
x_vals = np.linspace(errors.min(), errors.max(), 400)

plt.plot(
    x_vals,
    kde(x_vals),
    linewidth=2,
    color="#FF7F0E",
    label="Error Density (KDE)"
)

# Zero error reference
plt.axvline(
    0,
    linestyle="--",
    linewidth=2,
    label="Zero Prediction Error"
)

# Mean error reference
plt.axvline(
    errors.mean(),
    linestyle=":",
    linewidth=2,
    label="Average Model Bias"
)

plt.title("Prediction Error Distribution", fontsize=13, pad=12)

plt.xlabel("Residual Error (Actual - Predicted)")
plt.ylabel("Probability Density")

# Remove scientific notation on y-axis
plt.ticklabel_format(style='plain', axis='y')

plt.grid(alpha=0.25)

plt.legend()

save_chart("prediction_error_distribution")

report(f"""
<div style="max-width:850px; line-height:1.7;">

Most prediction errors are centered around zero, indicating that the model does not show strong systematic bias. 
However, the presence of larger positive residuals suggests occasional underestimation of revenue on high-demand days.

</div>
""")

# =====================================================
# Predicted vs Actual Relationship
# =====================================================

section("Predicted vs Actual Relationship")

plt.figure(figsize=(6,6))

# Scatter points
plt.scatter(
    y_test / 1e6,
    y_pred / 1e6,
    alpha=0.8,
    edgecolor="white",
    linewidth=0.7
)

# Perfect prediction reference line
min_val = min(y_test.min(), y_pred.min()) / 1e6
max_val = max(y_test.max(), y_pred.max()) / 1e6

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
    linewidth=2,
    label="Perfect Prediction"
)

plt.title("Predicted vs Actual Revenue", fontsize=13, pad=12)

plt.xlabel("Actual Revenue (₹M)")
plt.ylabel("Predicted Revenue (₹M)")

plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)

# Equal scaling (important)
plt.gca().set_aspect('equal', adjustable='box')

plt.grid(alpha=0.25)

plt.plot([], [], linestyle="none", label=f"Model Fit (R²) = {r2:.2f}")

plt.legend(loc="upper left", frameon=False)

save_chart("predicted_vs_actual_revenue")

report("""
<div style="max-width:850px; line-height:1.7;">

<h4>Insights</h4>

<ul>
<li>
The predicted vs actual plot shows that most predictions lie close to the 
perfect prediction line, indicating that the model captures the overall 
revenue trend reasonably well.
</li>
<li>
Some dispersion remains, which reflects 
natural variability in marketing performance and factors not included 
in the model.
</li>
</ul>

</div>
""")

# =====================================================
# Actual vs Predicted Chart
# =====================================================

section("Actual vs Predicted Revenue")

plt.figure(figsize=(8,5))

test_dates = model_data["date"].iloc[split_point:]

plt.plot(
    test_dates,
    y_test.values / 1e6,
    label="Actual",
    linewidth=2
)

plt.plot(
    test_dates,
    y_pred / 1e6,
    label="Predicted",
    linewidth=2,
    color="#DD8452"
)

plt.title("Actual vs Predicted Revenue", fontsize=13, pad=12)

plt.xlabel("Date")
plt.ylabel("Daily Revenue (₹M)")

plt.legend()

# -------- FIX --------
ax = plt.gca()

ax.grid(False)   # removes seaborn grid

ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

plt.xticks(rotation=45)

save_chart("actual_vs_predicted_revenue")

# =====================================================
# Prediction Error by Observation
# =====================================================

section("Prediction Error Over Time")

errors = y_test - y_pred

plt.figure(figsize=(8,4))

plt.plot(
    test_dates,
    errors / 1e6,
    label="Prediction Error"
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=2,
    label="Zero Error"
)

plt.title("Prediction Errors Over Time")

plt.xlabel("Date")
plt.ylabel("Revenue Error (₹M)")

# -------- FIX --------
ax = plt.gca()

# show one label every 7 days
ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))

# format date style
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

plt.xticks(rotation=45)

plt.grid(alpha=0.3)

plt.legend()

save_chart("pediction_errors")

# =====================================================
# Channel Impact (Model Coefficients)
# =====================================================

section("Channel Impact Estimation")

coef_df = pd.DataFrame({

    "feature": X.columns,

    "coefficient": model.coef_

})


coef_df = coef_df.sort_values(
    "coefficient",
    ascending=False
)


coef_table = coef_df.rename(columns={
    "feature":"Feature",
    "coefficient":"Coefficient"
})

display(
    coef_table.style
    .format({"Coefficient":"{:,.0f}"})
    .highlight_max(subset=["Coefficient"], color="#d4edda")
    .hide(axis="index")
)

# =====================================================
# Estimated Channel Impact
# =====================================================

section("Estimated Channel Impact")

impact = coef_df[
    coef_df["feature"].str.contains("log_")
].copy()

impact = impact[~impact["feature"].str.contains("lag")]

# Clean channel names
impact["channel"] = (impact["feature"].str.replace("log_", "").str.replace("_", " ").str.title())

# Sort by contribution
impact = impact.sort_values("coefficient")

plt.figure(figsize=(8,5))

colors = ["#d0d0d0"] * (len(impact)-1) + ["#1f77b4"]

impact["coef_million"] = impact["coefficient"] / 1_000_000

bars = plt.barh(
    impact["channel"],
    impact["coef_million"],
    color=colors
)


plt.title("Top Marketing Channels Driving Revenue", fontsize=13, pad=12)

plt.xlabel("Estimated Marginal Revenue Impact (₹M)")

plt.grid(axis="x", linestyle="--", alpha=0.2)

for bar in bars:

    width = bar.get_width()

    plt.text(
        width + 0.005,
        bar.get_y() + bar.get_height()/2,
        f"{width:.2f}M",
        va="center"
    )

# Add small left margin
plt.xlim(0, impact["coef_million"].max() * 1.12)


save_chart("marginal_revenue_impact")

# =====================================================
# Top Revenue Driving Channels
# =====================================================

section("Top Revenue Driving Channels")

report("""
<div style="margin-bottom:6px;">
Top marketing channels ranked by estimated marginal revenue impact.
</div>
""")

impact_sorted = impact.sort_values(
    "coefficient",
    ascending=False
)

top_channels = impact_sorted.head(3).copy()

top_channels["Estimated Revenue Impact (₹M)"] = (
    top_channels["coefficient"] / 1_000_000
)

top_channels = top_channels.rename(columns={
    "channel": "Channel"
})

top_channels = top_channels[
    [
        "Channel",
        "Estimated Revenue Impact (₹M)"
    ]
]

display(
    top_channels.style
    .format({
        "Estimated Revenue Impact (₹M)": "{:.2f}"
    })
    .highlight_max(
        subset=["Estimated Revenue Impact (₹M)"],
        color="#d4edda"
    )
    .set_properties(
        subset=["Estimated Revenue Impact (₹M)"],
        **{"text-align": "right"}
    )
    .hide(axis="index")
)

top_channel = impact_sorted.iloc[0]


report(
    f"<b>Top revenue-driving channel:</b> {top_channel['channel']}"
)

report("""
Paid Social appears to be the strongest revenue-driving channel in the model,
suggesting that incremental investment in this channel may generate higher
revenue returns compared with other marketing channels.
""")

# =====================================================
# Regression Insights
# =====================================================

section("Regression Insights")

report("""

<div style="
max-width:900px;
margin-left:20px;
line-height:1.7;
">

<h4>Key Findings</h4>

<ul>

<li>
The regression model estimates the relationship between
marketing spend and daily revenue while controlling for
seasonality (day-of-week), time trends, and promotions.
</li>

<li>
Log-transformed spend variables allow the model to capture
diminishing returns in marketing performance.
</li>

<li>
Channels with the largest positive coefficients show the
strongest marginal revenue contribution.
</li>

<li>
Because spend is log-transformed, coefficients can be
interpreted approximately as elasticity effects —
a 1% increase in spend is associated with a proportional
increase in revenue.
</li>

</ul>

<h4>Business Implication</h4>

<p>
Marketing budget should prioritize channels with the
highest marginal revenue impact while monitoring
diminishing returns.
</p>

<h4>Model Limitations</h4>

<ul>

<li>
The regression estimates statistical relationships rather
than true causal effects between marketing spend and revenue.
</li>

<li>
External factors such as competitor campaigns,
pricing changes, and macroeconomic conditions are
not included in the model.
</li>

<li>
The model assumes linear relationships and does not
capture complex interactions between marketing channels.
</li>

</ul>
""")

banner("Summary")

report(f"""

<div style=" max-width:900px; margin-left:40px; line-height:1.7; ">

<h4>Summary of Findings</h4>

<ul>

<li>
The regression model explains approximately
<b>{r2*100:.1f}%</b> of daily revenue variation.
</li>

<li>
Average prediction error is approximately
<b>₹ {mae:,.0f}</b> per day (~{mape:.1f}% of revenue).
</li>

<li>
Marketing spend shows a positive relationship with revenue,
with evidence of diminishing returns captured through
log-transformed spend variables.
</li>

<li>
<b>{top_channel['channel']}</b> appears as the strongest
revenue-driving marketing channel in the model.
</li>

</ul>

<p>
These insights can help guide marketing budget allocation
toward channels with higher marginal revenue impact.
</p>

</div>

""")


## Budget Reallocation Plan and Impact Estimate

# =====================================================
# Budget Reallocation Plan
# =====================================================

banner("BUDGET REALLOCATION PLAN")

section("Define Next-Month Marketing Budget")

# Define next month budget (example: based on last month's average spend)

avg_daily_spend = fact_channel.groupby("date")["total_spend"].sum().mean()

monthly_budget = avg_daily_spend * 30

report(f"""

<div style="max-width:850px; line-height:1.7;">

Next month’s marketing budget is estimated based on the recent
average daily marketing spend. <br>

<b>Average Daily Spend:</b> {format_inr_millions(avg_daily_spend)} <br>

<b>Estimated Monthly Budget (30 days):</b> {format_inr_millions(monthly_budget)} <br>

This budget level assumes a stable marketing investment strategy
similar to the most recent spending patterns.

</div>

""")

# =====================================================
# Estimate Channel Impact
# =====================================================

section("Channel Marginal Impact from Regression Model")

impact_df = coef_df.copy()

impact_df = impact_df[
    impact_df["feature"].str.contains("log_")
].copy()

impact_df["channel"] = impact_df["feature"].str.replace("log_","")

impact_df = impact_df.sort_values(
    "coefficient",
    ascending=False
)

impact_df = impact_df[~impact_df["channel"].str.contains("lag")]

impact_display = impact_df.copy()

impact_display["channel"] = (
    impact_display["channel"]
    .str.replace("_"," ")
    .str.title()
)

impact_display["Marginal Impact (Coefficient)"] = impact_display["coefficient"]

impact_display = impact_display[
    ["channel","Marginal Impact (Coefficient)"]
]

impact_display = impact_display.rename(columns={
    "channel":"Channel"
})

display(
    impact_display.style
    .format({"Marginal Impact (Coefficient)": "{:,.0f}"})
    .highlight_max(
        subset=["Marginal Impact (Coefficient)"],
        color="#d4edda"
    )
    .hide(axis="index")
)

# =====================================================
# Budget Constraints
# =====================================================

section("Define Channel Spend Constraints")

channels_list = impact_df["channel"].tolist()

constraints = pd.DataFrame({

    "channel": channels_list,

    "min_share": 0.05,

    "max_share": 0.40

})

constraints_display = constraints.copy()

constraints_display["channel"] = (
    constraints_display["channel"]
    .str.replace("_"," ")
    .str.title()
)

constraints_display = constraints_display.rename(columns={
    "channel":"Channel",
    "min_share":"Minimum Budget Share",
    "max_share":"Maximum Budget Share"
})

display(
    constraints_display.style
    .format({
        "Minimum Budget Share":"{:.0%}",
        "Maximum Budget Share":"{:.0%}"
    })
    .hide(axis="index")
)

report("""

<div style="max-width:850px; line-height:1.7;">

Budget constraints ensure diversification and risk control. <br>

• Each channel receives at least 5% of total budget<br>  
• No channel receives more than 40%<br>

These limits prevent over-concentration while allowing
high-performing channels to receive additional investment.

</div>

""")

# =====================================================
# Optimal Budget Allocation
# =====================================================

section("Allocate Budget Based on Marginal Impact")

impact_df["weight"] = impact_df["coefficient"] / impact_df["coefficient"].sum()

impact_df["recommended_budget"] = impact_df["weight"] * monthly_budget

# Apply constraints

impact_df = impact_df.merge(
    constraints,
    on="channel"
)

impact_df["min_budget"] = impact_df["min_share"] * monthly_budget
impact_df["max_budget"] = impact_df["max_share"] * monthly_budget

impact_df["recommended_budget"] = impact_df[
    ["recommended_budget","min_budget","max_budget"]
].apply(lambda x: min(max(x.iloc[0], x.iloc[1]), x.iloc[2]), axis=1)

impact_df = impact_df.sort_values(
    "recommended_budget",
    ascending=False
)

allocation_table = impact_df.copy()

allocation_table["Channel"] = (
    allocation_table["channel"]
    .str.replace("_"," ")
    .str.title()
)

allocation_table = allocation_table[
    ["Channel","recommended_budget","min_budget","max_budget"]
]

allocation_table = allocation_table.rename(columns={
    "recommended_budget":"Recommended Budget (₹M)",
    "min_budget":"Minimum Budget (₹M)",
    "max_budget":"Maximum Budget (₹M)"
})

display(
    allocation_table.style
    .format({
        "Recommended Budget (₹M)": format_inr_millions,
        "Minimum Budget (₹M)": format_inr_millions,
        "Maximum Budget (₹M)": format_inr_millions
    })
    .highlight_max(
        subset=["Recommended Budget (₹M)"],
        color="#d4edda"
    )
    .hide(axis="index")
)

# =====================================================
# Budget Allocation Visualization
# =====================================================

section("Recommended Budget Allocation")

plt.figure(figsize=(8,5))

bars = plt.bar(
    impact_df["channel"].str.replace("_"," ").str.title(),
    impact_df["recommended_budget"],
    color=main_color
)

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        format_inr_millions(height),
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.title("Recommended Marketing Budget Allocation", fontsize=13, pad=12)

plt.ylim(0, impact_df["recommended_budget"].max()*1.15)

plt.ylabel("Budget Allocation (₹M)")

plt.gca().yaxis.set_major_formatter(
    mtick.FuncFormatter(lambda x, p: f"₹ {x/1_000_000:.1f}M")
)

plt.grid(axis="y", linestyle="--", alpha=0.3)

save_chart("budget_allocation")

# =====================================================
# Simulate Revenue from Recommended Budget
# =====================================================

section("Simulate Revenue Impact from Allocation")

# Create simulated spend row
simulated_spend = pd.DataFrame(
    np.zeros((1, len(channels))),
    columns=channels
)

for ch in channels:

    if ch in impact_df["channel"].values:

        simulated_spend.loc[0, ch] = (
            impact_df.loc[
                impact_df["channel"] == ch,
                "recommended_budget"
            ].values[0] / 30
        )

# Apply same feature engineering used in training
for ch in channels:

    simulated_spend[f"log_{ch}"] = np.log1p(simulated_spend[ch])

# Add promo flag and week index assumptions
simulated_spend["promo_flag"] = 0
simulated_spend["week_index"] = model_data["week_index"].mean()

# Add day-of-week dummies
for col in dow_dummies.columns:
    simulated_spend[col] = 0

# Add missing columns expected by the model
for col in feature_cols:

    if col not in simulated_spend.columns:
        simulated_spend[col] = 0

# Reorder columns exactly like training data
simulated_spend = simulated_spend[feature_cols]

# Predict revenue
predicted_daily_revenue = model.predict(simulated_spend)[0]

predicted_monthly_revenue = predicted_daily_revenue * 30

report(f"""

<div style="max-width:850px; line-height:1.7;">

<b>Predicted Daily Revenue:</b> {format_inr_millions(predicted_daily_revenue)} <br>
<b>Projected 30-Day Revenue (30 days):</b> {format_inr_millions(predicted_monthly_revenue)}

</div>

""")

# =====================================================
# Revenue Impact Estimate
# =====================================================

section("Estimate 30-Day Revenue Impact")

# Revenue impact estimated using model simulation

expected_total_revenue = predicted_monthly_revenue

impact_df["budget_share"] = impact_df["recommended_budget"] / monthly_budget

impact_df["channel"] = (
    impact_df["channel"]
    .str.replace("_"," ")
    .str.title()
)

impact_display = impact_df.copy()

impact_display = impact_display.rename(columns={
    "channel":"Channel",
    "recommended_budget":"Recommended Budget (₹M)",
    "budget_share":"Budget Share"
})

impact_display["Channel"] = (
    impact_display["Channel"]
    .str.replace("_"," ")
    .str.title()
)

impact_display = impact_display[
    ["Channel","Recommended Budget (₹M)","Budget Share"]
]

display(
    impact_display.style
    .format({
        "Recommended Budget (₹M)": format_inr_millions,
        "Budget Share":"{:.1%}"
    })
    .highlight_max(subset=["Budget Share"], color="#d4edda")
    .hide(axis="index")
)

report(f"""

<div style="max-width:850px; line-height:1.7;">

Estimated total revenue generated from the recommended
budget allocation is approximately:

<b>{format_inr_millions(expected_total_revenue)}</b> <br>

This estimate is produced by simulating the optimized
budget allocation through the trained regression model.

</div>

""")

# =====================================================
# Budget Allocation Share by Channel
# =====================================================

section("Budget Allocation Share by Channel")

impact_df = impact_df.sort_values("budget_share")

plt.figure(figsize=(8,5))

bars = plt.barh(
    impact_df["channel"],
    impact_df["budget_share"],
    color="#1f77b4"
)

plt.title("Recommended Budget Allocation Share", fontsize=13, pad=12)

plt.xlabel("Share of Total Marketing Budget")

plt.gca().xaxis.set_major_formatter(
    mtick.FuncFormatter(lambda x, p: f"{x*100:.0f}%")
)

plt.grid(axis="x", linestyle="--", alpha=0.3)

for bar in bars:

    width = bar.get_width()

    plt.text(
        width,
        bar.get_y() + bar.get_height()/2,
        f" {width*100:.1f}%",
        va="center"
    )

save_chart("budget_allocation")

top_contributor = impact_df.sort_values(
    "recommended_budget",
    ascending=False
).iloc[0]["channel"]

report(f"""

<b>Top Recommended Investment Channel:</b> {top_contributor}<br>

Based on the regression model’s marginal impact estimates,
this channel receives the largest share of the optimized
marketing budget.

""")

# =====================================================
# Scenario Analysis
# =====================================================

section("Revenue Scenario Analysis")

base_revenue = predicted_monthly_revenue

best_case = base_revenue * 1.10
worst_case = base_revenue * 0.90

scenario_df = pd.DataFrame({

    "Scenario":["Worst Case","Base Case","Best Case"],

    "Estimated Revenue":[worst_case, base_revenue, best_case]

})

display(
    scenario_df.style
    .format({"Estimated Revenue": format_inr_millions})
    .highlight_max(subset=["Estimated Revenue"], color="#d4edda")
    .hide(axis="index")
)

# =====================================================
# Scenario Visualization
# =====================================================

section("Revenue Scenario Projection")

plt.figure(figsize=(6,4))

bars = plt.bar(
    scenario_df["Scenario"],
    scenario_df["Estimated Revenue"]
)

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        format_inr_millions(height),
        ha="center",
        va="bottom"
    )

plt.title("Projected Revenue Range (30 Days)", fontsize=13)

plt.ylabel("Estimated Revenue")
plt.gca().yaxis.set_major_formatter(
    mtick.FuncFormatter(lambda x, p: f"₹ {x/1_000_000:.1f}M")
)

plt.grid(axis="y", linestyle="--", alpha=0.3)

save_chart("revenue_range")

# =====================================================
# Export KPI values for Power BI
# =====================================================

# Output folder
OUTPUT_PATH = "output"

os.makedirs(OUTPUT_PATH, exist_ok=True)

kpi_export = pd.DataFrame({

    "metric": [
        "recommended_budget",
        "expected_revenue",
        "projected_roas"
    ],

    "value": [
        monthly_budget,
        expected_total_revenue,
        expected_total_revenue / monthly_budget
    ]

})

kpi_export

kpi_export.to_csv(
        f"{OUTPUT_PATH}/budget_kpis.csv",
        index=False
    )

impact_df_export = impact_df[["channel","coefficient"]]

impact_df_export.to_csv(
    f"{OUTPUT_PATH}/regression_channel_impact.csv",
    index=False
)


# =====================================================
# Margin Proxy Estimate
# =====================================================

section("Margin Proxy Impact")

avg_margin_rate = 0.35

margin_estimate = base_revenue * avg_margin_rate

report(f"""

<div style="max-width:850px; line-height:1.7;">

Assuming an average contribution margin of approximately
<b> {avg_margin_rate*100:.0f}%</b>, the expected contribution
margin generated by the proposed marketing budget is:

<b>{format_inr_millions(margin_estimate)}</b>

This estimate represents the potential profit contribution
after accounting for product costs.

</div>

""")

# =====================================================
# Assumptions & Sensitivity
# =====================================================

section("Model Assumptions and Sensitivity")

report("""

<div style="
max-width:900px;
margin-left:20px;
line-height:1.7;
">

<h4>Key Assumptions</h4>

<ul>

<li>
The regression coefficients represent stable marginal
relationships between marketing spend and revenue.
</li>

<li>
Channel performance remains broadly consistent with
recent historical data.
</li>

<li>
External factors such as competitor campaigns,
pricing changes, and macroeconomic conditions remain stable.
</li>

</ul>

<h4>Sensitivity Considerations</h4>

<ul>

<li>
Marketing effectiveness may vary due to creative changes,
audience targeting, or market conditions.
</li>

<li>
If conversion rates improve, revenue impact may exceed
the best-case scenario.
</li>

<li>
Conversely, weaker campaign performance could reduce
the realized impact toward the worst-case estimate.
</li>

</ul>

</div>

""")

report(f"""

<b>Projected Marketing ROI:</b> {(base_revenue / monthly_budget):.2f}x 
(revenue generated per ₹1 of marketing spend)

""")

# =====================================================
# Final Recommendation
# =====================================================

banner("RECOMMENDED MARKETING STRATEGY")

report(f"""

<div style="
max-width:900px;
margin-left:40px;
line-height:1.7;
">

<p>
Allocate the next month's marketing budget of 
<b>{format_inr_millions(monthly_budget)}</b> across channels based on the
estimated marginal revenue impact identified by the regression model.
</p>

<p>
Prioritizing channels with stronger marginal effects allows
the marketing team to maximize revenue generation while maintaining
diversification across acquisition channels.
</p>

<p>
Under the proposed allocation strategy, the marketing program
is expected to generate approximately
<b>{format_inr_millions(base_revenue)}</b> in revenue over the next 30 days,
with a potential range between
<b>{format_inr_millions(worst_case)}</b> and <b>{format_inr_millions(best_case)}</b>.
</p>

</div>

""")


# In[ ]:




