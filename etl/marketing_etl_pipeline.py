#!/usr/bin/env python
# coding: utf-8

# Marketing ROI ETL Pipeline
# 
# Steps:
# 1. Load raw datasets
# 2. Remove duplicates
# 3. Normalize channels
# 4. Handle missing ad spend
# 5. Handle revenue outliers
# 6. Save cleaned datasets
# 
# Author: Vaishnavi Bedre
# 
# Date: Feb 2026

# In[1]:


# ===============================
# Marketing ROI ETL Pipeline
# ===============================

import pandas as pd
import numpy as np
import json
import os

# Output folder
OUTPUT_PATH = "output"

os.makedirs(OUTPUT_PATH, exist_ok=True)


# In[2]:


# ===============================
# File Paths
# ===============================

USERS_PATH = "data/raw/users.csv"
CAMPAIGNS_PATH = "data/raw/campaigns.csv"
SPEND_PATH = "data/raw/ad_spend_daily.csv"
SESSIONS_PATH = "data/raw/sessions.csv"
ORDERS_PATH = "data/raw/orders.csv"
ORDER_ITEMS_PATH = "data/raw/order_items.csv"
PRODUCTS_PATH = "data/raw/products.json"


# In[3]:


# ===============================
# Load Raw Data
# ===============================

def load_data():

    # Load CSV Files
    users = pd.read_csv(USERS_PATH)
    campaigns = pd.read_csv(CAMPAIGNS_PATH)
    ad_spend = pd.read_csv(SPEND_PATH)
    sessions = pd.read_csv(SESSIONS_PATH)
    orders = pd.read_csv(ORDERS_PATH)
    order_items = pd.read_csv(ORDER_ITEMS_PATH)

    # Load JSON File
    with open(PRODUCTS_PATH, "r") as f:
        products_data = json.load(f)

    products = pd.DataFrame(products_data)

    print("All datasets loaded successfully")
    print("-----------------------------------")
    print("Users:", len(users))
    print("Campaigns:", len(campaigns))
    print("Ad Spend:", len(ad_spend))
    print("Sessions:", len(sessions))
    print("Orders:", len(orders))
    print("Order Items:", len(order_items))
    print("Products:", len(products))

    return users, campaigns, ad_spend, sessions, orders, order_items, products


# In[4]:


# ===============================
# Deduplication
# ===============================

def remove_duplicates(ad_spend, sessions, orders):

    print("Before Deduplication:")
    print("Ad Spend:", len(ad_spend))
    print("Sessions:", len(sessions))
    print("Orders:", len(orders))

    ad_spend = ad_spend.drop_duplicates()

    sessions = sessions.drop_duplicates(
        subset=["session_id"]
    )

    orders = orders.drop_duplicates(
        subset=["order_id"]
    )

    print("\nAfter Deduplication:")
    print("Ad Spend:", len(ad_spend))
    print("Sessions:", len(sessions))
    print("Orders:", len(orders))

    return ad_spend, sessions, orders


# In[5]:


# ===============================
# Normalize Channels
# ===============================

def normalize_channels(sessions, campaigns):

    print("Normalizing channels...")

    sessions["channel"] = (
        sessions["channel"]
        .str.lower()
        .str.strip()
        .str.replace(" ", "_")
    )

    campaigns["channel"] = (
        campaigns["channel"]
        .str.lower()
        .str.strip()
        .str.replace(" ", "_")
    )

    print("Channel normalization complete")

    return sessions, campaigns


# In[6]:


# ===============================
# Handle Missing Ad Spend Values
# ===============================

def clean_ad_spend(ad_spend):

    print("Cleaning Ad Spend Data...")

    ad_spend["spend"] = ad_spend["spend"].fillna(0)
    ad_spend["clicks"] = ad_spend["clicks"].fillna(0)
    ad_spend["impressions"] = ad_spend["impressions"].fillna(0)

    print("Ad Spend Cleaning Complete")

    return ad_spend


# In[7]:


# =========================
# Revenue Outlier Handling
# =========================

def handle_revenue_outliers(orders):

    # Calculate IQR
    Q1 = orders["net_revenue"].quantile(0.25)
    Q3 = orders["net_revenue"].quantile(0.75)

    IQR = Q3 - Q1

    upper_limit = Q3 + 1.5 * IQR

    # Identify Outliers
    orders["revenue_outlier"] = (
        orders["net_revenue"] > upper_limit
    )

    # Cap Outliers
    orders.loc[
        orders["net_revenue"] > upper_limit,
        "net_revenue"
    ] = upper_limit

    return orders, upper_limit


# In[8]:


# =========================
# Run ETL Pipeline
# =========================

def run_etl():

    # Load Data
    users, campaigns, ad_spend, sessions, orders, order_items, products = load_data()

    # Remove Duplicates
    ad_spend, sessions, orders = remove_duplicates(
        ad_spend,
        sessions,
        orders
    )

    # Normalize Channels
    sessions, campaigns = normalize_channels(
        sessions,
        campaigns
    )

    # Clean Ad Spend
    ad_spend = clean_ad_spend(ad_spend)


    # =========================
    # Create Net Revenue Column
    # =========================

    orders["net_revenue"] = orders["net_amount"]

    print("Net Revenue Column Created")


    # Handle Revenue Outliers
    orders, upper_limit = handle_revenue_outliers(orders)

    print("Revenue Outliers Handled")
    print("Upper Limit:", round(upper_limit,2))

    # =========================
    # Save Clean Data
    # =========================

    print("\nSaving Cleaned Data...")

    users.to_csv(f"{OUTPUT_PATH}/clean_users.csv", index=False)
    campaigns.to_csv(f"{OUTPUT_PATH}/clean_campaigns.csv", index=False)
    ad_spend.to_csv(f"{OUTPUT_PATH}/clean_ad_spend.csv", index=False)
    sessions.to_csv(f"{OUTPUT_PATH}/clean_sessions.csv", index=False)
    orders.to_csv(f"{OUTPUT_PATH}/clean_orders.csv", index=False)
    order_items.to_csv(f"{OUTPUT_PATH}/clean_order_items.csv", index=False)
    products.to_csv(f"{OUTPUT_PATH}/clean_products.csv", index=False)

    print("Clean datasets saved successfully")


    print("\nETL Step Completed")

    return users, campaigns, ad_spend, sessions, orders, order_items, products


# In[9]:


# ===============================
# Build fact_sessions Table
# ===============================

def build_fact_sessions(users, sessions, orders):

    print("\nBuilding fact_sessions...")

    # ---------------------------
    # Merge Sessions with Users
    # ---------------------------

    fact_sessions = sessions.merge(
        users,
        on="user_id",
        how="left"
    )

    print("Merged sessions with users")


    # ---------------------------
    # Merge Orders
    # ---------------------------

    fact_sessions = fact_sessions.merge(
        orders.drop(columns=["user_id"]),
        on="session_id",
        how="left"
    )

    print("Merged sessions with orders")


    # ---------------------------
    # Convert Date Columns
    # ---------------------------

    fact_sessions["session_ts"] = pd.to_datetime(
        fact_sessions["session_ts"]
    )

    fact_sessions["signup_date"] = pd.to_datetime(
        fact_sessions["signup_date"]
    )

    fact_sessions["order_ts"] = pd.to_datetime(
        fact_sessions["order_ts"]
    )


    # ---------------------------
    # Purchase Flag
    # ---------------------------

    fact_sessions["purchase_flag"] = (
        fact_sessions["order_id"]
        .notnull()
        .astype(int)
    )


    # ---------------------------
    # Revenue Fields
    # ---------------------------

    fact_sessions["gross_revenue"] = (fact_sessions["gross_amount"].fillna(0))

    fact_sessions["discount"] = (fact_sessions["discount_amount"].fillna(0))

    fact_sessions["net_revenue"] = (fact_sessions["net_revenue"].fillna(0))


    # ---------------------------
    # New User Flag (Correct Logic)
    # Based on signup_date
    # ---------------------------

    fact_sessions["is_new_user"] = (
        fact_sessions["session_ts"].dt.date
        ==
        fact_sessions["signup_date"].dt.date
    ).astype(int)


    # ---------------------------
    # Session to Order Time
    # ---------------------------

    fact_sessions["session_to_order_minutes"] = (
        (
            fact_sessions["order_ts"]
            - fact_sessions["session_ts"]
        )
        .dt.total_seconds() / 60
    )

    # Replace NaN with 0 for non-purchases

    fact_sessions["session_to_order_minutes"] = (
        fact_sessions["session_to_order_minutes"]
        .fillna(0)
    )


    # ---------------------------
    # Select Final Columns
    # ---------------------------

    fact_sessions = fact_sessions[[
        "session_id",
        "user_id",
        "session_ts",
        "device",
        "channel",
        "campaign_id",
        "is_new_user",
        "purchase_flag",
        "order_id",
        "gross_revenue",
        "discount",
        "net_revenue",
        "session_to_order_minutes"
    ]]


    # ---------------------------
    # Sort by Time (Professional)
    # ---------------------------

    fact_sessions = fact_sessions.sort_values(
        "session_ts"
    )


    # ---------------------------
    # Save Output
    # ---------------------------

    fact_sessions.to_csv(
        f"{OUTPUT_PATH}/fact_sessions.csv",
        index=False
    )

    print("fact_sessions.csv saved")


    return fact_sessions


# In[10]:


# ===============================
# Build fact_campaign_daily Table
# ===============================

def build_fact_campaign_daily(fact_sessions, ad_spend, campaigns):

    print("\nBuilding fact_campaign_daily...")

    # ---------------------------
    # Create Date Column
    # ---------------------------

    fact_sessions["session_ts"] = pd.to_datetime(
        fact_sessions["session_ts"]
    )

    fact_sessions["date"] = fact_sessions["session_ts"].dt.date

    # ---------------------------
    # Aggregate Sessions
    # ---------------------------

    sessions_daily = fact_sessions.groupby(
        ["date","campaign_id"]
    ).agg(

        attributed_sessions=("session_id","count"),

        attributed_orders=("purchase_flag","sum"),

        attributed_revenue=("net_revenue","sum")

    ).reset_index()

    # ---------------------------
    # Prepare Spend Data
    # ---------------------------

    ad_spend["date"] = pd.to_datetime(
        ad_spend["date"]
    ).dt.date

    # ---------------------------
    # Merge Spend with Sessions
    # ---------------------------

    fact_campaign = ad_spend.merge(
        sessions_daily,
        on=["date","campaign_id"],
        how="left"
    )

    # Replace Missing Values

    fact_campaign["attributed_sessions"] = fact_campaign[
        "attributed_sessions"
    ].fillna(0)

    fact_campaign["attributed_orders"] = fact_campaign[
        "attributed_orders"
    ].fillna(0)

    fact_campaign["attributed_revenue"] = fact_campaign[
        "attributed_revenue"
    ].fillna(0)

    # Ensure spend data has no missing values

    fact_campaign["spend"] = fact_campaign["spend"].fillna(0)

    fact_campaign["impressions"] = fact_campaign["impressions"].fillna(0)

    fact_campaign["clicks"] = fact_campaign["clicks"].fillna(0)

    # ---------------------------
    # Add Channel
    # ---------------------------

    fact_campaign = fact_campaign.merge(

        campaigns[["campaign_id","channel"]],
        on="campaign_id",
        how="left",
        suffixes=("", "_campaign")
    )

    # Ensure correct column name
    if "channel_campaign" in fact_campaign.columns:
        fact_campaign["channel"] = fact_campaign["channel_campaign"]

    # ---------------------------
    # Derived KPIs
    # ---------------------------

    fact_campaign["CPC"] = (
        fact_campaign["spend"]
        /
        fact_campaign["clicks"]
    ).replace(np.inf,0).fillna(0)


    fact_campaign["CTR"] = (
        fact_campaign["clicks"]
        /
        fact_campaign["impressions"]
    ).replace(np.inf,0).fillna(0)


    fact_campaign["CVR"] = (
        fact_campaign["attributed_orders"]
        /
        fact_campaign["attributed_sessions"]
    ).replace(np.inf,0).fillna(0)


    fact_campaign["ROAS"] = (
        fact_campaign["attributed_revenue"]
        /
        fact_campaign["spend"]
    ).replace(np.inf,0).fillna(0)


    fact_campaign["CAC_proxy"] = (
        fact_campaign["spend"]
        /
        fact_campaign["attributed_orders"]
    ).replace(np.inf,0).fillna(0)


    # ---------------------------
    # Round KPIs (Dashboard Ready)
    # ---------------------------

    fact_campaign["CPC"] = fact_campaign["CPC"].round(2)

    fact_campaign["CTR"] = fact_campaign["CTR"].round(4)

    fact_campaign["CVR"] = fact_campaign["CVR"].round(4)

    fact_campaign["ROAS"] = fact_campaign["ROAS"].round(2)

    fact_campaign["CAC_proxy"] = fact_campaign["CAC_proxy"].round(2)

    # ---------------------------
    # Fix Data Types
    # ---------------------------

    fact_campaign["attributed_sessions"] = (
        fact_campaign["attributed_sessions"].astype(int)
    )

    fact_campaign["attributed_orders"] = (
        fact_campaign["attributed_orders"].astype(int)
    )

    fact_campaign["clicks"] = (
        fact_campaign["clicks"].astype(int)
    )

    fact_campaign["impressions"] = (
        fact_campaign["impressions"].astype(int)
    )

    # ---------------------------
    # Final Columns
    # ---------------------------

    fact_campaign = fact_campaign[[

        "date",
        "campaign_id",
        "channel",

        "spend",
        "impressions",
        "clicks",

        "attributed_sessions",
        "attributed_orders",
        "attributed_revenue",

        "CPC",
        "CTR",
        "CVR",
        "ROAS",
        "CAC_proxy"

    ]]

    # ---------------------------
    # Format Date Column
    # ---------------------------

    fact_campaign["date"] = pd.to_datetime(
        fact_campaign["date"]
    )

    # ---------------------------
    # Sort Data
    # ---------------------------

    fact_campaign = fact_campaign.sort_values(
        ["date","campaign_id"]
    )


    # Save Output
    fact_campaign.to_csv(

        f"{OUTPUT_PATH}/fact_campaign_daily.csv",

        index=False

    )

    print("fact_campaign_daily.csv saved")

    return fact_campaign


# In[11]:


# ===============================
# Build fact_channel_daily Table
# ===============================

def build_fact_channel_daily(fact_campaign):

    print("\nBuilding fact_channel_daily...")

    # ---------------------------
    # Aggregate by Channel + Date
    # ---------------------------

    fact_channel = fact_campaign.groupby(
        ["date","channel"]
    ).agg(

        total_spend=("spend","sum"),

        attributed_orders=("attributed_orders","sum"),

        attributed_revenue=("attributed_revenue","sum")

    ).reset_index()

        # Convert date format

    fact_channel["date"] = pd.to_datetime(
        fact_channel["date"]
    )

        # ---------------------------
    # Day of Week
    # ---------------------------

    fact_channel["day_of_week"] = (
        fact_channel["date"]
        .dt.dayofweek
    )

    # ---------------------------
    # Promo Flag
    # ---------------------------

    # Rule: Top 10% revenue days = promotion.

    revenue_threshold = fact_channel[
        "attributed_revenue"
    ].quantile(0.90)


    fact_channel["promo_flag"] = (

        fact_channel["attributed_revenue"]

        > revenue_threshold

    ).astype(int)

    # ---------------------------
    # Week Index
    # ---------------------------

    start_date = fact_channel["date"].min()

    fact_channel["week_index"] = (

        (fact_channel["date"] - start_date)

        .dt.days

        // 7

    )

    # ---------------------------
    # Fix Data Types
    # ---------------------------

    fact_channel["total_spend"] = (
        fact_channel["total_spend"].astype(float)
    )

    fact_channel["attributed_orders"] = (
        fact_channel["attributed_orders"].astype(int)
    )

    # ---------------------------
    # Final Columns
    # ---------------------------

    fact_channel = fact_channel[[
        "date",
        "channel",
        "total_spend",
        "attributed_orders",
        "attributed_revenue",
        "day_of_week",
        "promo_flag",
        "week_index"
    ]]

    # ---------------------------
    # Sort Data
    # ---------------------------

    fact_channel = fact_channel.sort_values(
        ["date","channel"]
    )

    # ---------------------------
    # Save Output
    # ---------------------------

    fact_channel.to_csv(
        f"{OUTPUT_PATH}/fact_channel_daily.csv",
        index=False
    )

    print("fact_channel_daily.csv saved")

    return fact_channel



# In[12]:


# Run ETL

if __name__ == "__main__":

    users, campaigns, ad_spend, sessions, orders, order_items, products = run_etl()

    fact_sessions = build_fact_sessions(users, sessions, orders)
    fact_campaign = build_fact_campaign_daily(fact_sessions, ad_spend, campaigns)
    fact_channel = build_fact_channel_daily(fact_campaign)


# In[ ]:




