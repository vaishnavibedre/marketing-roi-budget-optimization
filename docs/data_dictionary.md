# Data Dictionary

## fact_sessions

Session level marketing attribution dataset.

| Column | Description |
|--------|-------------|
session_id | Unique session identifier
user_id | Unique user identifier
session_ts | Session timestamp
device | User device type
channel | Marketing channel
campaign_id | Campaign identifier
is_new_user | New user flag
purchase_flag | Purchase indicator
order_id | Order identifier
gross_revenue | Revenue before discounts
discount | Discount amount
net_revenue | Final revenue
session_to_order_minutes | Time to purchase

---

## fact_campaign_daily

Daily campaign performance dataset.

| Column | Description |
|--------|-------------|
date | Campaign date
campaign_id | Campaign identifier
channel | Marketing channel
spend | Daily marketing spend
impressions | Ad impressions
clicks | Ad clicks
attributed_sessions | Sessions driven
attributed_orders | Orders driven
attributed_revenue | Revenue generated
CPC | Cost per click
CTR | Click through rate
CVR | Conversion rate
ROAS | Return on ad spend
CAC_proxy | Cost per acquisition proxy

---

## fact_channel_daily

Channel level performance dataset.

| Column | Description |
|--------|-------------|
date | Date
channel | Marketing channel
total_spend | Total spend
attributed_orders | Orders
attributed_revenue | Revenue
day_of_week | Day index
promo_flag | Promotion indicator
week_index | Week number
