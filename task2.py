#task2 - Create an interactive hierarchical sunburst chart using the structure Country → Category → App Type → Rating Band. Segment size must represent total installs, while colour must represent the weighted average rating. Use only the top five eligible categories by global installs. Include apps with ratings of at least 4.0, more than 10,000 installs, over 1,000 reviews, sizes between 15 MB and 80 MB, and names containing no numbers. Exclude categories beginning with “A,” “C,” “G,” or “S.” Create rating bands of 4.0–4.2, 4.2–4.5, 4.5–4.7, and 4.7–5.0. Highlight segments exceeding one million installs and calculate each segment’s percentage contribution to its parent. Add drill-down interaction and custom hover information showing installs, weighted rating, review count, and percentage contribution. Translate Business into Tamil, Travel & Local into French, and Productivity into Spanish. Display this visualization only between 6 PM and 8 PM IST; outside this period, it must not appear on the dashboard.


from datetime import datetime
import zoneinfo
import numpy as np
import pandas as pd
import plotly.express as px

# Time Check
ist_time = datetime.now(zoneinfo.ZoneInfo("Asia/Kolkata"))
current_hour = ist_time.hour

if not (18 <= current_hour < 20):
    print(
        f"Current IST Time: {ist_time.strftime('%I:%M %p IST')}\n"
        "Access Denied: This visualization is configured to run ONLY between 6 PM and 8 PM IST."
    )
    exit()

print(
    f"Current IST Time: {ist_time.strftime('%I:%M %p IST')} - Access Granted.\n"
)

# Data Loading 
df = pd.read_csv("googleplaystore.csv")

def clean_installs(val):
    try:
        return float(str(val).replace("+", "").replace(",", "").strip())
    except:
        return np.nan


df["Installs_Clean"] = df["Installs"].apply(clean_installs)

# Clean Rating & Reviews
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")


# Clean Size (Convert to MB)
def clean_size_mb(val):
    val = str(val).strip()
    if "M" in val:
        return float(val.replace("M", ""))
    elif "k" in val:
        return float(val.replace("k", "")) / 1024.0
    return np.nan

df["Size_MB"] = df["Size"].apply(clean_size_mb)

# Add dummy Country variable if not present (to fulfill Country -> Category hierarchy)
if "Country" not in df.columns:
    df["Country"] = "Global"

# Filter 
mask = (
    (df["Rating"] >= 4.0)
    & (df["Installs_Clean"] > 10000)
    & (df["Reviews"] > 1000)
    & (df["Size_MB"] >= 15)
    & (df["Size_MB"] <= 80)
    & (~df["App"].astype(str).str.contains(r"\d", regex=True))
)

filtered_df = df[mask].copy()

# categories beginning with 'A', 'C', 'G', or 'S'
excluded_starts = ("A", "C", "G", "S")
filtered_df = filtered_df[
    ~filtered_df["Category"].str.upper().str.startswith(excluded_starts)
]

#  Top 5 eligible categories by total global installs
top_5_categories = (
    filtered_df.groupby("Category")["Installs_Clean"]
    .sum()
    .nlargest(5)
    .index.tolist()
)

filtered_df = filtered_df[filtered_df["Category"].isin(top_5_categories)]

translations = {
    "BUSINESS": "வணிகம்",
    "TRAVEL_AND_LOCAL": "Voyages et transits",
    "PRODUCTIVITY": "Productividad",
}
filtered_df["Category"] = filtered_df["Category"].replace(translations)

#Rating
bins = [4.0, 4.2, 4.5, 4.7, 5.01]
labels = ["4.0–4.2", "4.2–4.5", "4.5–4.7", "4.7–5.0"]
filtered_df["Rating Band"] = pd.cut(
    filtered_df["Rating"], bins=bins, labels=labels, right=False
)

filtered_df["Rating_x_Installs"] = (
    filtered_df["Rating"] * filtered_df["Installs_Clean"]
)

# Grouping by hierarchy: Country -> Category -> Type -> Rating Band
grouped = (
    filtered_df.groupby(["Country", "Category", "Type", "Rating Band"])
    .agg(
        Total_Installs=("Installs_Clean", "sum"),
        Total_Reviews=("Reviews", "sum"),
        Weighted_Rating_Sum=("Rating_x_Installs", "sum"),
    )
    .reset_index()
)

# Weighted Average Rating
grouped["Weighted_Rating"] = (
    grouped["Weighted_Rating_Sum"] / grouped["Total_Installs"]
)
grouped["Weighted_Rating"] = grouped["Weighted_Rating"].round(2)

# Highlighting segments exceeding 1 Million installs
grouped["Highlight"] = grouped["Total_Installs"].apply(
    lambda x: "🔥 >1M Installs" if x > 1000000 else "Standard"
)

#Sunburst visualisation
fig = px.sunburst(
    grouped,
    path=["Country", "Category", "Type", "Rating Band"],
    values="Total_Installs",
    color="Weighted_Rating",
    color_continuous_scale="RdYlGn",
    title="Interactive App Performance Sunburst Chart",
    hover_data={
        "Total_Installs": ":,.0f",
        "Total_Reviews": ":,.0f",
        "Weighted_Rating": ":.2f",
        "Highlight": True,
    },
)

fig.update_traces(
    textinfo="label+percent parent",
    hovertemplate="<b>%{label}</b><br>"
    + "Total Installs: %{value:,.0f}<br>"
    + "Percentage of Parent: %{percentParent:.2%}<br>"
    + "Weighted Rating: %{color:.2f}<br>"
    + "<extra></extra>",
)

fig.update_layout(
    margin=dict(t=50, l=10, r=10, b=10),
    coloraxis_colorbar=dict(title="Weighted<br>Rating"),
)

fig.show()