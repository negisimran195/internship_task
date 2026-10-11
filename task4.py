#TASK4 - Create an interactive streamgraph showing monthly and cumulative installs for app categories beginning with “T,” “P,” or “B.” Include only apps with ratings of at least 4.2, more than 1,000 reviews, sizes between 20 MB and 80 MB, at least 10,000 installs, and names containing no numbers. Create a complete monthly timeline and appropriately handle categories with missing months. Calculate month-over-month growth and detect unusual changes using both a growth threshold of 25% and a rolling Z-score above 2. Highlight anomalous periods using stronger colour intensity and annotations. Add a control to switch between monthly installs, cumulative installs, and growth percentage. Translate “Travel & Local” into French, “Productivity” into Spanish, and “Photography” into Japanese. Display this visualization only between 4 PM and 6 PM IST; outside this period, it must not appear on the dashboard.

import sys
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz

# Load Data
df = pd.read_csv("googleplaystore.csv")

# Clean & Filter Data
# Categories starting with T, P, or B
df = df[df["Category"].str.startswith(("T", "P", "B"), na=False)].copy()

# Rating >= 4.2
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
df = df[df["Rating"] >= 4.2]

# Reviews > 1000
df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")
df = df[df["Reviews"] > 1000]

# Size between 20 MB and 80 MB


def parse_size(size_str):
    if pd.isna(size_str) or size_str == "Varies with device":
        return np.nan
    size_str = str(size_str).strip()
    if size_str.endswith("M"):
        return float(size_str[:-1])
    elif size_str.endswith("k"):
        return float(size_str[:-1]) / 1024
    return np.nan


df["Size_MB"] = df["Size"].apply(parse_size)
df = df[(df["Size_MB"] >= 20) & (df["Size_MB"] <= 80)]

# Installs >= 10,000
df["Installs_Clean"] = (
    df["Installs"]
    .str.replace("+", "", regex=False)
    .str.replace(",", "", regex=False)
)
df["Installs_Clean"] = pd.to_numeric(df["Installs_Clean"], errors="coerce")
df = df[df["Installs_Clean"] >= 10000]

# Names containing no numbers
df = df[~df["App"].str.contains(r"\d", regex=True, na=False)]

# Date Parsing & Full Monthly Timeline
df["Last Updated"] = pd.to_datetime(df["Last Updated"], errors="coerce")
df = df.dropna(subset=["Last Updated"])
df["YearMonth"] = df["Last Updated"].dt.to_period("M")

# Translations
category_translations = {
    "TRAVEL_AND_LOCAL": "Voyage et Anti-local",  # Travel & Local -> French
    "PRODUCTIVITY": "Productividad",  # Productivity -> Spanish
    "PHOTOGRAPHY": "写真",  # Photography -> Japanese
}
df["Category_Translated"] = (
    df["Category"].map(category_translations).fillna(df["Category"])
)

# Group by Category and Month
grouped = (
    df.groupby(["Category_Translated", "YearMonth"])["Installs_Clean"]
    .sum()
    .reset_index()
)

# Complete Monthly Timeline Handling Missing Months
all_months = pd.period_range(
    start=grouped["YearMonth"].min(), end=grouped["YearMonth"].max(), freq="M"
)
all_categories = grouped["Category_Translated"].unique()

full_idx = pd.MultiIndex.from_product(
    [all_categories, all_months], names=["Category_Translated", "YearMonth"]
)
grouped = (
    grouped.set_index(["Category_Translated", "YearMonth"])
    .reindex(full_idx, fill_value=0)
    .reset_index()
)

# Sort chronologically
grouped = grouped.sort_values(["Category_Translated", "YearMonth"])
grouped["DateStr"] = grouped["YearMonth"].astype(str)

# Calculate Cumulative, Growth %, and Anomalies
metrics_data = {}

for cat in all_categories:
    cat_df = grouped[grouped["Category_Translated"] == cat].copy()

    # Monthly
    monthly = cat_df["Installs_Clean"].values

    # Cumulative
    cumulative = cat_df["Installs_Clean"].cumsum().values

    # Month-over-Month Growth %
    growth = cat_df["Installs_Clean"].pct_change().fillna(0).values * 100

    # Rolling Z-score on Growth
    growth_series = pd.Series(growth)
    rolling_mean = (
        growth_series.rolling(window=3, min_periods=1).mean().shift(1).fillna(0)
    )
    rolling_std = (
        growth_series.rolling(window=3, min_periods=1)
        .std()
        .shift(1)
        .fillna(1e-6)
    )
    z_scores = (growth_series - rolling_mean) / (rolling_std.replace(0, 1e-6))
    z_scores = z_scores.fillna(0).values

    # Anomalies: Growth > 25% AND Z-score > 2
    anomalies = (growth > 25) & (z_scores > 2)

    metrics_data[cat] = {
        "monthly": monthly,
        "cumulative": cumulative,
        "growth": growth,
        "anomalies": anomalies,
    }

# Time Constraint Check (4 PM to 6 PM IST)
ist = pytz.timezone("Asia/Kolkata")
now_ist = datetime.now(ist)

if not (16 <= now_ist.hour < 18):
    print(
        f" Current time ({now_ist.strftime('%I:%M %p %Z')}) is outside the active window (4:00 PM - 6:00 PM IST)."
    )
    print(" Visualization hidden per requirements.")
    sys.exit(0)

# Build Interactive Streamgraph Visualization
fig = go.Figure()

dates = grouped["DateStr"].unique()
base_colors = [
    "#1f77b4",
    "#aec7e8",
    "#ff7f0e",
    "#ffbb78",
    "#2ca02c",
    "#98df8a",
    "#d62728",
    "#ff9896",
    "#9467bd",
    "#c5b0d5",
]

# Add Traces for Each Metric
for cat_idx, cat in enumerate(all_categories):
    color = base_colors[cat_idx % len(base_colors)]

    # 1. Monthly Installs
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=metrics_data[cat]["monthly"],
            mode="lines",
            name=cat,
            stackgroup="one",  # Creates Streamgraph effect
            line=dict(width=1.5, color=color),
            fillcolor=color,
            visible=True,
        )
    )

    # 2. Cumulative Installs
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=metrics_data[cat]["cumulative"],
            mode="lines",
            name=cat,
            stackgroup="two",
            line=dict(width=1.5, color=color),
            fillcolor=color,
            visible=False,
        )
    )

    # 3. Growth Percentage
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=metrics_data[cat]["growth"],
            mode="lines+markers",
            name=cat,
            line=dict(width=2, color=color),
            visible=False,
        )
    )

# Highlight Anomalous Periods (Stronger Intensity Markers/Annotations)
n_cats = len(all_categories)
for cat_idx, cat in enumerate(all_categories):
    anom_indices = np.where(metrics_data[cat]["anomalies"])[0]
    for idx in anom_indices:
        anom_date = dates[idx]
        anom_val = metrics_data[cat]["monthly"][idx]

        # Highlight annotation on monthly view
        fig.add_annotation(
            x=anom_date,
            y=anom_val,
            text=f"<b>Anomaly!</b><br>{cat}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="red",
            arrowsize=1.5,
            arrowwidth=2,
            bgcolor="#ff4d4d",
            font=dict(color="white", size=10),
            bordercolor="red",
        )

# Dropdown Control to Switch Metrics
num_traces = len(fig.data)
visibility_monthly = [i % 3 == 0 for i in range(num_traces)]
visibility_cumulative = [i % 3 == 1 for i in range(num_traces)]
visibility_growth = [i % 3 == 2 for i in range(num_traces)]

fig.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=list(
                [
                    dict(
                        label="Monthly Installs",
                        method="update",
                        args=[
                            {"visible": visibility_monthly},
                            {
                                "title": "Monthly Installs Streamgraph (Categories: T, P, B)",
                                "yaxis": {"title": "Installs"},
                            },
                        ],
                    ),
                    dict(
                        label="Cumulative Installs",
                        method="update",
                        args=[
                            {"visible": visibility_cumulative},
                            {
                                "title": "Cumulative Installs Streamgraph (Categories: T, P, B)",
                                "yaxis": {"title": "Cumulative Installs"},
                            },
                        ],
                    ),
                    dict(
                        label="Growth Percentage (%)",
                        method="update",
                        args=[
                            {"visible": visibility_growth},
                            {
                                "title": "Month-over-Month Growth % (Categories: T, P, B)",
                                "yaxis": {"title": "Growth Rate (%)"},
                            },
                        ],
                    ),
                ]
            ),
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.0,
            xanchor="left",
            y=1.15,
            yanchor="top",
        )
    ],
    title="Monthly Installs Streamgraph (Categories: T, P, B)",
    xaxis=dict(title="Month", type="category"),
    yaxis=dict(title="Installs"),
    hovermode="x unified",
    template="plotly_white",
)

fig.show()