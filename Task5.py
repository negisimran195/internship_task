#task 5 - Create an interactive clustered heatmap comparing the top 10 app categories across weighted rating, total reviews, total installs, average size, engagement rate, and update frequency. Include only apps with ratings of at least 4.0, sizes above 10 MB, at least 10,000 installs, more than 1,000 reviews, and a last-updated month of January. Exclude app names containing numbers. Normalize all metrics using Z-score normalization and dynamically rank the categories using a weighted composite score. Apply hierarchical clustering to arrange categories according to their performance similarity and annotate the three highest and three lowest composite scores. Include a selector to switch between raw and normalized values. Display this graph only between 3 PM and 5 PM IST; outside this period, it must not appear on the dashboard.
 

import sys
import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pytz

#Time Check
ist = pytz.timezone('Asia/Kolkata')
current_time_ist = datetime.datetime.now(ist)

if not (15 <= current_time_ist.hour < 17):
    print(f"Current IST Time: {current_time_ist.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("RESTRICTION: The interactive heatmap graph is active only between 3 PM and 5 PM IST. Outside this window, execution stops.")
    sys.exit(0)

#Data Load
apps_df = pd.read_csv("googleplaystore.csv")
reviews_df = pd.read_csv("reviews.csv")

# Clean Rating & Reviews
apps_df['Rating_Clean'] = pd.to_numeric(apps_df['Rating'], errors='coerce')
apps_df['Reviews_Clean'] = pd.to_numeric(apps_df['Reviews'], errors='coerce')

# Clean Installs
apps_df['Installs_Clean'] = apps_df['Installs'].astype(str).str.replace('+', '', regex=False).str.replace(',', '', regex=False)
apps_df['Installs_Clean'] = pd.to_numeric(apps_df['Installs_Clean'], errors='coerce')

# Clean Size (convert M/k to MB)
def parse_size(val):
    if pd.isna(val) or val == 'Varies with device':
        return np.nan
    val = str(val).strip()
    if val.endswith('M') or val.endswith('m'):
        return float(val[:-1])
    elif val.endswith('k') or val.endswith('K'):
        return float(val[:-1]) / 1024.0
    try:
        return float(val)
    except:
        return np.nan

apps_df['Size_MB'] = apps_df['Size'].apply(parse_size)

# Extract Last Updated Month
apps_df['Last_Updated_DT'] = pd.to_datetime(apps_df['Last Updated'], errors='coerce')
apps_df['Updated_Month'] = apps_df['Last_Updated_DT'].dt.month

# Check for app names containing numbers
apps_df['Has_Number'] = apps_df['App'].astype(str).str.contains(r'\d', regex=True)

# Engagement Rate: (Reviews / Installs) * 100
apps_df['Engagement_Rate'] = np.where(apps_df['Installs_Clean'] > 0, (apps_df['Reviews_Clean'] / apps_df['Installs_Clean']) * 100, 0.0)

# Update Frequency: Inverse of days elapsed since dataset max updated date
max_dt = apps_df['Last_Updated_DT'].max()
apps_df['Days_Since_Update'] = (max_dt - apps_df['Last_Updated_DT']).dt.days
apps_df['Update_Frequency'] = 1 / (apps_df['Days_Since_Update'] + 1)

#Filter
filtered_df = apps_df[
    (apps_df['Rating_Clean'] >= 4.0) &
    (apps_df['Size_MB'] > 10.0) &
    (apps_df['Installs_Clean'] >= 10000) &
    (apps_df['Reviews_Clean'] > 1000) &
    (apps_df['Updated_Month'] == 1) &  # January
    (~apps_df['Has_Number'])            # No numbers in App name
].copy()

#  top 10 categories by total installs
top_10_cats = filtered_df.groupby('Category')['Installs_Clean'].sum().nlargest(10).index.tolist()
filtered_df = filtered_df[filtered_df['Category'].isin(top_10_cats)]

#metrics
cat_agg = filtered_df.groupby('Category').agg(
    Weighted_Rating=('Rating_Clean', lambda x: (x * filtered_df.loc[x.index, 'Reviews_Clean']).sum() / filtered_df.loc[x.index, 'Reviews_Clean'].sum()),
    Total_Reviews=('Reviews_Clean', 'sum'),
    Total_Installs=('Installs_Clean', 'sum'),
    Average_Size=('Size_MB', 'mean'),
    Engagement_Rate=('Engagement_Rate', 'mean'),
    Update_Frequency=('Update_Frequency', 'mean')
).reset_index()

metrics = ['Weighted_Rating', 'Total_Reviews', 'Total_Installs', 'Average_Size', 'Engagement_Rate', 'Update_Frequency']
metric_labels = ['Weighted Rating', 'Total Reviews', 'Total Installs', 'Average Size (MB)', 'Engagement Rate (%)', 'Update Frequency']

# Z-score Normalization using NumPy
for m in metrics:
    mean_val = cat_agg[m].mean()
    std_val = cat_agg[m].std(ddof=0)
    cat_agg[f'{m}_Z'] = (cat_agg[m] - mean_val) / (std_val if std_val != 0 else 1)

z_metrics = [f'{m}_Z' for m in metrics]

# Dynamic Weighted Composite Score (mean of normalized Z-scores)
cat_agg['Composite_Score'] = cat_agg[z_metrics].mean(axis=1)

# Sort by Composite Score to assign rank annotations
cat_agg = cat_agg.sort_values(by='Composite_Score', ascending=False).reset_index(drop=True)
cat_agg['Rank'] = cat_agg.index + 1

# Annotate 3 Highest and 3 Lowest Composite Scores
def format_annotation(row):
    if row['Rank'] <= 3:
        return f"{row['Category']} [High #{int(row['Rank'])}: {row['Composite_Score']:.2f}]"
    elif row['Rank'] >= 8:
        return f"{row['Category']} [Low #{11 - int(row['Rank'])}: {row['Composite_Score']:.2f}]"
    return f"{row['Category']} [{row['Composite_Score']:.2f}]"

cat_agg['Annotated_Category'] = cat_agg.apply(format_annotation, axis=1)



z_matrix = cat_agg[z_metrics].values
n = len(z_matrix)

# Compute Euclidean Distance Matrix
dist_matrix = np.sqrt(((z_matrix[:, np.newaxis, :] - z_matrix[np.newaxis, :, :]) ** 2).sum(axis=2))

# Hierarchical Greedy Nearest-Neighbor Cluster Ordering
order = [0]
unvisited = set(range(1, n))
while unvisited:
    last = order[-1]
    next_idx = min(unvisited, key=lambda i: dist_matrix[last, i])
    order.append(next_idx)
    unvisited.remove(next_idx)

clustered_df = cat_agg.iloc[order].reset_index(drop=True)

# Matrices for Plotly Heatmap
z_data = clustered_df[z_metrics].values
raw_data = clustered_df[metrics].values
y_labels = clustered_df['Annotated_Category'].tolist()

# Prepare formatted text annotations for cells
raw_text = np.empty(raw_data.shape, dtype=object)
for i in range(raw_data.shape[0]):
    for j in range(raw_data.shape[1]):
        v = raw_data[i, j]
        if metrics[j] in ['Total_Reviews', 'Total_Installs']:
            raw_text[i, j] = f"{v:,.0f}"
        else:
            raw_text[i, j] = f"{v:.2f}"

z_text = np.round(z_data, 2).astype(str)

#Heatmap
fig = go.Figure()

# Trace 0: Normalized Z-Scores
fig.add_trace(go.Heatmap(
    z=z_data,
    x=metric_labels,
    y=y_labels,
    text=z_text,
    texttemplate="%{text}",
    colorscale="Viridis",
    colorbar=dict(title="Z-Score"),
    name="Normalized",
    visible=True
))

# Trace 1: Raw Values
fig.add_trace(go.Heatmap(
    z=raw_data,
    x=metric_labels,
    y=y_labels,
    text=raw_text,
    texttemplate="%{text}",
    colorscale="YlGnBu",
    colorbar=dict(title="Raw Value"),
    name="Raw",
    visible=False
))

# Layout Dropdown Menu Selector
fig.update_layout(
    title="Interactive Clustered Heatmap: Top 10 Categories",
    xaxis=dict(title="Performance Metrics"),
    yaxis=dict(title="Category (Hierarchically Clustered)"),
    updatemenus=[
        dict(
            buttons=[
                dict(
                    label="Normalized (Z-Score)",
                    method="update",
                    args=[{"visible": [True, False]},
                          {"title": "Clustered Heatmap - Normalized (Z-Scores)"}]
                ),
                dict(
                    label="Raw Values",
                    method="update",
                    args=[{"visible": [False, True]},
                          {"title": "Clustered Heatmap - Raw Values"}]
                )
            ],
            direction="down",
            showactive=True,
            x=0.0,
            xanchor="left",
            y=1.18,
            yanchor="top"
        )
    ],
    width=980,
    height=650
)


fig.show()