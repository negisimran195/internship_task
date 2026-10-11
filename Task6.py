#task6 - Create an interactive radar chart comparing free and paid apps across average installs, weighted rating, total reviews, average size, revenue, and engagement rate within the top five categories by installs. Clean the Installs, Reviews, Size, Price, Rating, and Android Version columns before analysis. Calculate paid-app revenue as Installs × Price, weighted rating using review counts, and engagement rate as Reviews ÷ Installs × 100. Normalize every metric using percentile-based normalization. Include only apps with at least 10,000 installs, revenue above $10,000 for paid apps, Android version above 4.0, size above 15 MB, content rating “Everyone,” and app names containing no more than 30 characters. Add a dropdown to switch between individual categories and the overall comparison, calculate a composite performance score, and annotate the better-performing app type. Display this graph only between 1 PM and 2 PM IST; outside this period, it must not appear on the dashboard.

import sys
import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pytz

#Time check
ist = pytz.timezone('Asia/Kolkata')
current_time_ist = datetime.datetime.now(ist)

if not (13 <= current_time_ist.hour < 14):
    print(f"Current IST Time: {current_time_ist.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("RESTRICTION: The interactive graph display is active only between 1 PM and 2 PM IST. Outside this window, execution stops.")
    sys.exit(0)

#Data Load
df = pd.read_csv("googleplaystore.csv")

df['Installs_Clean'] = df['Installs'].astype(str).str.replace('+', '', regex=False).str.replace(',', '', regex=False)
df['Installs_Clean'] = pd.to_numeric(df['Installs_Clean'], errors='coerce')

df['Reviews_Clean'] = pd.to_numeric(df['Reviews'], errors='coerce')

df['Price_Clean'] = df['Price'].astype(str).str.replace('$', '', regex=False)
df['Price_Clean'] = pd.to_numeric(df['Price_Clean'], errors='coerce')

df['Rating_Clean'] = pd.to_numeric(df['Rating'], errors='coerce')

def clean_size(val):
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

df['Size_MB'] = df['Size'].apply(clean_size)

def clean_android_ver(val):
    if pd.isna(val) or val == 'Varies with device':
        return np.nan
    import re
    match = re.search(r'(\d+\.\d+|\d+)', str(val))
    if match:
        return float(match.group(1))
    return np.nan

df['Android_Ver_Clean'] = df['Android Ver'].apply(clean_android_ver)

# Clean 'Type'
df['Type'] = df['Type'].fillna('Free')

#Calculation 
df['Revenue'] = np.where(df['Type'] == 'Paid', df['Installs_Clean'] * df['Price_Clean'], 0.0)

df['Engagement_Rate'] = np.where(df['Installs_Clean'] > 0, (df['Reviews_Clean'] / df['Installs_Clean']) * 100, 0.0)

#Filter criteria
filtered_df = df[
    (df['Installs_Clean'] >= 10000) &
    ((df['Type'] == 'Free') | ((df['Type'] == 'Paid') & (df['Revenue'] > 10000))) &
    (df['Android_Ver_Clean'] > 4.0) &
    (df['Size_MB'] > 15) &
    (df['Content Rating'] == 'Everyone') &
    (df['App'].astype(str).str.len() <= 30)
].copy()

#  top 5 categories by total installs
top_5_categories = filtered_df.groupby('Category')['Installs_Clean'].sum().nlargest(5).index.tolist()
filtered_df = filtered_df[filtered_df['Category'].isin(top_5_categories)]

#metrics normalization
metrics = ['Installs_Clean', 'Weighted_Rating', 'Reviews_Clean', 'Size_MB', 'Revenue', 'Engagement_Rate']
metrics_display = ['Average Installs', 'Weighted Rating', 'Total Reviews', 'Average Size', 'Revenue', 'Engagement Rate']

grouped = []
categories_list = ['Overall'] + top_5_categories

for cat in categories_list:
    sub = filtered_df if cat == 'Overall' else filtered_df[filtered_df['Category'] == cat]
    
    for app_type in ['Free', 'Paid']:
        type_sub = sub[sub['Type'] == app_type]
        if len(type_sub) == 0:
            installs = weighted_rating = reviews = size = revenue = engagement = 0.0
        else:
            installs = type_sub['Installs_Clean'].mean()
            tot_reviews = type_sub['Reviews_Clean'].sum()
            weighted_rating = (type_sub['Rating_Clean'] * type_sub['Reviews_Clean']).sum() / tot_reviews if tot_reviews > 0 else 0.0
            reviews = tot_reviews
            size = type_sub['Size_MB'].mean()
            revenue = type_sub['Revenue'].mean()
            engagement = type_sub['Engagement_Rate'].mean()
        
        grouped.append({
            'Category': cat,
            'Type': app_type,
            'Installs_Clean': installs,
            'Weighted_Rating': weighted_rating,
            'Reviews_Clean': reviews,
            'Size_MB': size,
            'Revenue': revenue,
            'Engagement_Rate': engagement
        })

agg_df = pd.DataFrame(grouped)

# Percentile Normalization (0-100 scale using standard pandas)
norm_agg_df = agg_df.copy()
for m in metrics:
    norm_agg_df[m] = agg_df[m].rank(pct=True, method='min') * 100

# Calculate Composite Performance Score
norm_agg_df['Composite_Score'] = norm_agg_df[metrics].mean(axis=1)

#Radar chart
fig = go.Figure()

for idx, cat in enumerate(categories_list):
    cat_df = norm_agg_df[norm_agg_df['Category'] == cat]
    free_row = cat_df[cat_df['Type'] == 'Free'].iloc[0]
    paid_row = cat_df[cat_df['Type'] == 'Paid'].iloc[0]
    
    free_vals = [free_row[m] for m in metrics] + [free_row[metrics[0]]]
    paid_vals = [paid_row[m] for m in metrics] + [paid_row[metrics[0]]]
    theta_labels = metrics_display + [metrics_display[0]]
    
    # Trace for Free Apps
    fig.add_trace(go.Scatterpolar(
        r=free_vals,
        theta=theta_labels,
        fill='toself',
        name=f'Free Apps (Score: {free_row["Composite_Score"]:.1f})',
        visible=(cat == 'Overall'),
        line=dict(color='#1f77b4')
    ))
    
    # Trace for Paid Apps
    fig.add_trace(go.Scatterpolar(
        r=paid_vals,
        theta=theta_labels,
        fill='toself',
        name=f'Paid Apps (Score: {paid_row["Composite_Score"]:.1f})',
        visible=(cat == 'Overall'),
        line=dict(color='#ff7f0e')
    ))

# Build Dropdown Menu Options & Annotations
buttons = []
for idx, cat in enumerate(categories_list):
    cat_df = norm_agg_df[norm_agg_df['Category'] == cat]
    free_score = cat_df[cat_df['Type'] == 'Free'].iloc[0]['Composite_Score']
    paid_score = cat_df[cat_df['Type'] == 'Paid'].iloc[0]['Composite_Score']
    
    winner = "Free Apps" if free_score >= paid_score else "Paid Apps"
    diff = abs(free_score - paid_score)
    annotation_text = f"<b>Winner ({cat}):</b> {winner} (Composite Score Difference: +{diff:.1f})"
    
    visible_vector = [False] * (len(categories_list) * 2)
    visible_vector[idx * 2] = True
    visible_vector[idx * 2 + 1] = True
    
    buttons.append(dict(
        label=cat,
        method="update",
        args=[
            {"visible": visible_vector},
            {
                "title": f"Radar Comparison: Free vs Paid Apps - {cat}",
                "annotations": [{
                    "text": annotation_text,
                    "xref": "paper", "yref": "paper",
                    "x": 0.5, "y": -0.15,
                    "showarrow": False,
                    "font": dict(size=13, color="black"),
                    "align": "center",
                    "bgcolor": "#eef2f5",
                    "bordercolor": "#cccccc",
                    "borderwidth": 1
                }]
            }
        ]
    ))

# Layout configurations
overall_free = norm_agg_df[(norm_agg_df['Category']=='Overall') & (norm_agg_df['Type']=='Free')].iloc[0]['Composite_Score']
overall_paid = norm_agg_df[(norm_agg_df['Category']=='Overall') & (norm_agg_df['Type']=='Paid')].iloc[0]['Composite_Score']
overall_winner = "Free Apps" if overall_free >= overall_paid else "Paid Apps"
overall_diff = abs(overall_free - overall_paid)

fig.update_layout(
    updatemenus=[dict(
        active=0,
        buttons=buttons,
        x=1.15,
        y=1.15,
        showactive=True
    )],
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 100], title="Percentile Rank")
    ),
    title="Radar Comparison: Free vs Paid Apps - Overall",
    annotations=[{
        "text": f"<b>Winner (Overall):</b> {overall_winner} (Composite Score Difference: +{overall_diff:.1f})",
        "xref": "paper", "yref": "paper",
        "x": 0.5, "y": -0.15,
        "showarrow": False,
        "font": dict(size=13, color="black"),
        "align": "center",
        "bgcolor": "#eef2f5",
        "bordercolor": "#cccccc",
        "borderwidth": 1
    }],
    showlegend=True
)

fig.show()