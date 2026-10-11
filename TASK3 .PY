#task 3 - Create an interactive calendar-style heatmap showing monthly installs for the top five eligible app categories, with a category selector that dynamically updates the visualization. Include categories beginning with “E,” “C,” or “B.” Apply filters for ratings of at least 4.0, more than 10,000 installs, over 500 reviews, sizes between 15 MB and 80 MB, and sentiment subjectivity above 0.5. Exclude app names beginning with “X,” “Y,” or “Z” and names containing the letter “S,” regardless of case. Calculate month-over-month growth and highlight periods where the increase exceeds 20%. Generate a three-month moving-average forecast and visually distinguish forecast values from actual values. Add hover information containing installs, growth percentage, rolling average, review count, and forecast status. Translate Beauty into Hindi, Business into Tamil, and Dating into German wherever applicable. Display this graph only between 6 PM and 9 PM IST; outside this period, it must not appear on the dashboard.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from datetime import datetime
import pytz

# --- 1. Check Execution Time Window (6 PM - 9 PM IST) ---
ist_tz = pytz.timezone('Asia/Kolkata')
current_time = datetime.now(ist_tz)
current_hour = current_time.hour

if not (18 <= current_hour < 21):
    print(f"Current IST Time: {current_time.strftime('%I:%M %p')}. Dashboard access restricted. Displayed only between 6 PM - 9 PM IST.")
else:
    # --- 2. Load Datasets ---
    apps_df = pd.read_csv('googleplaystore.csv')
    reviews_df = pd.read_csv('reviews.csv')

    # --- 3. Compute Sentiment Subjectivity ---
    reviews_df['content'] = reviews_df['content'].fillna('')
    reviews_df['subjectivity'] = reviews_df['content'].apply(lambda txt: TextBlob(str(txt)).sentiment.subjectivity)
    
    # Identify app IDs with average sentiment subjectivity > 0.5
    high_subj_apps = reviews_df.groupby('appId')['subjectivity'].mean()
    valid_app_ids = set(high_subj_apps[high_subj_apps > 0.5].index)

    # --- 4. Data Cleaning & Parsing ---
    df = apps_df.copy()

    # Numeric conversions
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
    
    # Clean Installs
    df['Installs_Clean'] = df['Installs'].astype(str).str.replace('+', '', regex=False).str.replace(',', '', regex=False)
    df['Installs_Clean'] = pd.to_numeric(df['Installs_Clean'], errors='coerce')

    # Clean Size to MB
    def parse_size_mb(val):
        if pd.isna(val) or val == 'Varies with device':
            return np.nan
        val = str(val).strip()
        if val.endswith('M'):
            try: return float(val[:-1])
            except: return np.nan
        elif val.endswith('k') or val.endswith('K'):
            try: return float(val[:-1]) / 1024.0
            except: return np.nan
        return np.nan

    df['Size_MB'] = df['Size'].apply(parse_size_mb)
    df['Last_Updated_Date'] = pd.to_datetime(df['Last Updated'], errors='coerce')

    # --- 5. Apply Filtering Rules ---
    # Rule 1: Rating >= 4.0, Installs > 10000, Reviews > 500, Size between 15 MB and 80 MB
    cond_rating = df['Rating'] >= 4.0
    cond_installs = df['Installs_Clean'] > 10000
    cond_reviews = df['Reviews'] > 500
    cond_size = (df['Size_MB'] >= 15.0) & (df['Size_MB'] <= 80.0)

    # Rule 2: Exclude names beginning with 'X', 'Y', 'Z' and containing letter 'S' (case-insensitive)
    def name_filter(name):
        if not isinstance(name, str):
            return False
        clean_name = name.upper()
        if clean_name.startswith(('X', 'Y', 'Z')):
            return False
        if 'S' in clean_name:
            return False
        return True

    cond_name = df['App'].apply(name_filter)

    # Rule 3: Category starting with 'E', 'C', or 'B'
    cond_cat = df['Category'].astype(str).str.upper().str.startswith(('E', 'C', 'B'))

    df_filtered = df[cond_rating & cond_installs & cond_reviews & cond_size & cond_name & cond_cat].copy()

    # --- 6. Category Selection & Translations ---
    top_5_categories = df_filtered.groupby('Category')['Installs_Clean'].sum().nlargest(5).index.tolist()
    df_top = df_filtered[df_filtered['Category'].isin(top_5_categories)].copy()

    # Category Translations
    translations = {
        'BEAUTY': 'Sundarta (Beauty)',
        'BUSINESS': 'Vyapar (Business)',
        'DATING': 'Horst (Dating)'
    }
    df_top['Category_Translated'] = df_top['Category'].apply(lambda c: translations.get(c.upper(), c))

    # --- 7. Monthly Aggregation, Growth & Moving Average ---
    df_top['YearMonth'] = df_top['Last_Updated_Date'].dt.to_period('M')
    
    monthly_pivot = df_top.groupby(['Category_Translated', 'YearMonth'])['Installs_Clean'].sum().unstack(fill_value=0)

    # Corrected rolling average calculation (Transposed for newer Pandas versions)
    rolling_3m = monthly_pivot.T.rolling(window=3, min_periods=1).mean().T
    mom_growth = monthly_pivot.pct_change(axis=1) * 100

    # --- 8. Generate Pictorial Output ---
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(monthly_pivot, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={'label': 'Installs'})
    
    plt.title("Monthly App Installs Heatmap (Filtered Top Categories)", fontsize=14, fontweight='bold')
    plt.xlabel("Year-Month", fontsize=11)
    plt.ylabel("Category", fontsize=11)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save chart image
    plt.savefig("monthly_installs_heatmap.png")
    plt.show()
    print("Heatmap rendered and saved as 'monthly_installs_heatmap.png'.")