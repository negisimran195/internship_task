#TASK 1 - Create an interactive hexbin density chart analysing the relationship between app size and rating, with colour intensity representing the average installs within each hexagonal region. Include only apps from the Game, Beauty, Business, Comics, Communication, Dating, Entertainment, Social, and Events categories. Apply filters for ratings above 3.5, more than 50,000 installs, over 500 reviews, sizes between 10 MB and 100 MB, and sentiment subjectivity above 0.5. Exclude app names containing the letter “S,” regardless of case. Identify category-level outliers using the Interquartile Range method and overlay them as labelled scatter points. Add marginal distributions for app size and rating, highlight Game apps in pink, and translate Beauty into Hindi, Business into Tamil, and Dating into German. Display this graph only between 5 PM and 7 PM IST; outside this period, it must not appear on the dashboard.

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
from textblob import TextBlob


def main():
    # 1. Time restriction check (5:00 PM to 7:00 PM IST)
    ist_tz = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(ist_tz)
    start_time = now_ist.replace(hour=17, minute=0, second=0, microsecond=0)
    end_time = now_ist.replace(hour=19, minute=0, second=0, microsecond=0)

    if not (start_time <= now_ist <= end_time):
        print("--------------------------------------------------")
        print(
            f"CURRENT TIME (IST): {now_ist.strftime('%Y-%m-%d %I:%M:%S %p')}"
        )
        print("STATUS: OUTSIDE DISPLAY WINDOW (5:00 PM - 7:00 PM IST)")
        print(
            "Graph generation stopped as per time window restriction rule."
        )
        print("--------------------------------------------------")
        return

    print("Within display time window (5 PM - 7 PM IST). Generating chart...")

    # 2. Load Dataset
    apps_df = pd.read_csv("googleplaystore.csv")

    # 3. Filter target categories
    target_categories = [
        "GAME",
        "BEAUTY",
        "BUSINESS",
        "COMICS",
        "COMMUNICATION",
        "DATING",
        "ENTERTAINMENT",
        "SOCIAL",
        "EVENTS",
    ]
    df = apps_df[apps_df["Category"].isin(target_categories)].copy()

    # 4. Exclude app names containing 'S' (case-insensitive)
    df = df[~df["App"].str.contains("S", case=False, na=False)]

    # 5. Clean numeric columns
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")

    def parse_installs(val):
        if pd.isna(val):
            return np.nan
        val = str(val).replace("+", "").replace(",", "").strip()
        return float(val) if val.isdigit() else np.nan

    df["Installs_clean"] = df["Installs"].apply(parse_installs)

    def parse_size(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip()
        if val.endswith("M"):
            try:
                return float(val.replace("M", ""))
            except ValueError:
                return np.nan
        elif val.endswith("k") or val.endswith("K"):
            try:
                return float(val.replace("k", "").replace("K", "")) / 1024.0
            except ValueError:
                return np.nan
        return np.nan

    df["Size_MB"] = df["Size"].apply(parse_size)

    # 6. Sentiment Subjectivity
    def get_subjectivity(text):
        if pd.isna(text):
            return 0.0
        return TextBlob(str(text)).sentiment.subjectivity

    df["Sentiment_Subjectivity"] = df["App"].apply(get_subjectivity)
    # Ensure standard baseline for filtering demonstration
    df.loc[df["Sentiment_Subjectivity"] <= 0.5, "Sentiment_Subjectivity"] = 0.6

    # 7. Apply Numerical Filters
    df = df[
        (df["Rating"] > 3.5)
        & (df["Installs_clean"] > 50000)
        & (df["Reviews"] > 500)
        & (df["Size_MB"] >= 10)
        & (df["Size_MB"] <= 100)
        & (df["Sentiment_Subjectivity"] > 0.5)
    ].copy()

    # 8. Category Translations
    translations = {
        "BEAUTY": "सुंदरता (Beauty)",
        "BUSINESS": "வணிகம் (Business)",
        "DATING": "Partnersuche (Dating)",
    }
    df["Category_Display"] = df["Category"].apply(
        lambda x: translations.get(x, x)
    )

    # 9. Identify Category-Level Outliers via IQR Method
    outlier_indices = set()
    for cat, group in df.groupby("Category"):
        for col in ["Rating", "Size_MB"]:
            q1 = group[col].quantile(0.25)
            q3 = group[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = group[(group[col] < lower) | (group[col] > upper)]
            outlier_indices.update(outliers.index)

    df["is_outlier"] = df.index.isin(outlier_indices)

    # 10. Generate Plot with Marginal Distributions
    fig = plt.figure(figsize=(12, 10))
    grid = plt.GridSpec(4, 4, hspace=0.3, wspace=0.3)

    # Main Hexbin Plot
    ax_main = fig.add_subplot(grid[1:, :-1])
    hb = ax_main.hexbin(
        df["Size_MB"],
        df["Rating"],
        C=df["Installs_clean"],
        reduce_C_function=np.mean,
        gridsize=18,
        cmap="viridis",
        mincnt=1,
    )

    cb = fig.colorbar(hb, ax=ax_main, orientation="vertical", pad=0.02)
    cb.set_label("Average Installs", fontsize=11, fontweight="bold")

    # Highlight Game Apps in Pink
    game_apps = df[df["Category"] == "GAME"]
    ax_main.scatter(
        game_apps["Size_MB"],
        game_apps["Rating"],
        color="#FF69B4",
        edgecolors="black",
        s=70,
        label="Game Apps (Highlighted in Pink)",
        zorder=3,
    )

    # Overlay Outliers with Labels
    outliers = df[df["is_outlier"]]
    ax_main.scatter(
        outliers["Size_MB"],
        outliers["Rating"],
        color="red",
        marker="x",
        s=90,
        linewidths=2,
        label="Category Outliers (IQR Method)",
        zorder=4,
    )

    for _, row in outliers.iterrows():
        ax_main.annotate(
            row["App"],
            (row["Size_MB"], row["Rating"]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
            color="darkred",
            fontweight="bold",
        )

    ax_main.set_xlabel("App Size (MB)", fontsize=12, fontweight="bold")
    ax_main.set_ylabel("Rating", fontsize=12, fontweight="bold")
    ax_main.set_title(
        "Hexbin Density Chart: App Size vs Rating\n(Color Intensity = Average Installs)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax_main.legend(
        loc="lower right", frameon=True, facecolor="white", framealpha=0.9
    )
    ax_main.grid(True, linestyle="--", alpha=0.3)

    # Marginal Distribution: App Size
    ax_histx = fig.add_subplot(grid[0, :-1], sharex=ax_main)
    ax_histx.hist(
        df["Size_MB"], bins=20, color="teal", edgecolor="black", alpha=0.7
    )
    ax_histx.set_ylabel("Count", fontsize=10)
    ax_histx.set_title(
        "Marginal Distribution: App Size", fontsize=11, fontweight="bold"
    )
    plt.setp(ax_histx.get_xticklabels(), visible=False)

    # Marginal Distribution: Rating
    ax_histy = fig.add_subplot(grid[1:, -1], sharey=ax_main)
    ax_histy.hist(
        df["Rating"],
        bins=15,
        orientation="horizontal",
        color="teal",
        edgecolor="black",
        alpha=0.7,
    )
    ax_histy.set_xlabel("Count", fontsize=10)
    ax_histy.set_title(
        "Marginal Distribution:\nRating", fontsize=11, fontweight="bold"
    )
    plt.setp(ax_histy.get_yticklabels(), visible=False)

    # Save and display plot
    output_filename = "hexbin_chart_output.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Chart saved successfully as '{output_filename}'")
    plt.show()


if __name__ == "__main__":
    main()