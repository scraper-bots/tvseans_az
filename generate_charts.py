import os
import textwrap
from math import ceil
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
# ---------------------------
# Configuration / Environment
# ---------------------------
# Visual style (formatting only)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'legend.fontsize': 10,
})

# I/O
INPUT_CSV = 'tvseans_movies.csv'
OUTPUT_DIR = 'charts'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Utility helpers (formatting only)
# ---------------------------

def safe_series(df, *candidates):
    """Return the first matching series from df for the given candidate column names.
    If none present, returns an empty float64 Series.
    This function does NOT mutate the dataframe."""
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series(dtype='float64')


def to_numeric_local(series, **kwargs):
    """Convert a Series to numeric safely and return the converted Series (local variable)."""
    return pd.to_numeric(series, **kwargs)


def fd_bins(data):
    """Freedman–Diaconis bin count, fallback to 30 when degenerate."""
    if len(data) < 2:
        return 10
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return 30
    h = 2 * iqr / (len(data) ** (1/3))
    if h <= 0:
        return 30
    bins = int(ceil((data.max() - data.min()) / h))
    return max(10, min(bins, 100))


# ---------------------------
# Load data (original CSV; no mutation intended)
# ---------------------------
print(f"Loading data from '{INPUT_CSV}'")
df = pd.read_csv(INPUT_CSV, encoding='utf-8-sig')
print(f"Dataset contains {len(df)} rows and {len(df.columns)} columns")

# ---------------------------
# CHART 1: Top 15 Genres Distribution
# ---------------------------
print('1. Genre distribution...')
genres = []
for g in df.get('genres', pd.Series(dtype='object')).dropna():
    genres.extend([x.strip() for x in str(g).split(',') if x.strip()])

genre_counts = Counter(genres).most_common(15)
if genre_counts:
    names, counts = zip(*genre_counts)
else:
    names, counts = (), ()

fig, ax = plt.subplots(figsize=(12, 7))
ax.barh(range(len(names)), counts, color='steelblue', edgecolor='black', linewidth=0.6)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
ax.set_xlabel('Number of Movies')
ax.set_title('Top 15 Genres - Content Distribution')
ax.invert_yaxis()
for i, v in enumerate(counts):
    ax.text(v + max(1, max(counts) * 0.01), i, str(v), va='center')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '1_genre_distribution.png'), bbox_inches='tight')
plt.close(fig)

# ---------------------------
# CHART 2: Top 10 Countries
# ---------------------------
print('2. Country distribution...')
countries = []
for c in df.get('country', pd.Series(dtype='object')).dropna():
    countries.extend([x.strip() for x in str(c).split(',') if x.strip()])

country_counts = Counter(countries).most_common(10)
if country_counts:
    names, counts = zip(*country_counts)
else:
    names, counts = (), ()

fig, ax = plt.subplots(figsize=(12, 7))
colors = plt.cm.Paired(np.linspace(0, 1, max(1, len(names))))
ax.bar(range(len(names)), counts, color=colors, edgecolor='black')
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha='right')
ax.set_ylabel('Number of Movies')
ax.set_title('Top 10 Production Countries')
for i, v in enumerate(counts):
    ax.text(i, v + max(1, max(counts) * 0.02), str(v), ha='center', fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '2_country_distribution.png'), bbox_inches='tight')
plt.close(fig)

# ---------------------------
# CHART 3: Rating Distribution (IMDb)
# ---------------------------
print('3. Rating distribution...')
imdb_series = safe_series(df, 'rating_imdb', 'imdb', 'imdb_num')
imdb = to_numeric_local(imdb_series, errors='coerce').dropna()

fig, ax = plt.subplots(figsize=(12, 7))
if len(imdb) > 0:
    bins = fd_bins(imdb)
    ax.hist(imdb, bins=bins, edgecolor='black', alpha=0.8, color='coral')
    ax.axvline(imdb.mean(), color='red', linestyle='--', linewidth=2, label=f'Average: {imdb.mean():.2f}')
    ax.axvline(imdb.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {imdb.median():.2f}')
ax.set_xlabel('IMDb Rating')
ax.set_ylabel('Number of Movies')
ax.set_title('IMDb Rating Distribution')
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '3_rating_distribution.png'), bbox_inches='tight')
plt.close(fig)

# ---------------------------
# CHART 4: Movies by Year (2010-2025)
# ---------------------------
print('4. Year trend...')
year_series = to_numeric_local(safe_series(df, 'year'), errors='coerce').dropna().astype(int)
counts_by_year = year_series.value_counts().to_dict()
years = list(range(2010, 2026))
counts = [counts_by_year.get(y, 0) for y in years]

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(years, counts, marker='o', linewidth=2, markersize=6, color='darkgreen')
ax.fill_between(years, counts, alpha=0.25, color='lightgreen')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Movies')
ax.set_title('Movie Release Trend (2010-2025)')
ax.set_xticks(years)
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '4_year_trend.png'), bbox_inches='tight')
plt.close(fig)

# ---------------------------
# CHART 5: Top 15 Directors
# ---------------------------
print('5. Top directors...')
directors = []
for d in df.get('director', pd.Series(dtype='object')).dropna():
    s = str(d).strip()
    if s and s.lower() != 'nan':
        directors.extend([x.strip() for x in s.split(',') if x.strip()])

director_counts = Counter(directors).most_common(15)
if director_counts:
    names, counts = zip(*director_counts)
else:
    names, counts = (), ()

fig, ax = plt.subplots(figsize=(12, 7))
ax.barh(range(len(names)), counts, color='teal', edgecolor='black')
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
ax.set_xlabel('Number of Movies')
ax.set_title('Top 15 Most Prolific Directors')
ax.invert_yaxis()
for i, v in enumerate(counts):
    ax.text(v + max(1, max(counts) * 0.01), i, str(v), va='center')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '5_top_directors.png'), bbox_inches='tight')
plt.close(fig)

# ---------------------------
# CHART 6: Top 15 Actors
# ---------------------------
print('6. Top actors...')
actors = []
for a in df.get('actors', pd.Series(dtype='object')).dropna():
    s = str(a).strip()
    if s and s.lower() != 'nan':
        actors.extend([x.strip() for x in s.split(',') if x.strip()])

actor_counts = Counter(actors).most_common(15)
if actor_counts:
    names, counts = zip(*actor_counts)
else:
    names, counts = (), ()

fig, ax = plt.subplots(figsize=(12, 7))
ax.barh(range(len(names)), counts, color='indigo', edgecolor='black')
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
ax.set_xlabel('Number of Movies')
ax.set_title('Top 15 Most Featured Actors')
ax.invert_yaxis()
for i, v in enumerate(counts):
    ax.text(v + max(1, max(counts) * 0.01), i, str(v), va='center')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '6_top_actors.png'), bbox_inches='tight')
plt.close(fig)

# ---------------------------
# CHART 7: Age Rating Distribution (bar chart)
# (formatting only; replaced pie chart with vertical bar chart)
# ---------------------------
print('7. Age rating distribution...')
age_counts = df.get('age_rating', pd.Series(dtype='object')).value_counts().head(10)
labels = age_counts.index.tolist()
counts = age_counts.values

fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(labels))))
ax.bar(range(len(labels)), counts, color=colors, edgecolor='black', linewidth=0.6)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels([textwrap.fill(str(lbl), 20) for lbl in labels], rotation=45, ha='right')
ax.set_ylabel('Number of Movies')
ax.set_title('Content Age Rating Distribution (Top 10)')
# Add value labels above bars
for i, v in enumerate(counts):
    ax.text(i, v + max(1, max(counts) * 0.02), str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '7_age_rating_distribution.png'), bbox_inches='tight')
plt.close(fig)
# ---------------------------
# CHART 8: Rating Comparison (TvSeans vs IMDb vs Kinopoisk)
# (use local numeric conversions only)
# ---------------------------
print('8. Rating platform comparison...')
tv = to_numeric_local(safe_series(df, 'rating_tvseans', 'tvseans'), errors='coerce')
imdb = to_numeric_local(safe_series(df, 'rating_imdb', 'imdb', 'imdb_num'), errors='coerce')
kp = to_numeric_local(safe_series(df, 'rating_kinopoisk', 'kinopoisk', 'kinopoisk_num'), errors='coerce')

platforms = ['TvSeans', 'IMDb', 'Kinopoisk']
avgs = [tv.mean(), imdb.mean(), kp.mean()]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.bar(platforms, avgs, edgecolor='black', linewidth=0.8, color=['#05c7f2', '#f6c700', '#f50'])
ax.set_ylabel('Average Rating')
ax.set_title('Average Ratings Across Platforms')
ax.set_ylim(0, 10)
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
for bar, avg in zip(bars, avgs):
    if np.isnan(avg):
        label = 'n/a'
        y = 0
    else:
        label = f'{avg:.2f}'
        y = avg
    ax.text(bar.get_x() + bar.get_width() / 2., y + 0.08, label, ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '8_platform_comparison.png'), bbox_inches='tight')
plt.close(fig)

# ---------------------------
# CHART 9: User Engagement (Comments)
# ---------------------------
print('9. Engagement distribution...')
comments = to_numeric_local(safe_series(df, 'comments_count', 'comments', 'comments_num'), errors='coerce').dropna()

fig, ax = plt.subplots(figsize=(12, 7))
if len(comments) > 0:
    bins = fd_bins(comments)
    ax.hist(comments, bins=bins, edgecolor='black', alpha=0.75)
    mean_val = comments.mean()
    median_val = comments.median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Average: {mean_val:.1f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
ax.set_xlabel('Number of Comments')
ax.set_ylabel('Number of Movies')
ax.set_title('User Engagement Distribution (Comments per Movie)')
# Use a robust right limit (99th percentile) but keep at least 60 for readability
if len(comments) > 0:
    right = max(comments.quantile(0.99), comments.max())
    ax.set_xlim(left=0, right=max(60, right))
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '9_engagement_distribution.png'), bbox_inches='tight')
plt.close(fig)

# ---------------------------
# CHART 10: Content Quality by Genre (Top 10)
# ---------------------------
print('10. Quality by genre...')
# Reuse previously computed top genres (no mutation)
top_genres = [g for g, _ in genre_counts[:10]] if 'genre_counts' in globals() else []

genre_ratings = {}
for genre in top_genres:
    mask = df.get('genres', pd.Series(dtype='object')).fillna('').str.contains(fr'\b{re.escape(genre)}\b', regex=True)
    candidate = to_numeric_local(df.loc[mask, safe_series(df, 'rating_imdb', 'imdb', 'imdb_num').name], errors='coerce') if 'rating_imdb' in df.columns else to_numeric_local(safe_series(df, 'rating_imdb', 'imdb', 'imdb_num'), errors='coerce')
    # fallback: compute mean from a converted series directly
    if isinstance(candidate, pd.Series):
        avg_rating = candidate.mean()
    else:
        # if logic above cannot find a series, compute by converting 'rating_imdb' column directly
        avg_rating = to_numeric_local(safe_series(df, 'rating_imdb'), errors='coerce').loc[mask].mean()
    genre_ratings[genre] = avg_rating

# sort and plot
sorted_genres = sorted(genre_ratings.items(), key=lambda x: (x[1] if x[1] is not None else -np.inf), reverse=True)
if sorted_genres:
    genres_list, ratings_list = zip(*sorted_genres)
else:
    genres_list, ratings_list = (), ()

fig, ax = plt.subplots(figsize=(12, 7))
ax.barh(range(len(genres_list)), ratings_list, color='darkorange', edgecolor='black')
ax.set_yticks(range(len(genres_list)))
ax.set_yticklabels(genres_list)
ax.set_xlabel('Average IMDb Rating')
ax.set_title('Content Quality by Genre (Top 10)')
ax.invert_yaxis()
for i, v in enumerate(ratings_list):
    if v is not None and not (isinstance(v, float) and np.isnan(v)):
        ax.text(v + 0.05, i, f'{v:.2f}', va='center')
ax.set_xlim(0, 10)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '10_quality_by_genre.png'), bbox_inches='tight')
plt.close(fig)

print(f"\n\u2713 Generated charts in '{OUTPUT_DIR}/' folder")
