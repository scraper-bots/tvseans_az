import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 11

# Load data
df = pd.read_csv('tvseans_movies.csv', encoding='utf-8-sig')

# Create charts folder
import os
os.makedirs('charts', exist_ok=True)

print(f"Generating charts for {len(df)} movies...")

# CHART 1: Top 15 Genres Distribution
print("1. Genre distribution...")
genres = []
for g in df['genres'].dropna():
    genres.extend([x.strip() for x in str(g).split(',')])
genre_counts = Counter(genres).most_common(15)

plt.figure(figsize=(12, 7))
names, counts = zip(*genre_counts)
plt.barh(range(len(names)), counts, color='steelblue')
plt.yticks(range(len(names)), names)
plt.xlabel('Number of Movies')
plt.title('Top 15 Genres - Content Distribution', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
for i, v in enumerate(counts):
    plt.text(v + 10, i, str(v), va='center')
plt.tight_layout()
plt.savefig('charts/1_genre_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# CHART 2: Top 10 Countries
print("2. Country distribution...")
countries = []
for c in df['country'].dropna():
    countries.extend([x.strip() for x in str(c).split(',')])
country_counts = Counter(countries).most_common(10)

plt.figure(figsize=(12, 7))
names, counts = zip(*country_counts)
colors = plt.cm.Paired(np.linspace(0, 1, len(names)))
plt.bar(range(len(names)), counts, color=colors)
plt.xticks(range(len(names)), names, rotation=45, ha='right')
plt.ylabel('Number of Movies')
plt.title('Top 10 Production Countries', fontsize=14, fontweight='bold')
for i, v in enumerate(counts):
    plt.text(i, v + 20, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('charts/2_country_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# CHART 3: Rating Distribution (IMDb)
print("3. Rating distribution...")
df['imdb_num'] = pd.to_numeric(df['rating_imdb'], errors='coerce')
ratings = df['imdb_num'].dropna()

plt.figure(figsize=(12, 7))
plt.hist(ratings, bins=30, color='coral', edgecolor='black', alpha=0.7)
plt.axvline(ratings.mean(), color='red', linestyle='--', linewidth=2, label=f'Average: {ratings.mean():.2f}')
plt.axvline(ratings.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {ratings.median():.2f}')
plt.xlabel('IMDb Rating')
plt.ylabel('Number of Movies')
plt.title('IMDb Rating Distribution', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('charts/3_rating_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# CHART 4: Movies by Year (2010-2025)
print("4. Year trend...")
year_counts = df['year'].value_counts().sort_index()
years = [str(y) for y in range(2010, 2026)]
counts = [year_counts.get(y, 0) for y in years]

plt.figure(figsize=(12, 7))
plt.plot(years, counts, marker='o', linewidth=2, markersize=8, color='darkgreen')
plt.fill_between(range(len(years)), counts, alpha=0.3, color='lightgreen')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.title('Movie Release Trend (2010-2025)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('charts/4_year_trend.png', dpi=150, bbox_inches='tight')
plt.close()

# CHART 5: Top 15 Directors
print("5. Top directors...")
directors = []
for d in df['director'].dropna():
    dstr = str(d).strip()
    if dstr and dstr != 'nan':
        directors.extend([x.strip() for x in dstr.split(',')])
director_counts = Counter(directors).most_common(15)

plt.figure(figsize=(12, 7))
names, counts = zip(*director_counts)
plt.barh(range(len(names)), counts, color='teal')
plt.yticks(range(len(names)), names)
plt.xlabel('Number of Movies')
plt.title('Top 15 Most Prolific Directors', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
for i, v in enumerate(counts):
    plt.text(v + 0.2, i, str(v), va='center')
plt.tight_layout()
plt.savefig('charts/5_top_directors.png', dpi=150, bbox_inches='tight')
plt.close()

# CHART 6: Top 15 Actors
print("6. Top actors...")
actors = []
for a in df['actors'].dropna():
    astr = str(a).strip()
    if astr and astr != 'nan':
        actors.extend([x.strip() for x in astr.split(',')])
actor_counts = Counter(actors).most_common(15)

plt.figure(figsize=(12, 7))
names, counts = zip(*actor_counts)
plt.barh(range(len(names)), counts, color='indigo')
plt.yticks(range(len(names)), names)
plt.xlabel('Number of Movies')
plt.title('Top 15 Most Featured Actors', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
for i, v in enumerate(counts):
    plt.text(v + 0.3, i, str(v), va='center')
plt.tight_layout()
plt.savefig('charts/6_top_actors.png', dpi=150, bbox_inches='tight')
plt.close()

# CHART 7: Age Rating Distribution
print("7. Age rating distribution...")
age_rating_counts = df['age_rating'].value_counts().head(10)

plt.figure(figsize=(10, 7))
colors = plt.cm.Spectral(np.linspace(0, 1, len(age_rating_counts)))
plt.pie(age_rating_counts.values, labels=age_rating_counts.index, autopct='%1.1f%%',
        startangle=90, colors=colors, textprops={'fontsize': 11})
plt.title('Content Age Rating Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/7_age_rating_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# CHART 8: Rating Comparison (TvSeans vs IMDb vs Kinopoisk)
print("8. Rating platform comparison...")
df['tvseans_num'] = pd.to_numeric(df['rating_tvseans'], errors='coerce')
df['kinopoisk_num'] = pd.to_numeric(df['rating_kinopoisk'], errors='coerce')

platforms = ['TvSeans', 'IMDb', 'Kinopoisk']
averages = [
    df['tvseans_num'].mean(),
    df['imdb_num'].mean(),
    df['kinopoisk_num'].mean()
]

plt.figure(figsize=(10, 7))
bars = plt.bar(platforms, averages, color=['#05c7f2', '#f6c700', '#f50'])
plt.ylabel('Average Rating')
plt.title('Average Ratings Across Platforms', fontsize=14, fontweight='bold')
plt.ylim(0, 10)
for bar, avg in zip(bars, averages):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{avg:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/8_platform_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# CHART 9: User Engagement (Comments)
print("9. Engagement distribution...")
df['comments_num'] = pd.to_numeric(df['comments_count'], errors='coerce')
comments = df['comments_num'].dropna()

plt.figure(figsize=(12, 7))
plt.hist(comments, bins=30, color='purple', edgecolor='black', alpha=0.7)
plt.axvline(comments.mean(), color='red', linestyle='--', linewidth=2, label=f'Average: {comments.mean():.1f}')
plt.axvline(comments.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {comments.median():.1f}')
plt.xlabel('Number of Comments')
plt.ylabel('Number of Movies')
plt.title('User Engagement Distribution (Comments per Movie)', fontsize=14, fontweight='bold')
plt.legend()
plt.xlim(0, 60)
plt.tight_layout()
plt.savefig('charts/9_engagement_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# CHART 10: Content Quality by Genre (Top 10)
print("10. Quality by genre...")
top_genres = [g[0] for g in genre_counts[:10]]
genre_ratings = {}

for genre in top_genres:
    genre_movies = df[df['genres'].str.contains(genre, na=False)]
    avg_rating = genre_movies['imdb_num'].mean()
    genre_ratings[genre] = avg_rating

sorted_genres = sorted(genre_ratings.items(), key=lambda x: x[1], reverse=True)
genres_list, ratings_list = zip(*sorted_genres)

plt.figure(figsize=(12, 7))
bars = plt.barh(range(len(genres_list)), ratings_list, color='darkorange')
plt.yticks(range(len(genres_list)), genres_list)
plt.xlabel('Average IMDb Rating')
plt.title('Content Quality by Genre (Top 10)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
for i, v in enumerate(ratings_list):
    plt.text(v + 0.05, i, f'{v:.2f}', va='center')
plt.xlim(0, 10)
plt.tight_layout()
plt.savefig('charts/10_quality_by_genre.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✓ Generated 10 charts in 'charts/' folder")
