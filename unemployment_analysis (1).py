"""
Unemployment in India - Analysis with Python
==============================================
Data cleaning, exploration, COVID-19 impact analysis, seasonal trends,
and policy-relevant insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ----------------------------------------------------------------
# 1. Load and clean the data
# ----------------------------------------------------------------
df = pd.read_csv('/mnt/user-data/uploads/Unemployment_in_India.csv')

# Strip whitespace from column names
df.columns = [c.strip() for c in df.columns]
print("Columns:", df.columns.tolist())

# Drop fully empty rows (28 rows are all NaN)
df = df.dropna(how='all')
print(f"\nShape after dropping empty rows: {df.shape}")

# Strip whitespace from string columns
for col in ['Region', 'Date', 'Frequency', 'Area']:
    df[col] = df[col].str.strip()

# Parse dates
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Extract year and month for seasonal analysis
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['MonthName'] = df['Date'].dt.strftime('%b')

print("\nDate range:", df['Date'].min(), "to", df['Date'].max())
print("\nRegions:", df['Region'].nunique(), "->", sorted(df['Region'].unique()))
print("\nAreas:", df['Area'].unique())
print("\nFrequency:", df['Frequency'].unique())

print("\nSummary statistics:")
print(df[['Estimated Unemployment Rate (%)', 'Estimated Employed',
          'Estimated Labour Participation Rate (%)']].describe())

# ----------------------------------------------------------------
# 2. National-level monthly trend (average across regions)
# ----------------------------------------------------------------
national = df.groupby('Date').agg({
    'Estimated Unemployment Rate (%)': 'mean',
    'Estimated Labour Participation Rate (%)': 'mean'
}).reset_index()

# ----------------------------------------------------------------
# 3. COVID-19 impact analysis
# ----------------------------------------------------------------
# Define periods
pre_covid = df[df['Date'] < '2020-03-01']
covid_peak = df[(df['Date'] >= '2020-03-01') & (df['Date'] <= '2020-07-01')]
post_covid = df[df['Date'] > '2020-07-01']

print("\n" + "="*60)
print("COVID-19 IMPACT ANALYSIS")
print("="*60)
print(f"Pre-COVID (before Mar 2020) avg unemployment rate: "
      f"{pre_covid['Estimated Unemployment Rate (%)'].mean():.2f}%")
print(f"COVID Peak (Mar-Jul 2020) avg unemployment rate: "
      f"{covid_peak['Estimated Unemployment Rate (%)'].mean():.2f}%")
print(f"Post-COVID peak (after Jul 2020) avg unemployment rate: "
      f"{post_covid['Estimated Unemployment Rate (%)'].mean():.2f}%")

# Find the absolute peak month
peak_row = national.loc[national['Estimated Unemployment Rate (%)'].idxmax()]
print(f"\nHighest national avg unemployment rate: {peak_row['Estimated Unemployment Rate (%)']:.2f}% "
      f"on {peak_row['Date'].strftime('%B %Y')}")

# ----------------------------------------------------------------
# 4. Urban vs Rural comparison
# ----------------------------------------------------------------
area_comparison = df.groupby(['Date', 'Area'])['Estimated Unemployment Rate (%)'].mean().reset_index()

print("\n" + "="*60)
print("URBAN VS RURAL (Overall Average)")
print("="*60)
print(df.groupby('Area')['Estimated Unemployment Rate (%)'].mean())

# ----------------------------------------------------------------
# 5. Regional analysis - which states hit hardest during COVID
# ----------------------------------------------------------------
region_pre = pre_covid.groupby('Region')['Estimated Unemployment Rate (%)'].mean()
region_covid = covid_peak.groupby('Region')['Estimated Unemployment Rate (%)'].mean()
region_change = (region_covid - region_pre).sort_values(ascending=False)

print("\n" + "="*60)
print("TOP 5 STATES - INCREASE IN UNEMPLOYMENT (Pre-COVID -> COVID Peak)")
print("="*60)
print(region_change.head(5))

print("\nBOTTOM 5 STATES - LEAST AFFECTED")
print(region_change.tail(5))

# ----------------------------------------------------------------
# 6. Seasonal patterns (monthly averages across years)
# ----------------------------------------------------------------
monthly_avg = df.groupby('MonthName')['Estimated Unemployment Rate (%)'].mean()
month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
monthly_avg = monthly_avg.reindex(month_order)

print("\n" + "="*60)
print("MONTHLY AVERAGE UNEMPLOYMENT RATE (ALL YEARS COMBINED)")
print("="*60)
print(monthly_avg)

# ----------------------------------------------------------------
# 7. Visualizations
# ----------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(16, 16))

# (a) National trend over time
ax = axes[0, 0]
ax.plot(national['Date'], national['Estimated Unemployment Rate (%)'],
        marker='o', color='darkred', linewidth=2)
ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-07-01'),
           color='orange', alpha=0.2, label='COVID-19 Peak Period')
ax.set_title('National Average Unemployment Rate Over Time', fontsize=13, fontweight='bold')
ax.set_ylabel('Unemployment Rate (%)')
ax.legend()
ax.tick_params(axis='x', rotation=45)

# (b) Urban vs Rural trend
ax = axes[0, 1]
for area in df['Area'].unique():
    sub = area_comparison[area_comparison['Area'] == area]
    ax.plot(sub['Date'], sub['Estimated Unemployment Rate (%)'], marker='o', label=area)
ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-07-01'),
           color='orange', alpha=0.2)
ax.set_title('Urban vs Rural Unemployment Rate Over Time', fontsize=13, fontweight='bold')
ax.set_ylabel('Unemployment Rate (%)')
ax.legend()
ax.tick_params(axis='x', rotation=45)

# (c) Top 10 states by average unemployment rate
ax = axes[1, 0]
top_states = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False).head(10)
top_states.plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Top 10 States by Avg Unemployment Rate (Overall)', fontsize=13, fontweight='bold')
ax.set_xlabel('Unemployment Rate (%)')
ax.invert_yaxis()

# (d) States most affected by COVID (change in rate)
ax = axes[1, 1]
top_change = region_change.head(10)
top_change.plot(kind='barh', ax=ax, color='crimson')
ax.set_title('Top 10 States: Increase in Unemployment\n(Pre-COVID -> COVID Peak)', fontsize=13, fontweight='bold')
ax.set_xlabel('Increase in Unemployment Rate (percentage points)')
ax.invert_yaxis()

# (e) Monthly seasonal pattern
ax = axes[2, 0]
monthly_avg.plot(kind='bar', ax=ax, color='seagreen')
ax.set_title('Average Unemployment Rate by Month (Seasonal Pattern)', fontsize=13, fontweight='bold')
ax.set_ylabel('Unemployment Rate (%)')
ax.set_xlabel('Month')
ax.tick_params(axis='x', rotation=0)

# (f) Labour participation rate over time
ax = axes[2, 1]
ax.plot(national['Date'], national['Estimated Labour Participation Rate (%)'],
        marker='o', color='purple', linewidth=2)
ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-07-01'),
           color='orange', alpha=0.2, label='COVID-19 Peak Period')
ax.set_title('National Avg Labour Participation Rate Over Time', fontsize=13, fontweight='bold')
ax.set_ylabel('Labour Participation Rate (%)')
ax.legend()
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/unemployment_analysis.png', dpi=120, bbox_inches='tight')
print("\nVisualization saved as unemployment_analysis.png")

# ----------------------------------------------------------------
# 8. Save cleaned dataset
# ----------------------------------------------------------------
df.to_csv('/mnt/user-data/outputs/Unemployment_cleaned.csv', index=False)
print("Cleaned dataset saved as Unemployment_cleaned.csv")
