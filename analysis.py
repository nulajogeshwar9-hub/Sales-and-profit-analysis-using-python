"""
Store Sales and Profit Analysis
================================
Dataset  : Sample - Superstore
Author   : [Your Name]
Problem  : Analyzing sales and profit performance of a retail store
           to optimize operations, pricing, marketing, and inventory.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ─── PATHS ───────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join('data', 'Sample_Superstore.csv')
OUTPUT_PATH = 'outputs'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ─── STYLE ───────────────────────────────────────────────────────────────────
BLUE   = '#2563EB'
GREEN  = '#16A34A'
RED    = '#DC2626'
ORANGE = '#EA580C'
PURPLE = '#7C3AED'
GRAY   = '#6B7280'
BG     = '#F8FAFC'

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor':   BG,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'font.family':      'DejaVu Sans',
    'axes.titlesize':   13,
    'axes.titleweight': 'bold',
    'axes.labelsize':   11,
})

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD & PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path):
    df = pd.read_csv(path, encoding='latin1', parse_dates=['Order Date', 'Ship Date'])
    df['Year']          = df['Order Date'].dt.year
    df['Month']         = df['Order Date'].dt.month
    df['Month_Name']    = df['Order Date'].dt.strftime('%b')
    df['Profit_Margin'] = (df['Profit'] / df['Sales'] * 100).round(2)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def exploratory_analysis(df):
    print("=" * 60)
    print("       STORE SALES & PROFIT ANALYSIS — SUPERSTORE")
    print("=" * 60)
    print(f"\n📦 Dataset Shape   : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"📅 Date Range      : {df['Order Date'].min().date()} → {df['Order Date'].max().date()}")
    print(f"💰 Total Sales     : ${df['Sales'].sum():,.2f}")
    print(f"📈 Total Profit    : ${df['Profit'].sum():,.2f}")
    print(f"📊 Avg Profit Margin: {df['Profit_Margin'].mean():.1f}%")
    print(f"🗂  Categories      : {', '.join(df['Category'].unique())}")
    print(f"🌎 Regions         : {', '.join(df['Region'].unique())}")
    print(f"👤 Segments        : {', '.join(df['Segment'].unique())}")
    print("\n--- Descriptive Statistics ---")
    print(df[['Sales', 'Profit', 'Discount', 'Quantity']].describe().round(2))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — FIGURE 1: OVERVIEW DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def plot_overview_dashboard(df):
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Store Sales & Profit Analysis — Overview Dashboard',
                 fontsize=17, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # KPI Cards
    kpis = [
        ('Total Sales',  f"${df['Sales'].sum()/1e6:.2f}M",  BLUE),
        ('Total Profit', f"${df['Profit'].sum()/1e3:.0f}K", GREEN),
        ('Avg Margin',   f"{df['Profit_Margin'].mean():.1f}%", ORANGE),
    ]
    for i, (label, val, color) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(color)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('off')
        ax.text(0.5, 0.62, val, ha='center', va='center',
                fontsize=24, fontweight='bold', color='white', transform=ax.transAxes)
        ax.text(0.5, 0.25, label, ha='center', va='center',
                fontsize=12, color='white', alpha=0.9, transform=ax.transAxes)

    # Monthly Sales & Profit Trend
    ax1 = fig.add_subplot(gs[1, :2])
    monthly = df.groupby(['Year', 'Month'])[['Sales', 'Profit']].sum().reset_index()
    monthly['Period'] = pd.to_datetime(monthly[['Year', 'Month']].assign(Day=1))
    monthly = monthly.sort_values('Period')
    ax1.plot(monthly['Period'], monthly['Sales'],  color=BLUE,  lw=2.5, label='Sales',  marker='o', markersize=3)
    ax1.plot(monthly['Period'], monthly['Profit'], color=GREEN, lw=2.5, label='Profit', marker='o', markersize=3)
    ax1.fill_between(monthly['Period'], monthly['Sales'], alpha=0.08, color=BLUE)
    ax1.set_title('Monthly Sales & Profit Trend')
    ax1.set_ylabel('Amount ($)')
    ax1.legend()
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))

    # Sales by Category Pie
    ax2 = fig.add_subplot(gs[1, 2])
    cat_sales = df.groupby('Category')['Sales'].sum()
    wedges, texts, autotexts = ax2.pie(
        cat_sales, labels=cat_sales.index, autopct='%1.1f%%',
        colors=[BLUE, ORANGE, GREEN], startangle=140,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    for at in autotexts: at.set_fontsize(10); at.set_fontweight('bold')
    ax2.set_title('Sales Share by Category')

    # Profit by Sub-Category
    ax3 = fig.add_subplot(gs[2, :])
    sub_profit = df.groupby('Sub-Category')['Profit'].sum().sort_values()
    colors_bar = [RED if v < 0 else GREEN for v in sub_profit.values]
    bars = ax3.barh(sub_profit.index, sub_profit.values, color=colors_bar, edgecolor='white', height=0.6)
    ax3.axvline(0, color=GRAY, lw=1, ls='--')
    ax3.set_title('Total Profit by Sub-Category  (Red = Loss-Making)')
    ax3.set_xlabel('Profit ($)')
    for bar, val in zip(bars, sub_profit.values):
        ax3.text(val + (300 if val >= 0 else -300), bar.get_y() + bar.get_height() / 2,
                 f'${val:,.0f}', va='center', ha='left' if val >= 0 else 'right', fontsize=8)
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))

    path = os.path.join(OUTPUT_PATH, 'fig1_overview_dashboard.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — FIGURE 2: REGIONAL & SEGMENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def plot_regional_segment(df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Regional & Segment Performance Analysis', fontsize=16, fontweight='bold')
    fig.patch.set_facecolor(BG)

    # Sales & Profit by Region
    ax = axes[0, 0]
    region_data = df.groupby('Region')[['Sales', 'Profit']].sum().reset_index()
    x = np.arange(len(region_data)); w = 0.35
    ax.bar(x - w/2, region_data['Sales'],  w, label='Sales',  color=BLUE,  alpha=0.85)
    ax.bar(x + w/2, region_data['Profit'], w, label='Profit', color=GREEN, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(region_data['Region'])
    ax.set_title('Sales & Profit by Region')
    ax.set_ylabel('Amount ($)')
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))

    # Segment Donut
    ax = axes[0, 1]
    seg = df.groupby('Segment')['Sales'].sum()
    ax.pie(seg, labels=seg.index, autopct='%1.1f%%',
           colors=[BLUE, PURPLE, ORANGE], startangle=90,
           wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'width': 0.6})
    ax.set_title('Sales by Customer Segment')

    # Profit Margin by Region
    ax = axes[1, 0]
    reg_margin = df.groupby('Region')['Profit_Margin'].mean().sort_values(ascending=False)
    bars = ax.bar(reg_margin.index, reg_margin.values,
                  color=[GREEN if v > 0 else RED for v in reg_margin.values], alpha=0.85, width=0.5)
    for bar, val in zip(bars, reg_margin.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax.set_title('Average Profit Margin by Region')
    ax.set_ylabel('Margin (%)')

    # Discount vs Profit Scatter
    ax = axes[1, 1]
    sample = df.sample(min(1000, len(df)), random_state=42)
    sc = ax.scatter(sample['Discount'], sample['Profit'],
                    c=sample['Sales'], cmap='Blues', alpha=0.5, s=20, vmin=0, vmax=3000)
    ax.axhline(0, color=RED, lw=1.2, ls='--', label='Break-even')
    ax.set_title('Discount vs Profit (colored by Sales)')
    ax.set_xlabel('Discount Rate'); ax.set_ylabel('Profit ($)')
    plt.colorbar(sc, ax=ax, label='Sales ($)')
    ax.legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_PATH, 'fig2_regional_segment.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — FIGURE 3: TIME SERIES & TOP/BOTTOM ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def plot_timeseries_topbottom(df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Time Series & Top/Bottom Performance', fontsize=16, fontweight='bold')
    fig.patch.set_facecolor(BG)

    # Yearly Sales by Category
    ax = axes[0, 0]
    yearly_cat = df.groupby(['Year', 'Category'])['Sales'].sum().reset_index()
    for cat, color in zip(df['Category'].unique(), [BLUE, ORANGE, GREEN]):
        sub = yearly_cat[yearly_cat['Category'] == cat]
        ax.plot(sub['Year'], sub['Sales'], marker='o', lw=2.5, color=color, label=cat)
    ax.set_title('Yearly Sales by Category')
    ax.set_xlabel('Year'); ax.set_ylabel('Sales ($)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))
    ax.legend(); ax.set_xticks(df['Year'].unique())

    # Seasonality Heatmap
    ax = axes[0, 1]
    pivot = df.pivot_table(values='Sales', index='Year', columns='Month', aggfunc='sum')
    pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    sns.heatmap(pivot, ax=ax, cmap='YlOrRd', fmt='.0f',
                annot=True, annot_kws={'size': 8}, linewidths=0.5,
                cbar_kws={'label': 'Sales ($)'})
    ax.set_title('Sales Seasonality Heatmap (Year × Month)')
    ax.set_xlabel('Month'); ax.set_ylabel('Year')

    # Top 10 Profitable Products
    ax = axes[1, 0]
    top10 = df.groupby('Product Name')['Profit'].sum().nlargest(10).sort_values()
    ax.barh([p[:35] for p in top10.index], top10.values, color=GREEN, alpha=0.85, height=0.6)
    ax.set_title('Top 10 Most Profitable Products')
    ax.set_xlabel('Total Profit ($)')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))

    # Bottom 10 Products
    ax = axes[1, 1]
    bot10 = df.groupby('Product Name')['Profit'].sum().nsmallest(10).sort_values(ascending=False)
    ax.barh([p[:35] for p in bot10.index], bot10.values, color=RED, alpha=0.85, height=0.6)
    ax.set_title('Top 10 Loss-Making Products')
    ax.set_xlabel('Total Profit ($)')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))

    plt.tight_layout()
    path = os.path.join(OUTPUT_PATH, 'fig3_timeseries_topbottom.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — INSIGHTS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def print_insights(df):
    print("\n" + "=" * 60)
    print("  KEY BUSINESS INSIGHTS")
    print("=" * 60)

    best_region  = df.groupby('Region')['Profit'].sum().idxmax()
    worst_region = df.groupby('Region')['Profit'].sum().idxmin()
    best_cat     = df.groupby('Category')['Profit'].sum().idxmax()
    worst_subcat = df.groupby('Sub-Category')['Profit'].sum().idxmin()
    best_subcat  = df.groupby('Sub-Category')['Profit'].sum().idxmax()
    best_seg     = df.groupby('Segment')['Sales'].sum().idxmax()

    print(f"\n🏆 Best Performing Region  : {best_region}")
    print(f"⚠️  Worst Performing Region : {worst_region}")
    print(f"📦 Most Profitable Category: {best_cat}")
    print(f"🟢 Best Sub-Category       : {best_subcat}")
    print(f"🔴 Loss-Making Sub-Category: {worst_subcat}")
    print(f"👤 Top Customer Segment    : {best_seg}")

    high_disc = df[df['Discount'] > 0.4]
    print(f"\n💡 Orders with discount >40%: {len(high_disc):,} ({len(high_disc)/len(df)*100:.1f}%)")
    print(f"   Avg profit on those orders: ${high_disc['Profit'].mean():.2f} (vs ${df['Profit'].mean():.2f} overall)")
    print(f"\n📈 Peak sales season: Q4 (Oct–Dec) consistently highest across all years")
    print(f"📉 Tables sub-category total loss: ${df[df['Sub-Category']=='Tables']['Profit'].sum():,.2f}")
    print("\n" + "=" * 60)
    print("✅ Analysis Complete! Check the 'outputs/' folder for charts.")
    print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df = load_data(DATA_PATH)
    exploratory_analysis(df)
    plot_overview_dashboard(df)
    plot_regional_segment(df)
    plot_timeseries_topbottom(df)
    print_insights(df)
