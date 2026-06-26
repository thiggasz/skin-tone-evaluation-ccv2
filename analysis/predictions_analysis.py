import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
from analysis.results_analysis import get_matrix_file

sns.set_theme(style="whitegrid", palette="muted")

TRUE_FILE = ' '
ANALYSIS_FILE = ' '

def plot_correlation_heatmap(df, interest_columns, ax=None):
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
    
    df_filtered = df[interest_columns]
    
    corr_matrix = df_filtered.corr(method='spearman')
    
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, ax=ax)
    
    ax.set_title("Correlation with error Heatmap", fontsize=14)

def plot_dispersion_tendency(df, attribute_column, error_column="error", ax=None):
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
    
    sns.regplot(
        data=df, 
        x=attribute_column, 
        y=error_column, 
        scatter_kws={'alpha':0.3, 's':20}, 
        line_kws={'color':'red', 'linewidth':2},
        order=2,
        ax=ax
    )
    
    ax.set_title(f"Impact of {attribute_column} on the prediction error", fontsize=14)
    ax.set_xlabel(attribute_column)
    ax.set_ylabel(error_column)

def plot_error_boxplot(df, attribute_column, error_column='error', ax=None):    
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
    
    sns.boxplot(
        data=df, 
        x=error_column, 
        y=attribute_column,
        hue=error_column, 
        palette="Reds",
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 6},
        legend=False,
        ax=ax
    )
    
    ax.set_title(f'Distribution of {attribute_column} by error magnitude', fontsize=14, fontweight='bold')
    ax.set_xlabel('Error magnitude (Distância Absoluta)', fontsize=12)
    ax.set_ylabel(attribute_column.capitalize(), fontsize=12)
    
def plot_predictions_range(df, attribute_column, ax=None):
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
    
    unique_errors = sorted(df['error'].dropna().unique())
    colors = ['#2ca02c', '#98df8a', '#ffbb78', '#ff7f0e', '#d62728', '#8c564b']
    used_colors = colors[:len(unique_errors)]
    
    sns.histplot(
        data=df, 
        x=attribute_column, 
        hue='error', 
        multiple='fill', 
        bins='auto', 
        palette=used_colors, 
        edgecolor='black', 
        alpha=0.85,
        ax=ax
    )

    ax.set_title(f'Proportional error distribution by {attribute_column}', fontsize=14)
    ax.set_xlabel(attribute_column.capitalize(), fontsize=12)
    ax.set_ylabel('Images percentage (%)', fontsize=12)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.05, 1), title='Error level')
   
def plot_quantitative_distribution(df, attribute_column, ax=None):
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

    sns.histplot(
        data=df, 
        x=attribute_column, 
        bins='auto', 
        color='steelblue', 
        edgecolor='black',
        ax=ax
    )

    total_imgs = df[attribute_column].notna().sum()

    for p in ax.patches:
        height = p.get_height()
        if pd.isna(height) or height == 0:
            continue
        
        
    ax.set_title(f'Number of images distribution for {attribute_column}', fontsize=14, fontweight='bold')
    ax.set_xlabel(attribute_column.capitalize(), fontsize=12)
    ax.set_ylabel('Number of images', fontsize=12)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
   
def plot_error_bias(df, true_label_column='true_label', error_column='error', ax=None):
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

    df_grouped = df.groupby([true_label_column, error_column]).size().unstack(fill_value=0)
    df_percentage = df_grouped.div(df_grouped.sum(axis=1), axis=0) * 100
    
    colors = ['#2ca02c', '#ffbb78', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']
    used_colors = colors[:len(df_percentage.columns)]
    
    df_percentage.plot(kind='bar', stacked=True, color=used_colors, 
                       edgecolor='black', alpha=0.85, ax=ax)

    ax.set_title('Error distribution per True Class (Bias Analysis)', fontsize=14, fontweight='bold')
    ax.set_xlabel('True Label (Skin Tone)', fontsize=12)
    ax.set_ylabel('Images Percentage (%)', fontsize=12)
    ax.legend(title='Error Magnitude', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=0)

def create_dataframe(df_pred, df_analysis, output='prediction_analysis.csv', save_file=False):
    df_final = pd.merge(
        df_pred, 
        df_analysis, 
        on='file', 
        how='left' 
    )

    if save_file:
        df_final.to_csv(output, index=False)
    
    nulls = df_final['true_label'].isna().sum()
    if nulls > 0:
        print(f"{nulls} lines did not receive labels.")
        
    return df_final
        