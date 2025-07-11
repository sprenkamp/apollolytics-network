import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from dotenv import load_dotenv
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration parameters
SIMILARITY_THRESHOLD = 0.875  # Threshold for considering narratives relevant
AGGREGATION_PERIOD = 'monthly'  # Options: 'daily', 'weekly', 'monthly'

# Load environment variables
load_dotenv()

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "telegram_scraper")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

# Connect to database
engine = create_engine(
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

def load_narrative_data():
    """Load the classified narrative results from the database"""
    print("Loading narrative data from database...")
    df = pd.read_sql("SELECT * FROM relevant_classified_narrative_results", engine)
    print(f"Loaded {len(df)} rows of data")
    
    # Filter for relevant narratives (similarity > threshold)
    similarity_cols = [col for col in df.columns if col.lower().endswith('_similarity')]
    if similarity_cols:
        # Keep only rows where any narrative similarity > threshold
        mask = (df[similarity_cols] > SIMILARITY_THRESHOLD).any(axis=1)
        df = df[mask]
        print(f"Filtered to {len(df)} rows with narrative similarity > {SIMILARITY_THRESHOLD}")
    
    return df

def get_narrative_columns(df):
    """Get all narrative-related columns from the dataframe"""
    # Get similarity columns (end with _similarity)
    similarity_cols = [col for col in df.columns if col.lower().endswith('_similarity')]
    
    return similarity_cols

def get_country_narratives():
    """Define which narratives belong to which country"""
    russian_narratives = [
        'denazificationofukraine_similarity',
        'protectionofrussianspeakers_similarity', 
        'natoexpansionthreat_similarity',
        'biolabsconspiracy_similarity',
        'historicalunity_similarity',
        'westernrussophobia_similarity',
        'sanctionsaseconomicwarfare_similarity',
        'legitimizingannexedterritories_similarity',
        'discreditingukraineleadership_similarity'
    ]
    
    ukrainian_narratives = [
        'putinsdeath_similarity',
        'russiascollapse_similarity',
        'nordstreampipelinesabotage_similarity',
        'heroicmyths_similarity',
        'optimismstrategy_similarity',
        'notsidingwithukraine_similarity',
        'ukraineagainstnewfascism_similarity',
        'cannonfodder_similarity',
        'truthvsrussianlies_similarity'
    ]
    
    return russian_narratives, ukrainian_narratives

def prepare_time_series_data(df, narrative_cols, country):
    """Prepare time series data for plotting"""
    # Filter by country
    country_df = df[df['country'] == country].copy()
    
    if country_df.empty:
        print(f"No data found for {country}")
        return None
    
    # Convert date column to datetime if it exists
    date_col = None
    for col in ['messagedatetime', 'messagedate', 'date', 'datetime', 'timestamp', 'created_at']:
        if col in country_df.columns:
            date_col = col
            break
    
    if date_col is None:
        print(f"No date column found for {country}")
        return None
    
    country_df[date_col] = pd.to_datetime(country_df[date_col], errors='coerce')
    country_df = country_df.dropna(subset=[date_col])
    
    # Create aggregation period column based on configuration
    if AGGREGATION_PERIOD == 'daily':
        country_df['period'] = country_df[date_col].dt.date
        period_col = 'period'
    elif AGGREGATION_PERIOD == 'weekly':
        country_df['period'] = country_df[date_col].dt.to_period('W-MON').dt.start_time
        period_col = 'period'
    elif AGGREGATION_PERIOD == 'monthly':
        country_df['period'] = country_df[date_col].dt.to_period('M').dt.start_time
        period_col = 'period'
    else:
        print(f"Invalid aggregation period: {AGGREGATION_PERIOD}. Using weekly.")
        country_df['period'] = country_df[date_col].dt.to_period('W-MON').dt.start_time
        period_col = 'period'
    
    # Group by date and calculate metrics
    time_series_data = []
    
    for narrative_col in narrative_cols:
        if narrative_col in country_df.columns:
            # Get the narrative name (remove _similarity prefix)
            narrative_name = narrative_col.replace('_similarity', '').replace('_', ' ')
            
            # Filter for messages where THIS SPECIFIC narrative is above threshold
            narrative_mask = country_df[narrative_col] > SIMILARITY_THRESHOLD
            narrative_df = country_df[narrative_mask].copy()
            
            if not narrative_df.empty:
                # Check if 'type' column exists for groups vs channels
                if 'type' in narrative_df.columns:
                    # Group by period and type (groups vs channels)
                    period_stats = narrative_df.groupby([period_col, 'type'])[narrative_col].agg([
                        'mean', 'count', 'sum'
                    ]).reset_index()
                else:
                    # Fallback to just period grouping
                    period_stats = narrative_df.groupby(period_col)[narrative_col].agg([
                        'mean', 'count', 'sum'
                    ]).reset_index()
                    period_stats['type'] = 'unknown'
                
                period_stats['narrative'] = narrative_name
                period_stats['country'] = country
                time_series_data.append(period_stats)
    
    if time_series_data:
        return pd.concat(time_series_data, ignore_index=True)
    return None

def plot_narratives_over_time(df, output_dir='analysis_output'):
    """Create plots for narratives over time"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get narrative columns
    similarity_cols = get_narrative_columns(df)
    print(f"Found {len(similarity_cols)} similarity columns")
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Plot similarity scores over time
    if similarity_cols:
        plot_metric_over_time(df, output_dir)

def plot_metric_over_time(df, output_dir):
    """Plot narrative similarity over time for all narratives"""
    russian_narratives, ukrainian_narratives = get_country_narratives()
    
    # Create plots for each country with their specific narratives
    country_narratives = {
        'Russia': russian_narratives,
        'Ukraine': ukrainian_narratives
    }
    
    for country, narrative_cols in country_narratives.items():
        # Filter narratives that exist in the dataframe
        available_narratives = [col for col in narrative_cols if col in df.columns]
        
        if not available_narratives:
            print(f"No narrative data available for {country}")
            continue
        
        # Prepare data for this country
        country_data = prepare_time_series_data(df, available_narratives, country)
        
        if country_data is None or country_data.empty:
            print(f"No data available for {country}")
            continue
        
        narratives = country_data['narrative'].unique()
        num_narratives = len(narratives)
        
        # Determine subplot layout (3x3 for 9 narratives)
        rows, cols = 3, 3
        
        # Create aggregated version (total counts)
        fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
        period_title = AGGREGATION_PERIOD.capitalize()
        fig.suptitle(f'{country} - {period_title} Narrative Counts (Total)', fontsize=16, fontweight='bold')
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # Plot aggregated version
        for i, narrative in enumerate(narratives):
            if i < len(axes):
                narrative_data = country_data[country_data['narrative'] == narrative]
                
                if not narrative_data.empty:
                    # Aggregate across types (sum all types for each period)
                    aggregated_data = narrative_data.groupby('period')['sum'].sum().reset_index()
                    aggregated_data = aggregated_data.sort_values('period')
                    
                    y_values = aggregated_data['sum']
                    axes[i].plot(aggregated_data['period'], y_values, 
                                linewidth=2, marker='o', markersize=4, color='blue')
                    
                    axes[i].set_title(narrative, fontsize=12, fontweight='bold')
                    period_label = AGGREGATION_PERIOD.capitalize()
                    axes[i].set_ylabel(f'{period_label} Narrative Count', fontsize=10)
                    axes[i].grid(True, alpha=0.3)
                    axes[i].tick_params(axis='x', rotation=45)
                    
                    # Format x-axis dates based on aggregation period
                    if AGGREGATION_PERIOD == 'daily':
                        axes[i].xaxis.set_major_locator(plt.matplotlib.dates.WeekLocator())
                        axes[i].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
                    elif AGGREGATION_PERIOD == 'weekly':
                        axes[i].xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
                        axes[i].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
                    elif AGGREGATION_PERIOD == 'monthly':
                        axes[i].xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
                        axes[i].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
                    
                    # Set y-axis limits for period counts
                    if len(y_values) > 0:
                        y_max = y_values.max()
                        if y_max > 0:
                            axes[i].set_ylim(0, y_max * 1.1)  # Add 10% padding
                        else:
                            axes[i].set_ylim(0, 1)
        
        # Hide empty subplots
        for i in range(len(narratives), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        
        # Save aggregated plot
        filename = f'{country.lower()}_narratives_{AGGREGATION_PERIOD}_total.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved {filename}")
        plt.close()
        
        # Create split version (channels vs groups)
        fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
        fig.suptitle(f'{country} - {period_title} Narrative Counts (Channels vs Groups)', fontsize=16, fontweight='bold')
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # Plot split version
        for i, narrative in enumerate(narratives):
            if i < len(axes):
                narrative_data = country_data[country_data['narrative'] == narrative]
                
                if not narrative_data.empty:
                    # Sort by period
                    narrative_data = narrative_data.sort_values('period')
                    
                    # Check if we have type data (groups vs channels)
                    all_y_values = []  # Collect all y values for proper scaling
                    
                    if 'type' in narrative_data.columns and len(narrative_data['type'].unique()) > 1:
                        # Plot separate lines for groups and channels
                        colors = {'channel': 'blue', 'group': 'red'}
                        for msg_type in narrative_data['type'].unique():
                            type_data = narrative_data[narrative_data['type'] == msg_type]
                            y_values = type_data['sum']
                            all_y_values.extend(y_values.tolist())  # Collect all y values
                            
                            axes[i].plot(type_data['period'], y_values, 
                                        linewidth=2, marker='o', markersize=4, 
                                        color=colors.get(msg_type, 'green'),
                                        label=msg_type.capitalize())
                        
                        axes[i].legend(fontsize=8)
                    else:
                        # Single line plot (no type separation or only one type)
                        y_values = narrative_data['sum']
                        all_y_values = y_values.tolist()  # Collect y values
                        axes[i].plot(narrative_data['period'], y_values, 
                                    linewidth=2, marker='o', markersize=4, color='blue')
                    
                    axes[i].set_title(narrative, fontsize=12, fontweight='bold')
                    period_label = AGGREGATION_PERIOD.capitalize()
                    axes[i].set_ylabel(f'{period_label} Narrative Count', fontsize=10)
                    axes[i].grid(True, alpha=0.3)
                    axes[i].tick_params(axis='x', rotation=45)
                    
                    # Format x-axis dates based on aggregation period
                    if AGGREGATION_PERIOD == 'daily':
                        axes[i].xaxis.set_major_locator(plt.matplotlib.dates.WeekLocator())
                        axes[i].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
                    elif AGGREGATION_PERIOD == 'weekly':
                        axes[i].xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
                        axes[i].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
                    elif AGGREGATION_PERIOD == 'monthly':
                        axes[i].xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
                        axes[i].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
                    
                    # Set y-axis limits using ALL y values (both channels and groups)
                    if all_y_values:
                        y_max = max(all_y_values)
                        if y_max > 0:
                            axes[i].set_ylim(0, y_max * 1.1)  # Add 10% padding
                        else:
                            axes[i].set_ylim(0, 1)
        
        # Hide empty subplots
        for i in range(len(narratives), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        
        # Save split plot
        filename = f'{country.lower()}_narratives_{AGGREGATION_PERIOD}_split.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved {filename}")
        plt.close()

def plot_narrative_comparison(df, output_dir='analysis_output'):
    """Create comparison plots between Russia and Ukraine for each narrative"""
    similarity_cols = get_narrative_columns(df)
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    
    # Plot similarity comparison
    if similarity_cols:
        plot_comparison_metric(df, similarity_cols, output_dir)
    
    # Plot aggregated monthly comparison
    plot_aggregated_comparison(df, output_dir)

def plot_comparison_metric(df, narrative_cols, output_dir):
    """Create comparison plots for narrative similarity"""
    countries = ['Russia', 'Ukraine']
    
    # Calculate overall statistics for each narrative and country
    comparison_data = []
    
    for country in countries:
        country_df = df[df['country'] == country]
        
        for narrative_col in narrative_cols:
            if narrative_col in country_df.columns:
                narrative_name = narrative_col.replace('_similarity', '').replace('_', ' ')
                
                mean_val = country_df[narrative_col].mean()
                std_val = country_df[narrative_col].std()
                
                comparison_data.append({
                    'country': country,
                    'narrative': narrative_name,
                    'mean': mean_val,
                    'std': std_val,
                    'count': len(country_df)
                })
    
    if not comparison_data:
        return
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    narratives = comparison_df['narrative'].unique()
    x = np.arange(len(narratives))
    width = 0.35
    
    russia_data = comparison_df[comparison_df['country'] == 'Russia']
    ukraine_data = comparison_df[comparison_df['country'] == 'Ukraine']
    
    russia_means = [russia_data[russia_data['narrative'] == narrative]['mean'].iloc[0] 
                   if len(russia_data[russia_data['narrative'] == narrative]) > 0 else 0 
                   for narrative in narratives]
    ukraine_means = [ukraine_data[ukraine_data['narrative'] == narrative]['mean'].iloc[0] 
                    if len(ukraine_data[ukraine_data['narrative'] == narrative]) > 0 else 0 
                    for narrative in narratives]
    
    bars1 = ax.bar(x - width/2, russia_means, width, label='Russia', alpha=0.8, color='red')
    bars2 = ax.bar(x + width/2, ukraine_means, width, label='Ukraine', alpha=0.8, color='blue')
    
    ax.set_xlabel('Narratives', fontsize=12)
    ax.set_ylabel('Average Similarity Score', fontsize=12)
    ax.set_title('Narrative Similarity Comparison: Russia vs Ukraine', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(narratives, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):  # Check for NaN values
                if height < 100:
                    label_text = f'{height:.2f}'
                else:
                    label_text = f'{int(height)}'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       label_text,
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    filename = 'narrative_similarity_comparison.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()

def plot_aggregated_comparison(df, output_dir):
    """Create aggregated comparison plot showing monthly narrative counts for Russia vs Ukraine"""
    russian_narratives, ukrainian_narratives = get_country_narratives()
    
    # Prepare data for both countries
    all_data = []
    
    for country, narrative_cols in [('Russia', russian_narratives), ('Ukraine', ukrainian_narratives)]:
        # Filter narratives that exist in the dataframe
        available_narratives = [col for col in narrative_cols if col in df.columns]
        
        if not available_narratives:
            print(f"No narrative data available for {country}")
            continue
        
        # Prepare data for this country
        country_data = prepare_time_series_data(df, available_narratives, country)
        
        if country_data is not None and not country_data.empty:
            # Aggregate across types and narratives for each period
            aggregated_data = country_data.groupby('period')['sum'].sum().reset_index()
            aggregated_data['country'] = country
            all_data.append(aggregated_data)
    
    if not all_data:
        print("No data available for aggregated comparison")
        return
    
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Create the comparison plot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    colors = {'Russia': 'red', 'Ukraine': 'blue'}
    
    for country in ['Russia', 'Ukraine']:
        country_data = combined_data[combined_data['country'] == country]
        
        if not country_data.empty:
            country_data = country_data.sort_values('period')
            
            ax.plot(country_data['period'], country_data['sum'], 
                   linewidth=3, marker='o', markersize=6, 
                   color=colors[country], label=country, alpha=0.8)
    
    ax.set_xlabel('Time Period', fontsize=12)
    period_label = AGGREGATION_PERIOD.capitalize()
    ax.set_ylabel(f'{period_label} Total Narrative Count', fontsize=12)
    ax.set_title(f'Aggregated {period_label} Narrative Counts: Russia vs Ukraine', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Format x-axis dates based on aggregation period
    if AGGREGATION_PERIOD == 'daily':
        ax.xaxis.set_major_locator(plt.matplotlib.dates.WeekLocator())
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    elif AGGREGATION_PERIOD == 'weekly':
        ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    elif AGGREGATION_PERIOD == 'monthly':
        ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    
    plt.tight_layout()
    
    # Save plot
    filename = f'aggregated_narratives_{AGGREGATION_PERIOD}_comparison.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()

def main():
    """Main function to run the narrative analysis"""
    print("Starting narrative analysis over time...")
    
    # Load data
    df = load_narrative_data()
    
    if df.empty:
        print("No data loaded. Exiting.")
        return
    
    # Print basic statistics
    print(f"\nData summary:")
    print(f"Total messages: {len(df)}")
    print(f"Countries: {df['country'].value_counts().to_dict()}")
    
    # Find date column for range display
    date_col = None
    for col in ['messagedatetime', 'messagedate', 'date', 'datetime', 'timestamp', 'created_at']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df_filtered = df.dropna(subset=[date_col])
        if not df_filtered.empty:
            print(f"Date range: {df_filtered[date_col].min()} to {df_filtered[date_col].max()}")
        else:
            print("No valid dates found in the data")
    else:
        print("No date column found in the data")
    
    # Print available narrative columns
    similarity_cols = get_narrative_columns(df)
    print(f"\nAvailable narrative similarity columns ({len(similarity_cols)}):")
    for col in similarity_cols:
        print(f"  - {col}")
    
    # Create plots
    print("\nCreating narrative over time plots...")
    plot_narratives_over_time(df)
    
    print("\nCreating narrative comparison plots...")
    plot_narrative_comparison(df)
    
    print("\nAnalysis complete! Check the analysis_output directory for plots.")

if __name__ == "__main__":
    main()
