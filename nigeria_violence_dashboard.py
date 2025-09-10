import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import folium
from streamlit_folium import st_folium
from pathlib import Path
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import gc
from functools import lru_cache
import pickle
import hashlib
import os
import json
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Nigeria Violence Analysis Dashboard",
    page_icon="🇳🇬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Performance monitoring
if 'performance_metrics' not in st.session_state:
    st.session_state.performance_metrics = {}

def log_performance(func_name, duration):
    """Log performance metrics for monitoring"""
    if func_name not in st.session_state.performance_metrics:
        st.session_state.performance_metrics[func_name] = []
    st.session_state.performance_metrics[func_name].append(duration)


# Custom CSS - optimized for better performance
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .status-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 6px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .performance-info {
        background: #e3f2fd;
        padding: 0.5rem;
        border-radius: 4px;
        border-left: 3px solid #2196f3;
        margin: 0.5rem 0;
        font-size: 0.8rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
    }
    /* Optimize map rendering */
    .element-container iframe {
        width: 100% !important;
    }
    /* Loading spinner optimization */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
</style>
""", unsafe_allow_html=True)

# Data paths - adjust these for Nigeria
DATA_PATH = Path("data/")
PROCESSED_PATH = DATA_PATH / "processed"
CACHE_PATH = Path("cache/")

# Create cache directory only if we have write permissions
try:
    CACHE_PATH.mkdir(exist_ok=True)
    CACHE_ENABLED = True
except (PermissionError, OSError):
    CACHE_ENABLED = False
    st.warning("⚠️ Cache directory not writable. File caching disabled.")

POPULATION_RASTER = DATA_PATH / "nga_ppp_2020_UNadj.tif"
ACLED_DATA = DATA_PATH / "acled_nigeria_data.csv"

START_YEAR = 1997
END_YEAR = 2025

def get_cache_key(*args):
    """Generate cache key from arguments"""
    return hashlib.md5(str(args).encode()).hexdigest()

def save_to_cache(key, data):
    """Save data to cache file"""
    if not CACHE_ENABLED:
        return
    try:
        cache_file = CACHE_PATH / f"{key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        # Silently fail in cloud environments
        pass

def load_from_cache(key):
    """Load data from cache file"""
    if not CACHE_ENABLED:
        return None
    try:
        cache_file = CACHE_PATH / f"{key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        # Silently fail in cloud environments
        pass
    return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def generate_12_month_periods():
    """Generate 12-month periods every 6 months - optimized"""
    periods = []
    
    # Calendar year periods (Jan-Dec)
    for year in range(START_YEAR, END_YEAR + 1):
        periods.append({
            'label': f'Jan {year} - Dec {year}',
            'start_month': 1,
            'start_year': year,
            'end_month': 12,
            'end_year': year,
            'type': 'calendar'
        })
    
    # Mid-year periods (Jul-Jun)
    for year in range(START_YEAR, END_YEAR):
        periods.append({
            'label': f'Jul {year} - Jun {year+1}',
            'start_month': 7,
            'start_year': year,
            'end_month': 6,
            'end_year': year + 1,
            'type': 'mid_year'
        })
    
    return periods

@st.cache_data(ttl=3600, show_spinner=False)
def load_population_data():
    """Load and cache population data from processed JSON files"""
    import time
    start_time = time.time()
    
    # Check cache first
    cache_key = get_cache_key("nigeria_population_data", "v2")
    cached_data = load_from_cache(cache_key)
    if cached_data is not None:
        log_performance("load_population_data", time.time() - start_time)
        return cached_data
    
    try:
        # Load ward-level population data
        # Try comprehensive dataset first, fallback to original
        comprehensive_file = PROCESSED_PATH / "nigeria_ward_population_comprehensive.json"
        ward_file = PROCESSED_PATH / "nigeria_ward_population.json"
        
        if comprehensive_file.exists():
            # Load comprehensive ward population data
            with open(comprehensive_file, 'r') as f:
                ward_data = json.load(f)
            
            # Convert to DataFrame (comprehensive data is already in the right format)
            result_df = pd.DataFrame(ward_data)
            
        elif ward_file.exists():
            # Fallback to original dataset
            with open(ward_file, 'r') as f:
                ward_data = json.load(f)
            
            # Convert to DataFrame
            population_data = []
            for item in ward_data['data']:
                population_data.append({
                    'ADM3_PCODE': item.get('wardcode', ''),
                    'ADM3_EN': item.get('wardname', ''),
                    'ADM2_PCODE': item.get('lgacode', ''),
                    'ADM2_EN': item.get('lganame', ''),
                    'ADM1_PCODE': item.get('statecode', ''),
                    'ADM1_EN': item.get('statename', ''),
                    'ADM0_PCODE': 'NGA',  # Nigeria country code
                    'pop_count': float(item.get('population', 0)),
                    'pop_count_millions': float(item.get('population', 0)) / 1e6
                })
            
            result_df = pd.DataFrame(population_data)
        else:
            st.error(f"No ward population files found")
            return pd.DataFrame()
        
        # Cache the result
        save_to_cache(cache_key, result_df)
        
        log_performance("load_population_data", time.time() - start_time)
        return result_df
        
    except Exception as e:
        st.error(f"Error loading population data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def create_admin_levels(pop_data):
    """Create admin level aggregations from population data - optimized"""
    import time
    start_time = time.time()
    
    if pop_data.empty:
        return {'admin1': pd.DataFrame(), 'admin2': pd.DataFrame(), 'admin3': pop_data}
    
    # Use vectorized operations for better performance
    admin2_agg = pop_data.groupby(['ADM2_PCODE', 'ADM2_EN', 'ADM1_PCODE', 'ADM1_EN', 'ADM0_PCODE'], as_index=False).agg({
        'pop_count': 'sum',
        'pop_count_millions': 'sum'
    })
    
    admin1_agg = pop_data.groupby(['ADM1_PCODE', 'ADM1_EN', 'ADM0_PCODE'], as_index=False).agg({
        'pop_count': 'sum',
        'pop_count_millions': 'sum'
    })
    
    log_performance("create_admin_levels", time.time() - start_time)
    
    return {
        'admin3': pop_data,
        'admin2': admin2_agg,
        'admin1': admin1_agg
    }

@st.cache_data(ttl=3600, show_spinner=False)
def load_conflict_data():
    """Load and cache conflict data with optimized processing"""
    import time
    start_time = time.time()
    
    # Check cache first
    cache_key = get_cache_key("nigeria_conflict_data", "v2")
    cached_data = load_from_cache(cache_key)
    if cached_data is not None:
        log_performance("load_conflict_data", time.time() - start_time)
        return cached_data
    
    try:
        if not ACLED_DATA.exists():
            st.error(f"Conflict data not found: {ACLED_DATA}")
            return pd.DataFrame()
        
        # Load ACLED data directly (simplified approach)
        nigeria_acled = pd.read_csv(ACLED_DATA)
        
        if nigeria_acled.empty:
            return pd.DataFrame()
        
        # Convert event_date to datetime and extract month/year
        nigeria_acled['event_date'] = pd.to_datetime(nigeria_acled['event_date'])
        nigeria_acled['month'] = nigeria_acled['event_date'].dt.month
        nigeria_acled['year'] = nigeria_acled['event_date'].dt.year
        
        # Process the data to match our format
        # Define BRD: Exclude Protests and Riots, include all other violent events
        brd_events = nigeria_acled[
            (~nigeria_acled['event_type'].isin(['Protests', 'Riots'])) &
            (nigeria_acled['fatalities'] > 0)
        ].copy()
        
        # Categorize state vs non-state violence using interaction column
        def categorize_violence(interaction):
            if pd.isna(interaction):
                return 'unknown', 'unknown'
            
            interaction_lower = str(interaction).lower()
            
            # State violence: interactions involving "state forces"
            if 'state forces' in interaction_lower:
                return 'state', 'state'
            # Non-state violence: all other violent interactions
            else:
                return 'nonstate', 'nonstate'
        
        # Apply categorization
        brd_events[['violence_type', 'actor_type']] = brd_events['interaction'].apply(
            lambda x: pd.Series(categorize_violence(x))
        )
        
        # Load preprocessed ward-level conflict data
        ward_conflict_file = PROCESSED_PATH / "ward_conflict_data.csv"
        if ward_conflict_file.exists():
            print("Loading preprocessed ward-level conflict data...")
            conflict_processed = pd.read_csv(ward_conflict_file)
            
            # Rename columns to match expected format
            conflict_processed = conflict_processed.rename(columns={
                'wardcode': 'ADM3_PCODE',
                'wardname': 'ADM3_EN',
                'lganame': 'ADM2_EN',
                'statename': 'ADM1_EN'
            })
            
            # Add ADM codes (simplified mapping)
            conflict_processed['ADM1_PCODE'] = conflict_processed['ADM1_EN'].astype(str)
            conflict_processed['ADM2_PCODE'] = conflict_processed['ADM2_EN'].astype(str)
            
            print(f"Loaded {len(conflict_processed)} ward-level conflict records")
        else:
            print("Preprocessed ward data not found, falling back to LGA-level aggregation...")
            # Fallback to LGA-level aggregation
            conflict_processed = brd_events.groupby(['year', 'month', 'admin1', 'admin2', 'violence_type'], as_index=False).agg({
                'fatalities': 'sum'
            })
            
            # Pivot to get state and non-state columns
            conflict_pivot = conflict_processed.pivot_table(
                index=['year', 'month', 'admin1', 'admin2'],
                columns='violence_type',
                values='fatalities',
                fill_value=0
            ).reset_index()
            
            # Create a mapping from admin names to codes (simplified)
            conflict_pivot['ADM1_PCODE'] = conflict_pivot['admin1'].astype(str)
            conflict_pivot['ADM2_PCODE'] = conflict_pivot['admin2'].astype(str)
            
            # Rename columns to match our format
            conflict_processed = conflict_pivot.rename(columns={
                'admin1': 'ADM1_EN',
                'admin2': 'ADM2_EN'
            })
            # Add empty ward columns for LGA-level data
            conflict_processed['ADM3_PCODE'] = ''
            conflict_processed['ADM3_EN'] = ''
        
        # For preprocessed data, the columns are already in the correct format
        # For fallback LGA data, we need to process the pivot table
        if 'ACLED_BRD_total' not in conflict_processed.columns:
            # This is fallback LGA data, process the pivot table
            conflict_pivot.columns.name = None
            if 'state' in conflict_pivot.columns:
                conflict_pivot['ACLED_BRD_state'] = conflict_pivot['state']
            else:
                conflict_pivot['ACLED_BRD_state'] = 0
                
            if 'nonstate' in conflict_pivot.columns:
                conflict_pivot['ACLED_BRD_nonstate'] = conflict_pivot['nonstate']
            else:
                conflict_pivot['ACLED_BRD_nonstate'] = 0
                
            # Calculate total BRD
            conflict_pivot['ACLED_BRD_total'] = conflict_pivot['ACLED_BRD_state'] + conflict_pivot['ACLED_BRD_nonstate']
            
            # Update conflict_processed with the processed pivot data
            conflict_processed = conflict_pivot.copy()
        
        # Remove any rows with zero total BRD
        conflict_processed = conflict_processed[conflict_processed['ACLED_BRD_total'] > 0]
        
        if 'ADM3_PCODE' in conflict_processed.columns and conflict_processed['ADM3_PCODE'].notna().any():
            print(f"Loaded {len(conflict_processed)} ward-level BRD records from preprocessed data")
            print(f"Unique wards with conflict: {conflict_processed['ADM3_PCODE'].nunique():,}")
        else:
            print(f"Loaded {len(conflict_processed)} LGA-level BRD records (fallback mode)")
        print(f"Total BRD events: {conflict_processed['ACLED_BRD_total'].sum():,.0f}")
        print(f"State violence: {conflict_processed['ACLED_BRD_state'].sum():,.0f}")
        print(f"Non-state violence: {conflict_processed['ACLED_BRD_nonstate'].sum():,.0f}")
        
        # Cache the result
        save_to_cache(cache_key, conflict_processed)
        
        log_performance("load_conflict_data", time.time() - start_time)
        return conflict_processed
        
    except Exception as e:
        st.error(f"Error loading conflict data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def load_admin_boundaries():
    """Load administrative boundaries from GeoJSON files"""
    import time
    start_time = time.time()
    
    # Check cache first
    cache_key = get_cache_key("nigeria_admin_boundaries", "v2")
    cached_data = load_from_cache(cache_key)
    if cached_data is not None:
        log_performance("load_admin_boundaries", time.time() - start_time)
        return cached_data
    
    boundaries = {}
    try:
        # Try comprehensive ward boundaries first, fallback to original
        comprehensive_ward_file = DATA_PATH / "nga_ward_boundaries_comprehensive.geojson"
        ward_file = DATA_PATH / "nga_ward_boundaries.geojson"
        
        if comprehensive_ward_file.exists():
            # Load comprehensive ward boundaries
            ward_gdf = gpd.read_file(comprehensive_ward_file)
            ward_gdf = ward_gdf.to_crs('EPSG:4326')
            boundaries[3] = ward_gdf
        elif ward_file.exists():
            # Fallback to original boundaries
            ward_gdf = gpd.read_file(ward_file)
            ward_gdf = ward_gdf.to_crs('EPSG:4326')
            boundaries[3] = ward_gdf
        else:
            boundaries[3] = gpd.GeoDataFrame()
        
        # Load LGA boundaries
        lga_file = DATA_PATH / "nga_lga_boundaries.geojson"
        if lga_file.exists():
            lga_gdf = gpd.read_file(lga_file)
            lga_gdf = lga_gdf.to_crs('EPSG:4326')
            boundaries[2] = lga_gdf
        else:
            boundaries[2] = gpd.GeoDataFrame()
        
        # Load state boundaries
        state_file = DATA_PATH / "nga_state_boundaries.geojson"
        if state_file.exists():
            state_gdf = gpd.read_file(state_file)
            state_gdf = state_gdf.to_crs('EPSG:4326')
            boundaries[1] = state_gdf
        else:
            boundaries[1] = gpd.GeoDataFrame()
        
    except Exception as e:
        boundaries = {1: gpd.GeoDataFrame(), 2: gpd.GeoDataFrame(), 3: gpd.GeoDataFrame()}
    
    # Cache the result
    save_to_cache(cache_key, boundaries)
    
    log_performance("load_admin_boundaries", time.time() - start_time)
    return boundaries

def filter_data_by_period_impl(data, period_info):
    """Filter data based on custom date range - optimized implementation"""
    if len(data) == 0:
        return data
    
    start_year = period_info['start_year']
    end_year = period_info['end_year']
    start_month = period_info['start_month']
    end_month = period_info['end_month']
    
    # Now we have month-level data, so we can filter more precisely
    if start_year == end_year:
        # Same year: filter by year and month range
        mask = (data['year'] == start_year) & (data['month'] >= start_month) & (data['month'] <= end_month)
    else:
        # Different years: filter by year range and month conditions
        mask = (
            ((data['year'] == start_year) & (data['month'] >= start_month)) |
            ((data['year'] > start_year) & (data['year'] < end_year)) |
            ((data['year'] == end_year) & (data['month'] <= end_month))
        )
    
    return data[mask]

def classify_and_aggregate_data(pop_data, admin_data, conflict_data, period_info, rate_thresh, abs_thresh, agg_thresh, agg_level):
    """Classify wards and aggregate to selected administrative level - optimized"""
    import time
    start_time = time.time()
    
    # Filter conflict data for selected period using optimized function
    period_conflict = filter_data_by_period_impl(conflict_data, period_info)
    
    # Check if we have ward-level conflict data from spatial intersection
    if len(period_conflict) > 0 and 'ADM3_PCODE' in period_conflict.columns and period_conflict['ADM3_PCODE'].notna().any():
        # We have ward-level conflict data from spatial intersection
        # Aggregate conflict data by ward for the period
        conflict_ward = period_conflict.groupby(['ADM3_PCODE'], as_index=False).agg({
            'ACLED_BRD_state': 'sum',
            'ACLED_BRD_nonstate': 'sum',
            'ACLED_BRD_total': 'sum'
        })
        
        # Merge with population data at ward level
        merged = pd.merge(pop_data, conflict_ward, on='ADM3_PCODE', how='left')
        
        # Fill missing values with 0
        conflict_cols = ['ACLED_BRD_state', 'ACLED_BRD_nonstate', 'ACLED_BRD_total']
        merged[conflict_cols] = merged[conflict_cols].fillna(0)
    else:
        # No ward-level data available, set all ward conflict values to 0
        merged = pop_data.copy()
        merged['ACLED_BRD_state'] = 0
        merged['ACLED_BRD_nonstate'] = 0
        merged['ACLED_BRD_total'] = 0
    
    # Calculate death rates using vectorized operations
    merged['acled_total_death_rate'] = (merged['ACLED_BRD_total'] / (merged['pop_count_millions'] * 1e6)) * 1e5
    
    # Classify wards using vectorized operations
    merged['violence_affected'] = (
        (merged['acled_total_death_rate'] > rate_thresh) & 
        (merged['ACLED_BRD_total'] > abs_thresh)
    )
    
    # Aggregate to selected level using optimized groupby
    if agg_level == 'ADM1':
        group_cols = ['ADM1_PCODE', 'ADM1_EN']
    else:  # ADM2
        group_cols = ['ADM2_PCODE', 'ADM2_EN', 'ADM1_PCODE', 'ADM1_EN']
    
    aggregated = merged.groupby(group_cols, as_index=False).agg({
        'pop_count': 'sum',
        'violence_affected': 'sum',
        'ADM3_PCODE': 'count',
        'ACLED_BRD_total': 'sum'
    })
    
    aggregated.rename(columns={'ADM3_PCODE': 'total_wards'}, inplace=True)
    
    # Calculate shares using vectorized operations
    aggregated['share_wards_affected'] = aggregated['violence_affected'] / aggregated['total_wards']
    
    # Calculate population share using optimized operations
    affected_pop = merged[merged['violence_affected']].groupby(group_cols[0], as_index=False)['pop_count'].sum()
    affected_pop.rename(columns={'pop_count': 'affected_population'}, inplace=True)
    aggregated = pd.merge(aggregated, affected_pop, on=group_cols[0], how='left')
    aggregated['affected_population'] = aggregated['affected_population'].fillna(0)
    aggregated['share_population_affected'] = aggregated['affected_population'] / aggregated['pop_count']
    
    # Mark units above threshold using vectorized operations
    aggregated['above_threshold'] = aggregated['share_wards_affected'] > agg_thresh
    
    log_performance("classify_and_aggregate_data", time.time() - start_time)
    
    return aggregated, merged

@st.cache_data(ttl=3600, show_spinner=False)
def load_all_ward_timeseries_data():
    """Load all ward time series data at once for better performance"""
    import time
    start_time = time.time()
    
    try:
        ward_timeseries_path = PROCESSED_PATH / "ward_timeseries"
        
        if not ward_timeseries_path.exists():
            return {}
        
        # Load all JSON files at once
        all_data = {}
        json_files = list(ward_timeseries_path.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if 'ward_info' in data and 'wardcode' in data['ward_info']:
                        ward_code = data['ward_info']['wardcode']
                        all_data[ward_code] = data
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
        
        log_performance("load_all_ward_timeseries_data", time.time() - start_time)
        return all_data
        
    except Exception as e:
        st.error(f"Error loading ward time series data: {str(e)}")
        return {}

@st.cache_data(ttl=3600, show_spinner=False)
def get_ward_timeseries_data(ward_code, period_info=None):
    """Get time series data for a specific ward from cached data"""
    import time
    start_time = time.time()
    
    try:
        # Get all ward data from cache
        all_ward_data = load_all_ward_timeseries_data()
        
        if ward_code not in all_ward_data:
            return None
        
        data = all_ward_data[ward_code].copy()
        
        # Filter by date range if period_info is provided
        if period_info and 'time_series' in data:
            filtered_time_series = []
            start_year = period_info['start_year']
            start_month = period_info['start_month']
            end_year = period_info['end_year']
            end_month = period_info['end_month']
            
            for item in data['time_series']:
                item_year = item['year']
                item_month = item['month']
                
                # Check if the item falls within the selected date range
                if (start_year < item_year < end_year) or \
                   (start_year == item_year and item_month >= start_month) or \
                   (end_year == item_year and item_month <= end_month) or \
                   (start_year == end_year and start_month <= item_month <= end_month):
                    filtered_time_series.append(item)
            
            # Update the data with filtered time series
            data['time_series'] = filtered_time_series
        
        log_performance("get_ward_timeseries_data", time.time() - start_time)
        return data
        
    except Exception as e:
        st.error(f"Error getting ward time series data: {str(e)}")
        return None

def create_ward_timeseries_plot(ward_data):
    """Create time series plot for ward events and fatalities"""
    if not ward_data or 'time_series' not in ward_data:
        return None
    
    time_series = ward_data['time_series']
    ward_info = ward_data['ward_info']
    
    if not time_series:
        return None
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(time_series)
    df['date'] = pd.to_datetime(df['year_month'], format='%Y-%m')
    df = df.sort_values('date')
    
    # Create subplots with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            f"Event Count - {ward_info['wardname']}, {ward_info['lganame']}, {ward_info['statename']}",
            f"Total Fatalities - {ward_info['wardname']}, {ward_info['lganame']}, {ward_info['statename']}"
        ],
        vertical_spacing=0.1,
        row_heights=[0.5, 0.5]
    )
    
    # Add event count bars
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['event_count'],
            name='Event Count',
            marker=dict(color='#2E86AB', opacity=0.8),
            hovertemplate='<b>%{x|%Y-%m}</b><br>Events: %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add fatalities bars
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['total_fatalities'],
            name='Total Fatalities',
            marker=dict(color='#A23B72', opacity=0.8),
            hovertemplate='<b>%{x|%Y-%m}</b><br>Fatalities: %{y}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Update x-axes
    fig.update_xaxes(
        title_text="Date",
        showgrid=True,
        gridcolor='lightgray',
        row=2, col=1
    )
    
    # Update y-axes
    fig.update_yaxes(
        title_text="Number of Events",
        showgrid=True,
        gridcolor='lightgray',
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="Number of Fatalities",
        showgrid=True,
        gridcolor='lightgray',
        row=2, col=1
    )
    
    return fig

# Import mapping and chart functions
from mapping_functions import create_admin_map, create_ward_map
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def main():
    """Main Streamlit application with optimized performance"""
    import time
    app_start_time = time.time()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🇳🇬 Nigeria Violence Analysis Dashboard</h1>
        <p>Interactive analysis with 12-month periods and comprehensive ward mapping</p>
    </div>
    """, unsafe_allow_html=True)
    
    
    # Initialize session state for data caching
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.periods = None
        st.session_state.pop_data = None
        st.session_state.admin_data = None
        st.session_state.conflict_data = None
        st.session_state.boundaries = None
        st.session_state.ward_timeseries_loaded = False
    
    # Load data with progress indicators and session state caching
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            data_start_time = time.time()
            
            st.session_state.periods = generate_12_month_periods()
            st.session_state.pop_data = load_population_data()
            st.session_state.admin_data = create_admin_levels(st.session_state.pop_data)
            st.session_state.conflict_data = load_conflict_data()
            st.session_state.boundaries = load_admin_boundaries()
            
            st.session_state.data_loaded = True
            
            # Preload ward timeseries data for better performance
            if not st.session_state.ward_timeseries_loaded:
                with st.spinner("Preloading ward time series data..."):
                    load_all_ward_timeseries_data()
                    st.session_state.ward_timeseries_loaded = True
            
            data_load_time = time.time() - data_start_time
            log_performance("data_loading", data_load_time)
            
            # Show performance info
            st.markdown(f"""
            <div class="performance-info">
                ⚡ Data loaded in {data_load_time:.2f} seconds
            </div>
            """, unsafe_allow_html=True)
    
    # Use cached data
    periods = st.session_state.periods
    pop_data = st.session_state.pop_data
    admin_data = st.session_state.admin_data
    conflict_data = st.session_state.conflict_data
    boundaries = st.session_state.boundaries
    
    if pop_data.empty:
        st.error("Failed to load population data. Please check your data files.")
        st.stop()
    
    if conflict_data.empty:
        st.stop()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Analysis Controls")
    
    # Violence Classification
    st.sidebar.subheader("📊 Violence Classification")
    agg_thresh = st.sidebar.slider(
        "Aggregation Threshold",
        min_value=0.05, max_value=0.5, value=0.2, step=0.05,
        help="Minimum share of wards affected to mark administrative unit as high-violence"
    )
    
    # Analysis Settings
    st.sidebar.subheader("🗺️ Analysis & Display Settings")
    
    # Date selection options
    st.sidebar.subheader("📅 Date Range Selection")
    
    # Month names for display
    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    
    # Available years based on ACLED data range
    available_years = list(range(1997, 2026))
    
    # Start Date Selection
    st.sidebar.markdown("**Start Date:**")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        start_year = st.selectbox(
            "Start Year:",
            options=available_years,
            index=len(available_years) - 6,  # Default to 2020
            key="start_year",
            help="Select the start year"
        )
    
    with col2:
        start_month = st.selectbox(
            "Start Month:",
            options=list(range(1, 13)),
            format_func=lambda x: month_names[x-1],
            index=0,  # Default to January
            key="start_month",
            help="Select the start month"
        )
    
    # End Date Selection
    st.sidebar.markdown("**End Date:**")
    col3, col4 = st.sidebar.columns(2)
    
    with col3:
        end_year = st.selectbox(
            "End Year:",
            options=available_years,
            index=len(available_years) - 2,  # Default to 2024
            key="end_year",
            help="Select the end year"
        )
    
    with col4:
        end_month = st.selectbox(
            "End Month:",
            options=list(range(1, 13)),
            format_func=lambda x: month_names[x-1],
            index=11,  # Default to December
            key="end_month",
            help="Select the end month"
        )
    
    # Create datetime objects
    start_dt = pd.to_datetime(f"{start_year}-{start_month:02d}-01")
    end_dt = pd.to_datetime(f"{end_year}-{end_month:02d}-01") + pd.DateOffset(months=1) - pd.DateOffset(days=1)
    
    # Calculate number of months
    months_diff = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month) + 1
    
    # Create period info
    period_info = {
        'label': f"{month_names[start_month-1]} {start_year} - {month_names[end_month-1]} {end_year}",
        'start_date': start_dt,
        'end_date': end_dt,
        'start_year': start_year,
        'start_month': start_month,
        'end_year': end_year,
        'end_month': end_month,
        'months': months_diff,
        'type': 'custom_range'
    }
    
    
    # Calculate proportional thresholds based on period length
    base_rate_thresh = 4.0  # per 100k per 12 months
    base_abs_thresh = 5.0   # deaths per 12 months
    
    # Calculate proportional thresholds
    proportional_rate_thresh = (base_rate_thresh / 12) * period_info['months']
    proportional_abs_thresh = (base_abs_thresh / 12) * period_info['months']
    
    # Display calculated thresholds
    st.sidebar.markdown(f"**Calculated Thresholds for {period_info['months']} months:**")
    st.sidebar.markdown(f"• Death Rate: {proportional_rate_thresh:.2f} per 100k")
    st.sidebar.markdown(f"• Min Deaths: {proportional_abs_thresh:.1f}")
    
    # Allow manual adjustment if needed
    rate_thresh = st.sidebar.slider(
        "Death Rate Threshold (per 100k)",
        min_value=0.1, max_value=20.0, value=proportional_rate_thresh, step=0.1,
        help=f"Minimum death rate per 100,000 population (calculated: {proportional_rate_thresh:.2f} for {period_info['months']} months)"
    )
    abs_thresh = st.sidebar.slider(
        "Min Deaths Threshold",
        min_value=0.1, max_value=100.0, value=proportional_abs_thresh, step=0.1,
        help=f"Minimum absolute number of deaths (calculated: {proportional_abs_thresh:.1f} for {period_info['months']} months)"
    )
    
    agg_level = st.sidebar.selectbox(
        "Administrative Level",
        options=['ADM1', 'ADM2'],
        format_func=lambda x: 'Admin 1 (States)' if x == 'ADM1' else 'Admin 2 (LGAs)',
        index=1,
        help="Administrative level for aggregated analysis"
    )
    
    map_var = st.sidebar.selectbox(
        "Map Variable",
        options=['share_wards', 'share_population'],
        format_func=lambda x: 'Share of Wards Affected' if x == 'share_wards' else 'Share of Population Affected',
        index=0,
        help="Variable to display on administrative units map"
    )
    
    # period_info is already created from date range picker above
    
    # Status information
    st.markdown(f"""
    <div class="status-info">
        <strong>📊 Current Analysis Configuration</strong><br>
        <strong>Period:</strong> {period_info['label']} ({period_info['type'].replace('_', ' ').title()}) | 
        <strong>Level:</strong> {agg_level} | 
        <strong>Map Variable:</strong> {map_var}<br>
        <strong>Ward Classification:</strong> Death rate >{rate_thresh:.1f}/100k AND >{abs_thresh} deaths<br>
        <strong>Unit Threshold:</strong> >{agg_thresh:.1%} of wards affected → Marked as high-violence area
    </div>
    """, unsafe_allow_html=True)
    
    # Process data
    with st.spinner("Processing analysis..."):
        aggregated, ward_data = classify_and_aggregate_data(
            pop_data, admin_data, conflict_data, period_info, rate_thresh, abs_thresh, agg_thresh, agg_level
        )
    
    # Display metrics
    if len(aggregated) > 0:
        total_units = len(aggregated)
        above_threshold_count = aggregated['above_threshold'].sum()
        # Get total ward count from population data (which has all wards)
        total_wards = len(pop_data)
        affected_wards = aggregated['violence_affected'].sum()
        total_population = aggregated['pop_count'].sum()
        affected_population = aggregated['affected_population'].sum()
        total_deaths = aggregated['ACLED_BRD_total'].sum()
        
        # Display metrics in a grid
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🎯 High Violence Units</h4>
                <div style="font-size: 24px; font-weight: bold;">{above_threshold_count}</div>
                <div>out of {total_units} ({above_threshold_count/total_units*100:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🏘️ Affected Wards</h4>
                <div style="font-size: 24px; font-weight: bold;">{affected_wards}</div>
                <div>out of {total_wards} ({affected_wards/total_wards*100:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>👥 Affected Population</h4>
                <div style="font-size: 24px; font-weight: bold;">{affected_population:,.0f}</div>
                <div>out of {total_population:,.0f} ({affected_population/total_population*100:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Total Deaths</h4>
                <div style="font-size: 24px; font-weight: bold;">{total_deaths:,}</div>
                <div>in {period_info['label']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Maps section
    st.header("🗺️ Interactive Violence Maps")
    st.markdown("**📍 Administrative Units**: Aggregated analysis by states/LGAs | **🏘️ Ward Classification**: Individual ward violence classification")
    
    tab1, tab2 = st.tabs(["📍 Administrative Units", "🏘️ Ward Classification"])
    
    with tab1:
        st.subheader(f"Administrative Units Analysis - {agg_level}")
        if len(aggregated) > 0:
            admin_map = create_admin_map(
                aggregated, boundaries, agg_level, map_var, agg_thresh, period_info, rate_thresh, abs_thresh
            )
            if admin_map:
                # Use full width for the map
                st_folium(admin_map, width=None, height=600, returned_objects=["last_object_clicked"])
            else:
                st.error("Could not create administrative map due to missing boundary data.")
        else:
            st.error("No administrative data available for the selected period.")
    
    with tab2:
        st.subheader("Individual Ward Classification")
        if len(ward_data) > 0:
            ward_map = create_ward_map(
                ward_data, boundaries, period_info, rate_thresh, abs_thresh
            )
            if ward_map:
                # Use full width for the map
                st_folium(ward_map, width=None, height=600, returned_objects=["last_object_clicked"])
                
                # Add ward selection functionality
                st.markdown("---")
                st.subheader("📊 Ward Time Series Analysis")
                st.markdown("Select a ward from the dropdown below to view its historical events and fatalities over time.")
                
                # Create ward selection dropdown (cached for performance)
                if 'ward_options' not in st.session_state:
                    ward_options = ward_data[['ADM3_PCODE', 'ADM3_EN', 'ADM2_EN', 'ADM1_EN']].copy()
                    ward_options['display_name'] = ward_options['ADM3_EN'] + ' (' + ward_options['ADM2_EN'] + ', ' + ward_options['ADM1_EN'] + ')'
                    ward_options = ward_options.sort_values('display_name')
                    st.session_state.ward_options = ward_options
                else:
                    ward_options = st.session_state.ward_options
                
                # Create dropdown
                selected_ward_display = st.selectbox(
                    "Select a ward to analyze:",
                    options=ward_options['display_name'].tolist(),
                    index=0,
                    help="Choose a ward to view its time series data"
                )
                
                # Get the ward code for the selected ward
                if selected_ward_display:
                    selected_ward_code = ward_options[ward_options['display_name'] == selected_ward_display]['ADM3_PCODE'].iloc[0]
                    
                    # Display time series for selected ward
                    with st.spinner(f"Loading time series data for {selected_ward_display}..."):
                        ward_timeseries_data = get_ward_timeseries_data(selected_ward_code, period_info)
                        
                        if ward_timeseries_data:
                            # Display ward info
                            ward_info = ward_timeseries_data['ward_info']
                            st.markdown(f"""
                            **Selected Ward:** {ward_info['wardname']}  
                            **LGA:** {ward_info['lganame']}  
                            **State:** {ward_info['statename']}  
                            **Ward Code:** {ward_info['wardcode']}
                            """)
                            
                            # Create and display time series plot
                            timeseries_fig = create_ward_timeseries_plot(ward_timeseries_data)
                            if timeseries_fig:
                                st.plotly_chart(timeseries_fig, use_container_width=True)
                                
                                # Display summary statistics
                                time_series = ward_timeseries_data['time_series']
                                if time_series:
                                    total_events = sum(item['event_count'] for item in time_series)
                                    total_fatalities = sum(item['total_fatalities'] for item in time_series)
                                    date_range = f"{time_series[0]['year_month']} to {time_series[-1]['year_month']}"
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Total Events", f"{total_events:,}")
                                    with col2:
                                        st.metric("Total Fatalities", f"{total_fatalities:,}")
                                    with col3:
                                        st.metric("Date Range", date_range)
                            else:
                                st.error("Could not create time series plot.")
                        else:
                            st.warning(f"No time series data found for ward {selected_ward_code}")
                        
            else:
                st.error("Could not create ward map due to missing boundary data.")
        else:
            st.error("No ward data available for the selected period.")
    
    
    # Data Export Section
    st.header("📥 Data Export")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Aggregated Data", use_container_width=True):
            if len(aggregated) > 0:
                csv = aggregated.to_csv(index=False)
                st.download_button(
                    label="Download Aggregated Data CSV",
                    data=csv,
                    file_name=f"nigeria_aggregated_{period_info['label'].replace(' ', '_').replace('-', '_')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.error("No aggregated data to export.")
    
    with col2:
        if st.button("🏘️ Export Ward Data", use_container_width=True):
            if len(ward_data) > 0:
                csv = ward_data.to_csv(index=False)
                st.download_button(
                    label="Download Ward Data CSV",
                    data=csv,
                    file_name=f"nigeria_wards_{period_info['label'].replace(' ', '_').replace('-', '_')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.error("No ward data to export.")
    
    with col3:
        if st.button("📈 Export Analysis Summary", use_container_width=True):
            if len(aggregated) > 0:
                summary_data = {
                    'period': [period_info['label']],
                    'period_type': [period_info['type']],
                    'total_units': [total_units],
                    'high_violence_units': [above_threshold_count],
                    'total_wards': [total_wards],
                    'affected_wards': [affected_wards],
                    'total_population': [total_population],
                    'affected_population': [affected_population],
                    'total_deaths': [total_deaths],
                    'national_death_rate': [(total_deaths/total_population)*1e5],
                    'analysis_timestamp': [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                }
                summary_df = pd.DataFrame(summary_data)
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="Download Summary CSV",
                    data=csv,
                    file_name=f"nigeria_summary_{period_info['label'].replace(' ', '_').replace('-', '_')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.error("No summary data to export.")
    
    # Performance monitoring section (expandable)
    with st.expander("⚡ Performance Metrics", expanded=False):
        if st.session_state.performance_metrics:
            st.subheader("Function Performance")
            
            perf_data = []
            for func_name, times in st.session_state.performance_metrics.items():
                avg_time = sum(times) / len(times)
                max_time = max(times)
                min_time = min(times)
                call_count = len(times)
                
                perf_data.append({
                    'Function': func_name,
                    'Avg Time (s)': f"{avg_time:.3f}",
                    'Min Time (s)': f"{min_time:.3f}",
                    'Max Time (s)': f"{max_time:.3f}",
                    'Calls': call_count
                })
            
            if perf_data:
                perf_df = pd.DataFrame(perf_data)
                st.dataframe(perf_df, use_container_width=True)
                
                # Performance tips
                st.markdown("""
                **Performance Tips:**
                - Data is cached in session state for faster subsequent loads
                - File-based caching reduces processing time for repeated operations
                - Vectorized operations improve pandas performance
                - Map rendering uses canvas mode for better performance
                - Simplified popups and legends reduce rendering overhead
                """)
        else:
            st.info("No performance data available yet.")
    
    # Add refresh button to clear cache
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Refresh Data Cache", use_container_width=True):
            # Clear session state
            for key in ['data_loaded', 'periods', 'pop_data', 'admin_data', 'conflict_data', 'boundaries']:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Clear file cache if enabled
            if CACHE_ENABLED:
                try:
                    import shutil
                    if CACHE_PATH.exists():
                        shutil.rmtree(CACHE_PATH)
                        CACHE_PATH.mkdir(exist_ok=True)
                except Exception:
                    pass  # Silently fail in cloud environments
            
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()
    
    # Total app performance
    total_time = time.time() - app_start_time
    st.markdown(f"""
    <div class="performance-info">
        🏁 Total app processing time: {total_time:.2f} seconds
    </div>
    """, unsafe_allow_html=True)

# Main app navigation - simplified to single page
def app():
    """Main application - single page dashboard"""
    main()

if __name__ == "__main__":
    app()
