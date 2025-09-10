#!/usr/bin/env python3
"""
Preprocess ACLED conflict data with spatial intersection to wards
This script aggregates ACLED data and performs spatial intersection once,
then saves the results for fast loading in the dashboard.
"""

import pandas as pd
import geopandas as gpd
import json
from pathlib import Path
from shapely.geometry import Point
import time

# Data paths
DATA_PATH = Path("data")
PROCESSED_PATH = DATA_PATH / "processed"
ACLED_DATA = DATA_PATH / "acled_nigeria_data.csv"
WARD_BOUNDARIES = DATA_PATH / "nga_ward_boundaries_comprehensive.geojson"

def load_acled_data():
    """Load and preprocess ACLED data"""
    print("Loading ACLED data...")
    df = pd.read_csv(ACLED_DATA)
    
    # Convert event_date to datetime and extract month/year
    df['event_date'] = pd.to_datetime(df['event_date'])
    df['month'] = df['event_date'].dt.month
    df['year'] = df['event_date'].dt.year
    
    # Filter for BRD events (exclude Protests and Riots)
    brd_events = df[
        (~df['event_type'].isin(['Protests', 'Riots'])) &
        (df['fatalities'] > 0)
    ].copy()
    
    print(f"Total ACLED records: {len(df)}")
    print(f"BRD events: {len(brd_events)}")
    
    return brd_events

def categorize_violence(interaction):
    """Categorize violence by interaction type"""
    if pd.isna(interaction):
        return 'unknown', 'unknown'
    
    interaction_lower = str(interaction).lower()
    
    # State violence: interactions involving "state forces"
    if 'state forces' in interaction_lower:
        return 'state', 'state'
    # Non-state violence: all other violent interactions
    else:
        return 'nonstate', 'nonstate'

def aggregate_acled_data(brd_events):
    """Aggregate ACLED data by lat/lon/year/month"""
    print("Aggregating ACLED data by location and time...")
    
    # Categorize violence
    brd_events[['violence_type', 'actor_type']] = brd_events['interaction'].apply(
        lambda x: pd.Series(categorize_violence(x))
    )
    
    # Aggregate by lat/lon/year/month/violence_type
    aggregated = brd_events.groupby([
        'latitude', 'longitude', 'year', 'month', 'violence_type'
    ], as_index=False).agg({
        'fatalities': 'sum',
        'admin1': 'first',  # Keep admin info for reference
        'admin2': 'first',
        'location': 'first'
    })
    
    # Pivot to get state and non-state columns
    conflict_pivot = aggregated.pivot_table(
        index=['latitude', 'longitude', 'year', 'month', 'admin1', 'admin2', 'location'],
        columns='violence_type',
        values='fatalities',
        fill_value=0
    ).reset_index()
    
    # Rename columns and add totals
    conflict_pivot.columns.name = None
    if 'state' in conflict_pivot.columns:
        conflict_pivot['ACLED_BRD_state'] = conflict_pivot['state']
    else:
        conflict_pivot['ACLED_BRD_state'] = 0
        
    if 'nonstate' in conflict_pivot.columns:
        conflict_pivot['ACLED_BRD_nonstate'] = conflict_pivot['nonstate']
    else:
        conflict_pivot['ACLED_BRD_nonstate'] = 0
        
    conflict_pivot['ACLED_BRD_total'] = conflict_pivot['ACLED_BRD_state'] + conflict_pivot['ACLED_BRD_nonstate']
    
    print(f"Aggregated to {len(conflict_pivot)} location-time records")
    return conflict_pivot

def load_ward_boundaries():
    """Load ward boundaries"""
    print("Loading ward boundaries...")
    if not WARD_BOUNDARIES.exists():
        print(f"Error: Ward boundaries file not found: {WARD_BOUNDARIES}")
        return None
    
    gdf = gpd.read_file(WARD_BOUNDARIES)
    print(f"Loaded {len(gdf)} ward boundaries")
    return gdf

def perform_spatial_intersection(conflict_data, ward_boundaries):
    """Perform spatial intersection between conflict points and ward boundaries"""
    print("Performing spatial intersection...")
    
    # Create GeoDataFrame from conflict points
    conflict_gdf = gpd.GeoDataFrame(
        conflict_data,
        geometry=[Point(xy) for xy in zip(conflict_data['longitude'], conflict_data['latitude'])],
        crs='EPSG:4326'
    )
    
    # Ensure both GeoDataFrames have the same CRS
    if ward_boundaries.crs != conflict_gdf.crs:
        ward_boundaries = ward_boundaries.to_crs(conflict_gdf.crs)
    
    # Perform spatial join
    print("  - Joining conflict points with ward boundaries...")
    events_with_wards = gpd.sjoin(conflict_gdf, ward_boundaries, how='left', predicate='within')
    
    # Count how many events were successfully matched
    matched_events = events_with_wards['wardcode'].notna().sum()
    total_events = len(events_with_wards)
    print(f"  - Matched {matched_events}/{total_events} events to wards ({matched_events/total_events*100:.1f}%)")
    
    return events_with_wards

def aggregate_to_ward_level(events_with_wards):
    """Aggregate conflict data to ward level"""
    print("Aggregating to ward level...")
    
    # Filter to only events that were matched to wards
    matched_events = events_with_wards[events_with_wards['wardcode'].notna()].copy()
    
    if len(matched_events) == 0:
        print("Warning: No events were matched to wards!")
        return pd.DataFrame()
    
    # Aggregate by ward and time
    ward_conflict = matched_events.groupby([
        'wardcode', 'wardname', 'lganame', 'statename', 'year', 'month'
    ], as_index=False).agg({
        'ACLED_BRD_state': 'sum',
        'ACLED_BRD_nonstate': 'sum',
        'ACLED_BRD_total': 'sum'
    })
    
    print(f"Aggregated to {len(ward_conflict)} ward-time records")
    return ward_conflict

def save_processed_data(ward_conflict_data):
    """Save processed data to files"""
    print("Saving processed data...")
    
    # Ensure processed directory exists
    PROCESSED_PATH.mkdir(exist_ok=True)
    
    # Save as CSV for easy loading
    csv_file = PROCESSED_PATH / "ward_conflict_data.csv"
    ward_conflict_data.to_csv(csv_file, index=False)
    print(f"Saved to: {csv_file}")
    
    # Also save as JSON for compatibility
    json_file = PROCESSED_PATH / "ward_conflict_data.json"
    ward_conflict_data.to_json(json_file, orient='records', indent=2)
    print(f"Saved to: {json_file}")
    
    # Create summary statistics (convert numpy types to Python types for JSON serialization)
    summary = {
        'total_ward_time_records': int(len(ward_conflict_data)),
        'unique_wards': int(ward_conflict_data['wardcode'].nunique()),
        'date_range': {
            'start_year': int(ward_conflict_data['year'].min()),
            'end_year': int(ward_conflict_data['year'].max()),
            'start_month': int(ward_conflict_data['month'].min()),
            'end_month': int(ward_conflict_data['month'].max())
        },
        'total_deaths': {
            'state_violence': int(ward_conflict_data['ACLED_BRD_state'].sum()),
            'nonstate_violence': int(ward_conflict_data['ACLED_BRD_nonstate'].sum()),
            'total': int(ward_conflict_data['ACLED_BRD_total'].sum())
        },
        'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    summary_file = PROCESSED_PATH / "ward_conflict_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")
    
    return summary

def main():
    """Main processing function"""
    print("🚀 Starting ACLED ward-level conflict data preprocessing...")
    start_time = time.time()
    
    try:
        # Step 1: Load ACLED data
        brd_events = load_acled_data()
        
        # Step 2: Aggregate by location and time
        conflict_data = aggregate_acled_data(brd_events)
        
        # Step 3: Load ward boundaries
        ward_boundaries = load_ward_boundaries()
        if ward_boundaries is None:
            return
        
        # Step 4: Perform spatial intersection
        events_with_wards = perform_spatial_intersection(conflict_data, ward_boundaries)
        
        # Step 5: Aggregate to ward level
        ward_conflict_data = aggregate_to_ward_level(events_with_wards)
        
        if len(ward_conflict_data) == 0:
            print("❌ No ward-level data generated!")
            return
        
        # Step 6: Save processed data
        summary = save_processed_data(ward_conflict_data)
        
        # Print summary
        print("\n✅ Processing completed successfully!")
        print(f"⏱️  Total time: {time.time() - start_time:.1f} seconds")
        print(f"📊 Summary:")
        print(f"   - {summary['total_ward_time_records']:,} ward-time records")
        print(f"   - {summary['unique_wards']:,} unique wards with conflict")
        print(f"   - {summary['total_deaths']['total']:,} total deaths")
        print(f"   - {summary['total_deaths']['state_violence']:,} state violence deaths")
        print(f"   - {summary['total_deaths']['nonstate_violence']:,} non-state violence deaths")
        print(f"   - Date range: {summary['date_range']['start_year']}-{summary['date_range']['end_year']}")
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
