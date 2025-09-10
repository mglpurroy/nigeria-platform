#!/usr/bin/env python3
"""
Process ACLED data to create ward-level time series of fatalities and events.
This script performs spatial intersection between ACLED events and ward boundaries,
then aggregates data by ward and month.
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from datetime import datetime
import json
import os

def load_data():
    """Load ACLED data and ward boundaries."""
    print("Loading ACLED data...")
    acled_df = pd.read_csv('data/acled_nigeria_data.csv')
    
    print("Loading ward boundaries...")
    wards_gdf = gpd.read_file('data/nga_ward_boundaries.geojson')
    
    return acled_df, wards_gdf

def prepare_acled_data(acled_df):
    """Prepare ACLED data for spatial analysis."""
    print("Preparing ACLED data...")
    
    # Convert event_date to datetime
    acled_df['event_date'] = pd.to_datetime(acled_df['event_date'])
    
    # Create year-month column for aggregation
    acled_df['year_month'] = acled_df['event_date'].dt.to_period('M')
    
    # Create geometry points from lat/lon
    geometry = [Point(xy) for xy in zip(acled_df['longitude'], acled_df['latitude'])]
    acled_gdf = gpd.GeoDataFrame(acled_df, geometry=geometry, crs='EPSG:4326')
    
    # Filter out events with missing coordinates
    acled_gdf = acled_gdf.dropna(subset=['latitude', 'longitude'])
    
    print(f"ACLED data prepared: {len(acled_gdf)} events with valid coordinates")
    return acled_gdf

def perform_spatial_join(acled_gdf, wards_gdf):
    """Perform spatial join between ACLED events and ward boundaries."""
    print("Performing spatial join...")
    
    # Ensure both GeoDataFrames have the same CRS
    if acled_gdf.crs != wards_gdf.crs:
        acled_gdf = acled_gdf.to_crs(wards_gdf.crs)
    
    # Perform spatial join
    joined_gdf = gpd.sjoin(acled_gdf, wards_gdf, how='left', predicate='within')
    
    # Count events that couldn't be matched to wards
    unmatched = joined_gdf['wardname'].isna().sum()
    print(f"Events matched to wards: {len(joined_gdf) - unmatched}")
    print(f"Events not matched to any ward: {unmatched}")
    
    return joined_gdf

def create_ward_timeseries(joined_gdf):
    """Create time series data for each ward."""
    print("Creating ward-level time series...")
    
    # Filter out events that couldn't be matched to wards
    matched_events = joined_gdf.dropna(subset=['wardname'])
    
    # Group by ward and month, then aggregate
    ward_timeseries = matched_events.groupby(['wardname', 'wardcode', 'lganame', 'statename', 'year_month']).agg({
        'event_id_cnty': 'count',  # Count of events
        'fatalities': 'sum',       # Sum of fatalities
        'event_type': lambda x: x.value_counts().to_dict()  # Event type breakdown
    }).reset_index()
    
    # Rename columns
    ward_timeseries.columns = ['wardname', 'wardcode', 'lganame', 'statename', 'year_month', 
                              'event_count', 'total_fatalities', 'event_type_breakdown']
    
    # Convert year_month back to string for JSON serialization
    ward_timeseries['year_month'] = ward_timeseries['year_month'].astype(str)
    
    return ward_timeseries

def create_individual_ward_series(ward_timeseries):
    """Create individual time series files for each ward."""
    print("Creating individual ward time series files...")
    
    # Create output directory
    output_dir = 'data/processed/ward_timeseries'
    os.makedirs(output_dir, exist_ok=True)
    
    ward_series = {}
    
    for ward in ward_timeseries['wardname'].unique():
        ward_data = ward_timeseries[ward_timeseries['wardname'] == ward].copy()
        
        # Sort by date
        ward_data = ward_data.sort_values('year_month')
        
        # Create time series structure
        timeseries = {
            'ward_info': {
                'wardname': ward_data.iloc[0]['wardname'],
                'wardcode': ward_data.iloc[0]['wardcode'],
                'lganame': ward_data.iloc[0]['lganame'],
                'statename': ward_data.iloc[0]['statename']
            },
            'time_series': []
        }
        
        # Add monthly data
        for _, row in ward_data.iterrows():
            month_data = {
                'year_month': row['year_month'],
                'event_count': int(row['event_count']),
                'total_fatalities': int(row['total_fatalities']),
                'event_type_breakdown': row['event_type_breakdown']
            }
            timeseries['time_series'].append(month_data)
        
        # Save individual ward file
        filename = f"{ward.replace(' ', '_').replace('/', '_')}_{ward_data.iloc[0]['wardcode']}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(timeseries, f, indent=2)
        
        ward_series[ward] = timeseries
    
    return ward_series

def create_summary_statistics(ward_timeseries, ward_series):
    """Create summary statistics across all wards."""
    print("Creating summary statistics...")
    
    # Overall statistics
    total_events = ward_timeseries['event_count'].sum()
    total_fatalities = ward_timeseries['total_fatalities'].sum()
    total_wards = len(ward_series)
    
    # Date range
    min_date = ward_timeseries['year_month'].min()
    max_date = ward_timeseries['year_month'].max()
    
    # Ward-level statistics
    ward_stats = ward_timeseries.groupby(['wardname', 'wardcode', 'lganame', 'statename']).agg({
        'event_count': 'sum',
        'total_fatalities': 'sum'
    }).reset_index()
    
    ward_stats = ward_stats.sort_values('total_fatalities', ascending=False)
    
    summary = {
        'overview': {
            'total_wards': total_wards,
            'total_events': int(total_events),
            'total_fatalities': int(total_fatalities),
            'date_range': {
                'start': min_date,
                'end': max_date
            }
        },
        'top_wards_by_fatalities': ward_stats.head(20).to_dict('records'),
        'top_wards_by_events': ward_stats.sort_values('event_count', ascending=False).head(20).to_dict('records')
    }
    
    # Save summary
    with open('data/processed/ward_timeseries_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def main():
    """Main processing function."""
    print("Starting ACLED ward-level time series processing...")
    
    # Load data
    acled_df, wards_gdf = load_data()
    
    # Prepare ACLED data
    acled_gdf = prepare_acled_data(acled_df)
    
    # Perform spatial join
    joined_gdf = perform_spatial_join(acled_gdf, wards_gdf)
    
    # Create time series
    ward_timeseries = create_ward_timeseries(joined_gdf)
    
    # Create individual ward series
    ward_series = create_individual_ward_series(ward_timeseries)
    
    # Create summary statistics
    summary = create_summary_statistics(ward_timeseries, ward_series)
    
    # Save complete dataset
    ward_timeseries.to_csv('data/processed/ward_timeseries_complete.csv', index=False)
    
    print(f"\nProcessing complete!")
    print(f"Total wards processed: {summary['overview']['total_wards']}")
    print(f"Total events: {summary['overview']['total_events']}")
    print(f"Total fatalities: {summary['overview']['total_fatalities']}")
    print(f"Date range: {summary['overview']['date_range']['start']} to {summary['overview']['date_range']['end']}")
    print(f"Individual ward files saved to: data/processed/ward_timeseries/")
    print(f"Summary saved to: data/processed/ward_timeseries_summary.json")
    print(f"Complete dataset saved to: data/processed/ward_timeseries_complete.csv")

if __name__ == "__main__":
    main()
