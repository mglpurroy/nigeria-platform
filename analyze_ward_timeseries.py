#!/usr/bin/env python3
"""
Analyze ward-level time series data from ACLED events.
This script provides detailed analysis and creates visualizations of the time series data.
"""

import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

def load_timeseries_data():
    """Load the complete ward timeseries dataset."""
    print("Loading ward timeseries data...")
    df = pd.read_csv('data/processed/ward_timeseries_complete.csv')
    df['year_month'] = pd.to_datetime(df['year_month'])
    return df

def analyze_temporal_patterns(df):
    """Analyze temporal patterns in the data."""
    print("Analyzing temporal patterns...")
    
    # Monthly aggregation across all wards
    monthly_totals = df.groupby('year_month').agg({
        'event_count': 'sum',
        'total_fatalities': 'sum'
    }).reset_index()
    
    # Yearly aggregation
    df['year'] = df['year_month'].dt.year
    yearly_totals = df.groupby('year').agg({
        'event_count': 'sum',
        'total_fatalities': 'sum'
    }).reset_index()
    
    return monthly_totals, yearly_totals

def analyze_ward_patterns(df):
    """Analyze patterns by ward."""
    print("Analyzing ward patterns...")
    
    # Ward-level totals
    ward_totals = df.groupby(['wardname', 'wardcode', 'lganame', 'statename']).agg({
        'event_count': 'sum',
        'total_fatalities': 'sum',
        'year_month': ['min', 'max', 'count']  # First event, last event, months with events
    }).reset_index()
    
    # Flatten column names
    ward_totals.columns = ['wardname', 'wardcode', 'lganame', 'statename', 
                          'total_events', 'total_fatalities', 'first_event', 'last_event', 'months_with_events']
    
    # Calculate event intensity (events per month with activity)
    ward_totals['event_intensity'] = ward_totals['total_events'] / ward_totals['months_with_events']
    ward_totals['fatality_rate'] = ward_totals['total_fatalities'] / ward_totals['total_events']
    
    return ward_totals

def analyze_event_types(df):
    """Analyze event type patterns."""
    print("Analyzing event type patterns...")
    
    # Parse event type breakdowns
    event_type_data = []
    for _, row in df.iterrows():
        if pd.notna(row['event_type_breakdown']):
            breakdown = eval(row['event_type_breakdown'])  # Convert string dict to dict
            for event_type, count in breakdown.items():
                event_type_data.append({
                    'year_month': row['year_month'],
                    'wardname': row['wardname'],
                    'statename': row['statename'],
                    'event_type': event_type,
                    'count': count
                })
    
    event_type_df = pd.DataFrame(event_type_data)
    
    if not event_type_df.empty:
        # Overall event type distribution
        event_type_totals = event_type_df.groupby('event_type')['count'].sum().sort_values(ascending=False)
        
        # Event types by state
        event_type_by_state = event_type_df.groupby(['statename', 'event_type'])['count'].sum().unstack(fill_value=0)
        
        return event_type_totals, event_type_by_state
    else:
        return None, None

def create_visualizations(monthly_totals, yearly_totals, ward_totals, event_type_totals):
    """Create visualizations of the analysis."""
    print("Creating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory
    output_dir = 'data/processed/visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Monthly trends
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Events over time
    ax1.plot(monthly_totals['year_month'], monthly_totals['event_count'], linewidth=1, alpha=0.7)
    ax1.set_title('ACLED Events by Month (All Wards)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Events')
    ax1.grid(True, alpha=0.3)
    
    # Fatalities over time
    ax2.plot(monthly_totals['year_month'], monthly_totals['total_fatalities'], 
             color='red', linewidth=1, alpha=0.7)
    ax2.set_title('ACLED Fatalities by Month (All Wards)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Fatalities')
    ax2.set_xlabel('Year')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/monthly_trends.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Yearly trends
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.bar(yearly_totals['year'], yearly_totals['event_count'], alpha=0.7)
    ax1.set_title('ACLED Events by Year', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Events')
    ax1.set_xlabel('Year')
    
    ax2.bar(yearly_totals['year'], yearly_totals['total_fatalities'], 
            color='red', alpha=0.7)
    ax2.set_title('ACLED Fatalities by Year', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Fatalities')
    ax2.set_xlabel('Year')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/yearly_trends.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Top wards by events and fatalities
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    top_events = ward_totals.nlargest(15, 'total_events')
    top_fatalities = ward_totals.nlargest(15, 'total_fatalities')
    
    # Top wards by events
    ax1.barh(range(len(top_events)), top_events['total_events'])
    ax1.set_yticks(range(len(top_events)))
    ax1.set_yticklabels([f"{row['wardname']}, {row['statename']}" for _, row in top_events.iterrows()], fontsize=8)
    ax1.set_title('Top 15 Wards by Total Events', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Total Events')
    
    # Top wards by fatalities
    ax2.barh(range(len(top_fatalities)), top_fatalities['total_fatalities'], color='red')
    ax2.set_yticks(range(len(top_fatalities)))
    ax2.set_yticklabels([f"{row['wardname']}, {row['statename']}" for _, row in top_fatalities.iterrows()], fontsize=8)
    ax2.set_title('Top 15 Wards by Total Fatalities', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Total Fatalities')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_wards.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Event type distribution
    if event_type_totals is not None:
        plt.figure(figsize=(12, 8))
        event_type_totals.plot(kind='bar')
        plt.title('Distribution of Event Types', fontsize=14, fontweight='bold')
        plt.xlabel('Event Type')
        plt.ylabel('Total Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/event_type_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. State-level analysis
    state_totals = ward_totals.groupby('statename').agg({
        'total_events': 'sum',
        'total_fatalities': 'sum',
        'wardname': 'count'  # Number of wards with events
    }).reset_index()
    state_totals.columns = ['statename', 'total_events', 'total_fatalities', 'wards_with_events']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Events by state
    state_events = state_totals.nlargest(15, 'total_events')
    ax1.barh(range(len(state_events)), state_events['total_events'])
    ax1.set_yticks(range(len(state_events)))
    ax1.set_yticklabels(state_events['statename'], fontsize=10)
    ax1.set_title('Top 15 States by Total Events', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Total Events')
    
    # Fatalities by state
    state_fatalities = state_totals.nlargest(15, 'total_fatalities')
    ax2.barh(range(len(state_fatalities)), state_fatalities['total_fatalities'], color='red')
    ax2.set_yticks(range(len(state_fatalities)))
    ax2.set_yticklabels(state_fatalities['statename'], fontsize=10)
    ax2.set_title('Top 15 States by Total Fatalities', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Total Fatalities')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/state_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to: {output_dir}/")

def create_detailed_report(df, monthly_totals, yearly_totals, ward_totals, event_type_totals):
    """Create a detailed analysis report."""
    print("Creating detailed analysis report...")
    
    report = {
        'analysis_metadata': {
            'analysis_date': datetime.now().isoformat(),
            'total_wards_analyzed': len(ward_totals),
            'date_range': {
                'start': df['year_month'].min().strftime('%Y-%m'),
                'end': df['year_month'].max().strftime('%Y-%m')
            }
        },
        'overall_statistics': {
            'total_events': int(df['event_count'].sum()),
            'total_fatalities': int(df['total_fatalities'].sum()),
            'average_events_per_month': float(df.groupby('year_month')['event_count'].sum().mean()),
            'average_fatalities_per_month': float(df.groupby('year_month')['total_fatalities'].sum().mean())
        },
        'temporal_analysis': {
            'peak_month_events': {
                'month': monthly_totals.loc[monthly_totals['event_count'].idxmax(), 'year_month'].strftime('%Y-%m'),
                'events': int(monthly_totals['event_count'].max())
            },
            'peak_month_fatalities': {
                'month': monthly_totals.loc[monthly_totals['total_fatalities'].idxmax(), 'year_month'].strftime('%Y-%m'),
                'fatalities': int(monthly_totals['total_fatalities'].max())
            },
            'peak_year_events': {
                'year': int(yearly_totals.loc[yearly_totals['event_count'].idxmax(), 'year']),
                'events': int(yearly_totals['event_count'].max())
            },
            'peak_year_fatalities': {
                'year': int(yearly_totals.loc[yearly_totals['total_fatalities'].idxmax(), 'year']),
                'fatalities': int(yearly_totals['total_fatalities'].max())
            }
        },
        'ward_analysis': {
            'most_active_wards': ward_totals.nlargest(10, 'total_events')[['wardname', 'statename', 'total_events', 'total_fatalities']].to_dict('records'),
            'most_fatal_wards': ward_totals.nlargest(10, 'total_fatalities')[['wardname', 'statename', 'total_events', 'total_fatalities']].to_dict('records'),
            'highest_intensity_wards': ward_totals.nlargest(10, 'event_intensity')[['wardname', 'statename', 'event_intensity', 'total_events']].to_dict('records')
        }
    }
    
    if event_type_totals is not None:
        report['event_type_analysis'] = {
            'distribution': event_type_totals.to_dict(),
            'most_common_event_type': event_type_totals.index[0],
            'least_common_event_type': event_type_totals.index[-1]
        }
    
    # Save report
    with open('data/processed/ward_timeseries_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

def main():
    """Main analysis function."""
    print("Starting ward timeseries analysis...")
    
    # Load data
    df = load_timeseries_data()
    
    # Perform analyses
    monthly_totals, yearly_totals = analyze_temporal_patterns(df)
    ward_totals = analyze_ward_patterns(df)
    event_type_totals, event_type_by_state = analyze_event_types(df)
    
    # Create visualizations
    create_visualizations(monthly_totals, yearly_totals, ward_totals, event_type_totals)
    
    # Create detailed report
    report = create_detailed_report(df, monthly_totals, yearly_totals, ward_totals, event_type_totals)
    
    print(f"\nAnalysis complete!")
    print(f"Total wards with events: {len(ward_totals)}")
    print(f"Total events analyzed: {report['overall_statistics']['total_events']}")
    print(f"Total fatalities: {report['overall_statistics']['total_fatalities']}")
    print(f"Date range: {report['analysis_metadata']['date_range']['start']} to {report['analysis_metadata']['date_range']['end']}")
    print(f"Analysis report saved to: data/processed/ward_timeseries_analysis_report.json")
    print(f"Visualizations saved to: data/processed/visualizations/")

if __name__ == "__main__":
    main()
