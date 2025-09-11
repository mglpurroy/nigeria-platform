#!/usr/bin/env python3
"""
Pre-generate static PNG/PDF charts for all ward time series data.
This script creates cached static charts to improve dashboard performance.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import time
from tqdm import tqdm

def create_static_timeseries_charts(ward_data, output_dir="data/processed/static_charts"):
    """Create static PNG/PDF charts for ward time series data"""
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if not ward_data or 'time_series' not in ward_data:
        return None, None
    
    time_series = ward_data['time_series']
    ward_info = ward_data['ward_info']
    
    if not time_series:
        return None, None
    
    # Convert to DataFrame
    df = pd.DataFrame(time_series)
    df['date'] = pd.to_datetime(df['year_month'], format='%Y-%m')
    df = df.sort_values('date')
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f"Violence Analysis - {ward_info['wardname']}\n{ward_info['lganame']}, {ward_info['statename']}", 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Event Count
    ax1.bar(df['date'], df['event_count'], color='#1f77b4', alpha=0.7, width=20)
    ax1.set_title('Event Count Over Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Events', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Total Fatalities
    ax2.bar(df['date'], df['total_fatalities'], color='#d62728', alpha=0.7, width=20)
    ax2.set_title('Total Fatalities Over Time', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Fatalities', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Add summary statistics
    total_events = df['event_count'].sum()
    total_fatalities = df['total_fatalities'].sum()
    date_range = f"{df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}"
    
    # Add text box with summary
    summary_text = f"Summary:\nTotal Events: {total_events:,}\nTotal Fatalities: {total_fatalities:,}\nPeriod: {date_range}"
    fig.text(0.02, 0.02, summary_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save as PNG and PDF
    ward_name_clean = ward_info['wardname'].replace(' ', '_').replace('/', '_')
    png_path = os.path.join(output_dir, f"{ward_name_clean}_{ward_info['wardcode']}_timeseries.png")
    pdf_path = os.path.join(output_dir, f"{ward_name_clean}_{ward_info['wardcode']}_timeseries.pdf")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    return png_path, pdf_path

def load_all_ward_timeseries_data():
    """Load all ward time series data"""
    ward_timeseries_path = Path("data/processed/ward_timeseries")
    
    if not ward_timeseries_path.exists():
        print("❌ Ward timeseries directory not found!")
        return {}
    
    # Load all JSON files
    all_data = {}
    json_files = list(ward_timeseries_path.glob("*.json"))
    
    print(f"📁 Found {len(json_files)} ward timeseries files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if 'ward_info' in data and 'wardcode' in data['ward_info']:
                    ward_code = data['ward_info']['wardcode']
                    all_data[ward_code] = data
        except Exception as e:
            print(f"⚠️ Error loading {json_file}: {e}")
            continue
    
    print(f"✅ Loaded {len(all_data)} ward datasets")
    return all_data

def main():
    """Main function to generate static charts for all wards"""
    print("🚀 Starting static chart generation for all wards...")
    
    # Load all ward data
    all_ward_data = load_all_ward_timeseries_data()
    
    if not all_ward_data:
        print("❌ No ward data found. Exiting.")
        return
    
    # Create output directory
    output_dir = "data/processed/static_charts"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate charts for each ward
    successful_charts = 0
    failed_charts = 0
    
    print(f"📊 Generating charts for {len(all_ward_data)} wards...")
    
    for ward_code, ward_data in tqdm(all_ward_data.items(), desc="Generating charts"):
        try:
            png_path, pdf_path = create_static_timeseries_charts(ward_data, output_dir)
            if png_path and pdf_path:
                successful_charts += 1
            else:
                failed_charts += 1
        except Exception as e:
            print(f"⚠️ Error generating charts for ward {ward_code}: {e}")
            failed_charts += 1
    
    print(f"\n✅ Chart generation complete!")
    print(f"📊 Successful: {successful_charts}")
    print(f"❌ Failed: {failed_charts}")
    print(f"📁 Charts saved to: {output_dir}")
    
    # Create a summary file
    summary = {
        'total_wards': len(all_ward_data),
        'successful_charts': successful_charts,
        'failed_charts': failed_charts,
        'output_directory': output_dir,
        'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(output_dir, 'generation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Summary saved to: {os.path.join(output_dir, 'generation_summary.json')}")

if __name__ == "__main__":
    main()
