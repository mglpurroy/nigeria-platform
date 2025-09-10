#!/usr/bin/env python3
"""
Validate the ward-level time series data for quality and completeness.
"""

import pandas as pd
import json
import os
from datetime import datetime

def validate_complete_dataset():
    """Validate the complete dataset."""
    print("Validating complete dataset...")
    
    df = pd.read_csv('data/processed/ward_timeseries_complete.csv')
    
    print(f"Total records: {len(df)}")
    print(f"Unique wards: {df['wardname'].nunique()}")
    print(f"Date range: {df['year_month'].min()} to {df['year_month'].max()}")
    print(f"Total events: {df['event_count'].sum()}")
    print(f"Total fatalities: {df['total_fatalities'].sum()}")
    
    # Check for missing values
    missing_data = df.isnull().sum()
    if missing_data.any():
        print("Missing values found:")
        print(missing_data[missing_data > 0])
    else:
        print("No missing values found.")
    
    # Check for negative values
    negative_events = (df['event_count'] < 0).sum()
    negative_fatalities = (df['total_fatalities'] < 0).sum()
    
    if negative_events > 0:
        print(f"Warning: {negative_events} records with negative event counts")
    if negative_fatalities > 0:
        print(f"Warning: {negative_fatalities} records with negative fatalities")
    
    return df

def validate_individual_files():
    """Validate individual ward time series files."""
    print("\nValidating individual ward files...")
    
    timeseries_dir = 'data/processed/ward_timeseries'
    files = [f for f in os.listdir(timeseries_dir) if f.endswith('.json')]
    
    print(f"Total ward files: {len(files)}")
    
    valid_files = 0
    invalid_files = 0
    
    for file in files[:10]:  # Check first 10 files
        try:
            with open(os.path.join(timeseries_dir, file), 'r') as f:
                data = json.load(f)
            
            # Check structure
            if 'ward_info' in data and 'time_series' in data:
                ward_info = data['ward_info']
                time_series = data['time_series']
                
                if all(key in ward_info for key in ['wardname', 'wardcode', 'lganame', 'statename']):
                    if len(time_series) > 0:
                        # Check time series structure
                        sample = time_series[0]
                        if all(key in sample for key in ['year_month', 'event_count', 'total_fatalities', 'event_type_breakdown']):
                            valid_files += 1
                        else:
                            print(f"Invalid time series structure in {file}")
                            invalid_files += 1
                    else:
                        print(f"Empty time series in {file}")
                        invalid_files += 1
                else:
                    print(f"Invalid ward_info structure in {file}")
                    invalid_files += 1
            else:
                print(f"Invalid file structure in {file}")
                invalid_files += 1
                
        except Exception as e:
            print(f"Error reading {file}: {e}")
            invalid_files += 1
    
    print(f"Valid files (sample): {valid_files}")
    print(f"Invalid files (sample): {invalid_files}")

def validate_summary_files():
    """Validate summary files."""
    print("\nValidating summary files...")
    
    # Check summary file
    try:
        with open('data/processed/ward_timeseries_summary.json', 'r') as f:
            summary = json.load(f)
        
        print("Summary file structure:")
        print(f"- Overview: {list(summary['overview'].keys())}")
        print(f"- Top wards by fatalities: {len(summary['top_wards_by_fatalities'])}")
        print(f"- Top wards by events: {len(summary['top_wards_by_events'])}")
        
    except Exception as e:
        print(f"Error reading summary file: {e}")
    
    # Check analysis report
    try:
        with open('data/processed/ward_timeseries_analysis_report.json', 'r') as f:
            report = json.load(f)
        
        print("\nAnalysis report structure:")
        print(f"- Analysis metadata: {list(report['analysis_metadata'].keys())}")
        print(f"- Overall statistics: {list(report['overall_statistics'].keys())}")
        print(f"- Temporal analysis: {list(report['temporal_analysis'].keys())}")
        print(f"- Ward analysis: {list(report['ward_analysis'].keys())}")
        
    except Exception as e:
        print(f"Error reading analysis report: {e}")

def main():
    """Main validation function."""
    print("Starting ward timeseries data validation...")
    
    # Validate complete dataset
    df = validate_complete_dataset()
    
    # Validate individual files
    validate_individual_files()
    
    # Validate summary files
    validate_summary_files()
    
    print("\nValidation complete!")

if __name__ == "__main__":
    main()
