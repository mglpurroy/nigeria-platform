#!/usr/bin/env python3
"""
Debug script for conflict data processing
"""

import pandas as pd
import numpy as np

def debug_conflict_data():
    print("🔍 Debugging conflict data processing...")
    
    # Load raw ACLED data
    df = pd.read_csv('data/acled_nigeria_data.csv')
    print(f"Raw ACLED data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check data types
    print(f"\nData types:")
    print(df.dtypes)
    
    # Check for missing values
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    # Check admin1 and admin2 values
    print(f"\nUnique admin1 values (first 10): {df['admin1'].unique()[:10]}")
    print(f"Unique admin2 values (first 10): {df['admin2'].unique()[:10]}")
    
    # Check fatalities
    print(f"\nFatalities stats:")
    print(f"Min: {df['fatalities'].min()}")
    print(f"Max: {df['fatalities'].max()}")
    print(f"Mean: {df['fatalities'].mean()}")
    print(f"Non-zero fatalities: {(df['fatalities'] > 0).sum()}")
    
    # Test the processing logic
    print(f"\nTesting processing logic...")
    
    # Filter for non-zero fatalities
    df_filtered = df[df['fatalities'] > 0].copy()
    print(f"Records with fatalities > 0: {len(df_filtered)}")
    
    if len(df_filtered) > 0:
        # Test grouping
        try:
            grouped = df_filtered.groupby(['year', 'admin1', 'admin2'], as_index=False).agg({
                'fatalities': 'sum'
            })
            print(f"Grouped data shape: {grouped.shape}")
            print(f"Sample grouped data:")
            print(grouped.head())
            
            # Test the full processing
            grouped['ADM1_PCODE'] = grouped['admin1'].astype(str)
            grouped['ADM2_PCODE'] = grouped['admin2'].astype(str)
            grouped = grouped.rename(columns={
                'fatalities': 'ACLED_BRD_total',
                'admin1': 'ADM1_EN',
                'admin2': 'ADM2_EN'
            })
            grouped['ACLED_BRD_state'] = grouped['ACLED_BRD_total'] * 0.3
            grouped['ACLED_BRD_nonstate'] = grouped['ACLED_BRD_total'] * 0.7
            
            print(f"Final processed data shape: {grouped.shape}")
            print(f"Final columns: {list(grouped.columns)}")
            print(f"Sample final data:")
            print(grouped.head())
            
        except Exception as e:
            print(f"Error in processing: {e}")
    else:
        print("No records with fatalities > 0 found")

if __name__ == "__main__":
    debug_conflict_data()
