#!/usr/bin/env python3
"""
Simplified BRD processing for Nigeria Violence Dashboard
- Calculate total BRD for each time period and ward
- Classify wards as violence-affected based on BRD per 100k > threshold
- Aggregate to LGA and State levels with share of population/wards affected
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_comprehensive_data():
    """Load comprehensive ward population and conflict data"""
    print("📊 Loading comprehensive data...")
    
    # Load population data
    pop_file = Path("data/processed/nigeria_ward_population_comprehensive.json")
    if not pop_file.exists():
        raise FileNotFoundError(f"Population file not found: {pop_file}")
    
    with open(pop_file, 'r') as f:
        pop_data = json.load(f)
    
    pop_df = pd.DataFrame(pop_data)
    print(f"✅ Loaded population data: {len(pop_df)} wards")
    
    # Load conflict data
    conflict_file = Path("data/acled_nigeria_data.csv")
    if not conflict_file.exists():
        raise FileNotFoundError(f"Conflict file not found: {conflict_file}")
    
    conflict_df = pd.read_csv(conflict_file)
    print(f"✅ Loaded conflict data: {len(conflict_df)} events")
    
    return pop_df, conflict_df

def process_conflict_data(conflict_df):
    """Process ACLED data to extract BRD events"""
    print("🔍 Processing conflict data...")
    
    # Filter for BRD events (exclude Protests and Riots, fatalities > 0)
    brd_events = conflict_df[
        (~conflict_df['event_type'].isin(['Protests', 'Riots'])) &
        (conflict_df['fatalities'] > 0)
    ].copy()
    
    print(f"📊 BRD events: {len(brd_events)} (from {len(conflict_df)} total events)")
    
    # Categorize violence type based on interaction
    def categorize_violence(interaction):
        if pd.isna(interaction):
            return 'unknown', 'unknown'
        
        interaction_lower = str(interaction).lower()
        
        # State violence indicators
        state_indicators = ['military', 'police', 'state', 'government', 'security forces']
        if any(indicator in interaction_lower for indicator in state_indicators):
            return 'state', 'state'
        
        # Non-state violence indicators
        nonstate_indicators = ['civilians', 'militia', 'rebels', 'insurgents', 'terrorists', 'armed groups']
        if any(indicator in interaction_lower for indicator in nonstate_indicators):
            return 'nonstate', 'nonstate'
        
        # Default to nonstate if unclear
        return 'nonstate', 'nonstate'
    
    brd_events[['violence_type', 'actor_type']] = brd_events['interaction'].apply(
        lambda x: pd.Series(categorize_violence(x))
    )
    
    # Aggregate by year, month, and LGA (since we don't have ward-level conflict data)
    conflict_aggregated = brd_events.groupby(['year', 'month', 'admin1', 'admin2', 'violence_type'], as_index=False).agg({
        'fatalities': 'sum'
    })
    
    # Pivot to get state and nonstate columns
    conflict_pivot = conflict_aggregated.pivot_table(
        index=['year', 'month', 'admin1', 'admin2'],
        columns='violence_type',
        values='fatalities',
        fill_value=0
    ).reset_index()
    
    # Rename columns and calculate totals
    if 'state' in conflict_pivot.columns:
        conflict_pivot = conflict_pivot.rename(columns={'state': 'ACLED_BRD_state'})
    else:
        conflict_pivot['ACLED_BRD_state'] = 0
    
    if 'nonstate' in conflict_pivot.columns:
        conflict_pivot = conflict_pivot.rename(columns={'nonstate': 'ACLED_BRD_nonstate'})
    else:
        conflict_pivot['ACLED_BRD_nonstate'] = 0
    
    conflict_pivot['ACLED_BRD_total'] = conflict_pivot['ACLED_BRD_state'] + conflict_pivot['ACLED_BRD_nonstate']
    
    # Rename admin columns to match population data
    conflict_pivot = conflict_pivot.rename(columns={
        'admin1': 'ADM1_EN',
        'admin2': 'ADM2_EN'
    })
    
    # Filter out zero-death records
    conflict_pivot = conflict_pivot[conflict_pivot['ACLED_BRD_total'] > 0]
    
    print(f"✅ Processed conflict data: {len(conflict_pivot)} LGA-period records")
    print(f"   Total BRD deaths: {conflict_pivot['ACLED_BRD_total'].sum():,}")
    print(f"   State violence: {conflict_pivot['ACLED_BRD_state'].sum():,}")
    print(f"   Non-state violence: {conflict_pivot['ACLED_BRD_nonstate'].sum():,}")
    
    return conflict_pivot

def generate_time_periods():
    """Generate 12-month periods every 6 months"""
    periods = []
    
    # Calendar year periods (Jan-Dec)
    for year in range(1997, 2026):
        periods.append({
            'label': f'Jan {year} - Dec {year}',
            'start_month': 1,
            'start_year': year,
            'end_month': 12,
            'end_year': year,
            'type': 'calendar'
        })
    
    # Mid-year periods (Jul-Jun)
    for year in range(1997, 2025):
        periods.append({
            'label': f'Jul {year} - Jun {year+1}',
            'start_month': 7,
            'start_year': year,
            'end_month': 6,
            'end_year': year + 1,
            'type': 'mid_year'
        })
    
    return periods

def filter_conflict_by_period(conflict_df, period_info):
    """Filter conflict data for a specific time period"""
    start_year = period_info['start_year']
    end_year = period_info['end_year']
    start_month = period_info['start_month']
    end_month = period_info['end_month']
    
    if period_info['type'] == 'calendar':
        # Calendar year: Jan-Dec
        mask = (conflict_df['year'] == start_year) & (conflict_df['month'] >= start_month) & (conflict_df['month'] <= end_month)
    else:
        # Mid-year: Jul-Jun
        mask = ((conflict_df['year'] == start_year) & (conflict_df['month'] >= start_month)) | \
               ((conflict_df['year'] == end_year) & (conflict_df['month'] <= end_month))
    
    return conflict_df[mask]

def distribute_conflict_to_wards(pop_df, period_conflict):
    """Distribute LGA-level conflict data to wards proportionally by population"""
    if len(period_conflict) == 0:
        # No conflict data for this period
        pop_df['ACLED_BRD_state'] = 0
        pop_df['ACLED_BRD_nonstate'] = 0
        pop_df['ACLED_BRD_total'] = 0
        return pop_df
    
    # Merge population data with conflict data
    merged = pd.merge(pop_df, period_conflict, on=['ADM1_EN', 'ADM2_EN'], how='left')
    
    # Fill missing values
    conflict_cols = ['ACLED_BRD_state', 'ACLED_BRD_nonstate', 'ACLED_BRD_total']
    merged[conflict_cols] = merged[conflict_cols].fillna(0)
    
    # Calculate population share within each LGA
    lga_pop = merged.groupby(['ADM1_EN', 'ADM2_EN'])['pop_count'].sum().reset_index()
    lga_pop = lga_pop.rename(columns={'pop_count': 'lga_total_pop'})
    
    merged = pd.merge(merged, lga_pop, on=['ADM1_EN', 'ADM2_EN'], how='left')
    merged['pop_share'] = merged['pop_count'] / merged['lga_total_pop']
    merged['pop_share'] = merged['pop_share'].fillna(0)
    
    # Distribute conflict deaths proportionally
    for col in conflict_cols:
        merged[col] = merged[col] * merged['pop_share']
    
    # Clean up
    merged = merged.drop(['lga_total_pop', 'pop_share'], axis=1)
    
    return merged

def classify_wards(ward_data, threshold_per_100k):
    """Classify wards as violence-affected based on BRD per 100k population"""
    # Calculate BRD rate per 100k population
    ward_data['brd_rate_per_100k'] = (ward_data['ACLED_BRD_total'] / ward_data['pop_count']) * 100000
    
    # Handle division by zero
    ward_data['brd_rate_per_100k'] = ward_data['brd_rate_per_100k'].fillna(0)
    ward_data['brd_rate_per_100k'] = ward_data['brd_rate_per_100k'].replace([np.inf, -np.inf], 0)
    
    # Classify as violence-affected
    ward_data['violence_affected'] = ward_data['brd_rate_per_100k'] > threshold_per_100k
    
    return ward_data

def aggregate_to_admin_levels(ward_data, admin_level):
    """Aggregate ward data to LGA (ADM2) or State (ADM1) level"""
    if admin_level == 'ADM2':
        group_cols = ['ADM2_PCODE', 'ADM2_EN', 'ADM1_PCODE', 'ADM1_EN']
    else:  # ADM1
        group_cols = ['ADM1_PCODE', 'ADM1_EN']
    
    # Aggregate ward data
    aggregated = ward_data.groupby(group_cols, as_index=False).agg({
        'pop_count': 'sum',
        'violence_affected': 'sum',
        'ADM3_PCODE': 'count',
        'ACLED_BRD_total': 'sum',
        'ACLED_BRD_state': 'sum',
        'ACLED_BRD_nonstate': 'sum'
    })
    
    # Rename count column
    aggregated = aggregated.rename(columns={'ADM3_PCODE': 'total_wards'})
    
    # Calculate shares
    aggregated['share_wards_affected'] = aggregated['violence_affected'] / aggregated['total_wards']
    aggregated['share_wards_affected'] = aggregated['share_wards_affected'].fillna(0)
    
    # Calculate population share affected
    affected_pop = ward_data[ward_data['violence_affected']].groupby(group_cols[0], as_index=False)['pop_count'].sum()
    affected_pop = affected_pop.rename(columns={'pop_count': 'affected_population'})
    
    aggregated = pd.merge(aggregated, affected_pop, on=group_cols[0], how='left')
    aggregated['affected_population'] = aggregated['affected_population'].fillna(0)
    aggregated['share_population_affected'] = aggregated['affected_population'] / aggregated['pop_count']
    aggregated['share_population_affected'] = aggregated['share_population_affected'].fillna(0)
    
    return aggregated

def process_period_data(pop_df, conflict_df, period_info, threshold_per_100k):
    """Process data for a specific time period"""
    print(f"📅 Processing period: {period_info['label']}")
    
    # Filter conflict data for period
    period_conflict = filter_conflict_by_period(conflict_df, period_info)
    
    # Distribute conflict to wards
    ward_data = distribute_conflict_to_wards(pop_df.copy(), period_conflict)
    
    # Classify wards
    ward_data = classify_wards(ward_data, threshold_per_100k)
    
    # Aggregate to admin levels
    lga_data = aggregate_to_admin_levels(ward_data, 'ADM2')
    state_data = aggregate_to_admin_levels(ward_data, 'ADM1')
    
    return {
        'ward_data': ward_data,
        'lga_data': lga_data,
        'state_data': state_data,
        'period_info': period_info,
        'threshold_per_100k': threshold_per_100k
    }

def main():
    """Main processing function"""
    print("🚀 Starting simplified BRD processing...")
    
    # Load data
    pop_df, conflict_df = load_comprehensive_data()
    
    # Process conflict data
    conflict_processed = process_conflict_data(conflict_df)
    
    # Generate periods
    periods = generate_time_periods()
    print(f"📅 Generated {len(periods)} time periods")
    
    # Test with a few periods and thresholds
    test_periods = periods[:5]  # First 5 periods
    test_thresholds = [1.0, 5.0, 10.0]  # Different thresholds
    
    results = {}
    
    for threshold in test_thresholds:
        print(f"\n🎯 Testing threshold: {threshold} BRD per 100k population")
        results[threshold] = {}
        
        for period in test_periods:
            try:
                result = process_period_data(pop_df, conflict_processed, period, threshold)
                results[threshold][period['label']] = result
                
                # Print summary
                ward_data = result['ward_data']
                affected_wards = ward_data['violence_affected'].sum()
                total_wards = len(ward_data)
                total_deaths = ward_data['ACLED_BRD_total'].sum()
                
                print(f"   {period['label']}: {affected_wards:,}/{total_wards:,} wards affected ({affected_wards/total_wards*100:.1f}%), {total_deaths:,.0f} deaths")
                
            except Exception as e:
                print(f"   ❌ Error processing {period['label']}: {e}")
    
    # Save sample results
    output_file = Path("data/processed/simplified_brd_results.json")
    output_file.parent.mkdir(exist_ok=True)
    
    # Convert results to JSON-serializable format
    json_results = {}
    for threshold, period_data in results.items():
        json_results[str(threshold)] = {}
        for period_label, data in period_data.items():
            json_results[str(threshold)][period_label] = {
                'period_info': data['period_info'],
                'threshold_per_100k': data['threshold_per_100k'],
                'summary': {
                    'total_wards': len(data['ward_data']),
                    'affected_wards': int(data['ward_data']['violence_affected'].sum()),
                    'total_deaths': float(data['ward_data']['ACLED_BRD_total'].sum()),
                    'total_lgas': len(data['lga_data']),
                    'total_states': len(data['state_data'])
                }
            }
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n✅ Processing complete! Results saved to {output_file}")
    print(f"📊 Processed {len(test_periods)} periods with {len(test_thresholds)} thresholds")

if __name__ == "__main__":
    main()
