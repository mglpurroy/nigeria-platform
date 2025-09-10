#!/usr/bin/env python3
"""
Validate the processed Nigeria data and show sample results
"""

import json
import os

def validate_data():
    """Validate and display sample data from each administrative level"""
    
    print("NIGERIA VIOLENCE PLATFORM - DATA VALIDATION")
    print("=" * 60)
    
    # Check if files exist
    files = {
        'State': 'data/processed/nigeria_state_population.json',
        'LGA': 'data/processed/nigeria_lga_population.json', 
        'Ward': 'data/processed/nigeria_ward_population.json'
    }
    
    for level, filepath in files.items():
        if os.path.exists(filepath):
            print(f"\n✓ {level} Level Data: {filepath}")
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            print(f"  - Total Units: {data['total_units']:,}")
            print(f"  - Total Population: {data['total_population']:,.0f}")
            
            # Show top 3 by population
            sorted_data = sorted(data['data'], key=lambda x: x['population'], reverse=True)
            print(f"  - Top 3 by Population:")
            
            for i, unit in enumerate(sorted_data[:3]):
                if level == 'State':
                    name = unit.get('statename', 'Unknown')
                elif level == 'LGA':
                    name = f"{unit.get('lganame', 'Unknown')}, {unit.get('statename', 'Unknown')}"
                else:  # Ward
                    name = f"{unit.get('wardname', 'Unknown')}, {unit.get('lganame', 'Unknown')}, {unit.get('statename', 'Unknown')}"
                
                print(f"    {i+1}. {name}: {unit['population']:,.0f}")
            
            if level == 'Ward' and 'errors' in data:
                print(f"  - Processing Errors: {len(data['errors'])}")
        
        else:
            print(f"\n✗ {level} Level Data: {filepath} - FILE NOT FOUND")
    
    print("\n" + "=" * 60)
    print("DATA PROCESSING COMPLETE!")
    print("=" * 60)
    print("\nNext steps for the Nigeria Violence Platform:")
    print("1. Use these JSON files for mapping and visualization")
    print("2. Integrate with violence incident data (ACLED)")
    print("3. Create population-weighted analysis")
    print("4. Build interactive dashboard")
    print("5. Implement spatial analysis and clustering")

if __name__ == "__main__":
    validate_data()
