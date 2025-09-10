#!/usr/bin/env python3
"""
Test script for Nigeria Violence Dashboard
This script tests the core data loading functions without running the full Streamlit app
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nigeria_violence_dashboard import (
    load_population_data, 
    load_conflict_data, 
    load_admin_boundaries,
    create_admin_levels,
    generate_12_month_periods
)

def test_data_loading():
    """Test all data loading functions"""
    print("🧪 Testing Nigeria Violence Dashboard Data Loading...")
    print("=" * 60)
    
    # Test 1: Generate periods
    print("\n1. Testing period generation...")
    try:
        periods = generate_12_month_periods()
        print(f"✅ Generated {len(periods)} periods")
        print(f"   Sample: {periods[0]['label']} to {periods[-1]['label']}")
    except Exception as e:
        print(f"❌ Period generation failed: {e}")
        return False
    
    # Test 2: Load population data
    print("\n2. Testing population data loading...")
    try:
        pop_data = load_population_data()
        if not pop_data.empty:
            print(f"✅ Loaded population data for {len(pop_data)} wards")
            print(f"   Columns: {list(pop_data.columns)}")
            print(f"   Sample ward: {pop_data.iloc[0]['ADM3_EN']} in {pop_data.iloc[0]['ADM1_EN']}")
        else:
            print("❌ Population data is empty")
            return False
    except Exception as e:
        print(f"❌ Population data loading failed: {e}")
        return False
    
    # Test 3: Create admin levels
    print("\n3. Testing admin level creation...")
    try:
        admin_data = create_admin_levels(pop_data)
        print(f"✅ Created admin levels:")
        print(f"   States: {len(admin_data['admin1'])}")
        print(f"   LGAs: {len(admin_data['admin2'])}")
        print(f"   Wards: {len(admin_data['admin3'])}")
    except Exception as e:
        print(f"❌ Admin level creation failed: {e}")
        return False
    
    # Test 4: Load conflict data
    print("\n4. Testing conflict data loading...")
    try:
        conflict_data = load_conflict_data()
        if not conflict_data.empty:
            print(f"✅ Loaded conflict data: {len(conflict_data)} records")
            print(f"   Columns: {list(conflict_data.columns)}")
            print(f"   Year range: {conflict_data['year'].min()} to {conflict_data['year'].max()}")
        else:
            print("⚠️  Conflict data is empty (this may be expected)")
    except Exception as e:
        print(f"❌ Conflict data loading failed: {e}")
        return False
    
    # Test 5: Load boundaries
    print("\n5. Testing boundary data loading...")
    try:
        boundaries = load_admin_boundaries()
        print(f"✅ Loaded boundaries:")
        for level, gdf in boundaries.items():
            if not gdf.empty:
                print(f"   Level {level}: {len(gdf)} features")
            else:
                print(f"   Level {level}: No data")
    except Exception as e:
        print(f"❌ Boundary loading failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed successfully!")
    print("The dashboard should work correctly.")
    return True

if __name__ == "__main__":
    success = test_data_loading()
    sys.exit(0 if success else 1)
