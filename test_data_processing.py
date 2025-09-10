#!/usr/bin/env python3
"""
Test script for Nigeria data processing
Tests with a small sample to ensure everything works before full processing
"""

import json
import requests
import rasterio
import geopandas as gpd
import numpy as np
from rasterio.mask import mask
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_state_level():
    """Test processing with state level data (smallest dataset)"""
    logger.info("Testing state level processing...")
    
    # Download state boundaries
    state_url = 'https://services3.arcgis.com/BU6Aadhn6tbBEdyk/arcgis/rest/services/NGA_State_Boundaries_V2/FeatureServer/0/query?outFields=*&where=1%3D1&f=geojson'
    
    try:
        response = requests.get(state_url, timeout=30)
        response.raise_for_status()
        
        # Save and load data
        with open('data/test_state_boundaries.geojson', 'w') as f:
            json.dump(response.json(), f)
        
        gdf = gpd.read_file('data/test_state_boundaries.geojson')
        logger.info(f"Loaded {len(gdf)} states")
        
        # Test population extraction for first state
        population_file = 'data/nga_ppp_2020_UNadj.tif'
        
        with rasterio.open(population_file) as population_raster:
            # Ensure same CRS
            if gdf.crs != population_raster.crs:
                gdf = gdf.to_crs(population_raster.crs)
            
            # Test with first state
            first_state = gdf.iloc[0]
            logger.info(f"Testing with state: {first_state.get('statename', 'Unknown')}")
            
            # Extract population
            masked_data, transform = mask(population_raster, [first_state.geometry], crop=True, nodata=0)
            total_population = np.nansum(masked_data)
            
            logger.info(f"Population for {first_state.get('statename', 'Unknown')}: {total_population:,.0f}")
            
            return True
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

def main():
    """Run test"""
    logger.info("Starting Nigeria data processing test...")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Test state level processing
    success = test_state_level()
    
    if success:
        logger.info("Test completed successfully!")
        print("\nTest Results:")
        print("✓ State boundaries downloaded successfully")
        print("✓ Population raster loaded successfully")
        print("✓ Population extraction working correctly")
        print("\nReady to run full processing with: python process_nigeria_data_optimized.py")
    else:
        logger.error("Test failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
