#!/usr/bin/env python3
"""
Process Ward level data with error handling for invalid geometries
"""

import json
import requests
import rasterio
import geopandas as gpd
import numpy as np
from rasterio.mask import mask
import os
import logging
from shapely.geometry import shape
from shapely.validation import make_valid

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_ward_data():
    """Process ward level data with robust error handling"""
    logger.info("Processing ward level data...")
    
    # Download ward boundaries
    ward_url = 'https://services3.arcgis.com/BU6Aadhn6tbBEdyk/arcgis/rest/services/NGA_Ward_Boundaries/FeatureServer/0/query?outFields=*&where=1%3D1&f=geojson'
    
    try:
        response = requests.get(ward_url, timeout=60)
        response.raise_for_status()
        
        # Save raw data
        with open('data/nga_ward_boundaries.geojson', 'w') as f:
            json.dump(response.json(), f)
        
        logger.info("Downloaded ward boundaries")
        
    except Exception as e:
        logger.error(f"Error downloading ward boundaries: {e}")
        return
    
    # Load GeoDataFrame
    gdf = gpd.read_file('data/nga_ward_boundaries.geojson')
    logger.info(f"Loaded {len(gdf)} wards")
    
    # Load population raster
    population_file = 'data/nga_ppp_2020_UNadj.tif'
    
    with rasterio.open(population_file) as population_raster:
        # Ensure same CRS
        if gdf.crs != population_raster.crs:
            gdf = gdf.to_crs(population_raster.crs)
        
        # Process wards with error handling
        results = []
        errors = []
        
        for idx, row in gdf.iterrows():
            try:
                # Check if geometry is valid
                geometry = row.geometry
                if geometry is None:
                    logger.warning(f"Ward {idx}: No geometry found")
                    errors.append(f"Ward {idx}: No geometry")
                    continue
                
                # Try to fix invalid geometries
                if not geometry.is_valid:
                    logger.info(f"Ward {idx}: Fixing invalid geometry")
                    geometry = make_valid(geometry)
                    if geometry is None:
                        logger.warning(f"Ward {idx}: Could not fix geometry")
                        errors.append(f"Ward {idx}: Could not fix geometry")
                        continue
                
                # Extract population
                try:
                    masked_data, transform = mask(population_raster, [geometry], crop=True, nodata=0)
                    total_population = np.nansum(masked_data)
                except Exception as e:
                    logger.warning(f"Ward {idx}: Error extracting population: {e}")
                    total_population = 0.0
                
                # Create result record
                result = {
                    'id': idx,
                    'geometry': geometry.__geo_interface__,
                    'population': float(total_population),
                    'wardname': row.get('wardname', ''),
                    'wardcode': row.get('wardcode', ''),
                    'lganame': row.get('lganame', ''),
                    'lgacode': row.get('lgacode', ''),
                    'statename': row.get('statename', ''),
                    'statecode': row.get('statecode', ''),
                    'urban': row.get('urban', '')
                }
                
                results.append(result)
                
                if idx % 100 == 0:
                    logger.info(f"Processed {idx + 1}/{len(gdf)} wards")
                
            except Exception as e:
                logger.error(f"Ward {idx}: Processing error: {e}")
                errors.append(f"Ward {idx}: {str(e)}")
                continue
        
        # Save results
        output_file = 'data/processed/nigeria_ward_population.json'
        with open(output_file, 'w') as f:
            json.dump({
                'level': 'ward',
                'total_units': len(results),
                'total_population': sum(r['population'] for r in results),
                'errors': errors,
                'data': results
            }, f, indent=2)
        
        logger.info(f"Saved ward data to {output_file}")
        logger.info(f"Successfully processed: {len(results)} wards")
        logger.info(f"Errors encountered: {len(errors)}")
        
        if errors:
            logger.info("First 5 errors:")
            for error in errors[:5]:
                logger.info(f"  - {error}")
        
        return {
            'level': 'ward',
            'total_units': len(results),
            'total_population': sum(r['population'] for r in results),
            'errors': len(errors),
            'file': output_file
        }

def main():
    """Main function"""
    result = process_ward_data()
    
    if result:
        print("\n" + "="*50)
        print("WARD LEVEL PROCESSING SUMMARY")
        print("="*50)
        print(f"Total Units: {result['total_units']:,}")
        print(f"Total Population: {result['total_population']:,.0f}")
        print(f"Errors: {result['errors']}")
        print(f"Output File: {result['file']}")

if __name__ == "__main__":
    main()
