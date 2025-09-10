#!/usr/bin/env python3
"""
Nigeria Violence Platform - Data Processing Script
Processes administrative boundaries and population data for Nigeria
"""

import json
import requests
import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterio.mask import mask
from shapely.geometry import shape
import os
from typing import Dict, List, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NigeriaDataProcessor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.population_file = os.path.join(data_dir, "nga_ppp_2020_UNadj.tif")
        
        # URLs for administrative boundaries
        self.boundaries_urls = {
            'ward': 'https://services3.arcgis.com/BU6Aadhn6tbBEdyk/arcgis/rest/services/NGA_Ward_Boundaries/FeatureServer/0/query?outFields=*&where=1%3D1&f=geojson',
            'lga': 'https://services3.arcgis.com/BU6Aadhn6tbBEdyk/arcgis/rest/services/NGA_LGA_Boundaries_2/FeatureServer/0/query?outFields=*&where=1%3D1&f=geojson',
            'state': 'https://services3.arcgis.com/BU6Aadhn6tbBEdyk/arcgis/rest/services/NGA_State_Boundaries_V2/FeatureServer/0/query?outFields=*&where=1%3D1&f=geojson'
        }
        
        # Create output directory
        self.output_dir = os.path.join(data_dir, "processed")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def download_geojson(self, level: str) -> gpd.GeoDataFrame:
        """Download and load GeoJSON data for specified administrative level"""
        logger.info(f"Downloading {level} boundaries...")
        
        try:
            response = requests.get(self.boundaries_urls[level], timeout=30)
            response.raise_for_status()
            
            # Save raw data
            raw_file = os.path.join(self.data_dir, f"nga_{level}_boundaries.geojson")
            with open(raw_file, 'w') as f:
                json.dump(response.json(), f)
            
            # Load as GeoDataFrame
            gdf = gpd.read_file(raw_file)
            logger.info(f"Downloaded {len(gdf)} {level} boundaries")
            
            return gdf
            
        except Exception as e:
            logger.error(f"Error downloading {level} boundaries: {e}")
            raise
    
    def extract_population_for_geometry(self, geometry, population_raster) -> float:
        """Extract total population for a given geometry from raster data"""
        try:
            # Mask the raster with the geometry
            masked_data, transform = mask(population_raster, [geometry], crop=True, nodata=0)
            
            # Calculate total population (sum of all pixel values)
            total_population = np.nansum(masked_data)
            
            return float(total_population)
            
        except Exception as e:
            logger.warning(f"Error extracting population for geometry: {e}")
            return 0.0
    
    def process_administrative_level(self, level: str) -> Dict[str, Any]:
        """Process a single administrative level"""
        logger.info(f"Processing {level} level data...")
        
        # Download boundaries
        gdf = self.download_geojson(level)
        
        # Load population raster
        logger.info("Loading population raster...")
        with rasterio.open(self.population_file) as population_raster:
            # Ensure geometries are in the same CRS as the raster
            if gdf.crs != population_raster.crs:
                gdf = gdf.to_crs(population_raster.crs)
            
            # Extract population for each administrative unit
            results = []
            
            for idx, row in gdf.iterrows():
                logger.info(f"Processing {level} {idx + 1}/{len(gdf)}: {row.get('wardname', row.get('lganame', row.get('statename', 'Unknown')))}")
                
                # Extract population
                population = self.extract_population_for_geometry(row.geometry, population_raster)
                
                # Create result record
                result = {
                    'id': idx,
                    'geometry': row.geometry.__geo_interface__,
                    'population': population
                }
                
                # Add level-specific attributes
                if level == 'ward':
                    result.update({
                        'wardname': row.get('wardname', ''),
                        'wardcode': row.get('wardcode', ''),
                        'lganame': row.get('lganame', ''),
                        'lgacode': row.get('lgacode', ''),
                        'statename': row.get('statename', ''),
                        'statecode': row.get('statecode', ''),
                        'urban': row.get('urban', '')
                    })
                elif level == 'lga':
                    result.update({
                        'lganame': row.get('lganame', ''),
                        'lgacode': row.get('lgacode', ''),
                        'statename': row.get('statename', ''),
                        'statecode': row.get('statecode', ''),
                        'urban': row.get('urban', '')
                    })
                elif level == 'state':
                    result.update({
                        'statename': row.get('statename', ''),
                        'statecode': row.get('statecode', ''),
                        'region': row.get('region', '')
                    })
                
                results.append(result)
        
        # Save results
        output_file = os.path.join(self.output_dir, f"nigeria_{level}_population.json")
        with open(output_file, 'w') as f:
            json.dump({
                'level': level,
                'total_units': len(results),
                'total_population': sum(r['population'] for r in results),
                'data': results
            }, f, indent=2)
        
        logger.info(f"Saved {level} data to {output_file}")
        return {
            'level': level,
            'total_units': len(results),
            'total_population': sum(r['population'] for r in results),
            'file': output_file
        }
    
    def process_all_levels(self) -> Dict[str, Any]:
        """Process all administrative levels"""
        logger.info("Starting Nigeria data processing...")
        
        results = {}
        
        for level in ['state', 'lga', 'ward']:
            try:
                results[level] = self.process_administrative_level(level)
            except Exception as e:
                logger.error(f"Failed to process {level} level: {e}")
                results[level] = {'error': str(e)}
        
        # Save summary
        summary_file = os.path.join(self.output_dir, "processing_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Processing complete. Summary saved to {summary_file}")
        return results

def main():
    """Main function to run the data processing"""
    processor = NigeriaDataProcessor()
    results = processor.process_all_levels()
    
    # Print summary
    print("\n" + "="*50)
    print("NIGERIA DATA PROCESSING SUMMARY")
    print("="*50)
    
    for level, result in results.items():
        if 'error' not in result:
            print(f"{level.upper()} Level:")
            print(f"  - Total Units: {result['total_units']:,}")
            print(f"  - Total Population: {result['total_population']:,.0f}")
            print(f"  - Output File: {result['file']}")
        else:
            print(f"{level.upper()} Level: ERROR - {result['error']}")
        print()

if __name__ == "__main__":
    main()
