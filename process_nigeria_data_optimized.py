#!/usr/bin/env python3
"""
Nigeria Violence Platform - Optimized Data Processing Script
Processes administrative boundaries and population data for Nigeria with memory optimization
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedNigeriaDataProcessor:
    def __init__(self, data_dir: str = "data", max_workers: int = 4):
        self.data_dir = data_dir
        self.population_file = os.path.join(data_dir, "nga_ppp_2020_UNadj.tif")
        self.max_workers = max_workers
        
        # URLs for administrative boundaries
        self.boundaries_urls = {
            'ward': 'https://services3.arcgis.com/BU6Aadhn6tbBEdyk/arcgis/rest/services/NGA_Ward_Boundaries/FeatureServer/0/query?outFields=*&where=1%3D1&f=geojson',
            'lga': 'https://services3.arcgis.com/BU6Aadhn6tbBEdyk/arcgis/rest/services/NGA_LGA_Boundaries_2/FeatureServer/0/query?outFields=*&where=1%3D1&f=geojson',
            'state': 'https://services3.arcgis.com/BU6Aadhn6tbBEdyk/arcgis/rest/services/NGA_State_Boundaries_V2/FeatureServer/0/query?outFields=*&where=1%3D1&f=geojson'
        }
        
        # Create output directory
        self.output_dir = os.path.join(data_dir, "processed")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def download_geojson(self, level: str) -> str:
        """Download GeoJSON data and return file path"""
        logger.info(f"Downloading {level} boundaries...")
        
        try:
            response = requests.get(self.boundaries_urls[level], timeout=60)
            response.raise_for_status()
            
            # Save raw data
            raw_file = os.path.join(self.data_dir, f"nga_{level}_boundaries.geojson")
            with open(raw_file, 'w') as f:
                json.dump(response.json(), f)
            
            logger.info(f"Downloaded {level} boundaries to {raw_file}")
            return raw_file
            
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
    
    def process_geometry_batch(self, batch_data: List[tuple], population_raster) -> List[Dict]:
        """Process a batch of geometries"""
        results = []
        
        for idx, row_data in batch_data:
            geometry, attributes = row_data
            
            try:
                # Extract population
                population = self.extract_population_for_geometry(geometry, population_raster)
                
                # Create result record
                result = {
                    'id': idx,
                    'geometry': geometry.__geo_interface__,
                    'population': population,
                    **attributes
                }
                
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Error processing geometry {idx}: {e}")
                results.append({
                    'id': idx,
                    'geometry': geometry.__geo_interface__,
                    'population': 0.0,
                    **attributes,
                    'error': str(e)
                })
        
        return results
    
    def process_administrative_level(self, level: str, batch_size: int = 50) -> Dict[str, Any]:
        """Process a single administrative level with batching"""
        logger.info(f"Processing {level} level data...")
        
        # Download boundaries
        geojson_file = self.download_geojson(level)
        
        # Load population raster
        logger.info("Loading population raster...")
        with rasterio.open(self.population_file) as population_raster:
            # Load GeoDataFrame
            gdf = gpd.read_file(geojson_file)
            
            # Ensure geometries are in the same CRS as the raster
            if gdf.crs != population_raster.crs:
                gdf = gdf.to_crs(population_raster.crs)
            
            logger.info(f"Processing {len(gdf)} {level} units in batches of {batch_size}")
            
            # Prepare data for processing
            all_results = []
            total_batches = (len(gdf) + batch_size - 1) // batch_size
            
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min((batch_num + 1) * batch_size, len(gdf))
                
                logger.info(f"Processing batch {batch_num + 1}/{total_batches} ({start_idx}-{end_idx})")
                
                # Prepare batch data
                batch_data = []
                for idx in range(start_idx, end_idx):
                    row = gdf.iloc[idx]
                    
                    # Extract attributes based on level
                    attributes = {}
                    if level == 'ward':
                        attributes = {
                            'wardname': row.get('wardname', ''),
                            'wardcode': row.get('wardcode', ''),
                            'lganame': row.get('lganame', ''),
                            'lgacode': row.get('lgacode', ''),
                            'statename': row.get('statename', ''),
                            'statecode': row.get('statecode', ''),
                            'urban': row.get('urban', '')
                        }
                    elif level == 'lga':
                        attributes = {
                            'lganame': row.get('lganame', ''),
                            'lgacode': row.get('lgacode', ''),
                            'statename': row.get('statename', ''),
                            'statecode': row.get('statecode', ''),
                            'urban': row.get('urban', '')
                        }
                    elif level == 'state':
                        attributes = {
                            'statename': row.get('statename', ''),
                            'statecode': row.get('statecode', ''),
                            'region': row.get('region', '')
                        }
                    
                    batch_data.append((idx, (row.geometry, attributes)))
                
                # Process batch
                batch_results = self.process_geometry_batch(batch_data, population_raster)
                all_results.extend(batch_results)
        
        # Save results
        output_file = os.path.join(self.output_dir, f"nigeria_{level}_population.json")
        with open(output_file, 'w') as f:
            json.dump({
                'level': level,
                'total_units': len(all_results),
                'total_population': sum(r['population'] for r in all_results),
                'data': all_results
            }, f, indent=2)
        
        logger.info(f"Saved {level} data to {output_file}")
        return {
            'level': level,
            'total_units': len(all_results),
            'total_population': sum(r['population'] for r in all_results),
            'file': output_file
        }
    
    def process_all_levels(self) -> Dict[str, Any]:
        """Process all administrative levels"""
        logger.info("Starting Nigeria data processing...")
        
        results = {}
        
        # Process in order: state (smallest), lga (medium), ward (largest)
        for level in ['state', 'lga', 'ward']:
            try:
                start_time = time.time()
                results[level] = self.process_administrative_level(level)
                end_time = time.time()
                
                logger.info(f"Completed {level} level in {end_time - start_time:.2f} seconds")
                
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
    processor = OptimizedNigeriaDataProcessor()
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
