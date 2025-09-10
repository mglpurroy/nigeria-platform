#!/usr/bin/env python3
"""
Download comprehensive ward boundaries from ArcGIS REST service
and calculate population for all wards.
"""

import requests
import json
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask
import time
from pathlib import Path
import numpy as np
from shapely.geometry import shape
from shapely.validation import make_valid
import warnings
warnings.filterwarnings('ignore')

def download_comprehensive_wards():
    """Download all ward boundaries from ArcGIS REST service"""
    print("🌍 Downloading comprehensive ward boundaries...")
    
    base_url = 'https://services3.arcgis.com/BU6Aadhn6tbBEdyk/arcgis/rest/services/NGA_Ward_Boundaries/FeatureServer/0/query'
    
    # First, get the total count
    count_params = {
        'where': '1=1',
        'f': 'json',
        'returnCountOnly': 'true'
    }
    
    try:
        response = requests.get(base_url, params=count_params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            total_count = data.get('count', 0)
            print(f"📊 Total features in service: {total_count:,}")
        else:
            print(f"❌ Could not get count: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error getting count: {e}")
        return None
    
    # Download all features using pagination
    all_features = []
    batch_size = 1000
    offset = 0
    
    print(f"📥 Fetching features in batches of {batch_size:,}...")
    
    while offset < total_count:
        params = {
            'outFields': '*',
            'where': '1=1',
            'f': 'json',
            'returnGeometry': 'true',
            'resultOffset': str(offset),
            'resultRecordCount': str(batch_size)
        }
        
        try:
            batch_num = offset // batch_size + 1
            total_batches = (total_count + batch_size - 1) // batch_size
            print(f"   Batch {batch_num}/{total_batches} (offset: {offset:,})...")
            
            response = requests.get(base_url, params=params, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                if 'features' in data:
                    batch_features = data['features']
                    all_features.extend(batch_features)
                    print(f"   ✅ Got {len(batch_features):,} features")
                    
                    # Check if we got fewer features than requested (end of data)
                    if len(batch_features) < batch_size:
                        print(f"   📋 Reached end of data")
                        break
                        
                    offset += batch_size
                else:
                    print(f"   ❌ No features in response")
                    break
            else:
                print(f"   ❌ HTTP {response.status_code}")
                break
                
            # Small delay to be respectful to the service
            time.sleep(0.5)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            break
    
    print(f"📊 Total features downloaded: {len(all_features):,}")
    
    # Convert to GeoDataFrame
    print("🔄 Converting to GeoDataFrame...")
    
    features_data = []
    for feature in all_features:
        attrs = feature.get('attributes', {})
        geom_data = feature.get('geometry')
        
        # Convert geometry
        if geom_data and geom_data != 'null':
            try:
                # Handle ArcGIS geometry format (rings) vs standard GeoJSON
                if 'rings' in geom_data:
                    # Convert ArcGIS rings format to GeoJSON
                    rings = geom_data['rings']
                    if rings and len(rings) > 0:
                        # Create a polygon from the first ring
                        coords = rings[0]
                        if len(coords) >= 3:  # Need at least 3 points for a polygon
                            # Convert to GeoJSON format
                            geojson_geom = {
                                'type': 'Polygon',
                                'coordinates': [coords]
                            }
                            geometry = shape(geojson_geom)
                        else:
                            geometry = None
                    else:
                        geometry = None
                else:
                    # Standard GeoJSON format
                    geometry = shape(geom_data)
                
                # Try to fix invalid geometries
                if geometry and not geometry.is_valid:
                    geometry = make_valid(geometry)
                    
            except Exception as e:
                print(f"   ⚠️  Invalid geometry for {attrs.get('wardcode', 'unknown')}: {e}")
                geometry = None
        else:
            geometry = None
        
        # Create feature data
        feature_data = {
            'FID': attrs.get('FID'),
            'globalid': attrs.get('globalid'),
            'uniq_id': attrs.get('uniq_id'),
            'timestamp': attrs.get('timestamp'),
            'editor': attrs.get('editor'),
            'wardname': attrs.get('wardname'),
            'wardcode': attrs.get('wardcode'),
            'lganame': attrs.get('lganame'),
            'lgacode': attrs.get('lgacode'),
            'statename': attrs.get('statename'),
            'statecode': attrs.get('statecode'),
            'amapcode': attrs.get('amapcode'),
            'status': attrs.get('status'),
            'source': attrs.get('source'),
            'urban': attrs.get('urban'),
            'Shape__Area': attrs.get('Shape__Area'),
            'Shape__Length': attrs.get('Shape__Length'),
            'geometry': geometry
        }
        features_data.append(feature_data)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(features_data, crs='EPSG:4326')
    
    # Analyze the data
    valid_geometries = gdf[gdf.geometry.notnull()]
    null_geometries = gdf[gdf.geometry.isnull()]
    
    print(f"✅ Valid geometries: {len(valid_geometries):,}")
    print(f"⚠️  Null geometries: {len(null_geometries):,}")
    print(f"📈 Success rate: {len(valid_geometries)/len(gdf)*100:.1f}%")
    
    # Save the comprehensive dataset
    output_file = Path("data/nga_ward_boundaries_comprehensive.geojson")
    print(f"💾 Saving to {output_file}...")
    
    # Save only valid geometries
    valid_gdf = valid_geometries.copy()
    valid_gdf.to_file(output_file, driver='GeoJSON')
    
    print(f"✅ Saved {len(valid_gdf):,} ward boundaries to {output_file}")
    
    return valid_gdf

def calculate_population_for_wards(gdf, raster_file):
    """Calculate population for all wards using raster data"""
    print(f"🧮 Calculating population for {len(gdf):,} wards...")
    
    if not Path(raster_file).exists():
        print(f"❌ Raster file not found: {raster_file}")
        return None
    
    # Load raster
    print("📊 Loading population raster...")
    with rasterio.open(raster_file) as src:
        print(f"   Raster CRS: {src.crs}")
        print(f"   Raster bounds: {src.bounds}")
        print(f"   Raster shape: {src.shape}")
        
        # Ensure consistent CRS
        if gdf.crs != src.crs:
            print(f"🔄 Transforming geometries to raster CRS...")
            gdf = gdf.to_crs(src.crs)
        
        population_data = []
        total_wards = len(gdf)
        
        print(f"🔢 Processing {total_wards:,} wards...")
        
        # Process in batches for memory efficiency
        batch_size = 100
        for batch_start in range(0, total_wards, batch_size):
            batch_end = min(batch_start + batch_size, total_wards)
            batch_gdf = gdf.iloc[batch_start:batch_end]
            
            for idx, row in batch_gdf.iterrows():
                try:
                    if row.geometry is not None:
                        # Mask raster with ward geometry
                        geom = [row.geometry.__geo_interface__]
                        out_image, _ = mask(src, geom, crop=True, nodata=0)
                        
                        # Calculate population
                        pop_sum = out_image[out_image > 0].sum()
                        
                        population_data.append({
                            'ADM3_PCODE': row['wardcode'],
                            'ADM3_EN': row['wardname'],
                            'ADM2_PCODE': row['lgacode'],
                            'ADM2_EN': row['lganame'],
                            'ADM1_PCODE': row['statecode'],
                            'ADM1_EN': row['statename'],
                            'ADM0_PCODE': 'NGA',  # Nigeria country code
                            'pop_count': int(pop_sum),
                            'pop_count_millions': pop_sum / 1e6
                        })
                    else:
                        # Handle null geometries
                        population_data.append({
                            'ADM3_PCODE': row['wardcode'],
                            'ADM3_EN': row['wardname'],
                            'ADM2_PCODE': row['lgacode'],
                            'ADM2_EN': row['lganame'],
                            'ADM1_PCODE': row['statecode'],
                            'ADM1_EN': row['statename'],
                            'ADM0_PCODE': 'NGA',
                            'pop_count': 0,
                            'pop_count_millions': 0.0
                        })
                        
                except Exception as e:
                    print(f"   ⚠️  Error processing {row['wardcode']}: {e}")
                    # Add with zero population
                    population_data.append({
                        'ADM3_PCODE': row['wardcode'],
                        'ADM3_EN': row['wardname'],
                        'ADM2_PCODE': row['lgacode'],
                        'ADM2_EN': row['lganame'],
                        'ADM1_PCODE': row['statecode'],
                        'ADM1_EN': row['statename'],
                        'ADM0_PCODE': 'NGA',
                        'pop_count': 0,
                        'pop_count_millions': 0.0
                    })
            
            # Progress update
            progress = (batch_end / total_wards) * 100
            print(f"   📈 Progress: {progress:.1f}% ({batch_end:,}/{total_wards:,})")
    
    # Create DataFrame
    result_df = pd.DataFrame(population_data)
    
    # Save results
    output_file = Path("data/processed/nigeria_ward_population_comprehensive.json")
    output_file.parent.mkdir(exist_ok=True)
    
    print(f"💾 Saving population data to {output_file}...")
    result_df.to_json(output_file, orient='records', indent=2)
    
    # Summary statistics
    if len(result_df) > 0:
        total_population = result_df['pop_count'].sum()
        wards_with_population = (result_df['pop_count'] > 0).sum()
        
        print(f"✅ Population calculation complete!")
        print(f"   📊 Total population: {total_population:,}")
        print(f"   📍 Wards with population: {wards_with_population:,}/{len(result_df):,}")
        print(f"   📈 Coverage: {wards_with_population/len(result_df)*100:.1f}%")
    else:
        print("⚠️  No population data calculated - no valid geometries found")
    
    return result_df

def main():
    """Main function to download and process comprehensive ward data"""
    print("🚀 Starting comprehensive ward data download and processing...")
    
    # Download comprehensive ward boundaries
    gdf = download_comprehensive_wards()
    if gdf is None:
        print("❌ Failed to download ward boundaries")
        return
    
    # Calculate population
    raster_file = "data/nga_ppp_2020_UNadj.tif"
    pop_data = calculate_population_for_wards(gdf, raster_file)
    
    if pop_data is not None:
        print("🎉 Comprehensive ward data processing complete!")
        print(f"   📍 Total wards: {len(pop_data):,}")
        print(f"   🏛️  Total LGAs: {pop_data['ADM2_PCODE'].nunique():,}")
        print(f"   🗺️  Total states: {pop_data['ADM1_PCODE'].nunique():,}")
        print(f"   👥 Total population: {pop_data['pop_count'].sum():,}")
    else:
        print("❌ Failed to calculate population")

if __name__ == "__main__":
    main()
