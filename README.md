# Nigeria Violence Platform - Data Processing

This project processes administrative boundaries and population data for Nigeria to create a comprehensive dataset for violence analysis.

## Overview

The platform processes three administrative levels:
- **State Level**: 36 states + FCT
- **LGA Level**: Local Government Areas
- **Ward Level**: Electoral wards (smallest administrative unit)

## Data Sources

- **Population Data**: `data/nga_ppp_2020_UNadj.tif` (WorldPop 2020 UN-adjusted population estimates)
- **Administrative Boundaries**: 
  - Ward: [NGA Ward Boundaries](https://services3.arcgis.com/BU6Aadhn6tbBEdyk/arcgis/rest/services/NGA_Ward_Boundaries/FeatureServer/0/query?outFields=*&where=1%3D1&f=geojson)
  - LGA: [NGA LGA Boundaries](https://services3.arcgis.com/BU6Aadhn6tbBEdyk/arcgis/rest/services/NGA_LGA_Boundaries_2/FeatureServer/0/query?outFields=*&where=1%3D1&f=geojson)
  - State: [NGA State Boundaries](https://services3.arcgis.com/BU6Aadhn6tbBEdyk/arcgis/rest/services/NGA_State_Boundaries_V2/FeatureServer/0/query?outFields=*&where=1%3D1&f=geojson)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the population data file exists:
```
data/nga_ppp_2020_UNadj.tif
```

## Usage

### Test First (Recommended)
Run a small test to ensure everything works:
```bash
python test_data_processing.py
```

### Full Processing
Process all administrative levels:
```bash
python process_nigeria_data_optimized.py
```

### Alternative (Basic Version)
If you encounter memory issues, try the basic version:
```bash
python process_nigeria_data.py
```

## Output

The processing creates the following files in `data/processed/`:

- `nigeria_state_population.json` - State-level data with population
- `nigeria_lga_population.json` - LGA-level data with population  
- `nigeria_ward_population.json` - Ward-level data with population
- `processing_summary.json` - Summary of processing results

## Data Structure

Each JSON file contains:
```json
{
  "level": "state|lga|ward",
  "total_units": 36,
  "total_population": 206139589,
  "data": [
    {
      "id": 0,
      "geometry": {...},
      "population": 1234567,
      "statename": "Lagos",
      "statecode": "LA",
      ...
    }
  ]
}
```

## Performance Notes

- **State Level**: ~37 units, processes quickly
- **LGA Level**: ~774 units, moderate processing time
- **Ward Level**: ~8,000+ units, longest processing time

The optimized version processes data in batches to handle memory constraints with large datasets.

## Troubleshooting

1. **Memory Issues**: Use the optimized version with smaller batch sizes
2. **Network Issues**: The script will retry downloads automatically
3. **CRS Issues**: The script automatically handles coordinate reference system conversions

## Next Steps

After processing, the data can be used for:
- Violence incident mapping
- Population-weighted analysis
- Administrative boundary visualization
- Statistical analysis by administrative level
# nigeria-platform
