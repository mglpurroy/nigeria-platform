# 🇳🇬 Nigeria Violence Analysis Dashboard

A comprehensive interactive dashboard for analyzing violence patterns across Nigeria's administrative levels (States, LGAs, and Wards) using ACLED conflict data and population estimates.

## 🚀 Features

### 📊 **Interactive Analysis**
- **12-month period analysis** with calendar year and mid-year cycles
- **Multi-level administrative analysis** (States, LGAs, Wards)
- **Configurable violence thresholds** for death rates and absolute deaths
- **Real-time data processing** with caching for optimal performance

### 🗺️ **Interactive Maps**
- **Administrative Units Map**: Aggregated analysis by States/LGAs
- **Ward Classification Map**: Individual ward violence classification
- **Color-coded visualization** with customizable thresholds
- **Interactive popups** with detailed statistics

### 📈 **Comprehensive Analytics**
- **Violence hotspots identification**
- **Population impact analysis**
- **Statistical summaries** and key insights
- **Export functionality** for all data and analyses

### ⚡ **Performance Optimized**
- **Session state caching** for faster subsequent loads
- **File-based caching** for repeated operations
- **Vectorized operations** for improved pandas performance
- **Optimized map rendering** with canvas mode

## 📁 Data Sources

### Population Data
- **Source**: WorldPop 2020 UN-adjusted population estimates
- **File**: `data/nga_ppp_2020_UNadj.tif`
- **Processing**: Extracted to ward level and aggregated to LGA and State levels
- **Output**: JSON files with population data for each administrative level

### Conflict Data
- **Source**: ACLED (Armed Conflict Location & Event Data Project)
- **File**: `data/acled_nigeria_data.csv`
- **Coverage**: 1997-2025
- **BRD Definition**: Battle-Related Deaths excluding Protests and Riots
- **State vs Non-State**: Categorized using interaction column (State forces vs others)
- **Processing**: Aggregated at LGA level and distributed to wards proportionally
- **Total BRD**: 123,719 deaths (State: 51,268, Non-state: 72,451)

### Administrative Boundaries
- **Ward Level**: `data/nga_ward_boundaries.geojson`
- **LGA Level**: `data/nga_lga_boundaries.geojson`
- **State Level**: `data/nga_state_boundaries.geojson`
- **Source**: ArcGIS REST services

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mglpurroy/nigeria-platform.git
   cd nigeria-platform
   ```

2. **Create and activate virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify data files are present**:
   ```bash
   ls -la data/
   # Should show: nga_ppp_2020_UNadj.tif, acled_nigeria_data.csv, and GeoJSON files
   ```

## 🚀 Running the Dashboard

### Start the Dashboard
```bash
source venv/bin/activate
streamlit run nigeria_violence_dashboard.py
```

The dashboard will be available at: `http://localhost:8501`

### Test Data Loading
```bash
source venv/bin/activate
python test_dashboard.py
```

## 📊 Dashboard Components

### 🎛️ **Sidebar Controls**

#### Violence Classification
- **Death Rate Threshold**: Minimum death rate per 100,000 population
- **Min Deaths Threshold**: Minimum absolute number of deaths
- **Aggregation Threshold**: Minimum share of wards affected to mark unit as high-violence

#### Analysis Settings
- **Analysis Period**: Select 12-month period (calendar or mid-year)
- **Administrative Level**: Choose between States (ADM1) or LGAs (ADM2)
- **Map Variable**: Display share of wards affected or share of population affected

### 📈 **Main Dashboard Sections**

#### 1. **Key Metrics**
- High Violence Units count and percentage
- Affected Wards count and percentage
- Affected Population count and percentage
- Total Deaths in selected period

#### 2. **Interactive Maps**
- **Administrative Units**: Aggregated analysis with color-coded violence levels
- **Ward Classification**: Individual ward classification (Violence Affected, Below Threshold, No Violence)

#### 3. **Supporting Analysis**
- **Violence-Affected Areas**: Ranked by ward share
- **Population vs Deaths**: Scatter plot analysis
- **Distribution of Violence Levels**: Pie chart breakdown
- **Ward Classification**: Bar chart summary

#### 4. **Key Insights**
- **Violence Hotspots**: Top 5 areas with highest violence
- **Statistical Summary**: National statistics and impact metrics

#### 5. **Data Export**
- **Aggregated Data**: CSV export of administrative unit analysis
- **Ward Data**: CSV export of individual ward classifications
- **Analysis Summary**: CSV export of key metrics and statistics

## 🔧 Technical Details

### Data Processing Pipeline

1. **Population Data Loading**:
   - Loads ward-level population from processed JSON files
   - Creates State and LGA aggregations
   - Handles missing geometries and data validation

2. **Conflict Data Processing**:
   - Loads ACLED data and filters for Nigeria
   - Defines BRD: Excludes Protests and Riots, includes all other violent events
   - Categorizes state vs non-state violence using interaction column
   - Aggregates at LGA level (since ward-level data is limited)
   - Distributes LGA-level conflict to wards proportionally by population

3. **Administrative Boundaries**:
   - Loads GeoJSON files for all three administrative levels
   - Ensures consistent CRS (EPSG:4326)
   - Creates simplified geometries for better performance

4. **Analysis Engine**:
   - Filters data by selected time period
   - Applies violence classification thresholds
   - Aggregates to selected administrative level
   - Calculates shares and rates

### Performance Optimizations

- **Caching**: Session state and file-based caching for repeated operations
- **Vectorized Operations**: Pandas operations optimized for large datasets
- **Chunked Processing**: Memory-efficient data loading
- **Simplified Geometries**: Reduced map rendering complexity
- **Canvas Mode**: Optimized map rendering for better performance

## 📁 File Structure

```
nigeria-platform/
├── nigeria_violence_dashboard.py    # Main dashboard application
├── mapping_functions.py             # Map creation functions
├── chart_functions.py               # Chart creation functions
├── test_dashboard.py                # Data loading tests
├── debug_conflict.py                # Conflict data debugging
├── requirements.txt                 # Python dependencies
├── data/
│   ├── nga_ppp_2020_UNadj.tif      # Population raster
│   ├── acled_nigeria_data.csv      # ACLED conflict data
│   ├── nga_ward_boundaries.geojson # Ward boundaries
│   ├── nga_lga_boundaries.geojson  # LGA boundaries
│   ├── nga_state_boundaries.geojson # State boundaries
│   └── processed/                   # Processed population data
│       ├── nigeria_ward_population.json
│       ├── nigeria_lga_population.json
│       └── nigeria_state_population.json
└── cache/                          # Runtime cache (auto-created)
```

## 🐛 Troubleshooting

### Common Issues

1. **"No module named 'streamlit'"**:
   ```bash
   pip install streamlit
   ```

2. **"Conflict data is empty"**:
   - Check that `data/acled_nigeria_data.csv` exists
   - Verify the file has the correct column names
   - Clear cache: `rm -rf cache/`

3. **"Population data not found"**:
   - Ensure `data/processed/nigeria_ward_population.json` exists
   - Run the data processing scripts first if needed

4. **"Boundary data not found"**:
   - Check that all GeoJSON files exist in the `data/` directory
   - Verify file permissions and format

5. **Performance Issues**:
   - Clear cache: `rm -rf cache/`
   - Reduce map complexity by selecting fewer features
   - Use smaller time periods for analysis

### Debug Mode

Run the test script to verify all components:
```bash
python test_dashboard.py
```

## 📊 Data Validation

The dashboard includes comprehensive data validation:

- **Population Data**: 1,899 wards across 37 states
- **Conflict Data**: 6,955 LGA-level records (1997-2025)
- **Boundaries**: 2,000 wards, 774 LGAs, 37 states
- **Data Quality**: Handles missing geometries and invalid data

## 🔄 Updates & Maintenance

### Adding New Data
1. Update the data files in the `data/` directory
2. Clear the cache: `rm -rf cache/`
3. Restart the dashboard

### Modifying Analysis Parameters
- Edit the threshold values in the sidebar controls
- Adjust the time period ranges in `generate_12_month_periods()`
- Modify the violence classification logic in `classify_and_aggregate_data()`

## 📝 License

This project is part of the Nigeria Platform for Violence Analysis. Please refer to the main project documentation for licensing information.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly with `python test_dashboard.py`
5. Submit a pull request

## 📞 Support

For technical support or questions about the dashboard:
- Check the troubleshooting section above
- Review the test output for data loading issues
- Ensure all dependencies are correctly installed

---

**Dashboard Version**: 1.0.0  
**Last Updated**: September 2025  
**Compatible with**: Python 3.8+, Streamlit 1.28+
