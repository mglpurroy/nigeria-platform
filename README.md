# 🇳🇬 Nigeria Violence Analysis Dashboard

An interactive web application for analyzing violence patterns across Nigeria using ACLED conflict data, population statistics, and administrative boundaries. The dashboard provides comprehensive insights into battle-related deaths (BRD) at state, LGA, and ward levels with advanced filtering, mapping, and time series analysis.

## 🚀 Features

### 📊 **Interactive Dashboard**
- **Real-time Analysis**: Dynamic filtering by date range, violence type, and administrative level
- **Interactive Maps**: Folium-based mapping with ward-level detail and click interactions
- **Time Series Analysis**: Ward-level historical data with bar charts and trend analysis
- **Performance Optimized**: Bulk data loading and smart caching for instant responses

### 🗺️ **Geographic Coverage**
- **State Level**: 36 states + FCT (37 total)
- **LGA Level**: 774 Local Government Areas
- **Ward Level**: 9,308+ electoral wards (comprehensive dataset)

### 📈 **Data Analysis**
- **Battle-Related Deaths (BRD)**: State vs. non-state violence categorization
- **Population-Weighted Metrics**: BRD per 100k population calculations
- **Dynamic Thresholds**: Proportional scaling based on selected time periods
- **Spatial Intersection**: ACLED events mapped to ward boundaries using lat/lon coordinates

### ⚡ **Performance Features**
- **Bulk Data Loading**: All ward time series preloaded for instant access
- **Smart Caching**: Session state management with 1-hour TTL
- **Optimized I/O**: Reduced file operations and memory usage
- **Preprocessing**: Spatial intersection performed offline for faster dashboard loading

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.8+
- Git

### Quick Start
```bash
# Clone the repository
git clone https://github.com/mglpurroy/nigeria-platform.git
cd nigeria-platform

# Install dependencies
pip install -r requirements.txt

# Start the dashboard
streamlit run nigeria_violence_dashboard.py
```

### Docker Setup (Alternative)
```bash
# Using Docker Compose
docker compose up -d
pnpm db:push && pnpm db:seed
pnpm dev
```

## 📁 Data Sources

### **ACLED Conflict Data**
- **Source**: Armed Conflict Location & Event Data Project
- **Coverage**: Nigeria, 1997-2024
- **Events**: Battle-related deaths, protests, riots
- **Processing**: Spatial intersection with ward boundaries

### **Population Data**
- **Source**: WorldPop 2020 UN-adjusted estimates
- **Format**: Raster data (`.tif`) - excluded from repo due to size
- **Processing**: Extracted to administrative levels and saved as JSON

### **Administrative Boundaries**
- **Ward**: [NGA Ward Boundaries](https://services3.arcgis.com/BU6Aadhn6tbBEdyk/arcgis/rest/services/NGA_Ward_Boundaries/FeatureServer/0/query?outFields=*&where=1%3D1&f=geojson)
- **LGA**: [NGA LGA Boundaries](https://services3.arcgis.com/BU6Aadhn6tbBEdyk/arcgis/rest/services/NGA_LGA_Boundaries_2/FeatureServer/0/query?outFields=*&where=1%3D1&f=geojson)
- **State**: [NGA State Boundaries](https://services3.arcgis.com/BU6Aadhn6tbBEdyk/arcgis/rest/services/NGA_State_Boundaries_V2/FeatureServer/0/query?outFields=*&where=1%3D1&f=geojson)

## 🎯 Dashboard Usage

### **Date Range Selection**
- **Start/End Years**: 1997-2025
- **Monthly Granularity**: Select specific months for precise analysis
- **Dynamic Thresholds**: Automatically adjust based on period length

### **Violence Analysis**
- **BRD Thresholds**: 
  - Minimum deaths: 5 per 12 months (scales proportionally)
  - Rate threshold: 4 per 100k per 12 months (scales proportionally)
- **Violence Types**: State violence vs. non-state violence
- **Event Filtering**: Excludes protests and riots from BRD calculations

### **Interactive Features**
- **Map Interaction**: Click on wards to view time series
- **Ward Selection**: Dropdown with 9,308+ wards for detailed analysis
- **Real-time Filtering**: All visualizations update with date range changes

## 📊 Data Processing

### **Preprocessing Scripts**
```bash
# Process ACLED data and perform spatial intersection
python preprocess_ward_conflict_data.py

# Generate ward time series data
python process_acled_timeseries.py

# Validate data integrity
python validate_data.py
```

### **Output Files**
- `data/processed/ward_conflict_data.csv` - Ward-level conflict data
- `data/processed/ward_timeseries/` - Individual ward time series (641 files)
- `data/processed/nigeria_*_population.json` - Population data by level
- `data/processed/visualizations/` - Generated charts and analysis

## 🏗️ Architecture

### **Core Components**
- **`nigeria_violence_dashboard.py`** - Main Streamlit application
- **`mapping_functions.py`** - Geographic mapping utilities
- **`chart_functions.py`** - Data visualization functions
- **`preprocess_ward_conflict_data.py`** - Data preprocessing pipeline

### **Performance Optimizations**
- **Bulk Loading**: `load_all_ward_timeseries_data()` - Loads all ward data at startup
- **Smart Caching**: `@st.cache_data` decorators with TTL
- **Session State**: Persistent data across user interactions
- **Memory Management**: Efficient data structures and lazy loading

## 📈 Key Metrics

### **Data Coverage**
- **Total Wards**: 9,308 (comprehensive dataset)
- **Wards with Conflict**: 2,871 (30.8% coverage)
- **Total BRD Events**: 123,731 fatalities
- **Date Range**: 1997-2024 (27 years)
- **Spatial Match Rate**: 98.6% (ACLED to ward boundaries)

### **Performance Benchmarks**
- **Initial Load**: ~3-5 seconds
- **Ward Selection**: <0.1 seconds (vs 1-3s before optimization)
- **Map Rendering**: ~2-3 seconds
- **Date Range Changes**: <1 second

## 🔧 Configuration

### **Environment Variables**
```bash
# Optional: Custom data paths
DATA_PATH=/path/to/data
PROCESSED_PATH=/path/to/processed
```

### **Streamlit Configuration**
```toml
# .streamlit/config.toml
[server]
port = 8501
headless = true

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
```

## 🐛 Troubleshooting

### **Common Issues**
1. **Memory Issues**: Ensure sufficient RAM (8GB+ recommended)
2. **Slow Loading**: Check if preprocessed data exists in `data/processed/`
3. **Map Not Rendering**: Verify Folium installation and browser compatibility
4. **Data Missing**: Run preprocessing scripts to generate required files

### **Performance Tips**
- Use the comprehensive ward dataset for full coverage
- Preload data during initial setup for faster subsequent loads
- Clear browser cache if experiencing slow map rendering

## 📚 Documentation

- **`DASHBOARD_README.md`** - Detailed dashboard usage guide
- **`DATA_PROCESSING_SUMMARY.md`** - Data processing documentation
- **`ACLED_WARD_TIMESERIES_SUMMARY.md`** - Time series analysis overview

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ACLED** for comprehensive conflict data
- **WorldPop** for population estimates
- **ArcGIS** for administrative boundary data
- **Streamlit** for the web framework
- **Folium** for interactive mapping

## 📞 Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Check the documentation in the `/docs` folder
- Review the troubleshooting section above

---

**Built with ❤️ for violence analysis and peace research in Nigeria**