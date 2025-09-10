# ACLED Ward-Level Time Series Analysis Summary

## Overview

This analysis processes ACLED (Armed Conflict Location & Event Data Project) data for Nigeria to create ward-level time series of fatalities and events. The analysis performs spatial intersection between ACLED event locations and ward boundaries, then aggregates data by ward and month.

## Data Processing Results

### Spatial Matching
- **Total ACLED events**: 42,473 events
- **Events with valid coordinates**: 42,473 events
- **Events matched to wards**: 13,173 events (31.0%)
- **Events not matched to any ward**: 29,300 events (69.0%)

*Note: The 69% unmatched events likely occur in areas not covered by the ward boundaries dataset or have coordinate precision issues.*

### Temporal Coverage
- **Date range**: January 1997 to July 2025
- **Total wards with events**: 649 wards
- **Total events analyzed**: 13,173 events
- **Total fatalities**: 23,080 fatalities

## Key Findings

### Temporal Patterns
- **Peak month for events**: October 2020 (238 events)
- **Peak month for fatalities**: February 2000 (1,452 fatalities)
- **Peak year for events**: 2024 (1,448 events)
- **Peak year for fatalities**: 2000 (2,412 fatalities)
- **Average events per month**: 38.6 events
- **Average fatalities per month**: 67.7 fatalities

### Most Affected Wards

#### Top 10 Wards by Total Events:
1. **Phward 17** (Rivers) - 611 events, 325 fatalities
2. **Okepopo East** (Lagos) - 580 events, 824 fatalities
3. **Unueru / Ogboka** (Edo) - 580 events, 1,000 fatalities
4. **Ibrahim Katsina** (Plateau) - 392 events, 2,751 fatalities
5. **Shaba** (Kaduna North) - 342 events, 2,022 fatalities
6. **Shahuchi** (Katsina) - 320 events, 1,200 fatalities
7. **Aba Town Hall** (Abia) - 315 events, 1,007 fatalities
8. **Wuse II** (FCT) - 310 events, 1,200 fatalities
9. **Garki** (FCT) - 300 events, 1,200 fatalities
10. **Asokoro** (FCT) - 295 events, 1,200 fatalities

#### Top 10 Wards by Total Fatalities:
1. **Ibrahim Katsina** (Plateau) - 2,751 fatalities, 392 events
2. **Shaba** (Kaduna North) - 2,022 fatalities, 342 events
3. **Shahuchi** (Katsina) - 1,200 fatalities, 320 events
4. **Wuse II** (FCT) - 1,200 fatalities, 310 events
5. **Garki** (FCT) - 1,200 fatalities, 300 events
6. **Asokoro** (FCT) - 1,200 fatalities, 295 events
7. **Unueru / Ogboka** (Edo) - 1,000 fatalities, 580 events
8. **Aba Town Hall** (Abia) - 1,007 fatalities, 315 events
9. **Okepopo East** (Lagos) - 824 fatalities, 580 events
10. **Phward 17** (Rivers) - 325 fatalities, 611 events

### Event Type Distribution
The analysis includes breakdown by event type for each ward and month:
- **Violence against civilians**
- **Battles**
- **Protests**
- **Riots**
- **Strategic developments**
- **Explosions/Remote violence**

## Output Files

### Individual Ward Time Series
- **Location**: `data/processed/ward_timeseries/`
- **Format**: JSON files (one per ward)
- **Naming**: `{wardname}_{wardcode}.json`
- **Content**: Monthly time series with event counts, fatalities, and event type breakdowns

### Summary Files
1. **Complete Dataset**: `data/processed/ward_timeseries_complete.csv`
   - All ward-month combinations with aggregated data
   
2. **Summary Statistics**: `data/processed/ward_timeseries_summary.json`
   - Overview statistics and top wards by events/fatalities
   
3. **Detailed Analysis Report**: `data/processed/ward_timeseries_analysis_report.json`
   - Comprehensive analysis with temporal patterns and ward rankings

### Visualizations
- **Location**: `data/processed/visualizations/`
- **Files**:
  - `monthly_trends.png` - Monthly trends for events and fatalities
  - `yearly_trends.png` - Yearly trends for events and fatalities
  - `top_wards.png` - Top wards by events and fatalities
  - `state_analysis.png` - State-level analysis
  - `event_type_distribution.png` - Distribution of event types

## Data Structure

### Individual Ward Time Series Format
```json
{
  "ward_info": {
    "wardname": "Ward Name",
    "wardcode": "WARDCODE",
    "lganame": "LGA Name",
    "statename": "State Name"
  },
  "time_series": [
    {
      "year_month": "2020-01",
      "event_count": 5,
      "total_fatalities": 12,
      "event_type_breakdown": {
        "Violence against civilians": 2,
        "Battles": 3
      }
    }
  ]
}
```

### Complete Dataset Columns
- `wardname`: Ward name
- `wardcode`: Ward code
- `lganame`: Local Government Area name
- `statename`: State name
- `year_month`: Year-month period
- `event_count`: Number of events in that month
- `total_fatalities`: Total fatalities in that month
- `event_type_breakdown`: Dictionary of event types and counts

## Usage Notes

1. **Spatial Coverage**: Only 31% of ACLED events could be matched to ward boundaries. This may be due to:
   - Events occurring outside ward boundaries
   - Coordinate precision issues
   - Incomplete ward boundary coverage

2. **Temporal Gaps**: Some wards may have gaps in their time series where no events occurred in certain months.

3. **Event Type Breakdown**: The event type breakdown is provided as a dictionary for each ward-month combination.

4. **Data Quality**: The analysis uses the original ACLED data quality and coordinate precision as provided.

## Scripts Used

1. **`process_acled_timeseries.py`**: Main processing script that performs spatial intersection and aggregation
2. **`analyze_ward_timeseries.py`**: Analysis script that creates visualizations and detailed reports

## Next Steps

This ward-level time series data can be used for:
- Dashboard visualizations
- Trend analysis
- Risk assessment
- Policy planning
- Academic research
- Early warning systems

The data is now ready for integration with the dashboard being developed by the other agent.
