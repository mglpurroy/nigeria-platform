import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
import time

def create_admin_map(aggregated, boundaries, agg_level, map_var, agg_thresh, period_info, rate_thresh, abs_thresh):
    """Create administrative units map with optimized performance"""
    import time
    start_time = time.time()
    
    # Determine columns based on actual boundary file structure
    if agg_level == 'ADM1':
        # State level
        pcode_col = 'statecode'  # From boundary file
        name_col = 'statename'   # From boundary file
        agg_pcode_col = 'ADM1_PCODE'  # From aggregated data
        agg_name_col = 'ADM1_EN'      # From aggregated data
    else:
        # LGA level  
        pcode_col = 'lgacode'    # From boundary file
        name_col = 'lganame'     # From boundary file
        agg_pcode_col = 'ADM2_PCODE'  # From aggregated data
        agg_name_col = 'ADM2_EN'      # From aggregated data
    
    if map_var == 'share_wards':
        value_col = 'share_wards_affected'
        value_label = 'Share of Wards Affected'
    else:
        value_col = 'share_population_affected'
        value_label = 'Share of Population Affected'
    
    # Get appropriate boundary data
    map_level_num = 1 if agg_level == 'ADM1' else 2
    gdf = boundaries[map_level_num]
    
    if gdf.empty:
        st.error(f"No boundary data available for {agg_level}")
        return None
    
    # Merge data with boundaries using optimized merge
    merge_cols = [agg_pcode_col, value_col, 'above_threshold', 'violence_affected', 'total_wards', 'pop_count', 'ACLED_BRD_total']
    merged_gdf = gdf.merge(aggregated[merge_cols], left_on=pcode_col, right_on=agg_pcode_col, how='left')
    
    # Use vectorized fillna
    fill_values = {
        value_col: 0, 
        'above_threshold': False, 
        'violence_affected': 0, 
        'total_wards': 0,
        'pop_count': 0,
        'ACLED_BRD_total': 0
    }
    merged_gdf = merged_gdf.fillna(fill_values)
    
    # Create map with optimized settings - centered on Nigeria
    m = folium.Map(
        location=[9.0820, 8.6753],  # Nigeria center coordinates
        zoom_start=6, 
        tiles='OpenStreetMap',
        prefer_canvas=True
    )
    
    # Pre-calculate colors and status for better performance
    def get_color_status(value):
        if value > agg_thresh:
            return '#d73027', 0.8, "HIGH VIOLENCE"
        elif value > 0:
            return '#fd8d3c', 0.7, "Some Violence"
        else:
            return '#2c7fb8', 0.4, "Low/No Violence"
    
    # Add choropleth layer with optimized rendering
    for _, row in merged_gdf.iterrows():
        value = row[value_col]
        color, opacity, status = get_color_status(value)
        
        # Simplified popup content for better performance
        popup_content = f"""
        <div style="width: 280px; font-family: Arial, sans-serif;">
            <h4 style="color: {color}; margin: 0;">{row.get(name_col, 'Unknown')}</h4>
            <div style="background: {color}; color: white; padding: 3px; border-radius: 2px; text-align: center; margin: 5px 0;">
                <strong>{status}</strong>
            </div>
            <p><strong>{value_label}:</strong> {value:.1%}</p>
            <p><strong>Affected Wards:</strong> {row['violence_affected']}/{row['total_wards']}</p>
            <p><strong>Total Deaths:</strong> {row['ACLED_BRD_total']:,.0f}</p>
        </div>
        """
        
        folium.GeoJson(
            row.geometry,
            style_function=lambda x, color=color, opacity=opacity: {
                'fillColor': color,
                'color': 'black',
                'weight': 0.8,
                'fillOpacity': opacity
            },
            popup=folium.Popup(popup_content, max_width=300),
            tooltip=f"{row.get(name_col, 'Unknown')}: {value:.1%}"
        ).add_to(m)
    
    # Simplified legend
    legend_html = f'''
    <div style="position: fixed; top: 10px; right: 10px; width: 250px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:11px; padding: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                border-radius: 4px;">
    <h4 style="margin: 0 0 6px 0; color: #333;">{value_label}</h4>
    <div style="margin-bottom: 6px;">
        <div style="margin: 2px 0;"><span style="background:#d73027; color:white; padding:1px 3px; border-radius:1px; font-size:9px;">HIGH</span> >{agg_thresh:.1%}</div>
        <div style="margin: 2px 0;"><span style="background:#fd8d3c; color:white; padding:1px 3px; border-radius:1px; font-size:9px;">SOME</span> >0%</div>
        <div style="margin: 2px 0;"><span style="background:#2c7fb8; color:white; padding:1px 3px; border-radius:1px; font-size:9px;">LOW</span> 0%</div>
    </div>
    <div style="font-size:9px; color:#666;">
        <strong>Period:</strong> {period_info['label']}<br>
        <strong>Criteria:</strong> >{rate_thresh:.1f}/100k & >{abs_thresh} deaths<br>
        <strong>Black borders:</strong> State boundaries
    </div>
    </div>
    '''
    
    # Add State borders on top of admin units
    admin1_gdf = boundaries[1]
    if not admin1_gdf.empty:
        for _, row in admin1_gdf.iterrows():
            folium.GeoJson(
                row.geometry,
                style_function=lambda x: {
                    'fillColor': 'transparent',
                    'color': '#000000',
                    'weight': 2,
                    'fillOpacity': 0,
                    'opacity': 0.8
                },
                tooltip=f"State: {row.get('statename', 'Unknown')}"
            ).add_to(m)
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def create_ward_map(ward_data, boundaries, period_info, rate_thresh, abs_thresh):
    """Create ward classification map with optimized performance"""
    import time
    start_time = time.time()
    
    # Get ward boundaries
    ward_gdf = boundaries[3]
    
    if ward_gdf.empty:
        st.error("No ward boundary data available")
        return None
    
    # Merge with classification data using optimized merge
    # Ward boundaries use 'wardcode' column, ward data uses 'ADM3_PCODE'
    merge_cols = ['ADM3_PCODE', 'violence_affected', 'ACLED_BRD_total', 'acled_total_death_rate']
    merged_ward = ward_gdf.merge(ward_data[merge_cols], left_on='wardcode', right_on='ADM3_PCODE', how='left')
    
    # Use vectorized fillna
    fill_values = {
        'violence_affected': False, 
        'ACLED_BRD_total': 0,
        'acled_total_death_rate': 0
    }
    merged_ward = merged_ward.fillna(fill_values)
    
    # Create map with optimized settings - centered on Nigeria
    m = folium.Map(
        location=[9.0820, 8.6753],  # Nigeria center coordinates
        zoom_start=6, 
        tiles='OpenStreetMap',
        prefer_canvas=True
    )
    
    # Pre-calculate statistics for legend
    total_wards = len(ward_data)
    affected_wards = sum(ward_data['violence_affected'])
    affected_percentage = (affected_wards / total_wards * 100) if total_wards > 0 else 0
    
    # Pre-calculate colors and status for better performance
    def get_ward_color_status(row):
        if row['violence_affected']:
            return '#d73027', 0.8, "VIOLENCE AFFECTED"
        elif row['ACLED_BRD_total'] > 0:
            return '#fd8d3c', 0.6, "Below Threshold"
        else:
            return '#2c7fb8', 0.3, "No Violence"
    
    # Add ward layer with optimized rendering
    for _, row in merged_ward.iterrows():
        # Skip rows with missing geometries
        if row.geometry is None:
            continue
            
        color, opacity, status = get_ward_color_status(row)
        
        # Simplified popup content for better performance
        popup_content = f"""
        <div style="width: 250px; font-family: Arial, sans-serif;">
            <h4 style="color: {color}; margin: 0;">{row.get('wardname', 'Unknown')}</h4>
            <div style="background: {color}; color: white; padding: 3px; border-radius: 2px; text-align: center; margin: 5px 0;">
                <strong>{status}</strong>
            </div>
            <p><strong>LGA:</strong> {row.get('lganame', 'Unknown')}</p>
            <p><strong>Deaths:</strong> {row['ACLED_BRD_total']:,.0f}</p>
            <p><strong>Rate:</strong> {row['acled_total_death_rate']:.1f}/100k</p>
        </div>
        """
        
        # Create GeoJson with click functionality
        geojson_layer = folium.GeoJson(
            row.geometry,
            style_function=lambda x, color=color, opacity=opacity: {
                'fillColor': color,
                'color': 'black',
                'weight': 0.3,
                'fillOpacity': opacity
            },
            popup=folium.Popup(popup_content, max_width=270),
            tooltip=f"{row.get('wardname', 'Unknown')}: {status}"
        )
        
        # Add ward code to popup for click detection
        geojson_layer.add_child(
            folium.Popup(
                f"""
                <div style="width: 200px;">
                    <h4>{row.get('wardname', 'Unknown')}</h4>
                    <p><strong>Ward Code:</strong> {row.get('wardcode', 'Unknown')}</p>
                    <p><strong>LGA:</strong> {row.get('lganame', 'Unknown')}</p>
                    <p><strong>State:</strong> {row.get('statename', 'Unknown')}</p>
                    <p><strong>Deaths:</strong> {row['ACLED_BRD_total']:,.0f}</p>
                    <p><strong>Rate:</strong> {row['acled_total_death_rate']:.1f}/100k</p>
                    <p style="color: #666; font-size: 10px;">Click to select for time series analysis</p>
                </div>
                """,
                max_width=250
            )
        )
        
        geojson_layer.add_to(m)
    
    # Add State borders on top of wards
    admin1_gdf = boundaries[1]
    if not admin1_gdf.empty:
        for _, row in admin1_gdf.iterrows():
            folium.GeoJson(
                row.geometry,
                style_function=lambda x: {
                    'fillColor': 'transparent',
                    'color': '#000000',
                    'weight': 2,
                    'fillOpacity': 0,
                    'opacity': 0.8
                },
                tooltip=f"State: {row.get('statename', 'Unknown')}"
            ).add_to(m)
    
    # Simplified legend
    legend_html = f'''
    <div style="position: fixed; top: 10px; right: 10px; width: 240px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:11px; padding: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                border-radius: 4px;">
    <h4 style="margin: 0 0 6px 0; color: #333;">Ward Classification</h4>
    <div style="margin-bottom: 6px;">
        <div style="margin: 2px 0;"><span style="background:#d73027; color:white; padding:1px 3px; border-radius:1px; font-size:9px;">AFFECTED</span> Violence Affected</div>
        <div style="margin: 2px 0;"><span style="background:#fd8d3c; color:white; padding:1px 3px; border-radius:1px; font-size:9px;">BELOW</span> Below Threshold</div>
        <div style="margin: 2px 0;"><span style="background:#2c7fb8; color:white; padding:1px 3px; border-radius:1px; font-size:9px;">NONE</span> No Violence</div>
    </div>
    <div style="font-size:9px; color:#666;">
        <strong>Period:</strong> {period_info['label']}<br>
        <strong>Criteria:</strong> >{rate_thresh:.1f}/100k & >{abs_thresh} deaths<br>
        <strong>Affected:</strong> {affected_wards}/{total_wards} ({affected_percentage:.1f}%)<br>
        <strong>Black borders:</strong> State boundaries
    </div>
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    
    return m
