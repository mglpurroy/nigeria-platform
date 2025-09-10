import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def create_analysis_charts(aggregated, ward_data, period_info, agg_level, agg_thresh):
    """Create comprehensive analysis charts using Plotly"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Violence-Affected Areas (Ranked by Ward Share)',
            'Population vs Violence Deaths',
            'Distribution of Violence Levels',
            'Ward Classification Breakdown'
        ),
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "domain"}, {"type": "xy"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Chart 1: Horizontal bar chart of units with violence
    # Show all units, not just those with violence, for better visibility
    aggregated_sorted = aggregated.sort_values('share_wards_affected', ascending=True)
    
    if len(aggregated_sorted) > 0:
        name_col = f'{agg_level}_EN'
        colors = ['#d73027' if above else '#fd8d3c' if share > 0 else '#2c7fb8' 
                 for above, share in zip(aggregated_sorted['above_threshold'], aggregated_sorted['share_wards_affected'])]
        
        # Process names to ensure they fit properly and align with bars
        processed_names = []
        for name in aggregated_sorted[name_col]:
            if len(name) > 25:
                # For very long names, try to find a good break point
                if ' ' in name:
                    words = name.split(' ')
                    if len(words) > 2:
                        # Try to break at a logical point
                        processed_name = ' '.join(words[:2]) + '\n' + ' '.join(words[2:])
                    else:
                        processed_name = name[:22] + '...'
                else:
                    processed_name = name[:22] + '...'
            else:
                processed_name = name
        
        fig.add_trace(
            go.Bar(
                y=processed_names,
                x=aggregated_sorted['share_wards_affected'],
                orientation='h',
                marker_color=colors,
                showlegend=False,
                hovertemplate='<b>%{y}</b><br>Share: %{x:.1%}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_vline(x=agg_thresh, line_dash="dash", line_color="red", row=1, col=1)
    
    # Chart 2: Population vs Deaths scatter
    if len(aggregated) > 0:
        scatter_colors = ['#d73027' if above else '#2c7fb8' for above in aggregated['above_threshold']]
        
        fig.add_trace(
            go.Scatter(
                x=aggregated['pop_count']/1000,
                y=aggregated['ACLED_BRD_total'],
                mode='markers',
                marker=dict(color=scatter_colors, size=8),
                showlegend=False,
                hovertemplate='<b>Population:</b> %{x:.0f}k<br><b>Deaths:</b> %{y}<extra></extra>'
            ),
            row=1, col=2
        )
    
    # Chart 3: Distribution of violence levels
    if len(aggregated) > 0:
        aggregated_copy = aggregated.copy()
        aggregated_copy['violence_level'] = 'No Violence'
        aggregated_copy.loc[aggregated_copy['share_wards_affected'] > 0, 'violence_level'] = 'Some Violence'
        aggregated_copy.loc[aggregated_copy['above_threshold'], 'violence_level'] = 'High Violence'
        
        level_counts = aggregated_copy['violence_level'].value_counts()
        colors_pie = ['#2c7fb8', '#fd8d3c', '#d73027']
        
        fig.add_trace(
            go.Pie(
                labels=level_counts.index,
                values=level_counts.values,
                marker_colors=colors_pie,
                showlegend=False,
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            ),
            row=2, col=1
        )
    
    # Chart 4: Ward classification breakdown
    if len(ward_data) > 0:
        no_violence = len(ward_data[ward_data['ACLED_BRD_total'] == 0])
        below_threshold = len(ward_data[(ward_data['ACLED_BRD_total'] > 0) & (~ward_data['violence_affected'])])
        violence_affected = len(ward_data[ward_data['violence_affected']])
        
        categories = ['No Violence', 'Below Threshold', 'Violence Affected']
        values = [no_violence, below_threshold, violence_affected]
        colors = ['#2c7fb8', '#fd8d3c', '#d73027']
        
        fig.add_trace(
            go.Bar(
                x=categories,
                y=values,
                marker_color=colors,
                showlegend=False,
                hovertemplate='<b>%{x}</b><br>Count: %{y}<br>Percentage: %{y:.1%}<extra></extra>',
                text=[f'{v}<br>({v/len(ward_data)*100:.1f}%)' for v in values],
                textposition='auto'
            ),
            row=2, col=2
        )
    
    # Update layout with better spacing and sizing
    fig.update_layout(
        height=800,  # Increased height
        width=1200,  # Set explicit width
        title_text=f'Supporting Analysis - {period_info["label"]}',
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=50)  # Better margins
    )
    
    # Update axes with better formatting
    fig.update_xaxes(title_text="Share of Wards Affected", row=1, col=1)
    fig.update_xaxes(title_text="Population (thousands)", row=1, col=2)
    fig.update_xaxes(title_text="Category", row=2, col=2)
    
    # Update y-axes with better formatting for the bar chart
    fig.update_yaxes(
        title_text="Administrative Unit", 
        row=1, col=1,
        tickfont=dict(size=10),  # Smaller font for better fit
        automargin=True  # Auto-adjust margins
    )
    fig.update_yaxes(title_text="Total Deaths", row=1, col=2)
    fig.update_yaxes(title_text="Number of Wards", row=2, col=2)
    
    return fig
