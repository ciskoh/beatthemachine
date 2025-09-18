"""
Visualization Module for Gradient Descent Educational Game
Creates interactive Plotly figures for data visualization and loss tracking.
"""

import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Any


# ============================================================================
# VISUALIZATION CONFIGURATION - All visual settings in one place
# ============================================================================

PLOT_CONFIG = {
    # General plot dimensions
    'width': 600,
    'height': 500,
    
    # Background and grid settings
    'plot_bgcolor': 'white',
    'grid_color': 'lightgray',
    'grid_width': 1,
    
    # Axis settings
    'x_range': [0, 15],
    'y_range': [0, 15],
    'fixed_range': True,  # Prevent zooming
    
    # Data points styling
    'scatter': {
        'color': 'blue',
        'size': 8,
        'line_width': 0,
        'hover_decimal_places': 2,
    },
    
    # Regression line styling
    'regression_line': {
        'color': 'red',
        'width': 2,
        'dash': None,  # solid line
    },
    
    # True line styling (when shown)
    'true_line': {
        'color': 'green',
        'width': 2,
        'dash': 'dash',
    },
    
    # Loss plot specific
    'loss_plot': {
        'line_width': 2,
        'marker_size': 8,
        'marker_line_width': 1,
        'marker_line_color': 'white',
        # Color gradient from red (high loss) to blue (low loss)
        'color_gradient': {
            'high_color': (255, 0, 0),  # Red RGB
            'low_color': (0, 0, 255),   # Blue RGB
            'mid_color': (128, 0, 128),  # Purple RGB (for single points)
        },
        'show_improvement': True,
        'improvement_color': 'green',
    },
    
    # Annotations styling
    'annotations': {
        'equation': {
            'x_pos': 0.5,
            'y_pos': 9.5,
            'font_size': 14,
            'font_color': 'black',
            'bg_color': 'rgba(255, 255, 255, 0.8)',
            'border_color': 'black',
            'border_width': 1,
            'decimal_places': 2,
        },
        'rmse': {
            'x_pos': 9.5,
            'y_pos': 0.5,
            'font_size': 12,
            'font_color': 'black',
            'bg_color': 'rgba(255, 255, 255, 0.8)',
            'border_color': 'black',
            'border_width': 1,
            'decimal_places': 3,
        },
        'improvement': {
            'font_size': 10,
            'font_color': 'green',
            'bg_color': 'rgba(255, 255, 255, 0.8)',
        },
    },
    
    # Titles and labels
    'titles': {
        'scatter_plot': 'Linear Regression Fit',
        'loss_plot': 'Loss Evolution',
        'comparison_plot': 'Linear Regression Fit (with True Line)',
        'x_label': 'X',
        'y_label': 'Y',
        'epoch_label': 'Attempt (Epoch)',
        'loss_label': 'RMSE (Loss)',
    },
    
    # Hover template formats
    'hover': {
        'show_hover': True,
        'hover_mode': 'closest',
        'data_template': 'X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
        'loss_template': (
            'Attempt: %{x}<br>' +
            'RMSE: %{y:.3f}<br>' +
            'Slope: %{customdata[0]:.2f}<br>' +
            'Intercept: %{customdata[1]:.2f}' +
            '<extra></extra>'
        ),
    },
}


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================


def create_scatter_plot(
    data: Dict[str, np.ndarray], 
    current_slope: float = None, 
    current_intercept: float = None,
    current_rmse: float = None
) -> go.Figure:
    """
    Create an interactive scatter plot with data points and optional regression line.
    
    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Dictionary containing 'x' and 'y' arrays of data points
    current_slope : float, optional
        Current slope parameter for the regression line. If None, no line is drawn.
    current_intercept : float, optional
        Current intercept parameter for the regression line. If None, no line is drawn.
    current_rmse : float, optional
        Current RMSE value to display on the plot
    
    Returns
    -------
    go.Figure
        Plotly figure object with scatter plot and optional regression line
    """
    # Create figure
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=data['x'],
        y=data['y'],
        mode='markers',
        name='Data Points',
        marker=dict(
            color=PLOT_CONFIG['scatter']['color'],
            size=PLOT_CONFIG['scatter']['size'],
            line=dict(width=PLOT_CONFIG['scatter']['line_width'])
        ),
        hovertemplate=PLOT_CONFIG['hover']['data_template']
    ))
    
    # Add regression line only if slope and intercept are provided
    if current_slope is not None and current_intercept is not None:
        # Generate points for regression line
        x_line = np.array(PLOT_CONFIG['x_range'])
        y_line = current_slope * x_line + current_intercept
        
        # Add regression line
        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            name='Regression Line',
            line=dict(
                color=PLOT_CONFIG['regression_line']['color'],
                width=PLOT_CONFIG['regression_line']['width'],
                dash=PLOT_CONFIG['regression_line']['dash']
            ),
            hoverinfo='skip'
        ))
    
    # Update layout
    fig.update_layout(
        title=PLOT_CONFIG['titles']['scatter_plot'],
        xaxis=dict(
            title=PLOT_CONFIG['titles']['x_label'],
            range=PLOT_CONFIG['x_range'],
            fixedrange=PLOT_CONFIG['fixed_range'],
            showgrid=True,
            gridcolor=PLOT_CONFIG['grid_color'],
            gridwidth=PLOT_CONFIG['grid_width']
        ),
        yaxis=dict(
            title=PLOT_CONFIG['titles']['y_label'],
            range=PLOT_CONFIG['y_range'],
            fixedrange=PLOT_CONFIG['fixed_range'],
            showgrid=True,
            gridcolor=PLOT_CONFIG['grid_color'],
            gridwidth=PLOT_CONFIG['grid_width']
        ),
        width=PLOT_CONFIG['width'],
        height=PLOT_CONFIG['height'],
        showlegend=False,
        hovermode=PLOT_CONFIG['hover']['hover_mode'],
        plot_bgcolor=PLOT_CONFIG['plot_bgcolor']
    )
    
    # Add equation annotation only if slope and intercept are provided
    if current_slope is not None and current_intercept is not None:
        eq_config = PLOT_CONFIG['annotations']['equation']
        equation_text = f"y = {current_slope:.{eq_config['decimal_places']}f}x + {current_intercept:.{eq_config['decimal_places']}f}"
        
        fig.add_annotation(
            x=eq_config['x_pos'],
            y=eq_config['y_pos'],
            text=equation_text,
            showarrow=False,
            font=dict(size=eq_config['font_size'], color=eq_config['font_color']),
            bgcolor=eq_config['bg_color'],
            bordercolor=eq_config['border_color'],
            borderwidth=eq_config['border_width']
        )
    
    # Add RMSE annotation if provided (and if we have a line to measure against)
    if current_rmse is not None and current_slope is not None and current_intercept is not None:
        rmse_config = PLOT_CONFIG['annotations']['rmse']
        rmse_text = f"RMSE: {current_rmse:.{rmse_config['decimal_places']}f}"
        
        fig.add_annotation(
            x=rmse_config['x_pos'],
            y=rmse_config['y_pos'],
            text=rmse_text,
            showarrow=False,
            font=dict(size=rmse_config['font_size'], color=rmse_config['font_color']),
            bgcolor=rmse_config['bg_color'],
            bordercolor=rmse_config['border_color'],
            borderwidth=rmse_config['border_width']
        )
    
    return fig

def create_loss_plot(history: List[Dict[str, Any]]) -> go.Figure:
    """
    Create an interactive plot showing RMSE evolution over attempts with color gradient.
    
    Parameters
    ----------
    history : List[Dict[str, Any]]
        List of dictionaries, each containing:
        - 'epoch': int, attempt number
        - 'rmse': float, RMSE value
        - 'slope': float, slope parameter used
        - 'intercept': float, intercept parameter used
    
    Returns
    -------
    go.Figure
        Plotly figure object with loss evolution plot
    """
    # Handle empty history
    if not history:
        fig = go.Figure()
        fig.update_layout(
            title=PLOT_CONFIG['titles']['loss_plot'],
            xaxis=dict(title=PLOT_CONFIG['titles']['epoch_label'], range=[0, 10]),
            yaxis=dict(title=PLOT_CONFIG['titles']['loss_label'], range=[0, 10]),
            width=PLOT_CONFIG['width'],
            height=PLOT_CONFIG['height']
        )
        return fig
    
    # Extract data from history
    epochs = [h['epoch'] for h in history]
    rmse_values = [h['rmse'] for h in history]
    slopes = [h['slope'] for h in history]
    intercepts = [h['intercept'] for h in history]
    
    # Calculate color gradient
    colors = calculate_color_gradient(rmse_values)
    
    # Create figure
    fig = go.Figure()
    
    # Add line trace connecting points with gradient colors
    for i in range(len(epochs) - 1):
        fig.add_trace(go.Scatter(
            x=[epochs[i], epochs[i + 1]],
            y=[rmse_values[i], rmse_values[i + 1]],
            mode='lines',
            line=dict(
                color=colors[i],
                width=PLOT_CONFIG['loss_plot']['line_width']
            ),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add scatter points with color gradient
    fig.add_trace(go.Scatter(
        x=epochs,
        y=rmse_values,
        mode='markers',
        marker=dict(
            color=colors,
            size=PLOT_CONFIG['loss_plot']['marker_size'],
            line=dict(
                width=PLOT_CONFIG['loss_plot']['marker_line_width'],
                color=PLOT_CONFIG['loss_plot']['marker_line_color']
            )
        ),
        customdata=list(zip(slopes, intercepts)),
        hovertemplate=PLOT_CONFIG['hover']['loss_template'],
        showlegend=False
    ))
    
    # Calculate appropriate y-axis range
    max_rmse = max(rmse_values)
    y_range_max = max(PLOT_CONFIG['y_range'][1], max_rmse * 1.1)
    
    # Update layout
    fig.update_layout(
        title=PLOT_CONFIG['titles']['loss_plot'],
        xaxis=dict(
            title=PLOT_CONFIG['titles']['epoch_label'],
            range=[0.5, max(10, len(epochs) + 0.5)],
            showgrid=True,
            gridcolor=PLOT_CONFIG['grid_color'],
            gridwidth=PLOT_CONFIG['grid_width'],
            dtick=1
        ),
        yaxis=dict(
            title=PLOT_CONFIG['titles']['loss_label'],
            range=[0, y_range_max],
            showgrid=True,
            gridcolor=PLOT_CONFIG['grid_color'],
            gridwidth=PLOT_CONFIG['grid_width']
        ),
        width=PLOT_CONFIG['width'],
        height=PLOT_CONFIG['height'],
        hovermode=PLOT_CONFIG['hover']['hover_mode'],
        plot_bgcolor=PLOT_CONFIG['plot_bgcolor']
    )
    
    # Add improvement annotation if enabled
    if PLOT_CONFIG['loss_plot']['show_improvement'] and len(rmse_values) > 1:
        improvement = rmse_values[0] - rmse_values[-1]
        if improvement > 0:
            imp_config = PLOT_CONFIG['annotations']['improvement']
            improvement_text = f"Improvement: {improvement:.3f}"
            
            fig.add_annotation(
                x=0.98,
                y=0.98,
                xref='paper',
                yref='paper',
                text=improvement_text,
                showarrow=False,
                font=dict(size=imp_config['font_size'], color=imp_config['font_color']),
                bgcolor=imp_config['bg_color'],
                xanchor='right',
                yanchor='top'
            )
    
    return fig


def calculate_color_gradient(rmse_values: List[float]) -> List[str]:
    """
    Calculate color gradient from red (high RMSE) to blue (low RMSE).
    
    Parameters
    ----------
    rmse_values : List[float]
        List of RMSE values to map to colors
    
    Returns
    -------
    List[str]
        List of RGB color strings for each RMSE value
    """
    if not rmse_values:
        return []
    
    gradient_config = PLOT_CONFIG['loss_plot']['color_gradient']
    
    # Handle single value case
    if len(rmse_values) == 1:
        r, g, b = gradient_config['mid_color']
        return [f'rgb({r}, {g}, {b})']
    
    # Get min and max for normalization
    min_rmse = min(rmse_values)
    max_rmse = max(rmse_values)
    
    # Handle case where all values are the same
    if max_rmse == min_rmse:
        r, g, b = gradient_config['mid_color']
        return [f'rgb({r}, {g}, {b})'] * len(rmse_values)
    
    colors = []
    high_r, high_g, high_b = gradient_config['high_color']
    low_r, low_g, low_b = gradient_config['low_color']
    
    for rmse in rmse_values:
        # Normalize to 0-1 range (0 = worst/high, 1 = best/low)
        normalized = 1 - (rmse - min_rmse) / (max_rmse - min_rmse)
        
        # Linear interpolation between high and low colors
        red = int(high_r * (1 - normalized) + low_r * normalized)
        green = int(high_g * (1 - normalized) + low_g * normalized)
        blue = int(high_b * (1 - normalized) + low_b * normalized)
        
        color_str = f'rgb({red}, {green}, {blue})'
        colors.append(color_str)
    
    return colors


def create_comparison_plot(
    data: Dict[str, np.ndarray],
    current_slope: float,
    current_intercept: float,
    true_slope: float = None,
    true_intercept: float = None
) -> go.Figure:
    """
    Create a scatter plot comparing current fit with true fit (optional utility function).
    
    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Dictionary containing 'x' and 'y' arrays
    current_slope : float
        Current slope parameter
    current_intercept : float
        Current intercept parameter
    true_slope : float, optional
        True slope (for comparison)
    true_intercept : float, optional
        True intercept (for comparison)
    
    Returns
    -------
    go.Figure
        Plotly figure with comparison visualization
    """
    # Create base plot with current parameters
    fig = create_scatter_plot(data, current_slope, current_intercept)
    
    # Add true line if parameters provided
    if true_slope is not None and true_intercept is not None:
        x_line = np.array(PLOT_CONFIG['x_range'])
        y_line = true_slope * x_line + true_intercept
        
        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            name='True Line',
            line=dict(
                color=PLOT_CONFIG['true_line']['color'],
                width=PLOT_CONFIG['true_line']['width'],
                dash=PLOT_CONFIG['true_line']['dash']
            ),
            hoverinfo='skip'
        ))
        
        # Update title
        fig.update_layout(title=PLOT_CONFIG['titles']['comparison_plot'])
    
    return fig


# Example usage and testing
if __name__ == "__main__":
    # Example of how to modify settings
    print("Current scatter point color:", PLOT_CONFIG['scatter']['color'])
    print("Current regression line color:", PLOT_CONFIG['regression_line']['color'])
    
    # You can easily modify settings like this:
    # PLOT_CONFIG['scatter']['color'] = 'purple'
    # PLOT_CONFIG['regression_line']['width'] = 3
    
    # Create sample data for testing
    np.random.seed(42)
    test_data = {
        'x': np.random.uniform(0, 10, 30),
        'y': 1.5 * np.random.uniform(0, 10, 30) + 2 + np.random.normal(0, 0.5, 30)
    }
    
    # Test scatter plot
    print("\nCreating scatter plot...")
    scatter_fig = create_scatter_plot(test_data, 1.2, 2.5, 1.234)
    print("Scatter plot created successfully")
    
    # Test loss plot with sample history
    print("\nCreating loss plot...")
    test_history = [
        {'epoch': 1, 'rmse': 5.0, 'slope': 0.5, 'intercept': 1.0},
        {'epoch': 2, 'rmse': 4.2, 'slope': 0.8, 'intercept': 1.5},
        {'epoch': 3, 'rmse': 3.5, 'slope': 1.0, 'intercept': 1.8},
        {'epoch': 4, 'rmse': 2.8, 'slope': 1.2, 'intercept': 2.2},
        {'epoch': 5, 'rmse': 2.1, 'slope': 1.4, 'intercept': 2.4},
        {'epoch': 6, 'rmse': 1.5, 'slope': 1.45, 'intercept': 2.45},
        {'epoch': 7, 'rmse': 1.2, 'slope': 1.48, 'intercept': 2.48},
        {'epoch': 8, 'rmse': 1.1, 'slope': 1.5, 'intercept': 2.5},
    ]
    
    loss_fig = create_loss_plot(test_history)
    print("Loss plot created successfully")