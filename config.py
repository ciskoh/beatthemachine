"""
Configuration file for Gradient Descent Educational Game
Central location for all application settings and constants
"""

# ============================================================================
# DEFAULT VALUES FOR DATA GENERATION
# ============================================================================

DEFAULT_N_POINTS = 100
DEFAULT_NOISE_LEVEL = 3
DEFAULT_X_RANGE = (0, 10)
DEFAULT_Y_RANGE = (0, 10)

# Random parameter generation ranges
SLOPE_GENERATION_RANGE = (-5, -5)
INTERCEPT_GENERATION_RANGE = (-10, 10)

# ============================================================================
# UI SETTINGS
# ============================================================================

# Plot dimensions (used as defaults, can be overridden)
PLOT_WIDTH = 600
PLOT_HEIGHT = 500

# Slider configurations
SLIDER_PRECISION = 0.01
SLIDER_SLOPE_RANGE = (-5.0, 5.0)
SLIDER_INTERCEPT_RANGE = (-10.0, 10.0)

# Default slider values
DEFAULT_SLOPE = 0.0
DEFAULT_INTERCEPT = 0.0

# ============================================================================
# SUCCESS THRESHOLDS
# ============================================================================

# RMSE thresholds for different success levels
# These are added to the optimal RMSE to determine success
SUCCESS_THRESHOLD_EXCELLENT = 0.5  # Within 0.5 of optimal
SUCCESS_THRESHOLD_GOOD = 1.0       # Within 1.0 of optimal  
SUCCESS_THRESHOLD_ACCEPTABLE = 2.0  # Within 2.0 of optimal

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Color scheme for plots
COLOR_SCHEME = {
    'primary': '#2ca02c',      # Green (accent for success)
    'secondary': '#aaaaaa',    # Neutral grey
    'success': '#2ca02c',
    'error': '#ff4d4d',
    'warning': '#ffa500',
    'info': '#1f77b4',
    'background': '#000000',   # Black background
    'grid': '#333333'          # Subtle dark grey grid
}

PLOT_3D_CONFIG = {
    'width': 700,
    'height': 600,
    'colorscale': 'viridis',
    'marker_size': 5,
    'opacity': 0.9,
    'show_colorbar': False     # hides clutter
}


# ============================================================================
# APPLICATION BEHAVIOR
# ============================================================================

# Maximum number of attempts to store in history
MAX_HISTORY_LENGTH = 100

# Animation settings
SHOW_BALLOONS_ON_SUCCESS = True
SHOW_IMPROVEMENT_METRICS = True

# Educational features
SHOW_TUTORIAL_BY_DEFAULT = False
SHOW_OPTIMAL_PARAMETERS_BY_DEFAULT = False
SHOW_3D_VIEW_BY_DEFAULT = False

# ============================================================================
# MESSAGES AND TEXT
# ============================================================================

MESSAGES = {
    'welcome': "Welcome to the Gradient Descent Intuition Builder!",
    'no_attempts': "No attempts yet. Adjust parameters and click Submit!",
    'excellent': "🏆 Excellent! You found a great fit!",
    'good': "✨ Good job! Very close to optimal!",
    'acceptable': "👍 Getting there! Keep refining!",
    'keep_trying': "Keep adjusting! You can do better!",
    'new_best': "🎉 New best! RMSE: {rmse:.3f}",
    'dataset_reset': "New dataset generated!",
    'history_reset': "History cleared!"
}

# ============================================================================
# EDUCATIONAL CONTENT
# ============================================================================

TUTORIAL_CONTENT = {
    'objective': """
    **Objective**: Find the best line (y = mx + b) that fits the data points by minimizing RMSE (Root Mean Square Error). Use the loss evolution plot and the parameter space plot to guide your adjustments
    """,
    
    'instructions': """
    **Instructions**:
    1. Adjust the **Slope (m)** and **Intercept (b)** sliders
    2. Click **Submit Attempt** to test your parameters
    3. Watch the Loss Evolution plot to see your progress
    4. Try to minimize the RMSE - lower is better!
    5. The line color in the loss plot transitions from red (high error) to blue (low error)
    """,
    
    'tips': """
    **Tips**: 
    - Start by roughly estimating where the line should go
    - Make small adjustments based on the RMSE feedback
    - The optimal RMSE shows the best possible score (limited by noise)
    """,
    
    'gradient_descent_explanation': """
    **What is Gradient Descent?**
    
    Gradient descent is an optimization algorithm that finds the minimum of a function by repeatedly moving 
    in the direction of steepest descent. In this game, you're manually doing what gradient descent does 
    automatically - finding the parameters that minimize the loss function (RMSE).
    """,
    
    'rmse_explanation': """
    **What is RMSE?**
    
    RMSE (Root Mean Square Error) measures the average distance between your line's predictions and the 
    actual data points. Lower RMSE means a better fit. The formula is:
    RMSE = √(mean((predicted - actual)²))
    """
}

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================

# Debug mode settings
DEBUG_MODE = False
SHOW_COMPUTATION_TIME = False
LOG_USER_ACTIONS = False

# Performance settings
USE_CACHING = True
CACHE_EXPIRY_SECONDS = 3600  # 1 hour

# Random seed for reproducible demos (None for random)
DEMO_SEED = None

# ============================================================================
# EXPORT SETTINGS
# ============================================================================

# Settings for exporting results
ALLOW_EXPORT = True
EXPORT_FORMATS = ['csv', 'json']
EXPORT_INCLUDE_DATASET = True
EXPORT_INCLUDE_HISTORY = True

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_success_message(rmse: float, optimal_rmse: float) -> str:
    """
    Get appropriate success message based on RMSE performance.
    
    Parameters
    ----------
    rmse : float
        Current RMSE value
    optimal_rmse : float
        Optimal RMSE value (theoretical minimum)
    
    Returns
    -------
    str
        Appropriate message for the performance level
    """
    gap = rmse - optimal_rmse
    
    if gap <= SUCCESS_THRESHOLD_EXCELLENT:
        return MESSAGES['excellent']
    elif gap <= SUCCESS_THRESHOLD_GOOD:
        return MESSAGES['good']
    elif gap <= SUCCESS_THRESHOLD_ACCEPTABLE:
        return MESSAGES['acceptable']
    else:
        return MESSAGES['keep_trying']


def get_color_for_rmse(rmse: float, min_rmse: float, max_rmse: float) -> str:
    """
    Get color for RMSE value based on performance.
    
    Parameters
    ----------
    rmse : float
        Current RMSE value
    min_rmse : float
        Minimum RMSE in range
    max_rmse : float
        Maximum RMSE in range
    
    Returns
    -------
    str
        Hex color code
    """
    if max_rmse == min_rmse:
        return COLOR_SCHEME['primary']
    
    # Normalize to 0-1
    normalized = (rmse - min_rmse) / (max_rmse - min_rmse)
    
    # Interpolate between success (green) and error (red)
    if normalized < 0.5:
        return COLOR_SCHEME['success']
    else:
        return COLOR_SCHEME['error']


# ============================================================================
# VALIDATION SETTINGS
# ============================================================================

# Input validation
MIN_DATA_POINTS = 10
MAX_DATA_POINTS = 200
MIN_NOISE_LEVEL = 0.0
MAX_NOISE_LEVEL = 2.0

# Parameter bounds
ABSOLUTE_MIN_SLOPE = -10.0
ABSOLUTE_MAX_SLOPE = 10.0
ABSOLUTE_MIN_INTERCEPT = -20.0
ABSOLUTE_MAX_INTERCEPT = 20.0