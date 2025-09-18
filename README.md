Gradient Descent Educational Game - Technical Specifications (Revised)
Project Overview
An interactive web application that teaches gradient descent concepts through manual linear regression fitting. Students adjust slope and intercept parameters to fit a line to scattered data points while visualizing the loss function evolution with color-coded progress.
Technical Stack

Framework: Streamlit (latest stable version)
Plotting: Plotly (for interactive charts)
Computation: NumPy (for numerical operations)
Python Version: 3.9+

Project Structure
beatthemachine/
├── app.py
├── components/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── loss_functions.py
│   └── visualizations.py
├── config.py
├── requirements.txt
├── .gitignore
└── README.md

Core Features Specification
1. Data Generation Module - data_generator.py

Function: generate_dataset()

Parameters:

    n_points: Integer, default 30, number of data points
    noise_level: Float, default 0.3, standard deviation of Gaussian noise
    true_slope: Float, default random between 0.5 and 2
    true_intercept: Float, default random between 1 and 5
    x_range: Tuple, default (0, 10)
    seed: Optional integer for reproducibility

Returns a dictionary containing:

    x: numpy array of x coordinates
    y: numpy array of y coordinates
    true_slope: float representing actual slope used
    true_intercept: float representing actual intercept used
    optimal_rmse: float representing minimum achievable RMSE due to noise

Implementation Notes:

    Generate x values uniformly distributed in range 0 to 10
    Calculate y values using y = mx + b plus Gaussian noise
    Ensure generated points stay roughly within 0-10 range for x and y axis
    Store true parameters for calculating optimal RMSE

2. Loss Calculation Module - loss_functions.py

Function: calculate_rmse(x, y, slope, intercept)

    Takes x array, y array, and current parameter values
    Calculates Root Mean Square Error between actual y and predicted y values
    Returns single float value representing RMSE

3. Visualization Module - visualizations.py
Function: create_scatter_plot(data, current_slope, current_intercept)

Returns a Plotly figure object

Specifications:

    Plot dimensions: 600x500 pixels
    Fixed axis ranges: x from 0 to 10, y from 0 to 10
    Elements to display:
        Scatter points in blue, size 8
        Current regression line in red, width 2
        Grid enabled with light gray color

Layout requirements:

    Title: Linear Regression Fit
    X-axis label: X
    Y-axis label: Y
    Display current equation as annotation: y = slope x + intercept with 2 decimal places
    Show current RMSE value in corner as text annotation
    Hover info on points showing x, y coordinates rounded to 2 decimal places

Function: create_loss_plot(history)

Parameters:

    history: List of dictionaries, each containing epoch number, rmse value, slope value, and intercept value

Returns a Plotly figure object

Specifications:

    Plot dimensions: 600x500 pixels
    Elements:
        Line plot connecting RMSE values with color gradient
        Color transitions from red for highest RMSE to blue for lowest RMSE
        Scatter points at each attempt, size 8
        Points colored using same gradient as line

Color gradient implementation:

    Calculate color for each point based on RMSE value
    Use linear interpolation between red (255,0,0) and blue (0,0,255)
    Normalize based on min and max RMSE in history

Layout requirements:

    Title: Loss Evolution
    X-axis label: Attempt (Epoch)
    Y-axis label: RMSE (Loss)
    Grid enabled
    Y-axis starts at 0
    X-axis starts at 1
    Hover info showing: Attempt number, RMSE value, slope, intercept

4. Main Application - app.py
Session State Management

Initialize and maintain the following in session state:

    dataset: Current dataset dictionary from data generator
    history: List of attempt dictionaries
    attempt_count: Integer counter starting at 0
    best_rmse: Float tracking minimum RMSE achieved
    best_params: Dictionary with best slope and intercept values

Layout Structure

Page Configuration:

    Wide layout mode enabled
    Page title: Gradient Descent Intuition Builder
    Page icon: chart emoji

Sidebar Components in order:

    Title: Controls displayed as header
    Parameter Input Section:
        Slope slider:
            Range: -5.0 to 5.0
            Step: 0.01
            Default: 0.0
            Display format: 2 decimal places
            Label: Slope (m)
        Intercept slider:
            Range: -10.0 to 10.0
            Step: 0.01
            Default: 0.0
            Display format: 2 decimal places
            Label: Intercept (b)
    Action Buttons:
        Submit Attempt button (primary style)
        New Dataset button (secondary style)
    Statistics Display Section:
        Current RMSE displayed with 3 decimal places
        Best RMSE achieved with 3 decimal places
        Number of attempts as integer
        Improvement percentage from first attempt

Main Area Layout:

    Two column layout
    Left column: 60 percent width for scatter plot
    Right column: 40 percent width for loss plot

Interaction Flow

On Submit Attempt button click:

    Read current slider values for slope and intercept
    Calculate RMSE using current parameters
    Add entry to history with epoch, rmse, slope, intercept
    Increment attempt counter
    Update best RMSE and parameters if improved
    Refresh both plots with new data
    Display success message if RMSE below threshold

On New Dataset button click:

    Generate new dataset with random parameters
    Clear all history
    Reset attempt counter to 0
    Reset sliders to 0.0
    Reset best RMSE tracking
    Refresh both plots

User Feedback Features

Success Criteria:

    When RMSE is within 0.5 of optimal: Show success message
    When RMSE is best so far: Show improvement message
    Display messages using Streamlit success or info components

Information Displays:

    Info box at top explaining objective
    Current parameter values shown near sliders
    Best parameters shown in statistics section

5. Configuration - config.py

Default values section:

    DEFAULT_N_POINTS = 30
    DEFAULT_NOISE_LEVEL = 0.3
    DEFAULT_X_RANGE = (0, 10)
    DEFAULT_Y_RANGE = (0, 10)

UI settings section:

    PLOT_WIDTH = 600
    PLOT_HEIGHT = 500
    SLIDER_PRECISION = 0.01
    SLIDER_SLOPE_RANGE = (-5.0, 5.0)
    SLIDER_INTERCEPT_RANGE = (-10.0, 10.0)

Success thresholds section:

    SUCCESS_THRESHOLD = 0.5
    GOOD_THRESHOLD = 1.0
    ACCEPTABLE_THRESHOLD = 2.0

Color scheme section:

    DATA_POINTS_COLOR = blue hex value
    REGRESSION_LINE_COLOR = red hex value
    GRADIENT_START_COLOR = red RGB tuple
    GRADIENT_END_COLOR = blue RGB tuple

Deployment Instructions
Local Development

Install requirements using pip install -r requirements.txt then run streamlit run app.py from project directory
Streamlit Cloud Deployment

    Create GitHub repository with all project files
    Sign in to Streamlit Cloud with GitHub account
    Create new app pointing to repository
    Select app.py as main file
    Set Python version to 3.9 in advanced settings
    Deploy and share public URL

Performance Considerations

    Use st.cache_data decorator on dataset generation
    Limit history storage to last 100 attempts
    Use container placeholders for smooth plot updates
    Minimize redundant calculations

Testing Checklist

Functionality tests:

    Dataset generates within specified ranges
    RMSE calculation produces correct values
    Plots update without page refresh
    Session state persists between interactions
    New Dataset clears all appropriate state
    Submit Attempt adds to history correctly

Visual tests:

    Plots display at correct dimensions
    Color gradient transitions smoothly
    Axis ranges remain fixed
    Hover information displays correctly
    Layout responsive on different screen sizes

Educational value tests:

    Interface clearly shows relationship between parameters and fit
    Loss evolution demonstrates optimization concept
    Feedback helps guide parameter selection

Documentation Requirements
Code Documentation

All functions require docstrings with description, parameters, and return values. Type hints should be used for all function parameters and returns. Complex calculations need inline comments explaining logic.
User Documentation

README must include educational goals and learning objectives, step by step instructions for using the application, explanation of gradient descent concept in simple terms, and links to additional learning resources about machine learning and optimization.
Requirements File

The requirements.txt should list streamlit, plotly, and numpy with compatible version ranges that ensure stability while allowing minor updates.





