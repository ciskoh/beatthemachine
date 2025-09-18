"""
Loss Calculation Module for Gradient Descent Educational Game
Provides functions to calculate loss metrics for linear regression.
"""

import numpy as np
from typing import Union


def calculate_rmse(
    x: np.ndarray, 
    y: np.ndarray, 
    slope: float, 
    intercept: float
) -> float:
    """
    Calculate Root Mean Square Error between actual and predicted y values.
    
    RMSE measures the average magnitude of prediction errors, giving a sense
    of how far off the predictions are from actual values. Lower RMSE indicates
    better fit.
    
    Parameters
    ----------
    x : np.ndarray
        Array of x coordinates (independent variable)
    y : np.ndarray
        Array of actual y coordinates (dependent variable)
    slope : float
        Current slope parameter (m in y = mx + b)
    intercept : float
        Current intercept parameter (b in y = mx + b)
    
    Returns
    -------
    float
        Root Mean Square Error value
    
    Notes
    -----
    RMSE formula: sqrt(mean((y_actual - y_predicted)^2))
    """
    # Calculate predicted y values using the linear equation
    y_predicted = slope * x + intercept
    
    # Calculate squared differences
    squared_errors = (y - y_predicted) ** 2
    
    # Calculate mean of squared errors
    mean_squared_error = np.mean(squared_errors)
    
    # Return the square root (RMSE)
    rmse = np.sqrt(mean_squared_error)
    
    return float(rmse)


def calculate_mse(
    x: np.ndarray, 
    y: np.ndarray, 
    slope: float, 
    intercept: float
) -> float:
    """
    Calculate Mean Square Error between actual and predicted y values.
    
    MSE is the average of squared differences between predicted and actual values.
    It's the square of RMSE and is sometimes used instead of RMSE for optimization
    because it's computationally simpler (no square root).
    
    Parameters
    ----------
    x : np.ndarray
        Array of x coordinates (independent variable)
    y : np.ndarray
        Array of actual y coordinates (dependent variable)
    slope : float
        Current slope parameter
    intercept : float
        Current intercept parameter
    
    Returns
    -------
    float
        Mean Square Error value
    """
    # Calculate predicted y values
    y_predicted = slope * x + intercept
    
    # Calculate squared differences
    squared_errors = (y - y_predicted) ** 2
    
    # Return mean of squared errors
    mse = np.mean(squared_errors)
    
    return float(mse)


def calculate_mae(
    x: np.ndarray, 
    y: np.ndarray, 
    slope: float, 
    intercept: float
) -> float:
    """
    Calculate Mean Absolute Error between actual and predicted y values.
    
    MAE is an alternative to RMSE that uses absolute differences instead of
    squared differences. It's less sensitive to outliers than RMSE.
    
    Parameters
    ----------
    x : np.ndarray
        Array of x coordinates (independent variable)
    y : np.ndarray
        Array of actual y coordinates (dependent variable)
    slope : float
        Current slope parameter
    intercept : float
        Current intercept parameter
    
    Returns
    -------
    float
        Mean Absolute Error value
    """
    # Calculate predicted y values
    y_predicted = slope * x + intercept
    
    # Calculate absolute differences
    absolute_errors = np.abs(y - y_predicted)
    
    # Return mean of absolute errors
    mae = np.mean(absolute_errors)
    
    return float(mae)


def calculate_r_squared(
    x: np.ndarray, 
    y: np.ndarray, 
    slope: float, 
    intercept: float
) -> float:
    """
    Calculate R-squared (coefficient of determination) for the linear fit.
    
    R-squared represents the proportion of variance in the dependent variable
    that's predictable from the independent variable. Values range from 0 to 1,
    where 1 indicates perfect fit.
    
    Parameters
    ----------
    x : np.ndarray
        Array of x coordinates (independent variable)
    y : np.ndarray
        Array of actual y coordinates (dependent variable)
    slope : float
        Current slope parameter
    intercept : float
        Current intercept parameter
    
    Returns
    -------
    float
        R-squared value (between 0 and 1 for good fits, can be negative for poor fits)
    """
    # Calculate predicted y values
    y_predicted = slope * x + intercept
    
    # Calculate total sum of squares (variance of y)
    y_mean = np.mean(y)
    total_sum_squares = np.sum((y - y_mean) ** 2)
    
    # Calculate residual sum of squares
    residual_sum_squares = np.sum((y - y_predicted) ** 2)
    
    # Calculate R-squared
    # Handle edge case where total_sum_squares is 0 (all y values are the same)
    if total_sum_squares == 0:
        return 1.0 if residual_sum_squares == 0 else 0.0
    
    r_squared = 1 - (residual_sum_squares / total_sum_squares)
    
    return float(r_squared)


def get_prediction_line(
    x_range: tuple, 
    slope: float, 
    intercept: float, 
    n_points: int = 100
) -> tuple:
    """
    Generate points for plotting the regression line.
    
    Parameters
    ----------
    x_range : tuple
        (min_x, max_x) range for the line
    slope : float
        Slope of the line
    intercept : float
        Intercept of the line
    n_points : int, default=100
        Number of points to generate for smooth line plotting
    
    Returns
    -------
    tuple
        (x_line, y_line) arrays for plotting
    """
    x_line = np.linspace(x_range[0], x_range[1], n_points)
    y_line = slope * x_line + intercept
    
    return x_line, y_line


# Example usage and testing
if __name__ == "__main__":
    # Create some test data
    np.random.seed(42)
    n_points = 30
    
    # Generate synthetic data with known parameters
    true_slope = 1.5
    true_intercept = 2.0
    x_test = np.random.uniform(0, 10, n_points)
    y_test = true_slope * x_test + true_intercept + np.random.normal(0, 0.3, n_points)
    
    print("Testing Loss Functions")
    print("=" * 50)
    print(f"True parameters: slope={true_slope}, intercept={true_intercept}")
    
    # Test with true parameters (should give low error)
    print("\nWith true parameters:")
    rmse_true = calculate_rmse(x_test, y_test, true_slope, true_intercept)
    mse_true = calculate_mse(x_test, y_test, true_slope, true_intercept)
    mae_true = calculate_mae(x_test, y_test, true_slope, true_intercept)
    r2_true = calculate_r_squared(x_test, y_test, true_slope, true_intercept)
    
    print(f"  RMSE: {rmse_true:.4f}")
    print(f"  MSE:  {mse_true:.4f}")
    print(f"  MAE:  {mae_true:.4f}")
    print(f"  R²:   {r2_true:.4f}")
    
    # Test with wrong parameters (should give higher error)
    wrong_slope = 0.5
    wrong_intercept = 5.0
    
    print(f"\nWith wrong parameters: slope={wrong_slope}, intercept={wrong_intercept}")
    rmse_wrong = calculate_rmse(x_test, y_test, wrong_slope, wrong_intercept)
    mse_wrong = calculate_mse(x_test, y_test, wrong_slope, wrong_intercept)
    mae_wrong = calculate_mae(x_test, y_test, wrong_slope, wrong_intercept)
    r2_wrong = calculate_r_squared(x_test, y_test, wrong_slope, wrong_intercept)
    
    print(f"  RMSE: {rmse_wrong:.4f}")
    print(f"  MSE:  {mse_wrong:.4f}")
    print(f"  MAE:  {mae_wrong:.4f}")
    print(f"  R²:   {r2_wrong:.4f}")
    
    # Verify RMSE calculation
    print("\nVerification:")
    print(f"  RMSE² should equal MSE: {rmse_wrong**2:.4f} ≈ {mse_wrong:.4f}")
    print(f"  Error increased by factor of: {rmse_wrong/rmse_true:.2f}")