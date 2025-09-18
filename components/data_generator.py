"""
Data Generation Module for Gradient Descent Educational Game
Generates synthetic linear data with controlled noise for regression exercises.
"""

import numpy as np
from typing import Tuple, Optional, Dict




def generate_dataset(
    n_points: int = 30,
    noise_level: float = 0.3,
    true_slope: Optional[float] = None,
    true_intercept: Optional[float] = None,
    x_range: Tuple[float, float] = (0, 10),
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Generate a synthetic dataset for linear regression with controlled noise.
    
    Parameters
    ----------
    n_points : int, default=30
        Number of data points to generate
    noise_level : float, default=0.3
        Standard deviation of Gaussian noise to add to y values
    true_slope : float, optional
        True slope for the linear relationship. If None, randomly generated between 0.5 and 2
    true_intercept : float, optional
        True intercept for the linear relationship. If None, randomly generated between 1 and 5
    x_range : Tuple[float, float], default=(0, 10)
        Range for x values (min, max)
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing:
        - 'x': numpy array of x coordinates
        - 'y': numpy array of y coordinates  
        - 'true_slope': float, actual slope used
        - 'true_intercept': float, actual intercept used
        - 'optimal_rmse': float, minimum achievable RMSE due to noise
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Generate true parameters if not provided
    if true_slope is None:
        true_slope = np.random.uniform(0.5, 2.0)
    
    if true_intercept is None:
        true_intercept = np.random.uniform(1.0, 5.0)
    
    # Y range constraints
    y_min, y_max = 0, 10
    margin = 0.5  # Small margin for acceptable points slightly outside range
    
    # Method 1: Generate points and filter/regenerate those outside range
    max_attempts = 1000  # Prevent infinite loops
    attempts = 0
    
    x_list = []
    y_list = []
    
    while len(x_list) < n_points and attempts < max_attempts:
        attempts += 1
        
        # Generate a batch of points
        batch_size = n_points - len(x_list)
        x_batch = np.random.uniform(x_range[0], x_range[1], batch_size)
        
        # Calculate y values with noise
        y_true_batch = true_slope * x_batch + true_intercept
        noise_batch = np.random.normal(0, noise_level, batch_size)
        y_batch = y_true_batch + noise_batch
        
        # Keep only points within acceptable range
        valid_mask = (y_batch >= y_min - margin) & (y_batch <= y_max + margin)
        x_list.extend(x_batch[valid_mask])
        y_list.extend(y_batch[valid_mask])
    
    # If we couldn't generate enough valid points, adjust parameters
    if len(x_list) < n_points:
        # Fallback: adjust parameters to ensure points fit
        # Calculate maximum feasible slope to keep points in range
        max_y_from_line = true_slope * x_range[1] + true_intercept
        min_y_from_line = true_slope * x_range[0] + true_intercept
        
        # If line goes too high, scale it down
        if max_y_from_line > y_max - 2*noise_level:
            scale_factor = (y_max - 2*noise_level - true_intercept) / (x_range[1] * true_slope)
            true_slope *= min(scale_factor, 1.0)
        
        # Regenerate with adjusted parameters
        x = np.random.uniform(x_range[0], x_range[1], n_points)
        y_true = true_slope * x + true_intercept
        noise = np.random.normal(0, noise_level, n_points)
        y = y_true + noise
        
        # Final safety: remove extreme outliers
        valid_mask = (y >= y_min - margin) & (y <= y_max + margin)
        x = x[valid_mask]
        y = y[valid_mask]
        
        # If still not enough points, generate additional ones in safe zone
        if len(x) < n_points:
            n_additional = n_points - len(x)
            # Generate points in middle of x range where they're more likely to be valid
            x_safe = np.random.uniform(x_range[0] + 2, x_range[1] - 2, n_additional)
            y_safe = true_slope * x_safe + true_intercept + np.random.normal(0, noise_level, n_additional)
            x = np.concatenate([x, x_safe])
            y = np.concatenate([y, y_safe])
    else:
        # Use the successfully generated points
        x = np.array(x_list[:n_points])
        y = np.array(y_list[:n_points])
    
    # Ensure we have exactly n_points
    if len(x) > n_points:
        x = x[:n_points]
        y = y[:n_points]
    
    # Calculate optimal RMSE (this is approximately the noise level)
    # The optimal RMSE is what we'd get if we knew the true parameters
    optimal_rmse = noise_level
    
    # Create and return the dataset dictionary
    dataset = {
        'x': x,
        'y': y,
        'true_slope': true_slope,
        'true_intercept': true_intercept,
        'optimal_rmse': optimal_rmse
    }
    
    return dataset


def generate_dataset_with_info(
    n_points: int = 30,
    noise_level: float = 0.3,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Convenience function that generates a dataset and prints information about it.
    Useful for debugging and understanding the generated data.
    
    Parameters
    ----------
    n_points : int, default=30
        Number of data points to generate
    noise_level : float, default=0.3
        Standard deviation of Gaussian noise
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    Dict[str, np.ndarray]
        Same as generate_dataset()
    """
    dataset = generate_dataset(n_points=n_points, noise_level=noise_level, seed=seed)
    
    print(f"Generated dataset with {n_points} points")
    print(f"True equation: y = {dataset['true_slope']:.3f}x + {dataset['true_intercept']:.3f}")
    print(f"X range: [{dataset['x'].min():.2f}, {dataset['x'].max():.2f}]")
    print(f"Y range: [{dataset['y'].min():.2f}, {dataset['y'].max():.2f}]")
    print(f"Optimal RMSE (due to noise): {dataset['optimal_rmse']:.3f}")
    
    return dataset


# Example usage and testing
if __name__ == "__main__":
    # Test the function with different parameters
    print("Test 1: Default parameters")
    data1 = generate_dataset_with_info()
    
    print("\nTest 2: More points, higher noise")
    data2 = generate_dataset_with_info(n_points=50, noise_level=0.5)
    
    print("\nTest 3: With seed for reproducibility")
    data3 = generate_dataset_with_info(seed=42)
    data4 = generate_dataset_with_info(seed=42)
    print(f"Same seed produces same data: {np.allclose(data3['x'], data4['x'])}")
    
    print("\nTest 4: Custom slope and intercept")
    data5 = generate_dataset(true_slope=1.5, true_intercept=2.0, seed=123)
    print(f"Custom equation: y = {data5['true_slope']}x + {data5['true_intercept']}")