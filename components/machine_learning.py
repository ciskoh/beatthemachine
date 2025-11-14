import numpy as np
from sklearn.linear_model import SGDRegressor
import time
from config import MACHINE_TRAINING_DELAY

def train_machine(X, y):
    """
    Train a machine learning model using SGDRegressor for 10 epochs
    
    Parameters:
    X: numpy array of x values
    y: numpy array of y values  
    
    Returns:
    List of dictionaries containing epoch history
    """
    # Initialize machine history
    machine_history = []
    
    # Create and configure SGDRegressor
    model = SGDRegressor(
        learning_rate='constant',
        eta0=0.0002,
        max_iter=1,
        tol=None,  # Don't stop early
        warm_start=True,
        fit_intercept=True,
        shuffle=True,
        random_state=42,
        penalty='l2',
        alpha=0.01
    )
    
    # Set initial parameters to (0, 0) like humans start
    model.coef_ = np.array([0.0])
    model.intercept_ = np.array([0.0])
    
    # Reshape X for sklearn
    X_reshaped = X.reshape(-1, 1)
    
    # Train for 10 epochs
    for epoch in range(10):
        # One epoch of training
        model.partial_fit(X_reshaped, y)
        
        # Get current parameters
        slope = model.coef_[0]
        intercept = model.intercept_[0]
        
        # Calculate predictions and RMSE
        predictions = slope * X + intercept
        rmse = np.sqrt(np.mean((y - predictions) ** 2))
        
        # Store in history
        machine_history.append({
            'epoch': epoch + 1,
            'rmse': rmse,
            'slope': slope,
            'intercept': intercept
        })
        
        # Pause for visualization
        time.sleep(MACHINE_TRAINING_DELAY)
    
    return machine_history