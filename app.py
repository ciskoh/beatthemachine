"""
Gradient Descent Educational Game
Main Streamlit Application

This interactive application helps students understand gradient descent
by manually adjusting linear regression parameters and observing the loss evolution.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional

# Import custom modules
from components.data_generator import generate_dataset
from components.loss_functions import calculate_rmse
from components.visualizations import (
    create_scatter_plot,
    create_loss_plot,
    create_loss_landscape_3d
)

# Import configuration
import config


# Page configuration
st.set_page_config(
    page_title="Gradient Descent Intuition Builder",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Initialize session state
def init_session_state():
    """Initialize all session state variables."""
    
    if 'dataset' not in st.session_state:
        # Generate initial dataset using config values
        st.session_state.dataset = generate_dataset(
            n_points=config.DEFAULT_N_POINTS,
            noise_level=config.DEFAULT_NOISE_LEVEL,
            seed=config.DEMO_SEED
        )
    
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    if 'attempt_count' not in st.session_state:
        st.session_state.attempt_count = 0
    
    if 'best_rmse' not in st.session_state:
        st.session_state.best_rmse = float('inf')
    
    if 'best_params' not in st.session_state:
        st.session_state.best_params = {
            'slope': config.DEFAULT_SLOPE, 
            'intercept': config.DEFAULT_INTERCEPT
        }
    
    if 'current_slope' not in st.session_state:
        st.session_state.current_slope = config.DEFAULT_SLOPE
    
    if 'current_intercept' not in st.session_state:
        st.session_state.current_intercept = config.DEFAULT_INTERCEPT
    
    if 'show_line' not in st.session_state:
        st.session_state.show_line = False
    
    if 'last_submitted_slope' not in st.session_state:
        st.session_state.last_submitted_slope = None
    
    if 'last_submitted_intercept' not in st.session_state:
        st.session_state.last_submitted_intercept = None


def reset_history():
    """Reset the attempt history but keep the current dataset."""
    st.session_state.history = []
    st.session_state.attempt_count = 0
    st.session_state.best_rmse = float('inf')
    st.session_state.best_params = {
        'slope': config.DEFAULT_SLOPE, 
        'intercept': config.DEFAULT_INTERCEPT
    }
    st.session_state.show_line = False
    st.session_state.last_submitted_slope = None
    st.session_state.last_submitted_intercept = None


def new_dataset():
    """Generate a new dataset and reset everything."""
    st.session_state.dataset = generate_dataset(
        n_points=config.DEFAULT_N_POINTS,
        noise_level=config.DEFAULT_NOISE_LEVEL,
        seed=None  # Random seed for variety
    )
    reset_history()
    st.session_state.current_slope = config.DEFAULT_SLOPE
    st.session_state.current_intercept = config.DEFAULT_INTERCEPT


def submit_attempt(slope: float, intercept: float):
    """Process a new attempt with given parameters."""
    
    # Calculate RMSE for current parameters
    rmse = calculate_rmse(
        st.session_state.dataset['x'],
        st.session_state.dataset['y'],
        slope,
        intercept
    )
    
    # Update attempt count
    st.session_state.attempt_count += 1
    
    # Add to history (limit to max history length)
    st.session_state.history.append({
        'epoch': st.session_state.attempt_count,
        'rmse': rmse,
        'slope': slope,
        'intercept': intercept
    })
    
    # Limit history length
    if len(st.session_state.history) > config.MAX_HISTORY_LENGTH:
        st.session_state.history = st.session_state.history[-config.MAX_HISTORY_LENGTH:]
    
    # Update best if improved
    improved = False
    if rmse < st.session_state.best_rmse:
        st.session_state.best_rmse = rmse
        st.session_state.best_params = {
            'slope': slope,
            'intercept': intercept
        }
        improved = True
    
    # Store the submitted parameters and show the line
    st.session_state.last_submitted_slope = slope
    st.session_state.last_submitted_intercept = intercept
    st.session_state.show_line = True
    
    return improved


def calculate_success_level(rmse: float, optimal_rmse: float) -> str:
    """Determine success level based on RMSE."""
    gap = rmse - optimal_rmse
    
    if gap <= config.SUCCESS_THRESHOLD_EXCELLENT:
        return "excellent"
    elif gap <= config.SUCCESS_THRESHOLD_GOOD:
        return "good"
    elif gap <= config.SUCCESS_THRESHOLD_ACCEPTABLE:
        return "acceptable"
    else:
        return "keep_trying"


# Main app
def main():
    # Initialize session state
    init_session_state()
    
    # Header
    st.title(" Gradient Descent Intuition Builder")
    st.markdown("""
    Welcome to the Gradient Descent Intuition Builder! This interactive tool allows you to manually adjust
    the parameters of a linear regression model and see how it affects the fit to the data.
    """)
    with st.expander(" How to Play", expanded=False):
        st.info(config.TUTORIAL_CONTENT['objective'] + 
                config.TUTORIAL_CONTENT['instructions'] + 
                config.TUTORIAL_CONTENT['tips'])
    
    # Sidebar controls
    with st.sidebar:
        st.header(" Controls")
        
        
        # Parameter inputs
        st.subheader(" Parameters")
        
        current_slope = st.slider(
            "Slope (m)",
            min_value=config.SLIDER_SLOPE_RANGE[0],
            max_value=config.SLIDER_SLOPE_RANGE[1],
            value=st.session_state.current_slope,
            step=config.SLIDER_PRECISION,
            format="%.2f",
            help="Slope of the line (rise over run)"
        )
        
        current_intercept = st.slider(
            "Intercept (b)",
            min_value=config.SLIDER_INTERCEPT_RANGE[0],
            max_value=config.SLIDER_INTERCEPT_RANGE[1],
            value=st.session_state.current_intercept,
            step=config.SLIDER_PRECISION,
            format="%.2f",
            help="Y-value where the line crosses the y-axis"
        )
        
        # Update session state
        st.session_state.current_slope = current_slope
        st.session_state.current_intercept = current_intercept
        
        # Check if sliders have moved from last submitted values
        if (st.session_state.last_submitted_slope is not None and 
            st.session_state.last_submitted_intercept is not None):
            if (current_slope != st.session_state.last_submitted_slope or 
                current_intercept != st.session_state.last_submitted_intercept):
                # Sliders moved - hide the line
                st.session_state.show_line = False
        
        
        # Action buttons
        st.subheader("Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Submit Attempt", type="primary", use_container_width=True):
                improved = submit_attempt(current_slope, current_intercept)
                
                # Calculate current RMSE for feedback
                current_rmse = st.session_state.history[-1]['rmse']
                optimal_rmse = st.session_state.dataset['optimal_rmse']
                
                # Show feedback
                if improved:
                    st.success(config.MESSAGES['new_best'].format(rmse=current_rmse))
                
                success_level = calculate_success_level(current_rmse, optimal_rmse)
                
                if success_level == "excellent":
                    if config.SHOW_BALLOONS_ON_SUCCESS:
                        st.balloons()
                    st.success(config.MESSAGES['excellent'])
                elif success_level == "good":
                    st.success(config.MESSAGES['good'])
                elif success_level == "acceptable":
                    st.info(config.MESSAGES['acceptable'])
        
        with col2:
            if st.button("New Dataset", type="secondary", use_container_width=True):
                new_dataset()
                st.rerun()
        
        
        # Statistics display
        st.subheader("Statistics")
        
        if st.session_state.history:
            current_rmse = st.session_state.history[-1]['rmse']
            
            # Display metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Current RMSE",
                    f"{current_rmse:.3f}",
                    delta=None if len(st.session_state.history) < 2 else 
                    f"{current_rmse - st.session_state.history[-2]['rmse']:.3f}"
                )
                
                st.metric(
                    "Attempts",
                    st.session_state.attempt_count
                )
            
            with col2:
                st.metric(
                    "Best RMSE",
                    f"{st.session_state.best_rmse:.3f}"
                )
                
                # Calculate improvement percentage
                if len(st.session_state.history) > 1 and config.SHOW_IMPROVEMENT_METRICS:
                    first_rmse = st.session_state.history[0]['rmse']
                    if first_rmse > 0:
                        improvement = ((first_rmse - current_rmse) / first_rmse) * 100
                        st.metric(
                            "Improvement",
                            f"{improvement:.1f}%"
                        )
            
            # Show optimal RMSE info
            optimal_rmse = st.session_state.dataset['optimal_rmse']
            gap_to_optimal = st.session_state.best_rmse - optimal_rmse
            
            st.info(f"""
            **Target Info:**
            - Optimal RMSE: {optimal_rmse:.3f}
            - Gap to optimal: {gap_to_optimal:.3f}
            """)
        else:
            st.info(config.MESSAGES['no_attempts'])
    
    # Main area - three sections: left plot, right side with 2D loss on top and 3D below
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader(" Linear Regression Fit")
        
        # Determine what to show based on state
        if st.session_state.show_line and st.session_state.history:
            # Show line with the last submitted parameters
            current_rmse = st.session_state.history[-1]['rmse']
            fig_scatter = create_scatter_plot(
                st.session_state.dataset,
                st.session_state.last_submitted_slope,
                st.session_state.last_submitted_intercept,
                current_rmse
            )
        else:
            # No line - just show points
            fig_scatter = create_scatter_plot(
                st.session_state.dataset,
                None,
                None,
                None
            )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # 2D loss plot on top
        st.subheader("Loss Evolution")
        
        # Create loss plot
        fig_loss = create_loss_plot(st.session_state.history)
        
        # Adjust height for stacking
        fig_loss.update_layout(height=350)
        
        st.plotly_chart(fig_loss, use_container_width=True)
        
        # 3D plot below
        st.subheader("Parameter space")
        
        if st.session_state.history:
            # Extract data for 3D plot
            slopes = [h['slope'] for h in st.session_state.history]
            intercepts = [h['intercept'] for h in st.session_state.history]
            scores = [h['rmse'] for h in st.session_state.history]
            
            # Create 3D plot with smaller height for stacking
            fig_3d = create_loss_landscape_3d(
                slopes=slopes,
                intercepts=intercepts,
                scores=scores
            )
            
            # Adjust height for stacking
            fig_3d.update_layout(height=350)
            
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            # Show empty 3D plot with message
            st.info("Make your first attempt to see the loss landscape!")


if __name__ == "__main__":
    main()