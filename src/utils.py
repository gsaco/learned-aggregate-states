"""
Utility Functions for Learned Aggregate States Project

This module provides data processing, feature extraction, and visualization
utilities for the representation learning framework.

Features from the cross-sectional distribution:
    X_t = Φ(μ_t) includes:
    - Histogram-based encoding over asset grid
    - Moments (mean, std, skewness, kurtosis)
    - Inequality measures (Gini, top shares)
    - Mass at borrowing constraint
    - Productivity-related statistics

Author: Research Team
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Any
import matplotlib.pyplot as plt
from numba import njit, prange


# =============================================================================
# Vectorized Helper Functions (Performance Critical)
# =============================================================================

@njit(parallel=True)
def _compute_consumption_vectorized(
    a_t: np.ndarray,
    e_t: np.ndarray,
    c_policy: np.ndarray,
    a_grid: np.ndarray,
    e_grid: np.ndarray
) -> np.ndarray:
    """
    Vectorized consumption interpolation using numba.
    
    This is the performance-critical function that was causing slowdowns.
    Uses parallel loops for N agents instead of Python for-loop.
    """
    N = len(a_t)
    n_a = len(a_grid)
    n_e = len(e_grid)
    c_agents = np.zeros(N)
    
    for i in prange(N):
        a = a_t[i]
        e = e_t[i]
        
        # Find productivity index (closest match)
        ie = 0
        min_dist = np.abs(e_grid[0] - e)
        for j in range(1, n_e):
            dist = np.abs(e_grid[j] - e)
            if dist < min_dist:
                min_dist = dist
                ie = j
        
        # Binary search for asset position
        idx = np.searchsorted(a_grid, a)
        idx = max(1, min(idx, n_a - 1))
        
        # Linear interpolation
        denom = a_grid[idx] - a_grid[idx-1]
        if denom > 0:
            weight = (a - a_grid[idx-1]) / denom
        else:
            weight = 0.0
        
        c_agents[i] = (1.0 - weight) * c_policy[idx-1, ie] + weight * c_policy[idx, ie]
    
    return c_agents


def compute_aggregates_fast(
    a_t: np.ndarray,
    e_t: np.ndarray,
    c_policy: np.ndarray,
    a_grid: np.ndarray,
    e_grid: np.ndarray,
    alpha: float
) -> Tuple[float, float, float, float]:
    """
    Fast vectorized computation of aggregate variables.
    
    Uses numba-accelerated consumption interpolation.
    """
    K_t = np.mean(a_t)
    L_t = np.mean(e_t)
    
    # Use vectorized consumption computation
    c_agents = _compute_consumption_vectorized(a_t, e_t, c_policy, a_grid, e_grid)
    C_t = np.mean(c_agents)
    
    Y_t = K_t**alpha * L_t**(1 - alpha)
    
    return K_t, L_t, C_t, Y_t


def compute_features_batch(
    a_panel: np.ndarray,
    e_panel: np.ndarray,
    a_grid_bins: np.ndarray,
    n_bins: int = 50
) -> np.ndarray:
    """
    Compute features for all time periods at once (batch processing).
    
    Much faster than calling compute_features in a loop.
    
    Parameters
    ----------
    a_panel : np.ndarray
        Asset panel (N x T)
    e_panel : np.ndarray
        Productivity panel (N x T)
    a_grid_bins : np.ndarray
        Bin edges
    n_bins : int
        Number of bins
        
    Returns
    -------
    X : np.ndarray
        Feature matrix (T x p)
    """
    N, T = a_panel.shape
    bin_width = a_grid_bins[1] - a_grid_bins[0]
    p = n_bins + 10  # histogram + 10 summary stats
    X = np.zeros((T, p))
    
    for t in range(T):
        a_t = a_panel[:, t]
        e_t = e_panel[:, t]
        
        # 1. Histogram (use density=False to avoid NaN with empty bins)
        hist, _ = np.histogram(a_t, bins=a_grid_bins, density=False)
        hist = hist.astype(np.float64) / N  # Normalize to probability
        X[t, :n_bins] = hist
        
        # 2. Moments
        mean_a = np.mean(a_t)
        std_a = np.std(a_t)
        if std_a > 0:
            z = (a_t - mean_a) / std_a
            skew_a = np.mean(z**3)
            kurt_a = np.mean(z**4)
        else:
            skew_a, kurt_a = 0.0, 0.0
        
        # 3. Inequality (use partial sort for efficiency)
        # NOTE: Standard Gini requires non-negative values. When borrowing is allowed
        # (a_min < 0), we shift the distribution to compute a "relative Gini" which
        # measures inequality of the shifted distribution. This is a standard approach
        # in the literature for handling negative wealth.
        sorted_a = np.sort(a_t)
        
        # Shift to handle negative values (if any)
        min_a = sorted_a[0]
        if min_a < 0:
            shifted_a = sorted_a - min_a  # Shift so minimum is 0
        else:
            shifted_a = sorted_a
        
        cumsum = np.cumsum(shifted_a)
        total = cumsum[-1] if cumsum[-1] > 0 else 1.0
        
        # Gini coefficient using shifted values
        if total > 0:
            gini = 1 - 2 * np.sum(cumsum) / (N * total) + 1/N
            # Ensure Gini is in valid range [0, 1]
            gini = np.clip(gini, 0.0, 1.0)
        else:
            gini = 0.0
        
        # For wealth shares, use original values (handle negative total separately)
        total_wealth = np.sum(sorted_a)
        abs_total = np.abs(total_wealth) if total_wealth != 0 else 1.0
        
        # Compute shares using absolute total as reference
        top10_share = np.sum(sorted_a[int(0.9*N):]) / abs_total if abs_total > 0 else 0.0
        top1_share = np.sum(sorted_a[int(0.99*N):]) / abs_total if abs_total > 0 else 0.0
        bottom50_share = np.sum(sorted_a[:int(0.5*N)]) / abs_total if abs_total > 0 else 0.0
        
        # 4. Constraint mass
        mass_at_constraint = np.mean(a_t <= a_grid_bins[1])
        
        # 5. Labor
        mean_e = np.mean(e_t)
        
        # Store summary stats
        X[t, n_bins:] = [mean_a, std_a, skew_a, kurt_a, gini, 
                         top10_share, top1_share, bottom50_share,
                         mass_at_constraint, mean_e]
    
    return X


def compute_aggregates_batch(
    a_panel: np.ndarray,
    e_panel: np.ndarray,
    c_policy: np.ndarray,
    a_grid: np.ndarray,
    e_grid: np.ndarray,
    alpha: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute aggregates for all time periods at once.
    
    Parameters
    ----------
    a_panel : np.ndarray (N x T)
    e_panel : np.ndarray (N x T)
    c_policy : np.ndarray (n_a x n_e)
    a_grid : np.ndarray
    e_grid : np.ndarray
    alpha : float
    
    Returns
    -------
    K_series, L_series, C_series, Y_series : np.ndarray (T,)
    """
    N, T = a_panel.shape
    
    K_series = np.mean(a_panel, axis=0)
    L_series = np.mean(e_panel, axis=0)
    Y_series = K_series**alpha * L_series**(1 - alpha)
    
    C_series = np.zeros(T)
    for t in range(T):
        c_agents = _compute_consumption_vectorized(
            a_panel[:, t], e_panel[:, t], c_policy, a_grid, e_grid
        )
        C_series[t] = np.mean(c_agents)
    
    return K_series, L_series, C_series, Y_series


# =============================================================================
# Feature Extraction from Cross-Sectional Distribution
# =============================================================================

def compute_features(
    a_t: np.ndarray,
    e_t: np.ndarray,
    a_grid_bins: np.ndarray,
    n_bins: int = 50
) -> np.ndarray:
    """
    Compute feature vector X_t from cross-sectional distribution.
    
    Implements the feature map Φ: M → R^p from the proposal.
    
    Features:
        - Histogram over asset grid (n_bins dimensions)
        - Summary statistics: mean, std, skewness, kurtosis
        - Inequality measures: Gini, top 10%, top 1%, bottom 50%
        - Mass at/near borrowing constraint
        - Mean productivity (labor)
    
    Parameters
    ----------
    a_t : np.ndarray
        Asset holdings for all agents at time t
    e_t : np.ndarray
        Productivity levels for all agents at time t
    a_grid_bins : np.ndarray
        Bin edges for histogram
    n_bins : int
        Number of histogram bins
        
    Returns
    -------
    X_t : np.ndarray
        Feature vector (p = n_bins + 10 dimensions)
    """
    N = len(a_t)
    
    # 1. Histogram features (discretized distribution)
    hist, _ = np.histogram(a_t, bins=a_grid_bins, density=True)
    hist = hist * (a_grid_bins[1] - a_grid_bins[0])  # Normalize to sum ≈ 1
    
    # 2. Moments of asset distribution
    mean_a = np.mean(a_t)
    std_a = np.std(a_t)
    if std_a > 0:
        skew_a = np.mean(((a_t - mean_a) / std_a)**3)
        kurt_a = np.mean(((a_t - mean_a) / std_a)**4)
    else:
        skew_a, kurt_a = 0, 0
    
    # 3. Inequality measures
    # Handle negative wealth when borrowing is allowed
    sorted_a = np.sort(a_t)
    
    # Shift to handle negative values (if any)
    min_a = sorted_a[0]
    if min_a < 0:
        shifted_a = sorted_a - min_a  # Shift so minimum is 0
    else:
        shifted_a = sorted_a
    
    cumsum = np.cumsum(shifted_a)
    
    # Gini coefficient using shifted values
    if cumsum[-1] > 0:
        gini = 1 - 2 * np.sum(cumsum) / (N * cumsum[-1]) + 1/N
        gini = np.clip(gini, 0.0, 1.0)  # Ensure valid range
    else:
        gini = 0
    
    # Top/bottom shares using original values
    total_wealth = np.sum(a_t)
    abs_total = np.abs(total_wealth) if total_wealth != 0 else 1.0
    
    if abs_total > 0:
        top10_share = np.sum(sorted_a[int(0.9*N):]) / abs_total
        top1_share = np.sum(sorted_a[int(0.99*N):]) / abs_total
        bottom50_share = np.sum(sorted_a[:int(0.5*N)]) / abs_total
    else:
        top10_share, top1_share, bottom50_share = 0, 0, 0
    
    # 4. Mass at borrowing constraint (near zero)
    mass_at_constraint = np.mean(a_t <= a_grid_bins[1])
    
    # 5. Productivity-related (mean labor supply)
    mean_e = np.mean(e_t)
    
    # Combine all features
    summary_stats = np.array([
        mean_a, std_a, skew_a, kurt_a,
        gini, top10_share, top1_share, bottom50_share,
        mass_at_constraint, mean_e
    ])
    
    X_t = np.concatenate([hist, summary_stats])
    
    return X_t


def compute_aggregates(
    a_t: np.ndarray,
    e_t: np.ndarray,
    c_policy: np.ndarray,
    a_policy: np.ndarray,
    a_grid: np.ndarray,
    e_grid: np.ndarray,
    alpha: float
) -> Tuple[float, float, float, float]:
    """
    Compute aggregate variables from cross-section.
    
    Parameters
    ----------
    a_t : np.ndarray
        Asset holdings for all agents
    e_t : np.ndarray
        Productivity for all agents
    c_policy : np.ndarray
        Consumption policy function
    a_policy : np.ndarray
        Asset policy function
    a_grid : np.ndarray
        Asset grid
    e_grid : np.ndarray
        Productivity grid
    alpha : float
        Capital share
        
    Returns
    -------
    K_t : float
        Aggregate capital
    L_t : float
        Aggregate labor
    C_t : float
        Aggregate consumption
    Y_t : float
        Aggregate output
    """
    N = len(a_t)
    
    # Aggregate capital = mean assets
    K_t = np.mean(a_t)
    
    # Aggregate labor
    L_t = np.mean(e_t)
    
    # Compute consumption for each agent (interpolate policy)
    c_agents = np.zeros(N)
    for i in range(N):
        a = a_t[i]
        e = e_t[i]
        # Find productivity index
        ie = np.argmin(np.abs(e_grid - e))
        # Interpolate consumption
        idx = np.searchsorted(a_grid, a)
        idx = max(1, min(idx, len(a_grid) - 1))
        weight = (a - a_grid[idx-1]) / (a_grid[idx] - a_grid[idx-1])
        c_agents[i] = (1 - weight) * c_policy[idx-1, ie] + weight * c_policy[idx, ie]
    
    C_t = np.mean(c_agents)
    
    # Output (Cobb-Douglas)
    Y_t = K_t**alpha * L_t**(1 - alpha)
    
    return K_t, L_t, C_t, Y_t


def build_dataset(
    a_panel: np.ndarray,
    e_panel: np.ndarray,
    c_policy: np.ndarray,
    a_policy: np.ndarray,
    a_grid: np.ndarray,
    e_grid: np.ndarray,
    alpha: float,
    n_bins: int = 50,
    a_max: float = 50.0,
    Z_series: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Build complete dataset from simulated panel.
    
    Parameters
    ----------
    a_panel : np.ndarray
        Simulated asset panel (N x T)
    e_panel : np.ndarray
        Simulated productivity panel (N x T)
    c_policy : np.ndarray
        Consumption policy
    a_policy : np.ndarray
        Asset policy
    a_grid : np.ndarray
        Asset grid
    e_grid : np.ndarray
        Productivity grid
    alpha : float
        Capital share
    n_bins : int
        Number of histogram bins
    a_max : float
        Maximum assets for binning
    Z_series : np.ndarray, optional
        Aggregate TFP shocks (T,). If provided, these become the s_{t+1} shocks
        and output Y is multiplied by Z_t.
    verbose : bool
        Print progress
        
    Returns
    -------
    dataset : dict
        Dictionary with X, s, Y arrays
    """
    T_data = a_panel.shape[1]
    a_grid_bins = np.linspace(0, a_max, n_bins + 1)
    
    # Compute features for all periods at once (FAST batch processing)
    if verbose:
        print("Computing features X_t (batch)...")
    X = compute_features_batch(a_panel, e_panel, a_grid_bins, n_bins)
    
    # Compute aggregates for all periods at once (FAST vectorized)
    if verbose:
        print("Computing aggregates Y_t (batch)...")
    K_series, L_series, C_series, Y_series = compute_aggregates_batch(
        a_panel, e_panel, c_policy, a_grid, e_grid, alpha
    )
    
    # Handle aggregate TFP shocks
    if Z_series is not None and len(Z_series) == T_data:
        # TFP affects output: Y_t = Z_t * K_t^α * L_t^(1-α)
        # This is the key aggregate shock that makes approximate aggregation challenging
        Y_series = Y_series * Z_series
        s_series = Z_series
        has_shocks = True
        if verbose:
            print(f"Aggregate TFP shocks: mean={Z_series.mean():.4f}, std={Z_series.std():.4f}")
    else:
        # Placeholder for baseline without aggregate shocks
        s_series = np.ones(T_data)
        has_shocks = False
    
    # Target: Y_{t+1} = (K_{t+1}, C_{t+1}, Y_{t+1})
    Y = np.column_stack([K_series, C_series, Y_series])
    
    if verbose:
        print(f"Feature matrix shape: {X.shape}")
        print(f"Aggregates shape: {Y.shape}")
        print(f"Has aggregate shocks: {has_shocks}")
    
    return {
        'X': X,
        's': s_series,
        'Y': Y,
        'K_series': K_series,
        'L_series': L_series,
        'C_series': C_series,
        'Y_series': Y_series,
        'a_grid_bins': a_grid_bins,
        'n_bins': n_bins
    }


# =============================================================================
# Data Splitting and Normalization
# =============================================================================

def prepare_ml_dataset(
    dataset: Dict[str, np.ndarray],
    train_frac: float = 0.70,
    val_frac: float = 0.15
) -> Dict[str, np.ndarray]:
    """
    Prepare dataset for ML training with chronological split and normalization.
    
    Parameters
    ----------
    dataset : dict
        Output from build_dataset
    train_frac : float
        Fraction for training set
    val_frac : float
        Fraction for validation set
        
    Returns
    -------
    ml_data : dict
        Prepared dataset with splits and normalization params
    """
    X = dataset['X']
    s = dataset['s']
    Y = dataset['Y']
    
    # Align: X_t predicts Y_{t+1}
    X_data = X[:-1]
    s_data = s[1:]
    Y_data = Y[1:]
    
    # Chronological split
    T_total = len(Y_data)
    T_train = int(train_frac * T_total)
    T_val = int(val_frac * T_total)
    
    train_idx = slice(0, T_train)
    val_idx = slice(T_train, T_train + T_val)
    test_idx = slice(T_train + T_val, T_total)
    
    # Split data
    X_train = X_data[train_idx]
    X_val = X_data[val_idx]
    X_test = X_data[test_idx]
    
    s_train = s_data[train_idx]
    s_val = s_data[val_idx]
    s_test = s_data[test_idx]
    
    Y_train = Y_data[train_idx]
    Y_val = Y_data[val_idx]
    Y_test = Y_data[test_idx]
    
    # Normalize features
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std == 0] = 1  # Avoid division by zero
    
    X_train_norm = (X_train - X_mean) / X_std
    X_val_norm = (X_val - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std
    
    # Replace any remaining NaN/Inf values with 0 (safety check)
    X_train_norm = np.nan_to_num(X_train_norm, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_norm = np.nan_to_num(X_val_norm, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_norm = np.nan_to_num(X_test_norm, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize targets
    Y_mean = Y_train.mean(axis=0)
    Y_std = Y_train.std(axis=0)
    Y_std[Y_std == 0] = 1
    
    Y_train_norm = (Y_train - Y_mean) / Y_std
    Y_val_norm = (Y_val - Y_mean) / Y_std
    Y_test_norm = (Y_test - Y_mean) / Y_std
    
    # Replace any remaining NaN/Inf values in Y (safety check)
    Y_train_norm = np.nan_to_num(Y_train_norm, nan=0.0, posinf=0.0, neginf=0.0)
    Y_val_norm = np.nan_to_num(Y_val_norm, nan=0.0, posinf=0.0, neginf=0.0)
    Y_test_norm = np.nan_to_num(Y_test_norm, nan=0.0, posinf=0.0, neginf=0.0)
    
    return {
        # Raw splits
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        's_train': s_train, 's_val': s_val, 's_test': s_test,
        'Y_train': Y_train, 'Y_val': Y_val, 'Y_test': Y_test,
        # Normalized
        'X_train_norm': X_train_norm, 'X_val_norm': X_val_norm, 'X_test_norm': X_test_norm,
        'Y_train_norm': Y_train_norm, 'Y_val_norm': Y_val_norm, 'Y_test_norm': Y_test_norm,
        # Normalization params
        'X_mean': X_mean, 'X_std': X_std,
        'Y_mean': Y_mean, 'Y_std': Y_std,
        # Split info
        'T_train': T_train, 'T_val': T_val, 'T_test': T_total - T_train - T_val
    }


def extract_hand_crafted_states(
    X: np.ndarray,
    n_bins: int = 50,
    state_type: str = 'K'
) -> np.ndarray:
    """
    Extract hand-crafted aggregate states from feature vector.
    
    These are the traditional summary statistics used in Krusell-Smith
    style approximate aggregation.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (T x p)
    n_bins : int
        Number of histogram bins (to know where summary stats start)
    state_type : str
        Type of hand-crafted state:
        - 'K': only mean capital
        - 'K_Gini': capital and Gini
        - 'K_Gini_Top10': capital, Gini, and top 10% share
        - 'full_summary': all summary statistics (no histogram)
        
    Returns
    -------
    S : np.ndarray
        Hand-crafted state (T x d)
    """
    # Summary statistics start after histogram bins
    # Order: mean_a, std_a, skew_a, kurt_a, gini, top10, top1, bottom50, mass_constraint, mean_e
    summary_start = n_bins
    
    mean_a = X[:, summary_start]       # K_t
    std_a = X[:, summary_start + 1]
    skew_a = X[:, summary_start + 2]
    kurt_a = X[:, summary_start + 3]
    gini = X[:, summary_start + 4]
    top10 = X[:, summary_start + 5]
    top1 = X[:, summary_start + 6]
    bottom50 = X[:, summary_start + 7]
    mass_constraint = X[:, summary_start + 8]
    mean_e = X[:, summary_start + 9]
    
    if state_type == 'K':
        return mean_a.reshape(-1, 1)
    elif state_type == 'K_Gini':
        return np.column_stack([mean_a, gini])
    elif state_type == 'K_Gini_Top10':
        return np.column_stack([mean_a, gini, top10])
    elif state_type == 'K_Gini_Constraint':
        return np.column_stack([mean_a, gini, mass_constraint])
    elif state_type == 'full_summary':
        return X[:, summary_start:]
    else:
        raise ValueError(f"Unknown state_type: {state_type}")


# =============================================================================
# Visualization
# =============================================================================

def plot_R_d_curve(
    results: Dict[str, np.ndarray],
    baseline_results: Optional[Dict[str, float]] = None,
    title: str = "Prediction Error vs. Latent Dimension",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot R(d) curve showing minimal prediction error as function of dimension.
    
    Parameters
    ----------
    results : dict
        Output from compute_R_d
    baseline_results : dict, optional
        Results from hand-crafted state baselines
    title : str
        Plot title
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    d = results['d']
    R_d = results['R_d']
    R_d_std = results.get('R_d_std', np.zeros_like(R_d))
    
    # Plot R(d) curve with error bars
    ax.errorbar(d, R_d, yerr=R_d_std, fmt='o-', color='blue', 
                capsize=5, capthick=2, linewidth=2, markersize=8,
                label='Learned States')
    
    # Add baseline comparisons if provided
    if baseline_results is not None:
        colors = ['red', 'green', 'orange', 'purple']
        for i, (name, mse) in enumerate(baseline_results.items()):
            ax.axhline(mse, color=colors[i % len(colors)], linestyle='--', 
                      linewidth=1.5, label=f'{name}')
    
    ax.set_xlabel('Latent Dimension $d$', fontsize=12)
    ax.set_ylabel('Test MSE (normalized)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(d)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    figsize: Tuple[int, int] = (10, 4)
) -> plt.Figure:
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_learned_states_interpretation(
    Z: np.ndarray,
    summary_stats: np.ndarray,
    stat_names: List[str],
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot correlations between learned states and summary statistics.
    
    This implements the interpretation analysis from the proposal:
    Z_{t,k} ≈ α_k + Σ_m β_{k,m} M_{t,m}
    
    Parameters
    ----------
    Z : np.ndarray
        Learned states (T x d)
    summary_stats : np.ndarray
        Summary statistics (T x m)
    stat_names : List[str]
        Names of summary statistics
        
    Returns
    -------
    fig : plt.Figure
        Correlation heatmap
    """
    d = Z.shape[1]
    m = summary_stats.shape[1]
    
    # Compute correlations
    correlations = np.zeros((d, m))
    for i in range(d):
        for j in range(m):
            correlations[i, j] = np.corrcoef(Z[:, i], summary_stats[:, j])[0, 1]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(correlations, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    ax.set_xticks(range(m))
    ax.set_xticklabels(stat_names, rotation=45, ha='right')
    ax.set_yticks(range(d))
    ax.set_yticklabels([f'$Z_{{t,{i+1}}}$' for i in range(d)])
    
    ax.set_xlabel('Summary Statistics')
    ax.set_ylabel('Learned State Components')
    ax.set_title('Correlation: Learned States vs. Summary Statistics')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Correlation')
    
    # Add correlation values as text
    for i in range(d):
        for j in range(m):
            ax.text(j, i, f'{correlations[i, j]:.2f}', 
                   ha='center', va='center', fontsize=8,
                   color='white' if abs(correlations[i, j]) > 0.5 else 'black')
    
    fig.tight_layout()
    return fig


def plot_aggregate_series(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    Y_mean: np.ndarray,
    Y_std: np.ndarray,
    steady_state: Dict[str, float],
    n_periods: int = 200,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot true vs predicted aggregate series.
    
    Parameters
    ----------
    Y_true : np.ndarray
        True aggregates (normalized)
    Y_pred : np.ndarray
        Predicted aggregates (normalized)
    Y_mean, Y_std : np.ndarray
        Normalization parameters
    steady_state : dict
        Steady state values
    n_periods : int
        Number of periods to plot
    figsize : tuple
        Figure size
    """
    # Denormalize
    Y_true_orig = Y_true * Y_std + Y_mean
    Y_pred_orig = Y_pred * Y_std + Y_mean
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    labels = ['Capital $K_t$', 'Consumption $C_t$', 'Output $Y_t$']
    ss_keys = ['K', 'C', 'Y']
    
    for i, (ax, label, key) in enumerate(zip(axes, labels, ss_keys)):
        ax.plot(Y_true_orig[:n_periods, i], 'b-', linewidth=0.8, label='True')
        ax.plot(Y_pred_orig[:n_periods, i], 'r--', linewidth=0.8, label='Predicted')
        if key in steady_state:
            ax.axhline(steady_state[key], color='gray', linestyle=':', 
                      label='Steady State', alpha=0.7)
        ax.set_ylabel(label)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Period')
    fig.suptitle('True vs Predicted Aggregates', fontsize=14)
    fig.tight_layout()
    
    return fig


# =============================================================================
# Summary Statistics Names
# =============================================================================

def get_feature_names(n_bins: int = 50) -> List[str]:
    """Get names of all features in the feature vector."""
    hist_names = [f'hist_{i}' for i in range(n_bins)]
    summary_names = [
        'mean_a (K)', 'std_a', 'skew_a', 'kurt_a',
        'gini', 'top10_share', 'top1_share', 'bottom50_share',
        'mass_constraint', 'mean_e (L)'
    ]
    return hist_names + summary_names


def get_summary_stat_names() -> List[str]:
    """Get names of summary statistics (non-histogram features)."""
    return [
        '$K_t$', '$\\sigma_a$', 'Skew', 'Kurt',
        'Gini', 'Top 10%', 'Top 1%', 'Bottom 50%',
        'Constrained', '$L_t$'
    ]
