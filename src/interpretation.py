"""
Interpretation Module for Learned Aggregate States

This module provides tools for interpreting learned state representations
and comparing them to economic statistics.

From the proposal (Section: Interpreting Learned States):
> "For each dimension k, consider:
>  Z_{t,k} ≈ α_k + Σ_m β_{k,m} M_{t,m}
>  where M_{t,m} are hand-crafted statistics"

Key analyses:
1. Correlations between Z_t components and summary statistics
2. Regression of Z_t on economic statistics (R² and coefficients)
3. Economic labeling of learned factors
4. Feature importance and partial dependence analysis

Author: Research Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Correlation Analysis
# =============================================================================

def compute_correlation_matrix(
    Z: np.ndarray,
    summary_stats: np.ndarray,
    stat_names: List[str]
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute correlation matrix between learned states and summary statistics.
    
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
    correlations : np.ndarray
        Correlation matrix (d x m)
    corr_df : pd.DataFrame
        Formatted correlation DataFrame
    """
    d = Z.shape[1]
    m = summary_stats.shape[1]
    
    correlations = np.zeros((d, m))
    p_values = np.zeros((d, m))
    
    for i in range(d):
        for j in range(m):
            r, p = stats.pearsonr(Z[:, i], summary_stats[:, j])
            correlations[i, j] = r
            p_values[i, j] = p
    
    # Create DataFrame
    z_names = [f'Z_{i+1}' for i in range(d)]
    corr_df = pd.DataFrame(
        correlations,
        index=z_names,
        columns=stat_names
    )
    
    return correlations, corr_df


def regression_interpretation(
    Z: np.ndarray,
    summary_stats: np.ndarray,
    stat_names: List[str]
) -> Dict[str, Any]:
    """
    Regress each Z_k on summary statistics to interpret learned factors.
    
    From the proposal:
    Z_{t,k} ≈ α_k + Σ_m β_{k,m} M_{t,m} + η_{t,k}
    
    Parameters
    ----------
    Z : np.ndarray
        Learned states (T x d)
    summary_stats : np.ndarray
        Summary statistics (T x m)
    stat_names : List[str]
        Names of statistics
        
    Returns
    -------
    results : dict
        Dictionary with regression results for each Z component
    """
    d = Z.shape[1]
    m = summary_stats.shape[1]
    
    # Standardize for interpretable coefficients
    scaler = StandardScaler()
    M_scaled = scaler.fit_transform(summary_stats)
    
    results = {
        'r2': np.zeros(d),
        'coefficients': np.zeros((d, m)),
        'intercepts': np.zeros(d),
        'dominant_stats': [],
        'interpretations': []
    }
    
    for k in range(d):
        reg = LinearRegression()
        reg.fit(M_scaled, Z[:, k])
        
        # R² of regression
        results['r2'][k] = reg.score(M_scaled, Z[:, k])
        results['coefficients'][k] = reg.coef_
        results['intercepts'][k] = reg.intercept_
        
        # Find dominant statistics (top 3 by absolute coefficient)
        abs_coef = np.abs(reg.coef_)
        top_idx = np.argsort(abs_coef)[::-1][:3]
        
        dominant = [(stat_names[i], reg.coef_[i]) for i in top_idx]
        results['dominant_stats'].append(dominant)
        
        # Generate interpretation
        top_stat = stat_names[top_idx[0]]
        top_sign = 'positive' if reg.coef_[top_idx[0]] > 0 else 'negative'
        interpretation = f"Z_{k+1} strongly {top_sign}ly correlated with {top_stat}"
        
        if reg.coef_[top_idx[1]] != 0:
            second_stat = stat_names[top_idx[1]]
            second_sign = '+' if reg.coef_[top_idx[1]] > 0 else '-'
            interpretation += f", also {second_sign} {second_stat}"
        
        results['interpretations'].append(interpretation)
    
    return results


def label_learned_factors(
    Z: np.ndarray,
    summary_stats: np.ndarray,
    stat_names: List[str]
) -> Dict[str, str]:
    """
    Assign economic labels to learned factors based on correlations.
    
    Returns a mapping from Z_k to interpretable names like:
    - "Level state" (if highly correlated with K)
    - "Inequality state" (if highly correlated with Gini)
    - "Constraint state" (if correlated with mass at constraint)
    
    Parameters
    ----------
    Z : np.ndarray
        Learned states (T x d)
    summary_stats : np.ndarray
        Summary statistics (T x m)
    stat_names : List[str]
        Names of statistics
        
    Returns
    -------
    labels : dict
        Mapping from 'Z_k' to economic label
    """
    correlations, _ = compute_correlation_matrix(Z, summary_stats, stat_names)
    d = Z.shape[1]
    
    # Define economic categories based on typical stat names
    category_keywords = {
        'Level': ['K', 'mean', 'capital'],
        'Inequality': ['gini', 'Gini', 'top', 'Top', 'bottom', 'Bottom'],
        'Volatility': ['std', 'Std', 'σ'],
        'Constraint': ['constraint', 'Constraint', 'bound'],
        'Labor': ['L', 'labor', 'productivity', 'mean_e'],
        'Tail': ['skew', 'Skew', 'kurt', 'Kurt']
    }
    
    def get_category(stat_name: str) -> str:
        for category, keywords in category_keywords.items():
            if any(kw in stat_name for kw in keywords):
                return category
        return 'Other'
    
    labels = {}
    used_categories = set()
    
    for k in range(d):
        # Find highest absolute correlation
        best_idx = np.argmax(np.abs(correlations[k]))
        best_stat = stat_names[best_idx]
        best_corr = correlations[k, best_idx]
        
        category = get_category(best_stat)
        
        # Make label unique if needed
        if category in used_categories:
            label = f"{category} (secondary)"
        else:
            label = f"{category} state"
            used_categories.add(category)
        
        sign = '(+)' if best_corr > 0 else '(-)'
        labels[f'Z_{k+1}'] = f"{label} {sign}"
    
    return labels


# =============================================================================
# Visualization
# =============================================================================

def plot_correlation_heatmap(
    correlations: np.ndarray,
    stat_names: List[str],
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Learned States vs. Economic Statistics"
) -> plt.Figure:
    """
    Create heatmap of correlations between learned states and statistics.
    
    Parameters
    ----------
    correlations : np.ndarray
        Correlation matrix (d x m)
    stat_names : List[str]
        Names of statistics
    figsize : tuple
        Figure size
    title : str
        Plot title
        
    Returns
    -------
    fig : plt.Figure
    """
    d, m = correlations.shape
    z_names = [f'$Z_{{t,{i+1}}}$' for i in range(d)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(correlations, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    ax.set_xticks(range(m))
    ax.set_xticklabels(stat_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(d))
    ax.set_yticklabels(z_names, fontsize=12)
    
    ax.set_xlabel('Economic Statistics', fontsize=12)
    ax.set_ylabel('Learned State Components', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Pearson Correlation', fontsize=11)
    
    # Annotate with correlation values
    for i in range(d):
        for j in range(m):
            val = correlations[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   fontsize=9, color=color, fontweight='bold')
    
    fig.tight_layout()
    return fig


def plot_regression_interpretation(
    regression_results: Dict[str, Any],
    stat_names: List[str],
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Plot regression coefficients and R² for each learned component.
    
    Parameters
    ----------
    regression_results : dict
        Output from regression_interpretation
    stat_names : List[str]
        Names of statistics
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : plt.Figure
    """
    d = len(regression_results['r2'])
    coefficients = regression_results['coefficients']
    r2_values = regression_results['r2']
    
    fig, axes = plt.subplots(1, d + 1, figsize=figsize, 
                            gridspec_kw={'width_ratios': [1] * d + [0.5]})
    
    # Plot coefficients for each Z component
    for k in range(d):
        ax = axes[k]
        coefs = coefficients[k]
        colors = ['red' if c < 0 else 'blue' for c in coefs]
        
        y_pos = np.arange(len(stat_names))
        ax.barh(y_pos, coefs, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(stat_names if k == 0 else [], fontsize=9)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlabel('Coefficient', fontsize=10)
        ax.set_title(f'$Z_{{t,{k+1}}}$\n($R^2$ = {r2_values[k]:.3f})', fontsize=11)
        ax.grid(True, alpha=0.3, axis='x')
    
    # R² summary
    ax = axes[-1]
    ax.barh(range(d), r2_values, color='green', alpha=0.7)
    ax.set_yticks(range(d))
    ax.set_yticklabels([f'$Z_{{t,{k+1}}}$' for k in range(d)])
    ax.set_xlabel('$R^2$', fontsize=10)
    ax.set_title('Explained\nVariance', fontsize=11)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='x')
    
    fig.suptitle('Regression: $Z_{t,k} \\approx \\alpha_k + \\sum_m \\beta_{k,m} M_{t,m}$', 
                fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_learned_vs_handcrafted(
    learned_mse: np.ndarray,
    handcrafted_mse: Dict[str, float],
    d_values: List[int],
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Compare learned states vs hand-crafted states by dimension.
    
    Parameters
    ----------
    learned_mse : np.ndarray
        MSE for learned states at each d
    handcrafted_mse : dict
        MSE for hand-crafted states {name: mse}
    d_values : list
        Dimensions evaluated
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Learned states
    ax.plot(d_values, learned_mse, 'b-o', linewidth=2, markersize=10,
            label='Learned States', zorder=3)
    
    # Hand-crafted states
    markers = ['s', '^', 'D', 'v']
    colors = ['red', 'green', 'orange', 'purple']
    
    handcrafted_dims = {
        'S1_K': 1, 'S2_K_Gini': 2, 'S3_K_Gini_Top10': 3
    }
    
    for i, (name, mse) in enumerate(handcrafted_mse.items()):
        dim = handcrafted_dims.get(name, i + 1)
        ax.scatter([dim], [mse], s=150, marker=markers[i % len(markers)],
                   color=colors[i % len(colors)], zorder=4,
                   label=f'Hand-crafted: {name}', edgecolor='black', linewidth=1)
    
    ax.set_xlabel('State Dimension $d$', fontsize=12)
    ax.set_ylabel('Test MSE', fontsize=12)
    ax.set_title('Learned vs. Hand-Crafted Aggregate States', fontsize=14)
    ax.set_xticks(d_values)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add annotations for improvement
    for i, d in enumerate(d_values):
        hc_name = f'S{d}_K' if d == 1 else f'S{d}_K_Gini' if d == 2 else 'S3_K_Gini_Top10'
        if hc_name in handcrafted_mse:
            hc_mse = handcrafted_mse[hc_name]
            improvement = (hc_mse - learned_mse[i]) / hc_mse * 100
            if improvement > 0:
                ax.annotate(f'{improvement:.1f}% better',
                           xy=(d, learned_mse[i]), xytext=(d + 0.2, learned_mse[i] * 0.9),
                           fontsize=9, color='blue', fontweight='bold')
    
    fig.tight_layout()
    return fig


def plot_method_comparison_bar(
    comparison: Dict[str, Any],
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Create bar chart comparing all methods.
    
    Parameters
    ----------
    comparison : dict
        Output from compare_all_methods
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : plt.Figure
    """
    ranking = comparison['ranking']
    
    names = [r[0] for r in ranking]
    mses = [r[1] for r in ranking]
    categories = [r[2] for r in ranking]
    
    color_map = {
        'learned': 'steelblue',
        'handcrafted': 'coral',
        'ml_baseline': 'seagreen'
    }
    colors = [color_map[c] for c in categories]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, mses, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Test MSE', fontsize=12)
    ax.set_title('Comparison of All Methods (Lower is Better)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='Learned States'),
        Patch(facecolor='coral', label='Hand-crafted'),
        Patch(facecolor='seagreen', label='ML Baselines')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    # Add MSE values on bars
    for bar, mse in zip(bars, mses):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
               f'{mse:.4f}', va='center', fontsize=9)
    
    fig.tight_layout()
    return fig


def create_interpretation_summary(
    Z: np.ndarray,
    summary_stats: np.ndarray,
    stat_names: List[str]
) -> pd.DataFrame:
    """
    Create comprehensive interpretation summary table.
    
    Parameters
    ----------
    Z : np.ndarray
        Learned states (T x d)
    summary_stats : np.ndarray
        Summary statistics (T x m)
    stat_names : List[str]
        Names of statistics
        
    Returns
    -------
    summary_df : pd.DataFrame
        Summary table with interpretations
    """
    correlations, corr_df = compute_correlation_matrix(Z, summary_stats, stat_names)
    reg_results = regression_interpretation(Z, summary_stats, stat_names)
    labels = label_learned_factors(Z, summary_stats, stat_names)
    
    d = Z.shape[1]
    
    rows = []
    for k in range(d):
        z_name = f'Z_{k+1}'
        
        # Top correlations
        abs_corr = np.abs(correlations[k])
        top_idx = np.argsort(abs_corr)[::-1][:3]
        top_corr_str = ', '.join([
            f"{stat_names[i]}({correlations[k, i]:.2f})" 
            for i in top_idx
        ])
        
        row = {
            'Component': z_name,
            'Economic Label': labels[z_name],
            'R² (regression)': reg_results['r2'][k],
            'Top Correlations': top_corr_str,
            'Interpretation': reg_results['interpretations'][k]
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def extract_summary_stats_from_X(
    X: np.ndarray,
    n_bins: int = 50
) -> np.ndarray:
    """
    Extract summary statistics from feature matrix X.
    
    Summary stats are stored after the histogram bins in X.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (T x p)
    n_bins : int
        Number of histogram bins
        
    Returns
    -------
    summary_stats : np.ndarray
        Summary statistics (T x 10)
    """
    return X[:, n_bins:]


def get_economic_stat_names() -> List[str]:
    """Get readable names for economic statistics."""
    return [
        '$K_t$ (Capital)',
        '$\\sigma_a$ (Wealth Std)',
        'Skewness',
        'Kurtosis',
        'Gini Index',
        'Top 10% Share',
        'Top 1% Share',
        'Bottom 50% Share',
        'Mass at Constraint',
        '$L_t$ (Labor)'
    ]
