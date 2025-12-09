"""
Econometrics Diagnostics for Learned Aggregate States

This module provides tools for Block 1 diagnostics:
1. Uncertainty quantification: R̂(d) across multiple random seeds
2. Approximation vs. estimation decomposition
3. Training protocol documentation
4. LaTeX table generation

Key objects:
    - compute_R_d_across_seeds: Trains models across seeds, returns mean/std
    - compute_explained_variance: Computes Ξ(d) = 1 - R(d)/Var(Y)
    - generate_uncertainty_latex_table: Produces publication-ready LaTeX

Author: Research Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import os


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty analysis."""
    n_seeds: int = 5
    seed_base: int = 42
    dimensions: List[int] = field(default_factory=lambda: [1, 2, 3])
    train_fraction: float = 0.8
    

def set_all_seeds(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def compute_R_d_single_run(
    X_train: np.ndarray,
    s_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    s_val: np.ndarray,
    Y_val: np.ndarray,
    d: int,
    seed: int,
    training_config: Optional[Any] = None
) -> Dict[str, float]:
    """
    Train encoder-predictor for dimension d and evaluate.
    
    Returns
    -------
    Dict with keys:
        - 'R_hat_d': Validation MSE (empirical risk)
        - 'train_mse': Training MSE
        - 'epochs_used': Number of epochs before early stopping
    """
    try:
        import torch
        from .ml_models import EncoderPredictor, TrainingConfig
    except ImportError:
        # Fallback for standalone usage
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        import torch
        from ml_models import EncoderPredictor, TrainingConfig
    
    set_all_seeds(seed)
    
    if training_config is None:
        training_config = TrainingConfig(seed=seed)
    else:
        training_config.seed = seed
    
    # Setup model
    input_dim = X_train.shape[1]
    shock_dim = 1 if s_train.ndim == 1 else s_train.shape[1]
    output_dim = Y_train.shape[1] if Y_train.ndim > 1 else 1
    
    model = EncoderPredictor(
        input_dim=input_dim,
        latent_dim=d,
        shock_dim=shock_dim,
        output_dim=output_dim,
        config=training_config
    )
    
    device = torch.device(training_config.device)
    model.to(device)
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    s_train_t = torch.tensor(s_train.reshape(-1, 1) if s_train.ndim == 1 else s_train, 
                              dtype=torch.float32, device=device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32, device=device)
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    s_val_t = torch.tensor(s_val.reshape(-1, 1) if s_val.ndim == 1 else s_val,
                            dtype=torch.float32, device=device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32, device=device)
    
    # Training setup
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.lr,
        weight_decay=training_config.weight_decay
    )
    criterion = torch.nn.MSELoss()
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    epochs_used = 0
    
    # Training loop
    for epoch in range(training_config.max_epochs):
        model.train()
        optimizer.zero_grad()
        
        _, Y_pred = model(X_train_t, s_train_t)
        train_loss = criterion(Y_pred, Y_train_t)
        
        train_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            _, Y_val_pred = model(X_val_t, s_val_t)
            val_loss = criterion(Y_val_pred, Y_val_t).item()
        
        epochs_used = epoch + 1
        
        # Early stopping check
        if val_loss < best_val_loss - training_config.min_delta:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= training_config.patience:
                break
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        _, Y_pred_train = model(X_train_t, s_train_t)
        train_mse = criterion(Y_pred_train, Y_train_t).item()
        
        _, Y_pred_val = model(X_val_t, s_val_t)
        val_mse = criterion(Y_pred_val, Y_val_t).item()
    
    return {
        'R_hat_d': val_mse,
        'train_mse': train_mse,
        'epochs_used': epochs_used
    }


def compute_R_d_across_seeds(
    X: np.ndarray,
    s: np.ndarray,
    Y: np.ndarray,
    d: int,
    config: Optional[UncertaintyConfig] = None,
    training_config: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Train models across multiple random seeds and compute statistics.
    
    This addresses the uncertainty quantification requirement by reporting
    both point estimates and standard errors for R̂(d).
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (N, p)
    s : np.ndarray
        Aggregate shocks (N,) or (N, q)
    Y : np.ndarray
        Target aggregates (N, m)
    d : int
        Bottleneck dimension
    config : UncertaintyConfig
        Configuration for uncertainty analysis
    training_config : TrainingConfig, optional
        Configuration for neural network training
        
    Returns
    -------
    Dict with keys:
        - 'mean_R_hat': Mean validation MSE across seeds
        - 'std_R_hat': Standard deviation of validation MSE
        - 'se_R_hat': Standard error (std / sqrt(n_seeds))
        - 'all_R_hat': List of individual seed results
        - 'mean_train_mse': Mean training MSE (for overfitting diagnosis)
        - 'mean_epochs': Mean epochs used (for early stopping check)
    """
    if config is None:
        config = UncertaintyConfig()
    
    results_by_seed = []
    
    for seed_idx in range(config.n_seeds):
        seed = config.seed_base + seed_idx
        set_all_seeds(seed)
        
        # Random train/val split
        n = len(X)
        idx = np.random.permutation(n)
        split = int(n * config.train_fraction)
        train_idx = idx[:split]
        val_idx = idx[split:]
        
        X_train, X_val = X[train_idx], X[val_idx]
        s_train, s_val = s[train_idx], s[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]
        
        result = compute_R_d_single_run(
            X_train, s_train, Y_train,
            X_val, s_val, Y_val,
            d=d, seed=seed, training_config=training_config
        )
        results_by_seed.append(result)
    
    all_R_hat = [r['R_hat_d'] for r in results_by_seed]
    all_train = [r['train_mse'] for r in results_by_seed]
    all_epochs = [r['epochs_used'] for r in results_by_seed]
    
    return {
        'mean_R_hat': np.mean(all_R_hat),
        'std_R_hat': np.std(all_R_hat),
        'se_R_hat': np.std(all_R_hat) / np.sqrt(config.n_seeds),
        'all_R_hat': all_R_hat,
        'mean_train_mse': np.mean(all_train),
        'std_train_mse': np.std(all_train),
        'mean_epochs': np.mean(all_epochs),
        'train_val_gap': np.mean(all_R_hat) - np.mean(all_train)
    }


def compute_full_uncertainty_analysis(
    X: np.ndarray,
    s: np.ndarray,
    Y: np.ndarray,
    config: Optional[UncertaintyConfig] = None,
    training_config: Optional[Any] = None
) -> pd.DataFrame:
    """
    Run full uncertainty analysis for all dimensions.
    
    Returns DataFrame with columns:
        d, mean_R, std_R, se_R, mean_train, train_val_gap, mean_epochs
    """
    if config is None:
        config = UncertaintyConfig()
    
    results = []
    for d in config.dimensions:
        res = compute_R_d_across_seeds(X, s, Y, d, config, training_config)
        results.append({
            'd': d,
            'mean_R': res['mean_R_hat'],
            'std_R': res['std_R_hat'],
            'se_R': res['se_R_hat'],
            'mean_train': res['mean_train_mse'],
            'train_val_gap': res['train_val_gap'],
            'mean_epochs': res['mean_epochs']
        })
    
    return pd.DataFrame(results)


def compute_explained_variance(
    R_d: float,
    Y: np.ndarray
) -> float:
    """
    Compute explained variance ratio Ξ(d) = 1 - R(d) / Var(Y).
    
    This measure quantifies how much of Y's variance is predictable
    from a d-dimensional bottleneck representation.
    
    Parameters
    ----------
    R_d : float
        Prediction risk at dimension d
    Y : np.ndarray
        Target variable(s)
        
    Returns
    -------
    Xi_d : float
        Explained variance ratio (analogous to R²)
    """
    var_Y = np.var(Y, axis=0).mean() if Y.ndim > 1 else np.var(Y)
    return 1 - R_d / var_Y if var_Y > 0 else 0.0


def compute_approximation_vs_estimation_bound(
    R_d_train: float,
    R_d_val: float,
    n_train: int,
    d: int,
    input_dim: int,
    significance_level: float = 0.95
) -> Dict[str, float]:
    """
    Decompose total error into approximation and estimation components.
    
    Following the bias-variance decomposition:
        R̂_n(d) - R(d) = [R_{F,G}(d) - R(d)] + [R̂_n(d) - R_{F,G}(d)]
                       = (approximation)      + (estimation)
    
    We estimate the estimation component from train-val gap and
    provide heuristic bounds.
    
    Parameters
    ----------
    R_d_train : float
        Training MSE
    R_d_val : float
        Validation MSE
    n_train : int
        Number of training samples
    d : int
        Bottleneck dimension
    input_dim : int
        Feature dimension
    significance_level : float
        Confidence level for bounds
        
    Returns
    -------
    Dict with estimation diagnostics
    """
    # Train-val gap suggests estimation error magnitude
    estimation_gap = R_d_val - R_d_train
    
    # Heuristic: estimation error scales as O(dim / n)
    # where dim is effective parameter count
    # For neural net: roughly (input_dim + hidden) * d + predictor params
    approx_param_count = (input_dim * 64 + 64 * 32 + 32 * d) + (d * 32 + 32 * 16 + 16 * 3)
    complexity_ratio = approx_param_count / n_train
    
    return {
        'train_mse': R_d_train,
        'val_mse': R_d_val,
        'estimation_gap': estimation_gap,
        'gap_positive': estimation_gap > 0,  # Positive gap = no severe overfitting
        'complexity_ratio': complexity_ratio,
        'is_overparameterized': complexity_ratio > 0.1  # Heuristic threshold
    }


# =============================================================================
# LaTeX Table Generation
# =============================================================================

def generate_uncertainty_latex_table(
    df: pd.DataFrame,
    caption: str = "Uncertainty in learned aggregate risk",
    label: str = "tab:uncertainty",
    config_name: str = "Representative calibration"
) -> str:
    """
    Generate publication-ready LaTeX table for uncertainty analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Output from compute_full_uncertainty_analysis
    caption : str
        Table caption
    label : str
        LaTeX label
    config_name : str
        Name of the calibration configuration
        
    Returns
    -------
    latex_str : str
        Complete LaTeX table code
    """
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{cccccc}",
        r"\toprule",
        r"$d$ & $\bar{\hat{R}}(d)$ & $\text{std}(\hat{R})$ & $\text{SE}(\hat{R})$ & Train MSE & Gap \\",
        r"\midrule"
    ]
    
    for _, row in df.iterrows():
        line = (
            f"{int(row['d'])} & "
            f"{row['mean_R']:.4f} & "
            f"{row['std_R']:.4f} & "
            f"{row['se_R']:.4f} & "
            f"{row['mean_train']:.4f} & "
            f"{row['train_val_gap']:.4f} \\\\"
        )
        latex_lines.append(line)
    
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.2cm}",
        fr"\parbox{{\textwidth}}{{\footnotesize \textit{{Notes:}} Results for {config_name}. "
        r"$\bar{\hat{R}}(d)$ is the mean empirical risk across 5 random seeds, "
        r"$\text{std}(\hat{R})$ is the standard deviation, "
        r"$\text{SE}(\hat{R})$ is the standard error, "
        r"Train MSE is the in-sample error, and Gap = $\bar{\hat{R}} - \text{Train MSE}$ "
        r"measures overfitting (positive values indicate generalization).}}",
        r"\end{table}"
    ])
    
    return "\n".join(latex_lines)


def generate_explained_variance_table(
    dimensions: List[int],
    R_values: List[float],
    Y: np.ndarray,
    label: str = "tab:explained_variance"
) -> str:
    """
    Generate LaTeX table showing explained variance Ξ(d) for each dimension.
    """
    var_Y = np.var(Y, axis=0).mean() if Y.ndim > 1 else np.var(Y)
    
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Explained variance by bottleneck dimension}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{ccc}",
        r"\toprule",
        r"$d$ & $R(d)$ & $\Xi(d) = 1 - R(d)/\text{Var}(Y)$ \\",
        r"\midrule"
    ]
    
    for d, R_d in zip(dimensions, R_values):
        Xi_d = compute_explained_variance(R_d, Y)
        latex_lines.append(f"{d} & {R_d:.4f} & {Xi_d:.2%} \\\\")
    
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])
    
    return "\n".join(latex_lines)


# =============================================================================
# Training Protocol Documentation
# =============================================================================

def document_training_protocol(
    config: Optional[Any] = None
) -> str:
    """
    Generate documentation of the training protocol for methods section.
    
    Returns markdown/text description of:
    - Architecture choices
    - Hyperparameters
    - Early stopping criterion
    - Data handling
    """
    try:
        from .ml_models import TrainingConfig
    except ImportError:
        from ml_models import TrainingConfig
    
    if config is None:
        config = TrainingConfig()
    
    doc = f"""
Training Protocol for Encoder-Predictor Networks
=================================================

Architecture:
- Encoder f_θ: MLP with hidden layers {config.encoder_hidden_dims}
- Predictor g_ψ: MLP with hidden layers {config.predictor_hidden_dims}
- Activation: {config.activation}
- Dropout: {config.dropout}

Optimization:
- Optimizer: Adam
- Learning rate: {config.lr}
- Weight decay (L2): {config.weight_decay}
- Batch size: {config.batch_size}
- Maximum epochs: {config.max_epochs}

Early Stopping:
- Patience: {config.patience} epochs
- Minimum improvement threshold: {config.min_delta}
- Criterion: Validation MSE

Data:
- Train/validation split: 80/20 (configurable)
- Features: Wealth histogram bins + summary statistics
- Targets: Next-period aggregates (K, C, Y)

Uncertainty Quantification:
- Number of random seeds: 5 (default)
- Reported metrics: Mean ± standard error across seeds
"""
    return doc


# =============================================================================
# Convenience Functions for Paper Figures
# =============================================================================

def load_results_and_compute_uncertainty(
    results_path: str = "results/table1_R_d_by_config.csv"
) -> pd.DataFrame:
    """
    Load existing results and compute implied statistics.
    
    This is a utility for working with pre-computed results when
    full retraining is not needed.
    """
    df = pd.read_csv(results_path)
    
    # Extract R(d) columns
    R_cols = [c for c in df.columns if c.startswith('R(d=')]
    
    # Compute mean/std across configurations (as proxy for robustness)
    stats = {}
    for col in R_cols:
        d = int(col.replace('R(d=', '').replace(')', ''))
        stats[f'd={d}'] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max()
        }
    
    return pd.DataFrame(stats).T


if __name__ == "__main__":
    # Example usage: print training protocol documentation
    print(document_training_protocol())
    
    # Example: load existing results
    try:
        stats_df = load_results_and_compute_uncertainty()
        print("\nR(d) statistics across configurations:")
        print(stats_df)
    except FileNotFoundError:
        print("Results file not found. Run experiments first.")
