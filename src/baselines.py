"""
Baseline Models for Learned Aggregate States Project

This module implements baseline predictors for comparison with learned states:
1. ML baselines without dimension reduction (Linear, Random Forest, Gradient Boosting)
2. Utility functions for training and evaluating baselines

From the proposal:
> "For completeness, consider:
>  - Linear models: regress Y_{t+1} on (X_t, s_{t+1}) directly.
>  - Tree-based models: random forests or gradient-boosted trees on (X_t, s_{t+1})."

Author: Research Team
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


@dataclass
class BaselineConfig:
    """Configuration for baseline models."""
    # Ridge regression
    ridge_alpha: float = 1.0
    
    # Random Forest
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = 10
    rf_min_samples_leaf: int = 5
    rf_n_jobs: int = -1
    
    # Gradient Boosting
    gb_n_estimators: int = 100
    gb_learning_rate: float = 0.1
    gb_max_depth: int = 5
    gb_min_samples_leaf: int = 5
    
    # General
    random_state: int = 42


class LinearBaseline:
    """
    Linear regression baseline (Ridge).
    
    Maps (X_t, s_{t+1}) directly to Y_{t+1} using linear regression.
    This serves as a simple baseline that doesn't use any nonlinear 
    transformations or learned representations.
    """
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = None
        
    def fit(
        self,
        X_train: np.ndarray,
        s_train: np.ndarray,
        Y_train: np.ndarray
    ) -> 'LinearBaseline':
        """Fit the linear model."""
        # Combine features
        Xs = np.column_stack([X_train, s_train.reshape(-1, 1)])
        
        # Fit Ridge regression (handles multioutput automatically)
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(Xs, Y_train)
        
        return self
    
    def predict(
        self,
        X: np.ndarray,
        s: np.ndarray
    ) -> np.ndarray:
        """Predict Y_{t+1}."""
        Xs = np.column_stack([X, s.reshape(-1, 1)])
        return self.model.predict(Xs)
    
    def evaluate(
        self,
        X_test: np.ndarray,
        s_test: np.ndarray,
        Y_test: np.ndarray,
        Y_mean: np.ndarray,
        Y_std: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model on test set."""
        Y_pred = self.predict(X_test, s_test)
        return _compute_metrics(Y_pred, Y_test, Y_mean, Y_std)


class RandomForestBaseline:
    """
    Random Forest baseline.
    
    Uses ensemble of decision trees for nonlinear prediction without
    explicit dimension reduction. Tests whether learned representations
    add value beyond unrestricted flexibility.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 10,
        min_samples_leaf: int = 5,
        n_jobs: int = -1,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.model = None
        
    def fit(
        self,
        X_train: np.ndarray,
        s_train: np.ndarray,
        Y_train: np.ndarray
    ) -> 'RandomForestBaseline':
        """Fit the Random Forest model."""
        Xs = np.column_stack([X_train, s_train.reshape(-1, 1)])
        
        # Use MultiOutputRegressor for multiple targets
        base_rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        self.model = MultiOutputRegressor(base_rf)
        self.model.fit(Xs, Y_train)
        
        return self
    
    def predict(
        self,
        X: np.ndarray,
        s: np.ndarray
    ) -> np.ndarray:
        """Predict Y_{t+1}."""
        Xs = np.column_stack([X, s.reshape(-1, 1)])
        return self.model.predict(Xs)
    
    def evaluate(
        self,
        X_test: np.ndarray,
        s_test: np.ndarray,
        Y_test: np.ndarray,
        Y_mean: np.ndarray,
        Y_std: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model on test set."""
        Y_pred = self.predict(X_test, s_test)
        return _compute_metrics(Y_pred, Y_test, Y_mean, Y_std)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importances (averaged across outputs)."""
        importances = []
        for estimator in self.model.estimators_:
            importances.append(estimator.feature_importances_)
        return np.mean(importances, axis=0)


class GradientBoostingBaseline:
    """
    Gradient Boosting baseline.
    
    Uses gradient boosted trees for sequential correction of predictions.
    Strong nonlinear baseline that often achieves excellent performance.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        min_samples_leaf: int = 5,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model = None
        
    def fit(
        self,
        X_train: np.ndarray,
        s_train: np.ndarray,
        Y_train: np.ndarray
    ) -> 'GradientBoostingBaseline':
        """Fit the Gradient Boosting model."""
        Xs = np.column_stack([X_train, s_train.reshape(-1, 1)])
        
        # Use MultiOutputRegressor for multiple targets
        base_gb = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )
        self.model = MultiOutputRegressor(base_gb)
        self.model.fit(Xs, Y_train)
        
        return self
    
    def predict(
        self,
        X: np.ndarray,
        s: np.ndarray
    ) -> np.ndarray:
        """Predict Y_{t+1}."""
        Xs = np.column_stack([X, s.reshape(-1, 1)])
        return self.model.predict(Xs)
    
    def evaluate(
        self,
        X_test: np.ndarray,
        s_test: np.ndarray,
        Y_test: np.ndarray,
        Y_mean: np.ndarray,
        Y_std: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model on test set."""
        Y_pred = self.predict(X_test, s_test)
        return _compute_metrics(Y_pred, Y_test, Y_mean, Y_std)


def _compute_metrics(
    Y_pred: np.ndarray,
    Y_test: np.ndarray,
    Y_mean: np.ndarray,
    Y_std: np.ndarray
) -> Dict[str, float]:
    """
    Compute evaluation metrics for predictions.
    
    Returns same metrics as neural network evaluation for fair comparison.
    """
    # MSE on normalized scale
    mse_norm = np.mean((Y_pred - Y_test)**2)
    
    # Convert to original scale
    Y_pred_orig = Y_pred * Y_std + Y_mean
    Y_test_orig = Y_test * Y_std + Y_mean
    mse_orig = np.mean((Y_pred_orig - Y_test_orig)**2)
    
    # Variance of test targets
    var_test = np.var(Y_test, axis=0)
    var_test[var_test == 0] = 1
    
    # Relative MSE
    relative_mse = np.mean((Y_pred - Y_test)**2, axis=0) / var_test
    
    # R² for each component
    ss_res = np.sum((Y_test - Y_pred)**2, axis=0)
    ss_tot = np.sum((Y_test - Y_test.mean(axis=0))**2, axis=0)
    r2 = 1 - ss_res / np.maximum(ss_tot, 1e-10)
    
    return {
        'mse_norm': mse_norm,
        'mse_orig': mse_orig,
        'relative_mse': relative_mse.mean(),
        'relative_mse_K': relative_mse[0],
        'relative_mse_C': relative_mse[1],
        'relative_mse_Y': relative_mse[2],
        'r2_K': r2[0],
        'r2_C': r2[1],
        'r2_Y': r2[2],
        'r2_mean': r2.mean()
    }


def train_all_ml_baselines(
    X_train: np.ndarray,
    s_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    s_test: np.ndarray,
    Y_test: np.ndarray,
    Y_mean: np.ndarray,
    Y_std: np.ndarray,
    config: Optional[BaselineConfig] = None,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Train and evaluate all ML baselines.
    
    Parameters
    ----------
    X_train, s_train, Y_train : np.ndarray
        Training data (normalized)
    X_test, s_test, Y_test : np.ndarray
        Test data (normalized)
    Y_mean, Y_std : np.ndarray
        Normalization parameters
    config : BaselineConfig
        Configuration for baselines
    verbose : bool
        Print progress
        
    Returns
    -------
    results : dict
        Dictionary with results for each baseline
    """
    if config is None:
        config = BaselineConfig()
    
    results = {}
    
    # 1. Linear regression
    if verbose:
        print("  Training Linear (Ridge) baseline...")
    linear = LinearBaseline(alpha=config.ridge_alpha)
    linear.fit(X_train, s_train, Y_train)
    results['Linear'] = {
        'model': linear,
        'metrics': linear.evaluate(X_test, s_test, Y_test, Y_mean, Y_std)
    }
    if verbose:
        print(f"    MSE = {results['Linear']['metrics']['mse_norm']:.6f}, "
              f"R² = {results['Linear']['metrics']['r2_mean']:.4f}")
    
    # 2. Random Forest
    if verbose:
        print("  Training Random Forest baseline...")
    rf = RandomForestBaseline(
        n_estimators=config.rf_n_estimators,
        max_depth=config.rf_max_depth,
        min_samples_leaf=config.rf_min_samples_leaf,
        n_jobs=config.rf_n_jobs,
        random_state=config.random_state
    )
    rf.fit(X_train, s_train, Y_train)
    results['RandomForest'] = {
        'model': rf,
        'metrics': rf.evaluate(X_test, s_test, Y_test, Y_mean, Y_std),
        'feature_importance': rf.get_feature_importance()
    }
    if verbose:
        print(f"    MSE = {results['RandomForest']['metrics']['mse_norm']:.6f}, "
              f"R² = {results['RandomForest']['metrics']['r2_mean']:.4f}")
    
    # 3. Gradient Boosting
    if verbose:
        print("  Training Gradient Boosting baseline...")
    gb = GradientBoostingBaseline(
        n_estimators=config.gb_n_estimators,
        learning_rate=config.gb_learning_rate,
        max_depth=config.gb_max_depth,
        min_samples_leaf=config.gb_min_samples_leaf,
        random_state=config.random_state
    )
    gb.fit(X_train, s_train, Y_train)
    results['GradientBoosting'] = {
        'model': gb,
        'metrics': gb.evaluate(X_test, s_test, Y_test, Y_mean, Y_std)
    }
    if verbose:
        print(f"    MSE = {results['GradientBoosting']['metrics']['mse_norm']:.6f}, "
              f"R² = {results['GradientBoosting']['metrics']['r2_mean']:.4f}")
    
    return results


def compare_all_methods(
    learned_results: Dict[str, Any],
    handcrafted_results: Dict[str, Any],
    ml_baseline_results: Dict[str, Any],
    d_values: List[int] = [1, 2, 3]
) -> Dict[str, Any]:
    """
    Create comprehensive comparison of all methods.
    
    Parameters
    ----------
    learned_results : dict
        Results from compute_R_d for learned states
    handcrafted_results : dict
        Results from hand-crafted state predictors
    ml_baseline_results : dict
        Results from ML baselines
    d_values : list
        Dimensions used for learned states
        
    Returns
    -------
    comparison : dict
        Comparison summary with rankings and analysis
    """
    comparison = {
        'learned': {},
        'handcrafted': {},
        'ml_baselines': {},
        'ranking': []
    }
    
    all_methods = []
    
    # Learned states
    for i, d in enumerate(d_values):
        name = f'Learned (d={d})'
        mse = learned_results['R_d'][i]
        comparison['learned'][name] = {'mse': mse, 'dim': d}
        all_methods.append((name, mse, 'learned', d))
    
    # Hand-crafted states
    for name, data in handcrafted_results.items():
        mse = data['mse']
        dim = data['dim']
        comparison['handcrafted'][name] = {'mse': mse, 'dim': dim}
        all_methods.append((name, mse, 'handcrafted', dim))
    
    # ML baselines
    for name, data in ml_baseline_results.items():
        mse = data['metrics']['mse_norm']
        comparison['ml_baselines'][name] = {'mse': mse, 'dim': 'full'}
        all_methods.append((name, mse, 'ml_baseline', 'full'))
    
    # Rank all methods
    all_methods.sort(key=lambda x: x[1])
    comparison['ranking'] = all_methods
    
    # Find best in each category
    comparison['best_learned'] = min(
        comparison['learned'].items(), 
        key=lambda x: x[1]['mse']
    )
    comparison['best_handcrafted'] = min(
        comparison['handcrafted'].items(),
        key=lambda x: x[1]['mse']
    ) if comparison['handcrafted'] else None
    comparison['best_ml_baseline'] = min(
        comparison['ml_baselines'].items(),
        key=lambda x: x[1]['mse']
    )
    
    return comparison
