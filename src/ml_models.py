"""
Machine Learning Models for Learned Aggregate States

This module implements the encoder-predictor architecture for learning
low-dimensional aggregate state representations from wealth distributions.

The framework follows the proposal:
    - Encoder f_θ: R^p → R^d maps high-dimensional features X_t to learned state Z_t
    - Predictor g_ψ: R^d × S → R^q predicts next-period aggregates Y_{t+1}
    
Key concept: R(d) = inf_{θ,ψ} E[||Y_{t+1} - g_ψ(f_θ(X_t), s_{t+1})||²]
is the minimal achievable prediction error for dimension d.

Author: Research Team
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """Configuration for training encoder-predictor models."""
    # Architecture
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    predictor_hidden_dims: List[int] = field(default_factory=lambda: [32, 16])
    activation: str = 'relu'
    dropout: float = 0.0
    
    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    max_epochs: int = 500
    patience: int = 30  # Early stopping patience
    min_delta: float = 1e-6  # Minimum improvement for early stopping
    
    # Misc
    seed: int = 42
    device: str = 'cpu'


class Encoder(nn.Module):
    """
    Encoder network f_θ: R^p → R^d
    
    Maps high-dimensional feature vector X_t (histogram + summary stats)
    to low-dimensional learned aggregate state Z_t.
    
    Parameters
    ----------
    input_dim : int
        Dimension of input features X_t
    latent_dim : int
        Dimension of learned state Z_t (this is d)
    hidden_dims : List[int]
        Hidden layer dimensions
    activation : str
        Activation function ('relu', 'tanh', 'elu')
    dropout : float
        Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = [64, 32],
        activation: str = 'relu',
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(self._get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        
        # Final layer to latent space (no activation)
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU(0.1)
        }
        return activations.get(name, nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: X_t → Z_t"""
        return self.network(x)


class Predictor(nn.Module):
    """
    Predictor network g_ψ: R^d × S → R^q
    
    Maps learned aggregate state Z_t and aggregate shock s_{t+1}
    to prediction of next-period aggregates Y_{t+1}.
    
    Parameters
    ----------
    latent_dim : int
        Dimension of learned state Z_t
    shock_dim : int
        Dimension of aggregate shocks s_{t+1}
    output_dim : int
        Dimension of output Y_{t+1} (typically 3: K, C, Y)
    hidden_dims : List[int]
        Hidden layer dimensions
    activation : str
        Activation function
    dropout : float
        Dropout probability
    """
    
    def __init__(
        self,
        latent_dim: int,
        shock_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [32, 16],
        activation: str = 'relu',
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.shock_dim = shock_dim
        self.output_dim = output_dim
        
        # Build network
        layers = []
        prev_dim = latent_dim + shock_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(self._get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        
        # Final layer to output (no activation - we want unbounded predictions)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU(0.1)
        }
        return activations.get(name, nn.ReLU())
    
    def forward(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Forward pass: (Z_t, s_{t+1}) → Ŷ_{t+1}"""
        # Concatenate latent state and shock
        zs = torch.cat([z, s], dim=-1)
        return self.network(zs)


class EncoderPredictor(nn.Module):
    """
    Combined encoder-predictor model.
    
    This is the main model for learning aggregate states:
        Ŷ_{t+1} = g_ψ(f_θ(X_t), s_{t+1})
    
    Parameters
    ----------
    input_dim : int
        Dimension of input features X_t
    latent_dim : int
        Dimension of learned state Z_t (this is d)
    shock_dim : int
        Dimension of aggregate shocks s_{t+1}
    output_dim : int
        Dimension of output Y_{t+1}
    config : TrainingConfig
        Training configuration
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        shock_dim: int = 1,
        output_dim: int = 3,
        config: Optional[TrainingConfig] = None
    ):
        super().__init__()
        
        if config is None:
            config = TrainingConfig()
        
        self.config = config
        self.latent_dim = latent_dim
        
        self.encoder = Encoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=config.encoder_hidden_dims,
            activation=config.activation,
            dropout=config.dropout
        )
        
        self.predictor = Predictor(
            latent_dim=latent_dim,
            shock_dim=shock_dim,
            output_dim=output_dim,
            hidden_dims=config.predictor_hidden_dims,
            activation=config.activation,
            dropout=config.dropout
        )
    
    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: (X_t, s_{t+1}) → (Z_t, Ŷ_{t+1})
        
        Returns both the learned state and prediction for analysis.
        """
        z = self.encoder(x)
        y_pred = self.predictor(z, s)
        return z, y_pred
    
    def predict(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Get only the prediction (for evaluation)."""
        _, y_pred = self.forward(x, s)
        return y_pred
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get only the learned state (for analysis)."""
        return self.encoder(x)


class DirectPredictor(nn.Module):
    """
    Direct predictor without bottleneck (baseline).
    
    Maps (X_t, s_{t+1}) directly to Y_{t+1} without going through
    a low-dimensional representation. Used as an upper bound on
    what's achievable with the full feature set.
    
    Parameters
    ----------
    input_dim : int
        Dimension of features X_t
    shock_dim : int
        Dimension of shocks s_{t+1}
    output_dim : int
        Dimension of output Y_{t+1}
    hidden_dims : List[int]
        Hidden layer dimensions
    """
    
    def __init__(
        self,
        input_dim: int,
        shock_dim: int = 1,
        output_dim: int = 3,
        hidden_dims: List[int] = [128, 64, 32],
        activation: str = 'relu',
        dropout: float = 0.0
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim + shock_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        xs = torch.cat([x, s], dim=-1)
        return self.network(xs)


class HandCraftedPredictor(nn.Module):
    """
    Predictor using hand-crafted aggregate states (Krusell-Smith style).
    
    Uses pre-specified summary statistics like (K_t) or (K_t, Gini_t)
    instead of learned representations.
    
    Parameters
    ----------
    state_dim : int
        Dimension of hand-crafted state
    shock_dim : int
        Dimension of shocks
    output_dim : int
        Dimension of output
    hidden_dims : List[int]
        Hidden layer dimensions
    """
    
    def __init__(
        self,
        state_dim: int,
        shock_dim: int = 1,
        output_dim: int = 3,
        hidden_dims: List[int] = [32, 16],
        activation: str = 'relu'
    ):
        super().__init__()
        
        layers = []
        prev_dim = state_dim + shock_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        xs = torch.cat([state, s], dim=-1)
        return self.network(xs)


# =============================================================================
# Training and Evaluation
# =============================================================================

class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 30, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


def train_encoder_predictor(
    model: nn.Module,
    X_train: np.ndarray,
    s_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    s_val: np.ndarray,
    Y_val: np.ndarray,
    config: TrainingConfig
) -> Dict[str, List[float]]:
    """
    Train encoder-predictor model.
    
    Parameters
    ----------
    model : nn.Module
        Model to train (EncoderPredictor or DirectPredictor)
    X_train, s_train, Y_train : np.ndarray
        Training data
    X_val, s_val, Y_val : np.ndarray
        Validation data
    config : TrainingConfig
        Training configuration
        
    Returns
    -------
    history : dict
        Training history with 'train_loss' and 'val_loss'
    """
    # Set seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    device = torch.device(config.device)
    model = model.to(device)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    s_train_t = torch.FloatTensor(s_train).reshape(-1, 1).to(device)
    Y_train_t = torch.FloatTensor(Y_train).to(device)
    
    X_val_t = torch.FloatTensor(X_val).to(device)
    s_val_t = torch.FloatTensor(s_val).reshape(-1, 1).to(device)
    Y_val_t = torch.FloatTensor(Y_val).to(device)
    
    # Create data loader
    train_dataset = TensorDataset(X_train_t, s_train_t, Y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    # Optimizer and loss
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.lr, 
        weight_decay=config.weight_decay
    )
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=False
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta)
    
    # Training history
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_state_dict = None
    
    for epoch in range(config.max_epochs):
        # Training
        model.train()
        train_losses = []
        
        for X_batch, s_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass (handle different model types)
            if isinstance(model, EncoderPredictor):
                _, Y_pred = model(X_batch, s_batch)
            else:
                Y_pred = model(X_batch, s_batch)
            
            loss = criterion(Y_pred, Y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        with torch.no_grad():
            if isinstance(model, EncoderPredictor):
                _, Y_val_pred = model(X_val_t, s_val_t)
            else:
                Y_val_pred = model(X_val_t, s_val_t)
            val_loss = criterion(Y_val_pred, Y_val_t).item()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if early_stopping(val_loss):
            break
    
    # Restore best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    
    return history


def evaluate_model(
    model: nn.Module,
    X_test: np.ndarray,
    s_test: np.ndarray,
    Y_test: np.ndarray,
    Y_mean: np.ndarray,
    Y_std: np.ndarray,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Evaluate model on test set.
    
    Computes:
    - MSE (normalized and original scale)
    - Relative MSE (normalized by variance)
    - R² for each output component
    
    Parameters
    ----------
    model : nn.Module
        Trained model
    X_test, s_test, Y_test : np.ndarray
        Test data (normalized)
    Y_mean, Y_std : np.ndarray
        Normalization parameters
    device : str
        Device for computation
        
    Returns
    -------
    metrics : dict
        Dictionary of evaluation metrics
    """
    model.eval()
    device = torch.device(device)
    model = model.to(device)
    
    X_test_t = torch.FloatTensor(X_test).to(device)
    s_test_t = torch.FloatTensor(s_test).reshape(-1, 1).to(device)
    
    with torch.no_grad():
        if isinstance(model, EncoderPredictor):
            _, Y_pred = model(X_test_t, s_test_t)
        else:
            Y_pred = model(X_test_t, s_test_t)
        Y_pred = Y_pred.cpu().numpy()
    
    # MSE on normalized scale
    mse_norm = np.mean((Y_pred - Y_test)**2)
    
    # Convert to original scale
    Y_pred_orig = Y_pred * Y_std + Y_mean
    Y_test_orig = Y_test * Y_std + Y_mean
    
    # MSE on original scale
    mse_orig = np.mean((Y_pred_orig - Y_test_orig)**2)
    
    # Variance of test targets (for relative MSE)
    var_test = np.var(Y_test, axis=0)
    var_test[var_test == 0] = 1  # Avoid division by zero
    
    # Relative MSE (proportion of variance unexplained)
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


def compute_R_d(
    X_train: np.ndarray,
    s_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    s_val: np.ndarray,
    Y_val: np.ndarray,
    X_test: np.ndarray,
    s_test: np.ndarray,
    Y_test: np.ndarray,
    Y_mean: np.ndarray,
    Y_std: np.ndarray,
    d_values: List[int] = [1, 2, 3, 4, 5],
    n_runs: int = 3,
    config: Optional[TrainingConfig] = None,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Compute R(d) = minimal achievable prediction error for different dimensions d.
    
    For each d, trains multiple models with different random seeds and returns
    the best (and distribution of) test errors.
    
    Parameters
    ----------
    X_train, s_train, Y_train : np.ndarray
        Training data (normalized)
    X_val, s_val, Y_val : np.ndarray
        Validation data (normalized)
    X_test, s_test, Y_test : np.ndarray
        Test data (normalized)
    Y_mean, Y_std : np.ndarray
        Normalization parameters
    d_values : List[int]
        Dimensions to evaluate
    n_runs : int
        Number of random seeds per dimension
    config : TrainingConfig
        Training configuration
    verbose : bool
        Print progress
        
    Returns
    -------
    results : dict
        Dictionary with R(d) values and other metrics
    """
    if config is None:
        config = TrainingConfig()
    
    input_dim = X_train.shape[1]
    shock_dim = 1
    output_dim = Y_train.shape[1]
    
    results = {
        'd': np.array(d_values),
        'R_d': np.zeros(len(d_values)),
        'R_d_std': np.zeros(len(d_values)),
        'r2_mean': np.zeros(len(d_values)),
        'all_runs': []
    }
    
    for i, d in enumerate(d_values):
        if verbose:
            print(f"\nTraining models with d = {d}...")
        
        run_metrics = []
        best_mse = float('inf')
        
        for run in range(n_runs):
            # Create model
            config_run = TrainingConfig(
                **{k: v for k, v in config.__dict__.items()},
            )
            config_run.seed = config.seed + run * 100
            
            model = EncoderPredictor(
                input_dim=input_dim,
                latent_dim=d,
                shock_dim=shock_dim,
                output_dim=output_dim,
                config=config_run
            )
            
            # Train
            history = train_encoder_predictor(
                model, X_train, s_train, Y_train,
                X_val, s_val, Y_val, config_run
            )
            
            # Evaluate
            metrics = evaluate_model(
                model, X_test, s_test, Y_test, Y_mean, Y_std, config.device
            )
            
            run_metrics.append(metrics)
            
            if metrics['mse_norm'] < best_mse:
                best_mse = metrics['mse_norm']
            
            if verbose:
                print(f"  Run {run+1}/{n_runs}: MSE = {metrics['mse_norm']:.6f}, "
                      f"R² = {metrics['r2_mean']:.4f}")
        
        # Aggregate results
        mses = [m['mse_norm'] for m in run_metrics]
        r2s = [m['r2_mean'] for m in run_metrics]
        
        results['R_d'][i] = np.min(mses)  # Best achievable
        results['R_d_std'][i] = np.std(mses)
        results['r2_mean'][i] = np.max(r2s)
        results['all_runs'].append(run_metrics)
        
        if verbose:
            print(f"  → R({d}) = {results['R_d'][i]:.6f} (std: {results['R_d_std'][i]:.6f})")
    
    return results


def extract_learned_states(
    model: EncoderPredictor,
    X: np.ndarray,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Extract learned aggregate states Z_t from features X_t.
    
    Parameters
    ----------
    model : EncoderPredictor
        Trained encoder-predictor model
    X : np.ndarray
        Feature matrix (T x p)
    device : str
        Device for computation
        
    Returns
    -------
    Z : np.ndarray
        Learned states (T x d)
    """
    model.eval()
    device = torch.device(device)
    model = model.to(device)
    
    X_t = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        Z = model.encode(X_t)
    
    return Z.cpu().numpy()


# =============================================================================
# Statistical Inference for Publication
# =============================================================================

def bootstrap_confidence_interval(
    model: nn.Module,
    X_test: np.ndarray,
    s_test: np.ndarray,
    Y_test: np.ndarray,
    Y_mean: np.ndarray,
    Y_std: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    device: str = 'cpu'
) -> Dict[str, np.ndarray]:
    """
    Compute bootstrap confidence intervals for model performance metrics.
    
    This is essential for publication-quality results to show uncertainty
    in the R(d) estimates.
    
    Parameters
    ----------
    model : nn.Module
        Trained model
    X_test, s_test, Y_test : np.ndarray
        Test data (normalized)
    Y_mean, Y_std : np.ndarray
        Normalization parameters
    n_bootstrap : int
        Number of bootstrap samples
    alpha : float
        Significance level (default 0.05 for 95% CI)
    device : str
        Device for computation
        
    Returns
    -------
    results : dict
        Bootstrap results with point estimates and confidence intervals
    """
    model.eval()
    device_t = torch.device(device)
    model = model.to(device_t)
    
    n_test = len(X_test)
    
    # Get predictions
    X_test_t = torch.FloatTensor(X_test).to(device_t)
    s_test_t = torch.FloatTensor(s_test).reshape(-1, 1).to(device_t)
    
    with torch.no_grad():
        if isinstance(model, EncoderPredictor):
            _, Y_pred = model(X_test_t, s_test_t)
        else:
            Y_pred = model(X_test_t, s_test_t)
        Y_pred = Y_pred.cpu().numpy()
    
    # Bootstrap resampling
    mse_samples = np.zeros(n_bootstrap)
    r2_samples = np.zeros((n_bootstrap, 3))  # For each component
    
    for b in range(n_bootstrap):
        # Sample with replacement
        idx = np.random.choice(n_test, size=n_test, replace=True)
        Y_test_b = Y_test[idx]
        Y_pred_b = Y_pred[idx]
        
        # Compute MSE
        mse_samples[b] = np.mean((Y_pred_b - Y_test_b)**2)
        
        # Compute R² for each component
        for k in range(3):
            ss_res = np.sum((Y_test_b[:, k] - Y_pred_b[:, k])**2)
            ss_tot = np.sum((Y_test_b[:, k] - Y_test_b[:, k].mean())**2)
            r2_samples[b, k] = 1 - ss_res / max(ss_tot, 1e-10)
    
    # Compute point estimates
    mse_point = np.mean((Y_pred - Y_test)**2)
    ss_res = np.sum((Y_test - Y_pred)**2, axis=0)
    ss_tot = np.sum((Y_test - Y_test.mean(axis=0))**2, axis=0)
    r2_point = 1 - ss_res / np.maximum(ss_tot, 1e-10)
    
    # Compute confidence intervals
    mse_ci = np.percentile(mse_samples, [100*alpha/2, 100*(1-alpha/2)])
    r2_ci = np.percentile(r2_samples, [100*alpha/2, 100*(1-alpha/2)], axis=0)
    
    return {
        'mse': mse_point,
        'mse_ci': mse_ci,
        'mse_std': np.std(mse_samples),
        'r2': r2_point,
        'r2_mean': r2_point.mean(),
        'r2_ci': r2_ci,
        'n_bootstrap': n_bootstrap,
        'alpha': alpha
    }


def compute_R_d_with_ci(
    X_train: np.ndarray,
    s_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    s_val: np.ndarray,
    Y_val: np.ndarray,
    X_test: np.ndarray,
    s_test: np.ndarray,
    Y_test: np.ndarray,
    Y_mean: np.ndarray,
    Y_std: np.ndarray,
    d_values: List[int] = [1, 2, 3],
    n_runs: int = 5,
    n_bootstrap: int = 500,
    config: Optional[TrainingConfig] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compute R(d) with bootstrap confidence intervals.
    
    Publication-quality version that reports:
    - R(d) point estimates (best across runs)
    - R(d) confidence intervals via bootstrap
    - R² for each target component with CIs
    
    Parameters
    ----------
    d_values : List[int]
        Dimensions to evaluate
    n_runs : int
        Number of random seeds per dimension (more = more robust)
    n_bootstrap : int
        Number of bootstrap samples for CIs
    config : TrainingConfig
        Training configuration
    verbose : bool
        Print progress
        
    Returns
    -------
    results : dict
        Comprehensive results with CIs
    """
    if config is None:
        config = TrainingConfig()
    
    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]
    
    results = {
        'd': np.array(d_values),
        'R_d': np.zeros(len(d_values)),
        'R_d_std': np.zeros(len(d_values)),
        'R_d_ci_low': np.zeros(len(d_values)),
        'R_d_ci_high': np.zeros(len(d_values)),
        'r2_K': np.zeros(len(d_values)),
        'r2_C': np.zeros(len(d_values)),
        'r2_Y': np.zeros(len(d_values)),
        'r2_mean': np.zeros(len(d_values)),
        'best_models': []
    }
    
    for i, d in enumerate(d_values):
        if verbose:
            print(f"\nTraining models with d = {d}...")
        
        best_mse = float('inf')
        best_model = None
        run_mses = []
        
        for run in range(n_runs):
            # Create model with different seed
            config_run = TrainingConfig(
                **{k: v for k, v in config.__dict__.items()},
            )
            config_run.seed = config.seed + run * 100
            
            model = EncoderPredictor(
                input_dim=input_dim,
                latent_dim=d,
                shock_dim=1,
                output_dim=output_dim,
                config=config_run
            )
            
            # Train
            _ = train_encoder_predictor(
                model, X_train, s_train, Y_train,
                X_val, s_val, Y_val, config_run
            )
            
            # Quick evaluation
            metrics = evaluate_model(
                model, X_test, s_test, Y_test, Y_mean, Y_std, config.device
            )
            
            run_mses.append(metrics['mse_norm'])
            
            if metrics['mse_norm'] < best_mse:
                best_mse = metrics['mse_norm']
                best_model = model
            
            if verbose:
                print(f"  Run {run+1}/{n_runs}: MSE = {metrics['mse_norm']:.6f}, "
                      f"R² = {metrics['r2_mean']:.4f}")
        
        # Bootstrap CI for best model
        if verbose:
            print(f"  Computing bootstrap CIs ({n_bootstrap} samples)...")
        
        boot_results = bootstrap_confidence_interval(
            best_model, X_test, s_test, Y_test, Y_mean, Y_std,
            n_bootstrap=n_bootstrap, device=config.device
        )
        
        # Store results
        results['R_d'][i] = boot_results['mse']
        results['R_d_std'][i] = np.std(run_mses)
        results['R_d_ci_low'][i] = boot_results['mse_ci'][0]
        results['R_d_ci_high'][i] = boot_results['mse_ci'][1]
        results['r2_K'][i] = boot_results['r2'][0]
        results['r2_C'][i] = boot_results['r2'][1]
        results['r2_Y'][i] = boot_results['r2'][2]
        results['r2_mean'][i] = boot_results['r2_mean']
        results['best_models'].append(best_model)
        
        if verbose:
            print(f"  → R({d}) = {results['R_d'][i]:.6f} "
                  f"[{results['R_d_ci_low'][i]:.6f}, {results['R_d_ci_high'][i]:.6f}]")
            print(f"  → R²: K={results['r2_K'][i]:.4f}, C={results['r2_C'][i]:.4f}, "
                  f"Y={results['r2_Y'][i]:.4f}")
    
    return results
