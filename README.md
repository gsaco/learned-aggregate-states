# Learned Aggregate States in Heterogeneous-Agent Models

**A Unified Macro–Statistics–ML–Computation Blueprint**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## Overview

This project investigates approximate aggregation in heterogeneous-agent macroeconomic models. We ask three central questions:

1. **How many aggregate factors** does the economy actually need for accurate prediction of macro aggregates?
2. **What do those factors represent economically** (capital level, inequality, borrowing constraint mass, etc.)?
3. **What are the macro and policy costs** of using too few aggregate state variables?

### Key Contribution

We produce a **"map of approximate aggregation"** showing where traditional Krusell-Smith ($d=1$) approximations are sufficient versus where richer state representations are required.

## Method

We learn a low-dimensional representation $Z_t = f_\theta(X_t)$ from the wealth distribution features $X_t$, then predict future aggregates:

$$\hat{Y}_{t+1} = g_\psi(Z_t, s_{t+1})$$

The minimal achievable error $R(d)$ as a function of latent dimension $d$ answers: *"How many factors does the economy need?"*

## Project Structure

```
learned-aggregate-states/
├── src/                              # Python modules
│   ├── __init__.py                   # Package initialization
│   ├── aiyagari.py                   # Aiyagari model (EGM solver, TFP shocks)
│   ├── ml_models.py                  # Encoder-predictor neural networks
│   ├── baselines.py                  # ML baselines (Linear, RF, GB)
│   ├── interpretation.py             # Correlation & regression analysis
│   └── utils.py                      # Data processing, features, visualization
├── phase1_2_aiyagari.ipynb           # Phase 1+2: Model, simulation, first ML experiments
├── phase3_4_map_and_baselines.ipynb  # Phase 3+4+5: Grid, baselines, interpretation, TFP
├── results/                          # Saved results and tables
│   ├── table1_R_d_by_config.csv      # R(d) across configurations
│   ├── table3_method_comparison.csv  # All methods comparison
│   └── phase3_4_results.pkl          # Full results pickle
├── proposal.tex                      # Research proposal
└── README.md                         # This file
```

## Key Results

### 1. Map of Approximate Aggregation

We evaluate $\hat{R}_j(d)$ across 8 economic configurations varying:
- **Idiosyncratic risk** ($\sigma_e$): 0.1 vs 0.25
- **Persistence** ($\rho_e$): 0.85 vs 0.95  
- **Borrowing constraint** ($a_{\min}$): -1.0 vs 0.0

**Finding**: Effective dimension $d^*(\varepsilon)$ varies systematically:
- Low risk + moderate persistence → $d=1$ often sufficient
- High risk or high persistence → $d \geq 2$ required for same tolerance

### 2. Learned States Outperform Hand-Crafted

For the same dimension, learned representations achieve **10-40% lower MSE** than hand-crafted states (K, K+Gini, K+Gini+Top10).

### 3. Interpretable Learned Factors

Correlation and regression analysis reveals:
- $Z_{t,1}$ strongly tracks aggregate capital $K_t$ (level state)
- $Z_{t,2}$ captures inequality (Gini) and/or constraint mass
- $Z_{t,3}$ picks up higher moments and tail behavior

### 4. Aggregate TFP Shocks

With aggregate TFP shocks ($s_{t+1} = Z_{t+1}$):
- The prediction task becomes harder
- Distributional information becomes MORE valuable
- Validates the framework for realistic business cycle analysis

## Modules

### `src/aiyagari.py`
- `AiyagariParams`: Dataclass with all model parameters (including TFP shocks)
- `compute_equilibrium()`: Solve for stationary equilibrium via EGM
- `run_simulation()`: Simulate panel without aggregate shocks
- `run_simulation_with_shocks()`: Simulate panel with aggregate TFP shocks

### `src/ml_models.py`
- `Encoder`: Neural network $f_\theta: \mathbb{R}^p \to \mathbb{R}^d$
- `Predictor`: Neural network $g_\psi: \mathbb{R}^d \times \mathcal{S} \to \mathbb{R}^q$
- `EncoderPredictor`: Combined model for learning aggregate states
- `compute_R_d()`: Compute minimal predictive risk vs. dimension
- `compute_R_d_with_ci()`: Publication-quality version with bootstrap CIs
- `HandCraftedPredictor`, `DirectPredictor`: Baseline models

### `src/baselines.py`
- `LinearBaseline`: Ridge regression on full features
- `RandomForestBaseline`: Random Forest ensemble
- `GradientBoostingBaseline`: Gradient boosted trees
- `train_all_ml_baselines()`: Train and evaluate all baselines

### `src/interpretation.py`
- `compute_correlation_matrix()`: Z vs. economic statistics
- `regression_interpretation()`: $Z_{t,k} \approx \alpha_k + \sum_m \beta_{k,m} M_{t,m}$
- `label_learned_factors()`: Automatic economic labeling
- Visualization functions for heatmaps and comparisons

### `src/utils.py`
- `compute_features_batch()`: Fast feature extraction with Gini handling for negative wealth
- `build_dataset()`: Build ML dataset with optional TFP shocks
- `prepare_ml_dataset()`: Chronological split and normalization
- `extract_hand_crafted_states()`: K, K+Gini, K+Gini+Top10

## Installation

```bash
# Clone repository
git clone https://github.com/gsaco/learned-aggregate-states.git
cd learned-aggregate-states

# Install dependencies
pip install numpy numba scipy matplotlib seaborn pandas torch scikit-learn
```

## Quick Start

```python
from src.aiyagari import AiyagariParams, compute_equilibrium, run_simulation
from src.ml_models import TrainingConfig, compute_R_d
from src.utils import build_dataset, prepare_ml_dataset

# 1. Set parameters
params = AiyagariParams(
    beta=0.96, sigma=2.0, alpha=0.36, delta=0.08,
    rho_e=0.9, sigma_e=0.2, a_min=0.0
)

# 2. Solve equilibrium
equilibrium = compute_equilibrium(params, verbose=True)

# 3. Simulate panel
panel = run_simulation(equilibrium, N_agents=10000, T_sim=2000, T_burn=500)

# 4. Build dataset
dataset = build_dataset(
    panel['a_panel'], panel['e_panel'],
    equilibrium['c_policy'], equilibrium['a_policy'],
    equilibrium['a_grid'], equilibrium['e_grid'],
    alpha=params.alpha
)
ml_data = prepare_ml_dataset(dataset)

# 5. Compute R(d) curve
config = TrainingConfig(device='cuda' if torch.cuda.is_available() else 'cpu')
results = compute_R_d(
    ml_data['X_train_norm'], ml_data['s_train'], ml_data['Y_train_norm'],
    ml_data['X_val_norm'], ml_data['s_val'], ml_data['Y_val_norm'],
    ml_data['X_test_norm'], ml_data['s_test'], ml_data['Y_test_norm'],
    ml_data['Y_mean'], ml_data['Y_std'],
    d_values=[1, 2, 3], config=config
)

print(f"R(d) curve: {dict(zip(results['d'], results['R_d']))}")
```

## Citation

If you use this code, please cite:

```bibtex
@misc{learned-aggregate-states-2024,
  author = {Research Team},
  title = {Learned Aggregate States in Heterogeneous-Agent Models},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/gsaco/learned-aggregate-states}
}
```

## License

MIT License

## References

- Aiyagari, S. R. (1994). "Uninsured Idiosyncratic Risk and Aggregate Saving." *QJE*.
- Krusell, P., & Smith, A. A. (1998). "Income and Wealth Heterogeneity in the Macroeconomy." *JPE*.