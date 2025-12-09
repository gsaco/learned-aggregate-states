"""
Learned Aggregate States Package

This package implements the research framework for learning low-dimensional
aggregate state representations in heterogeneous-agent macro models.

Modules:
    - aiyagari: Core Aiyagari model implementation
    - ml_models: Encoder-predictor neural network architectures
    - utils: Data processing, feature extraction, and visualization
"""

from .aiyagari import (
    AiyagariParams,
    rouwenhorst,
    solve_household,
    stationary_distribution,
    compute_equilibrium,
    simulate_panel,
    run_simulation,
    r_from_K,
    w_from_K,
    Y_from_K
)

from .ml_models import (
    TrainingConfig,
    Encoder,
    Predictor,
    EncoderPredictor,
    DirectPredictor,
    HandCraftedPredictor,
    train_encoder_predictor,
    evaluate_model,
    compute_R_d,
    extract_learned_states
)

from .utils import (
    compute_features,
    compute_aggregates,
    build_dataset,
    prepare_ml_dataset,
    extract_hand_crafted_states,
    plot_R_d_curve,
    plot_training_history,
    plot_learned_states_interpretation,
    get_feature_names,
    get_summary_stat_names
)

from .baselines import (
    BaselineConfig,
    LinearBaseline,
    RandomForestBaseline,
    GradientBoostingBaseline,
    train_all_ml_baselines,
    compare_all_methods
)

from .interpretation import (
    compute_correlation_matrix,
    regression_interpretation,
    label_learned_factors,
    plot_correlation_heatmap,
    plot_regression_interpretation,
    plot_learned_vs_handcrafted,
    plot_method_comparison_bar,
    create_interpretation_summary,
    extract_summary_stats_from_X,
    get_economic_stat_names
)

__version__ = '0.1.0'
__author__ = 'Research Team'
