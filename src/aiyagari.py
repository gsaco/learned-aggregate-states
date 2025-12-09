"""
Aiyagari Model Implementation

This module implements the standard one-asset Aiyagari (1994) model
for studying heterogeneous-agent macroeconomics.

Reference:
    Aiyagari, S. R. (1994). Uninsured Idiosyncratic Risk and Aggregate Saving.
    The Quarterly Journal of Economics, 109(3), 659-684.

Author: Research Team
"""

import numpy as np
from numba import njit, prange
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class AiyagariParams:
    """Parameters for the Aiyagari model."""
    # Preferences
    beta: float = 0.96       # Discount factor
    sigma: float = 2.0       # CRRA coefficient
    
    # Production (Cobb-Douglas)
    alpha: float = 0.36      # Capital share
    delta: float = 0.08      # Depreciation rate
    
    # Borrowing constraint
    a_min: float = 0.0       # Lower bound on assets (no borrowing)
    
    # Asset grid
    a_max: float = 50.0      # Upper bound on assets
    n_a: int = 200           # Number of asset grid points
    
    # Idiosyncratic productivity (AR(1))
    rho_e: float = 0.9       # Persistence
    sigma_e: float = 0.2     # Conditional std
    n_e: int = 7             # Number of productivity states
    
    # Aggregate TFP shocks (AR(1))
    # log(Z') = rho_z * log(Z) + eps_z, eps_z ~ N(0, sigma_z^2)
    rho_z: float = 0.9       # TFP persistence
    sigma_z: float = 0.0     # TFP shock std (0 = no aggregate shocks)
    n_z: int = 3             # Number of TFP states (if using discrete)


def rouwenhorst(n: int, rho: float, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rouwenhorst method for discretizing AR(1) process.
    
    Discretizes: e' = rho * e + eps, where eps ~ N(0, sigma^2 * (1 - rho^2))
    
    Parameters
    ----------
    n : int
        Number of states
    rho : float
        Persistence parameter
    sigma : float
        Conditional standard deviation
        
    Returns
    -------
    e_grid : np.ndarray
        Grid of productivity states (in levels, i.e., exp(e))
    Pi : np.ndarray
        Transition matrix (n x n)
    """
    sigma_unconditional = sigma / np.sqrt(1 - rho**2)
    e_max = sigma_unconditional * np.sqrt(n - 1)
    e_grid = np.linspace(-e_max, e_max, n)
    
    p = (1 + rho) / 2
    q = p
    
    Pi = np.array([[p, 1-p], [1-q, q]])
    
    for i in range(3, n + 1):
        Pi_new = np.zeros((i, i))
        Pi_new[:i-1, :i-1] += p * Pi
        Pi_new[:i-1, 1:i] += (1-p) * Pi
        Pi_new[1:i, :i-1] += (1-q) * Pi
        Pi_new[1:i, 1:i] += q * Pi
        Pi_new[1:i-1, :] /= 2
        Pi = Pi_new
    
    return np.exp(e_grid), Pi


def stationary_distribution_markov(Pi: np.ndarray) -> np.ndarray:
    """
    Compute stationary distribution of a Markov chain.
    
    Parameters
    ----------
    Pi : np.ndarray
        Transition matrix (rows sum to 1)
        
    Returns
    -------
    stat_dist : np.ndarray
        Stationary distribution
    """
    eigvals, eigvecs = np.linalg.eig(Pi.T)
    stat_dist = eigvecs[:, np.argmax(np.abs(eigvals))].real
    stat_dist = stat_dist / stat_dist.sum()
    return stat_dist


# =============================================================================
# CRRA Utility Functions (JIT-compiled)
# =============================================================================

@njit
def u(c: float, sigma: float) -> float:
    """CRRA utility function."""
    if sigma == 1.0:
        return np.log(c)
    else:
        return (c**(1 - sigma) - 1) / (1 - sigma)


@njit
def u_prime(c: float, sigma: float) -> float:
    """Marginal utility of consumption."""
    return c**(-sigma)


@njit
def u_prime_inv(m: float, sigma: float) -> float:
    """Inverse of marginal utility."""
    return m**(-1/sigma)


# =============================================================================
# Factor Prices and Aggregates
# =============================================================================

def r_from_K(K: float, L: float, alpha: float, delta: float) -> float:
    """Compute interest rate from capital stock."""
    return alpha * (K / L)**(alpha - 1) - delta


def w_from_K(K: float, L: float, alpha: float) -> float:
    """Compute wage from capital stock."""
    return (1 - alpha) * (K / L)**alpha


def Y_from_K(K: float, L: float, alpha: float) -> float:
    """Compute output from capital stock (Cobb-Douglas)."""
    return K**alpha * L**(1 - alpha)


# =============================================================================
# Household Problem: Endogenous Grid Method (EGM)
# =============================================================================

@njit
def egm_step(
    c_next: np.ndarray,
    a_grid: np.ndarray,
    e_grid: np.ndarray,
    Pi: np.ndarray,
    r: float,
    w: float,
    beta: float,
    sigma: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One step of EGM iteration.
    
    Parameters
    ----------
    c_next : np.ndarray
        Consumption policy from next iteration (n_a x n_e)
    a_grid : np.ndarray
        Asset grid
    e_grid : np.ndarray
        Productivity grid
    Pi : np.ndarray
        Transition matrix
    r : float
        Interest rate
    w : float
        Wage
    beta : float
        Discount factor
    sigma : float
        CRRA coefficient
        
    Returns
    -------
    c_new : np.ndarray
        Updated consumption policy (n_a x n_e)
    a_policy : np.ndarray
        Asset policy function (n_a x n_e)
    """
    n_a = len(a_grid)
    n_e = len(e_grid)
    
    c_new = np.zeros((n_a, n_e))
    a_policy = np.zeros((n_a, n_e))
    
    for ie in range(n_e):
        # Expected marginal utility tomorrow
        Eu_prime = np.zeros(n_a)
        for ie_next in range(n_e):
            Eu_prime += Pi[ie, ie_next] * u_prime(c_next[:, ie_next], sigma)
        
        # Consumption from Euler equation
        c_endo = u_prime_inv(beta * (1 + r) * Eu_prime, sigma)
        
        # Endogenous asset grid
        a_endo = (c_endo + a_grid - w * e_grid[ie]) / (1 + r)
        
        # Interpolate back to exogenous grid
        for ia in range(n_a):
            a = a_grid[ia]
            if a <= a_endo[0]:
                # Borrowing constraint binds
                c_new[ia, ie] = (1 + r) * a + w * e_grid[ie] - a_grid[0]
                a_policy[ia, ie] = a_grid[0]
            elif a >= a_endo[-1]:
                c_new[ia, ie] = c_endo[-1] + (1 + r) * (a - a_endo[-1])
                a_policy[ia, ie] = a_grid[-1]
            else:
                # Linear interpolation
                idx = np.searchsorted(a_endo, a)
                weight = (a - a_endo[idx-1]) / (a_endo[idx] - a_endo[idx-1])
                c_new[ia, ie] = c_endo[idx-1] + weight * (c_endo[idx] - c_endo[idx-1])
                a_policy[ia, ie] = a_grid[idx-1] + weight * (a_grid[idx] - a_grid[idx-1])
    
    return c_new, a_policy


def solve_household(
    a_grid: np.ndarray,
    e_grid: np.ndarray,
    Pi: np.ndarray,
    r: float,
    w: float,
    beta: float,
    sigma: float,
    tol: float = 1e-8,
    max_iter: int = 10000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve household problem via EGM iteration.
    
    Parameters
    ----------
    a_grid : np.ndarray
        Asset grid
    e_grid : np.ndarray
        Productivity grid
    Pi : np.ndarray
        Transition matrix
    r : float
        Interest rate
    w : float
        Wage
    beta : float
        Discount factor
    sigma : float
        CRRA coefficient
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations
        
    Returns
    -------
    c_policy : np.ndarray
        Optimal consumption policy (n_a x n_e)
    a_policy : np.ndarray
        Optimal asset policy (n_a x n_e)
    """
    n_a, n_e = len(a_grid), len(e_grid)
    
    # Initialize with consuming all income
    c = np.zeros((n_a, n_e))
    for ie in range(n_e):
        c[:, ie] = r * a_grid + w * e_grid[ie] + 0.1
    
    for it in range(max_iter):
        c_new, a_policy = egm_step(c, a_grid, e_grid, Pi, r, w, beta, sigma)
        diff = np.max(np.abs(c_new - c))
        c = c_new
        if diff < tol:
            break
    
    return c, a_policy


# =============================================================================
# Stationary Distribution
# =============================================================================

@njit
def get_lottery_weights(
    a_policy: np.ndarray,
    a_grid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute lottery weights for asset policy on grid (numba-accelerated)."""
    n_a, n_e = a_policy.shape
    idx_low = np.zeros((n_a, n_e), dtype=np.int64)
    weight_low = np.zeros((n_a, n_e))
    
    for ie in range(n_e):
        for ia in range(n_a):
            a_next = a_policy[ia, ie]
            # Binary search
            idx = np.searchsorted(a_grid, a_next)
            idx = max(1, min(idx, n_a - 1))
            idx_low[ia, ie] = idx - 1
            denom = a_grid[idx] - a_grid[idx-1]
            if denom > 0:
                weight_low[ia, ie] = (a_grid[idx] - a_next) / denom
            else:
                weight_low[ia, ie] = 0.5
            weight_low[ia, ie] = max(0.0, min(1.0, weight_low[ia, ie]))
    
    return idx_low, weight_low


@njit
def _iterate_distribution(
    mu: np.ndarray,
    idx_low: np.ndarray,
    weight_low: np.ndarray,
    Pi: np.ndarray,
    tol: float,
    max_iter: int
) -> np.ndarray:
    """
    Numba-accelerated distribution iteration.
    
    This is the performance-critical inner loop for computing
    the stationary distribution.
    """
    n_a, n_e = mu.shape
    
    for it in range(max_iter):
        mu_new = np.zeros((n_a, n_e))
        
        for ie in range(n_e):
            for ia in range(n_a):
                mass = mu[ia, ie]
                if mass > 1e-15:  # Skip negligible mass
                    il = idx_low[ia, ie]
                    wl = weight_low[ia, ie]
                    for ie_next in range(n_e):
                        prob = Pi[ie, ie_next]
                        mu_new[il, ie_next] += mass * wl * prob
                        if il + 1 < n_a:
                            mu_new[il + 1, ie_next] += mass * (1.0 - wl) * prob
        
        # Check convergence
        diff = 0.0
        for ie in range(n_e):
            for ia in range(n_a):
                d = abs(mu_new[ia, ie] - mu[ia, ie])
                if d > diff:
                    diff = d
        
        # Copy new to old
        for ie in range(n_e):
            for ia in range(n_a):
                mu[ia, ie] = mu_new[ia, ie]
        
        if diff < tol:
            break
    
    return mu


def stationary_distribution(
    a_policy: np.ndarray,
    a_grid: np.ndarray,
    Pi: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 10000
) -> np.ndarray:
    """
    Compute stationary distribution via iteration (FAST numba version).
    
    Parameters
    ----------
    a_policy : np.ndarray
        Asset policy function (n_a x n_e)
    a_grid : np.ndarray
        Asset grid
    Pi : np.ndarray
        Productivity transition matrix
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations
        
    Returns
    -------
    mu : np.ndarray
        Stationary distribution (n_a x n_e)
    """
    n_a, n_e = len(a_grid), Pi.shape[0]
    idx_low, weight_low = get_lottery_weights(a_policy, a_grid)
    
    # Initialize uniform
    mu = np.ones((n_a, n_e)) / (n_a * n_e)
    
    # Use numba-accelerated iteration
    mu = _iterate_distribution(mu, idx_low, weight_low, Pi, tol, max_iter)
    
    return mu


# =============================================================================
# General Equilibrium
# =============================================================================

def compute_equilibrium(
    params: AiyagariParams,
    K_init: float = 5.0,
    tol: float = 1e-5,
    max_iter: int = 100,
    damping: float = 0.3,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Find stationary equilibrium via iteration on aggregate capital K.
    
    Parameters
    ----------
    params : AiyagariParams
        Model parameters
    K_init : float
        Initial guess for capital
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations
    damping : float
        Damping parameter for updating K
    verbose : bool
        Print progress
        
    Returns
    -------
    result : dict
        Dictionary with equilibrium objects
    """
    # Set up grids
    a_grid = np.linspace(params.a_min, params.a_max, params.n_a)
    e_grid, Pi = rouwenhorst(params.n_e, params.rho_e, params.sigma_e)
    
    # Stationary distribution of productivity
    stat_dist_e = stationary_distribution_markov(Pi)
    L_ss = np.dot(stat_dist_e, e_grid)
    
    K = K_init
    
    for it in range(max_iter):
        r = r_from_K(K, L_ss, params.alpha, params.delta)
        w = w_from_K(K, L_ss, params.alpha)
        
        # Solve household problem
        c_policy, a_policy = solve_household(
            a_grid, e_grid, Pi, r, w, params.beta, params.sigma
        )
        
        # Compute stationary distribution
        mu = stationary_distribution(a_policy, a_grid, Pi)
        
        # Aggregate capital supply
        K_supply = np.sum(mu * a_grid[:, None])
        
        diff = np.abs(K_supply - K)
        if verbose:
            print(f"Iter {it+1}: K_demand={K:.4f}, K_supply={K_supply:.4f}, "
                  f"diff={diff:.6f}, r={r:.4f}")
        
        if diff < tol:
            if verbose:
                print("Equilibrium found!")
            break
        
        K = damping * K_supply + (1 - damping) * K
    
    # Compute aggregates
    K_ss = K_supply
    Y_ss = Y_from_K(K_ss, L_ss, params.alpha)
    C_ss = np.sum(mu * c_policy)
    r_ss = r_from_K(K_ss, L_ss, params.alpha, params.delta)
    w_ss = w_from_K(K_ss, L_ss, params.alpha)
    
    return {
        'K_ss': K_ss,
        'Y_ss': Y_ss,
        'C_ss': C_ss,
        'L_ss': L_ss,
        'r_ss': r_ss,
        'w_ss': w_ss,
        'c_policy': c_policy,
        'a_policy': a_policy,
        'mu_ss': mu,
        'a_grid': a_grid,
        'e_grid': e_grid,
        'Pi': Pi,
        'stat_dist_e': stat_dist_e,
        'params': params
    }


# =============================================================================
# Panel Simulation
# =============================================================================

@njit(parallel=True)
def simulate_panel(
    a_policy: np.ndarray,
    a_grid: np.ndarray,
    e_grid: np.ndarray,
    Pi: np.ndarray,
    N: int,
    T: int,
    a_init: np.ndarray,
    e_init_idx: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate panel of N agents for T periods.
    
    Parameters
    ----------
    a_policy : np.ndarray
        Asset policy function
    a_grid : np.ndarray
        Asset grid
    e_grid : np.ndarray
        Productivity grid
    Pi : np.ndarray
        Transition matrix
    N : int
        Number of agents
    T : int
        Number of periods
    a_init : np.ndarray
        Initial assets for each agent
    e_init_idx : np.ndarray
        Initial productivity index for each agent
        
    Returns
    -------
    a_sim : np.ndarray
        Simulated asset panel (N x T)
    e_sim : np.ndarray
        Simulated productivity panel (N x T)
    e_idx_sim : np.ndarray
        Simulated productivity index panel (N x T)
    """
    n_a = len(a_grid)
    n_e = len(e_grid)
    
    # Storage
    a_sim = np.zeros((N, T))
    e_sim = np.zeros((N, T))
    e_idx_sim = np.zeros((N, T), dtype=np.int64)
    
    # Cumulative transition probabilities for sampling
    Pi_cumsum = np.zeros((n_e, n_e))
    for ie in range(n_e):
        Pi_cumsum[ie, 0] = Pi[ie, 0]
        for ie_next in range(1, n_e):
            Pi_cumsum[ie, ie_next] = Pi_cumsum[ie, ie_next-1] + Pi[ie, ie_next]
    
    for i in prange(N):
        a = a_init[i]
        ie = e_init_idx[i]
        
        for t in range(T):
            a_sim[i, t] = a
            e_sim[i, t] = e_grid[ie]
            e_idx_sim[i, t] = ie
            
            # Interpolate policy
            idx = np.searchsorted(a_grid, a)
            idx = max(1, min(idx, n_a - 1))
            weight = (a - a_grid[idx-1]) / (a_grid[idx] - a_grid[idx-1])
            a_next = (1 - weight) * a_policy[idx-1, ie] + weight * a_policy[idx, ie]
            a_next = max(a_grid[0], min(a_grid[-1], a_next))
            
            # Draw next productivity
            u = np.random.random()
            ie_next = 0
            for k in range(n_e):
                if u <= Pi_cumsum[ie, k]:
                    ie_next = k
                    break
            
            a = a_next
            ie = ie_next
    
    return a_sim, e_sim, e_idx_sim


def run_simulation(
    equilibrium: Dict[str, Any],
    N_agents: int = 10000,
    T_sim: int = 2000,
    T_burn: int = 500,
    seed: int = 123
) -> Dict[str, np.ndarray]:
    """
    Run panel simulation from equilibrium.
    
    Parameters
    ----------
    equilibrium : dict
        Output from compute_equilibrium
    N_agents : int
        Number of agents
    T_sim : int
        Total simulation periods
    T_burn : int
        Burn-in periods to drop
    seed : int
        Random seed
        
    Returns
    -------
    panel : dict
        Dictionary with simulated panel data
    """
    np.random.seed(seed)
    
    mu_ss = equilibrium['mu_ss']
    a_grid = equilibrium['a_grid']
    e_grid = equilibrium['e_grid']
    Pi = equilibrium['Pi']
    a_policy = equilibrium['a_policy']
    n_e = len(e_grid)
    
    # Initialize from stationary distribution
    flat_mu = mu_ss.flatten()
    flat_mu = flat_mu / flat_mu.sum()
    init_idx = np.random.choice(len(flat_mu), size=N_agents, p=flat_mu)
    a_init_idx = init_idx // n_e
    e_init_idx = init_idx % n_e
    a_init = a_grid[a_init_idx]
    
    # Simulate
    a_sim, e_sim, e_idx_sim = simulate_panel(
        a_policy, a_grid, e_grid, Pi, N_agents, T_sim, a_init, e_init_idx.astype(np.int64)
    )
    
    # Drop burn-in
    a_panel = a_sim[:, T_burn:]
    e_panel = e_sim[:, T_burn:]
    
    return {
        'a_panel': a_panel,
        'e_panel': e_panel,
        'N_agents': N_agents,
        'T_data': a_panel.shape[1]
    }


# =============================================================================
# Simulation with Aggregate TFP Shocks
# =============================================================================

def simulate_tfp_shocks(
    T: int,
    rho_z: float,
    sigma_z: float,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Simulate aggregate TFP shocks following AR(1) process.
    
    log(Z_t) = rho_z * log(Z_{t-1}) + eps_t, eps_t ~ N(0, sigma_z^2)
    
    Parameters
    ----------
    T : int
        Number of periods
    rho_z : float
        Persistence of TFP process
    sigma_z : float
        Standard deviation of TFP innovations
    seed : int, optional
        Random seed
        
    Returns
    -------
    Z : np.ndarray
        TFP levels (T,), normalized so mean ≈ 1
    """
    if seed is not None:
        np.random.seed(seed + 999)  # Different seed from other randomness
    
    # Simulate log(Z)
    log_Z = np.zeros(T)
    innovations = np.random.normal(0, sigma_z, T)
    
    for t in range(1, T):
        log_Z[t] = rho_z * log_Z[t-1] + innovations[t]
    
    # Convert to levels (Z = exp(log_Z))
    Z = np.exp(log_Z)
    
    return Z


def run_simulation_with_shocks(
    equilibrium: Dict[str, Any],
    params: AiyagariParams,
    N_agents: int = 10000,
    T_sim: int = 2000,
    T_burn: int = 500,
    seed: int = 123
) -> Dict[str, np.ndarray]:
    """
    Run panel simulation with aggregate TFP shocks.
    
    When TFP shocks are present (sigma_z > 0), this provides a more realistic
    test of approximate aggregation. The shock s_{t+1} = Z_{t+1} becomes
    a meaningful predictor variable rather than a constant.
    
    Note: This is a "partial equilibrium in aggregates" approach - we use
    the steady-state policy functions but apply time-varying TFP to output.
    For a full Krusell-Smith style solution, one would need to solve for
    policy functions conditional on Z and beliefs about future aggregates.
    
    Parameters
    ----------
    equilibrium : dict
        Output from compute_equilibrium
    params : AiyagariParams
        Model parameters (includes TFP shock parameters)
    N_agents : int
        Number of agents
    T_sim : int
        Total simulation periods
    T_burn : int
        Burn-in periods to drop
    seed : int
        Random seed
        
    Returns
    -------
    panel : dict
        Dictionary with simulated panel data and TFP shocks
    """
    np.random.seed(seed)
    
    mu_ss = equilibrium['mu_ss']
    a_grid = equilibrium['a_grid']
    e_grid = equilibrium['e_grid']
    Pi = equilibrium['Pi']
    a_policy = equilibrium['a_policy']
    n_e = len(e_grid)
    
    # Initialize from stationary distribution
    flat_mu = mu_ss.flatten()
    flat_mu = flat_mu / flat_mu.sum()
    init_idx = np.random.choice(len(flat_mu), size=N_agents, p=flat_mu)
    a_init_idx = init_idx // n_e
    e_init_idx = init_idx % n_e
    a_init = a_grid[a_init_idx]
    
    # Simulate micro panel (idiosyncratic dynamics)
    a_sim, e_sim, e_idx_sim = simulate_panel(
        a_policy, a_grid, e_grid, Pi, N_agents, T_sim, a_init, e_init_idx.astype(np.int64)
    )
    
    # Simulate aggregate TFP shocks
    if params.sigma_z > 0:
        Z_series = simulate_tfp_shocks(T_sim, params.rho_z, params.sigma_z, seed)
    else:
        Z_series = np.ones(T_sim)  # No shocks
    
    # Drop burn-in
    a_panel = a_sim[:, T_burn:]
    e_panel = e_sim[:, T_burn:]
    Z_panel = Z_series[T_burn:]
    
    return {
        'a_panel': a_panel,
        'e_panel': e_panel,
        'Z_series': Z_panel,        # Aggregate TFP shocks
        'N_agents': N_agents,
        'T_data': a_panel.shape[1],
        'has_aggregate_shocks': params.sigma_z > 0
    }
