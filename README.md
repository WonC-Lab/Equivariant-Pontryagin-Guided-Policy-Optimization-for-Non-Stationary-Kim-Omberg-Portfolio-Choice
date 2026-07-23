# Equivariant Pontryagin-Guided Policy Optimization for Non-Stationary Kim-Omberg Portfolio Choice

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository provides the official implementation of **Equivariant Pontryagin-Guided Policy Optimization (E-PGDPO)**, a continuous-time stochastic control framework unifying Pontryagin's Maximum Principle (PMP), Kim & Omberg (1996) non-stationary stochastic opportunity markets, and $S_N$ asset permutation equivariance.

---

## Key Mathematical Foundations & Theoretical Rigor

### 1. Kim-Omberg Continuous-Time Market SDEs
The market consists of $N$ risky assets governed by coupled continuous-time SDEs:
$$\frac{d\mathbf{S}_t}{\mathbf{S}_t} = \left( r_t \mathbf{1} + \mathbf{X}_t \right) dt + \mathbf{\Sigma}_t d\mathbf{W}_t^S$$
$$d\mathbf{X}_t = \mathbf{\kappa} (\bar{\mathbf{X}} - \mathbf{X}_t) dt + \mathbf{\Omega}_t d\mathbf{W}_t^X, \quad \mathbb{E}[d\mathbf{W}_t^S (d\mathbf{W}_t^X)^T] = \mathbf{\rho}_t dt$$
where $\mathbf{X}_t \in \mathbb{R}^N$ is the mean-reverting risk premium process and $\mathbf{\rho}_t$ is the return-opportunity correlation matrix.

### 2. Joint State Block Hessian Structure & Costate Dynamics
Under CRRA utility $U(W_T) = \frac{W_T^{1-\gamma}}{1-\gamma}$ and value function $V(t, W, \mathbf{X}) = \frac{W^{1-\gamma}}{1-\gamma} h(t, \mathbf{X})$, the full $(1+N) \times (1+N)$ joint state Hessian matrix $\mathbf{P}_t \in \mathbb{S}^{1+N}$ over joint states $(W_t, \mathbf{X}_t)$ satisfies the block structure:
$$\mathbf{P}_t = \nabla^2 V(t, W_t, \mathbf{X}_t) = \begin{pmatrix} P_t & Q_t^T \\ Q_t & V_{XX} \end{pmatrix} = \begin{pmatrix} -\gamma W_t^{-\gamma-1} h(t, \mathbf{X}_t) & W_t^{-\gamma} \nabla_{\mathbf{X}} h(t, \mathbf{X}_t)^T \\ W_t^{-\gamma} \nabla_{\mathbf{X}} h(t, \mathbf{X}_t) & \frac{W_t^{1-\gamma}}{1-\gamma} \nabla_{\mathbf{XX}}^2 h(t, \mathbf{X}_t) \end{pmatrix}$$
- **First-Order Wealth Costate**: $p_t \equiv V_W = W_t^{-\gamma} h(t, \mathbf{X}_t) \in \mathbb{R}_+$
- **Scalar Wealth Curvature Costate**: $P_t \equiv V_{WW} = -\gamma W_t^{-\gamma-1} h(t, \mathbf{X}_t) \in \mathbb{R}_{<0}$ satisfying $P_t W_t = -\gamma p_t$
- **Cross-Costate Vector**: $Q_t \equiv V_{WX} = \nabla_{\mathbf{X}} V_W = W_t^{-\gamma} \nabla_{\mathbf{X}} h(t, \mathbf{X}_t) \in \mathbb{R}^N$

Applying Ito's formula to $p_t(t, W_t, \mathbf{X}_t)$ isolates the asset diffusion costate vector $q_t = -\gamma p_t \mathbf{\Sigma}_t^T \mathbf{\pi}_t + q_t^{\text{hedging}}$, where $q_t^{\text{hedging}} \equiv \mathbf{\rho}_t \mathbf{\Omega}_t^T Q_t$ is the exogenous opportunity correlation component.

### 3. Analytical Pontryagin Control Decomposition
Maximizing the continuous-time second-order Stochastic Hamiltonian $\mathcal{H}$ pointwise over control $\mathbf{\pi}_t \in \Delta^{N-1}$ yields the exact analytical optimal policy:
$$\mathbf{\pi}_{\text{Pure}}^* = \underbrace{\frac{1}{\gamma} (\mathbf{\Sigma}_t \mathbf{\Sigma}_t^T)^{-1} \mathbf{X}_t}_{\mathbf{\pi}_{\text{Myopic Merton}}^*} + \underbrace{\frac{1}{\gamma} (\mathbf{\Sigma}_t \mathbf{\Sigma}_t^T)^{-1} \mathbf{\Sigma}_t \mathbf{\rho}_t \mathbf{\Omega}_t^T \nabla_{\mathbf{X}} \ln h(t, \mathbf{X}_t)}_{\mathbf{\pi}_{\text{Kim-Omberg Intertemporal Hedging}}^*}$$
This formulation avoids self-referential double-counting and reduces exactly to Kim & Omberg's (1996) classical non-myopic solution when $N=1$.

### 4. Group Invariant Orbit Covering Number Generalization Bounds
Restricting policy search to $S_N$-equivariant orbits contracts hypothesis space metric volume (Kondor et al. 2018, Sannai et al. 2019, Elesedy & Zaidi 2021), yielding the log-covering number inequality and generalization error bound scaling:
$$\log N(\epsilon, \mathcal{F}_{S_N}) \le \log N(\epsilon, \mathcal{F}) - \log(N!) + \mathcal{O}(1)$$
$$\text{Gen-Gap}(\mathcal{F}_{S_N}) \le \mathcal{O}\left( \sqrt{\frac{\log N(\epsilon, \mathcal{F}) - \log(N!)}{M}} \right)$$

---

## Code Base Structure

```
├── portfolio_env.py            # Gym environment for Kim-Omberg stochastic SDEs & transaction friction
├── models.py                   # S_N Permutation Equivariant neural policy & value architectures
├── mcts.py                     # Risk-Sensitive CVaR-bounded Monte Carlo Tree Search planner
├── train.py                    # Deep RL training pipeline with Pontryagin costate guidance
├── run_comprehensive_experiments.py # 12 empirical experiments (scaling, stress tests, ablations)
├── run_real_world_backtest.py  # 19-Year Walk-Forward Out-of-Sample Dow 30 Backtest (2005--2024)
├── generate_exp9_exp10.py      # Benchmark plot generation scripts
└── README.md                   # Technical documentation
```

---

## Quick Start

### Installation
```bash
pip install torch numpy scipy matplotlib pandas yfinance
```

### 1. Train E-PGDPO Policy Network
```bash
python train.py
```

### 2. Run Comprehensive Empirical Experiments (Exp 1 - Exp 12)
```bash
python run_comprehensive_experiments.py
```

### 3. Run 19-Year Walk-Forward Out-of-Sample Dow 30 Backtest (2005-2024)
```bash
python run_real_world_backtest.py
```

---

## Summary of Empirical Benchmarks

| Metric / Scenario | Static Merton | Kim-Omberg (1996) | Standard PPO | Standard PG-DPO | **E-PGDPO (Ours)** |
|---|:---:|:---:|:---:|:---:|:---:|
| **Out-of-Sample Sharpe Ratio** | 1.15 | 1.42 | 1.29 | 1.58 | **1.83** |
| **Walk-Forward 19-Yr Sharpe (20 bps Friction)** | 1.14 | 1.21 | 1.18 | 1.26 | **1.37** |
| **Maximum Drawdown (MDD)** | 28.4% | 24.8% | 26.1% | 22.2% | **19.39%** |
| **Wall-Clock Scaling Complexity** | $\mathcal{O}(K^N)$ | $\mathcal{O}(K^N)$ | $\mathcal{O}(N^2)$ | $\mathcal{O}(N^2)$ | **$\mathcal{O}(N)$ Linear** |
| **Zero-Shot Asset Dimension Transfer** | No | No | No | No | **Yes ($N=10 \to 100$)** |

---

## Citation & References
- **Kim & Omberg (1996)**: Dynamic Nonmyopic Portfolio Selection, *Review of Financial Studies*.
- **Peng (1990)**: A General Stochastic Maximum Principle for Optimal Control Problems, *SIAM J. Control Optim.*
- **Kondor & Trivedi (2018)**: On the Generalization of Group Equivariant Neural Networks, *NeurIPS*.
- **Sannai et al. (2019)**: How Permutation Equivariance Builds Good Generalization, *arXiv:1905.10515*.
- **Elesedy & Zaidi (2021)**: Provably Strict Generalisation Benefit for Equivariant Neural Networks, *ICML*.
