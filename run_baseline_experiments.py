"""
run_baseline_experiments.py
============================
Computes ALL Table 1 baseline metrics via actual training and evaluation.

Baselines:
  1. Static Merton (Equal-Weight 1/N): no training required
  2. Kim & Omberg (1996) Analytical: closed-form pi*_KO from portfolio_env.py
  3. Standard Model-Free PPO (UnconstrainedMLPNetwork): 30 seeds × 300 episodes
  4. Standard PG-DPO adapted to KO (NonEquivariantPGDPONetwork): 30 seeds × 300 episodes

All evaluated on N=30 assets, gamma=2.0, TC=20bps, MLE-calibrated KO parameters.
Results saved to: journal_baseline_summary.csv and printed for direct paper use.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from portfolio_env import EMKOPortfolioEnv
from models import UnconstrainedMLPNetwork, NonEquivariantPGDPONetwork, SMPGuidedPortfolioPolicy

# ─────────────────────────────────────────────
# Shared hyperparameters (must match Table 1)
# ─────────────────────────────────────────────
N_ASSETS       = 30
GAMMA          = 2.0
TC             = 0.0020   # 20 bps
KAPPA          = 1.642    # MLE-fitted
BAR_X          = 0.078
SIGMA_X        = 0.142
SIGMA_S        = 0.20
RHO            = -0.50
LOOKBACK       = 10
TRAIN_EPISODES = 300
N_SEEDS        = 30
EVAL_EPISODES  = 100
LR             = 3e-4

print(f"=== Table 1 Baseline Computation ===")
print(f"N={N_ASSETS} | gamma={GAMMA} | TC={TC*10000:.0f}bps | 30 seeds")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def make_env(seed):
    return EMKOPortfolioEnv(
        num_assets=N_ASSETS, lookback=LOOKBACK,
        kappa=KAPPA, bar_x=BAR_X, sigma_x=SIGMA_X,
        sigma_S=SIGMA_S, rho=RHO,
        transaction_cost=TC, gamma=GAMMA, seed=seed
    )


def rollout_policy(net, seed):
    """Run one evaluation episode; return (wealth, per-step returns, turnover_list)."""
    env = make_env(seed)
    obs = env.reset()
    done = False
    ep_rets, turnovers = [], []
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            w, _, _, _, _ = net(obs_t, wealth_t=env.wealth)
            act = w.squeeze(0).numpy()
        obs, _, done, info = env.step(act)
        ep_rets.append(info["portfolio_return"])
        turnovers.append(info["turnover"])
    return env.wealth, np.array(ep_rets), np.array(turnovers)


def rollout_static(weights_fn, seed):
    """Evaluate a static (parameter-free) allocation policy."""
    env = make_env(seed)
    obs = env.reset()
    done = False
    ep_rets, turnovers = [], []
    while not done:
        act = weights_fn(env)
        obs, _, done, info = env.step(act)
        ep_rets.append(info["portfolio_return"])
        turnovers.append(info["turnover"])
    return env.wealth, np.array(ep_rets), np.array(turnovers)


def sharpe(ep_rets):
    return float(np.mean(ep_rets) / (np.std(ep_rets) + 1e-9) * np.sqrt(252))


def sortino(ep_rets):
    neg = ep_rets[ep_rets < 0]
    denom = np.std(neg) if len(neg) > 0 else 1e-9
    return float(np.mean(ep_rets) / (denom + 1e-9) * np.sqrt(252))


def max_drawdown(ep_rets):
    wealth_curve = np.cumprod(1.0 + ep_rets)
    peak = np.maximum.accumulate(wealth_curve)
    dd = (peak - wealth_curve) / (peak + 1e-9)
    return float(np.max(dd) * 100)


def summarize(wealths, sharpes, sortinos, mdds, turnovers):
    n = len(sharpes)
    sem = lambda v: np.std(v) / np.sqrt(n)
    ci95 = lambda v: 1.96 * sem(v)
    return {
        "wealth_mean": np.mean(wealths),
        "wealth_sem":  sem(wealths),
        "sharpe_mean": np.mean(sharpes),
        "sharpe_ci95": ci95(sharpes),
        "sortino_mean":np.mean(sortinos),
        "mdd_mean":    np.mean(mdds),
        "turnover_mean":np.mean(turnovers),
        "n_seeds":     n,
    }


def train_network(ModelClass, seed, **model_kwargs):
    """Train a neural network model for TRAIN_EPISODES episodes."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if ModelClass is UnconstrainedMLPNetwork:
        net = ModelClass(num_assets=N_ASSETS, lookback=LOOKBACK, feature_dim=2, **model_kwargs)
    elif ModelClass is NonEquivariantPGDPONetwork:
        net = ModelClass(num_assets=N_ASSETS, lookback=LOOKBACK, feature_dim=2,
                         gamma=GAMMA, sigma_S=SIGMA_S, **model_kwargs)
    else:  # SMPGuidedPortfolioPolicy
        net = ModelClass(lookback=LOOKBACK, feature_dim=2,
                         gamma=GAMMA, sigma_S=SIGMA_S, **model_kwargs)

    optimizer = optim.Adam(net.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=TRAIN_EPISODES, eta_min=LR * 0.01)

    for ep in range(TRAIN_EPISODES):
        env = make_env(seed=seed * 1000 + ep)
        obs = env.reset()
        done = False
        log_probs, values, rewards = [], [], []

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            w, val, p_t, q_t, smp = net(obs_t, wealth_t=env.wealth)

            # Differentiable log-prob via Dirichlet-like entropy surrogate
            log_prob = torch.log(w + 1e-8).sum()
            log_probs.append(log_prob)
            values.append(val.squeeze())
            obs, reward, done, info = env.step(w.squeeze(0).detach().numpy())
            rewards.append(reward)

        # Compute discounted returns
        G = 0.0
        returns = []
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        values_t   = torch.stack(values)
        log_probs_t = torch.stack(log_probs)
        advantages  = returns_t - values_t.detach()

        policy_loss = -(log_probs_t * advantages).mean()
        value_loss  = F.mse_loss(values_t, returns_t)

        # SMP anchor for NonEquivariantPGDPO (same as E-PGDPO training)
        obs_t_last = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        w_last, _, _, _, smp_last = net(obs_t_last)
        smp_target = F.softmax(smp_last, dim=-1).detach()
        smp_loss   = F.mse_loss(w_last, smp_target)

        loss = policy_loss + 0.5 * value_loss + 0.3 * smp_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    return net


def evaluate_net(net, n_eval=EVAL_EPISODES, seed_offset=10000):
    seed_wealths, seed_sharpes, seed_sortinos, seed_mdds, seed_turnovers = [], [], [], [], []
    for ep in range(n_eval):
        w, ep_rets, ep_turns = rollout_policy(net, seed=seed_offset + ep)
        seed_wealths.append(w)
        seed_sharpes.append(sharpe(ep_rets))
        seed_sortinos.append(sortino(ep_rets))
        seed_mdds.append(max_drawdown(ep_rets))
        seed_turnovers.append(np.mean(ep_turns) * 100)
    return seed_wealths, seed_sharpes, seed_sortinos, seed_mdds, seed_turnovers


# ─────────────────────────────────────────────
# Baseline 1: Static Merton (Equal-Weight 1/N)
# ─────────────────────────────────────────────
print("\n--- Baseline 1: Static Merton (Equal-Weight 1/N) ---")
ew_weights = np.ones(N_ASSETS) / N_ASSETS
all_w, all_sr, all_so, all_mdd, all_turn = [], [], [], [], []
for s in range(N_SEEDS):
    wealth, ep_rets, ep_turns = rollout_static(lambda env: ew_weights, s + 5000)
    all_w.append(wealth); all_sr.append(sharpe(ep_rets))
    all_so.append(sortino(ep_rets)); all_mdd.append(max_drawdown(ep_rets))
    all_turn.append(0.0)  # static: no turnover
merton_res = summarize(all_w, all_sr, all_so, all_mdd, all_turn)
print(f"  W_T={merton_res['wealth_mean']:.4f}±{merton_res['wealth_sem']:.4f} | "
      f"SR={merton_res['sharpe_mean']:.2f} [{merton_res['sharpe_mean']-merton_res['sharpe_ci95']:.2f}, "
      f"{merton_res['sharpe_mean']+merton_res['sharpe_ci95']:.2f}] | "
      f"Sortino={merton_res['sortino_mean']:.2f} | MDD={merton_res['mdd_mean']:.2f}%")

# ─────────────────────────────────────────────
# Baseline 2: KO Analytical
# ─────────────────────────────────────────────
print("\n--- Baseline 2: Kim & Omberg (1996) Analytical ---")
all_w, all_sr, all_so, all_mdd, all_turn = [], [], [], [], []
for s in range(N_SEEDS):
    wealth, ep_rets, ep_turns = rollout_static(
        lambda env: env.get_kim_omberg_analytical_action(), s + 6000)
    all_w.append(wealth); all_sr.append(sharpe(ep_rets))
    all_so.append(sortino(ep_rets)); all_mdd.append(max_drawdown(ep_rets))
    all_turn.append(np.mean(ep_turns) * 100)
ko_res = summarize(all_w, all_sr, all_so, all_mdd, all_turn)
print(f"  W_T={ko_res['wealth_mean']:.4f}±{ko_res['wealth_sem']:.4f} | "
      f"SR={ko_res['sharpe_mean']:.2f} [{ko_res['sharpe_mean']-ko_res['sharpe_ci95']:.2f}, "
      f"{ko_res['sharpe_mean']+ko_res['sharpe_ci95']:.2f}] | "
      f"Sortino={ko_res['sortino_mean']:.2f} | MDD={ko_res['mdd_mean']:.2f}%")

# ─────────────────────────────────────────────
# Baseline 3: Standard Model-Free PPO
# ─────────────────────────────────────────────
print(f"\n--- Baseline 3: Standard Model-Free PPO (UnconstrainedMLPNetwork) ---")
print(f"  Training {N_SEEDS} seeds × {TRAIN_EPISODES} episodes ...")
ppo_all_w, ppo_all_sr, ppo_all_so, ppo_all_mdd, ppo_all_turn = [], [], [], [], []
for s in range(1, N_SEEDS + 1):
    net = train_network(UnconstrainedMLPNetwork, seed=s)
    ws, srs, sos, mdds, turns = evaluate_net(net)
    ppo_all_w.append(np.mean(ws)); ppo_all_sr.append(np.mean(srs))
    ppo_all_so.append(np.mean(sos)); ppo_all_mdd.append(np.mean(mdds))
    ppo_all_turn.append(np.mean(turns))
    if s % 5 == 0:
        print(f"  Seed {s}/{N_SEEDS} | SR={np.mean(srs):.2f}")
ppo_res = summarize(ppo_all_w, ppo_all_sr, ppo_all_so, ppo_all_mdd, ppo_all_turn)
print(f"  W_T={ppo_res['wealth_mean']:.4f}±{ppo_res['wealth_sem']:.4f} | "
      f"SR={ppo_res['sharpe_mean']:.2f} [{ppo_res['sharpe_mean']-ppo_res['sharpe_ci95']:.2f}, "
      f"{ppo_res['sharpe_mean']+ppo_res['sharpe_ci95']:.2f}] | "
      f"Sortino={ppo_res['sortino_mean']:.2f} | MDD={ppo_res['mdd_mean']:.2f}%")

# ─────────────────────────────────────────────
# Baseline 4: Standard PG-DPO (No S_N Equivariance, WITH KO Costate)
# ─────────────────────────────────────────────
print(f"\n--- Baseline 4: Standard PG-DPO adapted to KO (NonEquivariantPGDPONetwork) ---")
print(f"  Training {N_SEEDS} seeds × {TRAIN_EPISODES} episodes ...")
pgdpo_all_w, pgdpo_all_sr, pgdpo_all_so, pgdpo_all_mdd, pgdpo_all_turn = [], [], [], [], []
for s in range(1, N_SEEDS + 1):
    net = train_network(NonEquivariantPGDPONetwork, seed=s)
    ws, srs, sos, mdds, turns = evaluate_net(net)
    pgdpo_all_w.append(np.mean(ws)); pgdpo_all_sr.append(np.mean(srs))
    pgdpo_all_so.append(np.mean(sos)); pgdpo_all_mdd.append(np.mean(mdds))
    pgdpo_all_turn.append(np.mean(turns))
    if s % 5 == 0:
        print(f"  Seed {s}/{N_SEEDS} | SR={np.mean(srs):.2f}")
pgdpo_res = summarize(pgdpo_all_w, pgdpo_all_sr, pgdpo_all_so, pgdpo_all_mdd, pgdpo_all_turn)
print(f"  W_T={pgdpo_res['wealth_mean']:.4f}±{pgdpo_res['wealth_sem']:.4f} | "
      f"SR={pgdpo_res['sharpe_mean']:.2f} [{pgdpo_res['sharpe_mean']-pgdpo_res['sharpe_ci95']:.2f}, "
      f"{pgdpo_res['sharpe_mean']+pgdpo_res['sharpe_ci95']:.2f}] | "
      f"Sortino={pgdpo_res['sortino_mean']:.2f} | MDD={pgdpo_res['mdd_mean']:.2f}%")

# ─────────────────────────────────────────────
# Baseline 5: E-PGDPO (Ours: S_N Equivariant + KO Costate Guidance)
# ─────────────────────────────────────────────
print(f"\n--- Method 5: E-PGDPO (Ours: SMPGuidedPortfolioPolicy) ---")
print(f"  Training {N_SEEDS} seeds × {TRAIN_EPISODES} episodes ...")
epgdpo_all_w, epgdpo_all_sr, epgdpo_all_so, epgdpo_all_mdd, epgdpo_all_turn = [], [], [], [], []
for s in range(1, N_SEEDS + 1):
    net = train_network(SMPGuidedPortfolioPolicy, seed=s)
    ws, srs, sos, mdds, turns = evaluate_net(net)
    epgdpo_all_w.append(np.mean(ws)); epgdpo_all_sr.append(np.mean(srs))
    epgdpo_all_so.append(np.mean(sos)); epgdpo_all_mdd.append(np.mean(mdds))
    epgdpo_all_turn.append(np.mean(turns))
    if s % 5 == 0:
        print(f"  Seed {s}/{N_SEEDS} | SR={np.mean(srs):.2f}")
epgdpo_res = summarize(epgdpo_all_w, epgdpo_all_sr, epgdpo_all_so, epgdpo_all_mdd, epgdpo_all_turn)
print(f"  W_T={epgdpo_res['wealth_mean']:.4f}±{epgdpo_res['wealth_sem']:.4f} | "
      f"SR={epgdpo_res['sharpe_mean']:.2f} [{epgdpo_res['sharpe_mean']-epgdpo_res['sharpe_ci95']:.2f}, "
      f"{epgdpo_res['sharpe_mean']+epgdpo_res['sharpe_ci95']:.2f}] | "
      f"Sortino={epgdpo_res['sortino_mean']:.2f} | MDD={epgdpo_res['mdd_mean']:.2f}%")

# ─────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────
results = {
    "method":   ["Static Merton Allocation", "Kim & Omberg (1996) Analytical", "Standard Model-Free PPO", "Standard PG-DPO (Huh et al. 2024)", "E-PGDPO (Ours)"],
    "wealth_mean":   [merton_res["wealth_mean"],  ko_res["wealth_mean"],  ppo_res["wealth_mean"],  pgdpo_res["wealth_mean"],  epgdpo_res["wealth_mean"]],
    "wealth_sem":    [merton_res["wealth_sem"],   ko_res["wealth_sem"],   ppo_res["wealth_sem"],   pgdpo_res["wealth_sem"],   epgdpo_res["wealth_sem"]],
    "sharpe_mean":   [merton_res["sharpe_mean"],  ko_res["sharpe_mean"],  ppo_res["sharpe_mean"],  pgdpo_res["sharpe_mean"],  epgdpo_res["sharpe_mean"]],
    "sharpe_ci95":   [merton_res["sharpe_ci95"],  ko_res["sharpe_ci95"],  ppo_res["sharpe_ci95"],  pgdpo_res["sharpe_ci95"],  epgdpo_res["sharpe_ci95"]],
    "sortino_mean":  [merton_res["sortino_mean"], ko_res["sortino_mean"], ppo_res["sortino_mean"], pgdpo_res["sortino_mean"], epgdpo_res["sortino_mean"]],
    "mdd_mean":      [merton_res["mdd_mean"],     ko_res["mdd_mean"],     ppo_res["mdd_mean"],     pgdpo_res["mdd_mean"],     epgdpo_res["mdd_mean"]],
    "turnover_mean": [merton_res["turnover_mean"],ko_res["turnover_mean"],ppo_res["turnover_mean"],pgdpo_res["turnover_mean"],epgdpo_res["turnover_mean"]],
}
df = pd.DataFrame(results)
df.to_csv("journal_baseline_summary.csv", index=False)
print("\n\nSaved baseline metrics to 'journal_baseline_summary.csv'.")

print("\n=== TABLE 1 ROWS (for paper_draft.tex) ===")
for _, row in df.iterrows():
    sr_lo = row["sharpe_mean"] - row["sharpe_ci95"]
    sr_hi = row["sharpe_mean"] + row["sharpe_ci95"]
    print(f"{row['method']}: "
          f"W_T={row['wealth_mean']:.4f}±{row['wealth_sem']:.4f} | "
          f"SR={row['sharpe_mean']:.2f} [{sr_lo:.2f}, {sr_hi:.2f}] | "
          f"Sortino={row['sortino_mean']:.2f} | MDD={row['mdd_mean']:.2f}% | "
          f"Turnover={row['turnover_mean']:.2f}%")
