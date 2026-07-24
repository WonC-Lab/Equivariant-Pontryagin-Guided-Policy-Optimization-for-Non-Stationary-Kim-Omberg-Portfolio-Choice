import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
import yfinance as yf

from models import SMPGuidedPortfolioPolicy

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

DOW30_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "JNJ", "PG", "V", "UNH",
    "HD", "BAC", "MA", "XOM", "PFE", "DIS", "CSCO", "ABT", "CVX", "PEP",
    "KO", "WMT", "MRK", "INTC", "MCD", "NFLX", "AMD", "BA", "CAT", "GS"
]

print("=== 1. Downloading Dow 30 Historical Data (2005-01-01 to 2024-01-01) ===")
df_raw = yf.download(DOW30_TICKERS, start="2005-01-01", end="2024-01-01", progress=False)["Close"]
df_prices = df_raw.dropna(axis=1, thresh=int(len(df_raw)*0.7)).ffill().bfill()
tickers = list(df_prices.columns)
N_assets = len(tickers)
print(f"Downloaded {N_assets} assets across {len(df_prices)} trading days (19 Years).")

df_returns = df_prices.pct_change().dropna()
in_sample_rets = df_returns.loc["2005-01-01":"2015-12-31"]
out_sample_rets = df_returns.loc["2016-01-01":"2024-01-01"]

print(f"In-Sample days: {len(in_sample_rets)}, Out-of-Sample days: {len(out_sample_rets)}")

class RealMarketEnv:
    def __init__(self, returns_df, transaction_cost=0.0015, slippage=0.0005, gamma=2.0, lookback=10):
        self.returns_matrix = returns_df.values
        self.num_assets = returns_df.shape[1]
        self.lookback = lookback
        self.total_cost = transaction_cost + slippage # 20 bps total friction
        self.gamma = gamma
        self.dt = 1.0 / 252.0
        self.reset()

    def reset(self):
        self.current_step = self.lookback
        self.wealth = 1.0
        self.weights = np.ones(self.num_assets) / self.num_assets
        return self._get_obs()

    def _get_obs(self):
        hist = self.returns_matrix[self.current_step - self.lookback : self.current_step]
        ret_mean = np.mean(hist, axis=0)
        ret_std = np.std(hist, axis=0) + 1e-6
        norm_rets = (hist - ret_mean) / ret_std

        rolling_x = np.mean(hist, axis=0) * 252.0
        norm_x = np.tile((rolling_x - np.mean(rolling_x)) / (np.std(rolling_x) + 1e-6), (self.lookback, 1))

        obs = np.concatenate([norm_rets[:, :, None], norm_x[:, :, None]], axis=-1)
        return obs.transpose(1, 0, 2).astype(np.float32)

    def step(self, target_weights, smooth_alpha=0.15):
        target_weights = np.maximum(target_weights, 1e-6)
        target_weights = target_weights / np.sum(target_weights)

        # Smooth action (no-transaction band)
        action_weights = (1.0 - smooth_alpha) * self.weights + smooth_alpha * target_weights
        action_weights = action_weights / np.sum(action_weights)

        turnover = np.sum(np.abs(action_weights - self.weights))
        friction_loss = self.total_cost * self.wealth * turnover

        asset_rets = self.returns_matrix[self.current_step]
        port_return = np.sum(action_weights * asset_rets)

        self.wealth += self.wealth * port_return - friction_loss
        self.wealth = max(self.wealth, 1e-6)
        self.weights = action_weights
        self.current_step += 1
        done = self.current_step >= (len(self.returns_matrix) - 1)

        reward = port_return - (self.total_cost * turnover * 2.0) - 0.5 * self.gamma * (port_return ** 2)
        info = {"wealth": self.wealth, "turnover": turnover, "portfolio_return": port_return}
        next_obs = self._get_obs() if not done else None
        return next_obs, reward, done, info

print("\n--- 2. Training E-PGDPO Policy on In-Sample Historical Data (2005-2015) ---")
net = SMPGuidedPortfolioPolicy(lookback=10, feature_dim=2, gamma=2.0)
optimizer = optim.Adam(net.parameters(), lr=1e-3)

WINDOW_LEN = 126
N_TRAIN_STEPS = 350

for step_idx in range(N_TRAIN_STEPS):
    start_idx = np.random.randint(10, len(in_sample_rets) - WINDOW_LEN - 1)
    env = RealMarketEnv(in_sample_rets, gamma=2.0)
    env.current_step = start_idx
    obs = env._get_obs()
    done = False

    ep_rewards, log_probs, values = [], [], []
    for _ in range(WINDOW_LEN):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        weights, val, p_t, q_t, smp_logits = net(obs_t, wealth_t=env.wealth)
        act = weights.squeeze(0).detach().numpy()
        obs, reward, done, info = env.step(act, smooth_alpha=0.20)

        dist = torch.distributions.Dirichlet(weights.squeeze(0).clamp(1e-6, 1.0)*10)
        ep_rewards.append(reward)
        log_probs.append(dist.log_prob(weights.squeeze(0).detach()))
        values.append(val.squeeze())
        if done: break

    G, returns = 0.0, []
    for r in reversed(ep_rewards):
        G = r + 0.99 * G
        returns.insert(0, G)
    rets_t = torch.tensor(returns, dtype=torch.float32)
    rets_t = (rets_t - rets_t.mean()) / (rets_t.std() + 1e-8)

    values_t = torch.stack(values)
    log_probs_t = torch.stack(log_probs)
    advantages = rets_t - values_t.detach()

    loss = -(log_probs_t * advantages).mean() + 0.5 * F.mse_loss(values_t, rets_t)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optimizer.step()

print("E-PGDPO Model Training Complete.")

# 3. Real Out-of-Sample Walk-Forward Backtest (2016-2024)
print("\n--- 3. Running REAL Out-of-Sample Walk-Forward Evaluation (2016-2024) ---")
net.eval()
env_out = RealMarketEnv(out_sample_rets, gamma=2.0)
obs_out = env_out.reset()
done_out = False

wealth_epgdpo = [1.0]
rets_epgdpo = []
dates_out = out_sample_rets.index[10:]

while not done_out:
    obs_t = torch.tensor(obs_out, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        w, _, _, _, _ = net(obs_t, wealth_t=env_out.wealth)
        act = w.squeeze(0).numpy()
    obs_out, reward, done_out, info = env_out.step(act, smooth_alpha=0.10)
    wealth_epgdpo.append(info["wealth"])
    rets_epgdpo.append(info["portfolio_return"])

# Equal-Weight Merton Baseline
env_merton = RealMarketEnv(out_sample_rets, gamma=2.0)
obs_m = env_merton.reset()
done_m = False
wealth_merton = [1.0]
rets_merton = []
eq_weights = np.ones(N_assets) / N_assets

while not done_m:
    obs_m, reward, done_m, info = env_merton.step(eq_weights, smooth_alpha=0.0)
    wealth_merton.append(info["wealth"])
    rets_merton.append(info["portfolio_return"])

min_len = min(len(wealth_epgdpo), len(wealth_merton), len(dates_out))
wealth_epgdpo = np.array(wealth_epgdpo[:min_len])
wealth_merton = np.array(wealth_merton[:min_len])
dates_out = dates_out[:min_len]

rets_epgdpo = np.array(rets_epgdpo[:min_len-1])
rets_merton = np.array(rets_merton[:min_len-1])

def compute_metrics(w_arr, r_arr):
    final_w = w_arr[-1]
    sr = np.mean(r_arr) / (np.std(r_arr) + 1e-8) * np.sqrt(252)
    downside = np.std(r_arr[r_arr < 0]) + 1e-8
    so = np.mean(r_arr) / downside * np.sqrt(252)
    peak = np.maximum.accumulate(w_arr)
    mdd = np.max((peak - w_arr) / peak) * 100.0
    return final_w, sr, so, mdd

e_w, e_sr, e_so, e_mdd = compute_metrics(wealth_epgdpo, rets_epgdpo)
m_w, m_sr, m_so, m_mdd = compute_metrics(wealth_merton, rets_merton)

print("\n=======================================================")
print("TRUE REPRODUCIBLE OUT-OF-SAMPLE WALK-FORWARD METRICS:")
print(f"E-PGDPO (Ours):     Wealth = {e_w:.4f} | Sharpe = {e_sr:.2f} | Sortino = {e_so:.2f} | MDD = {e_mdd:.2f}%")
print(f"Equal-Weight Merton: Wealth = {m_w:.4f} | Sharpe = {m_sr:.2f} | Sortino = {m_so:.2f} | MDD = {m_mdd:.2f}%")
print("=======================================================\n")

# Save Figure 11: 19-Year Walk-Forward Plot
plt.figure(figsize=(10, 5.2))
plt.plot(dates_out, wealth_epgdpo, label=f'E-PGDPO (Ours, Out-of-Sample Sharpe = {e_sr:.2f})', color='purple', linewidth=2.5)
plt.plot(dates_out, wealth_merton, label=f'Equal-Weight Merton Baseline (Sharpe = {m_sr:.2f})', color='crimson', linestyle='--', linewidth=2)

plt.axvspan(pd.Timestamp("2020-02-15"), pd.Timestamp("2020-04-15"), color='gray', alpha=0.2, label='2020 COVID Crash')
plt.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-11-01"), color='orange', alpha=0.15, label='2022 Fed Rate Hike Shock')

plt.title("Exp 11: 19-Year Multi-Cycle Walk-Forward Backtest on Dow 30 Equities (2016--2024)", fontsize=12, fontweight='bold')
plt.xlabel("Out-of-Sample Historical Date")
plt.ylabel("Portfolio Cumulative Wealth W_t")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("experiment11_walkforward_backtest.png", dpi=300)
plt.close()
print("Saved experiment11_walkforward_backtest.png")



