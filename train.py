import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from portfolio_env import MultiAssetPortfolioEnv
from models import PermutationEquivariantPortfolioNet

def train_portfolio_agent(num_episodes=100, num_assets=5, lookback=10, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = MultiAssetPortfolioEnv(num_assets=num_assets, lookback=lookback, seed=seed)
    model = PermutationEquivariantPortfolioNet(lookback=lookback, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    returns_history = []
    sharpe_history = []
    
    print(f"=== Starting Permutation-Equivariant Portfolio Training ({num_episodes} Episodes) ===")
    
    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        done = False
        episode_rewards = []
        net_returns = []
        
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0) # (1, num_assets, lookback)
            model.eval()
            with torch.no_grad():
                action_weights, val = model(obs_tensor)
                weights_np = action_weights.squeeze(0).numpy()
                
            # Add small exploration noise on simplex
            noise = np.random.dirichlet(np.ones(num_assets) * 0.5) * 0.05
            weights_np = weights_np + noise
            weights_np = weights_np / np.sum(weights_np)
            
            next_obs, reward, done, info = env.step(weights_np)
            
            # Optimization step
            model.train()
            optimizer.zero_grad()
            
            pred_weights, pred_val = model(obs_tensor)
            
            # Policy loss: maximize reward weighted allocation
            target_return = torch.tensor([[info["net_return"]]], dtype=torch.float32)
            policy_loss = -torch.mean(torch.sum(pred_weights * torch.tensor(weights_np), dim=-1) * target_return)
            value_loss = F.mse_loss(pred_val, target_return)
            
            loss = policy_loss + 0.5 * value_loss
            loss.backward()
            optimizer.step()
            
            obs = next_obs
            episode_rewards.append(reward)
            net_returns.append(info["net_return"])
            
        tot_return = np.sum(net_returns)
        sharpe = np.mean(net_returns) / (np.std(net_returns) + 1e-8) * np.sqrt(252)
        returns_history.append(tot_return)
        sharpe_history.append(sharpe)
        
        if episode % 20 == 0 or episode == 1:
            print(f"Episode {episode:3d}/{num_episodes} | Total Net Return: {tot_return:.4f} | Ann. Sharpe Ratio: {sharpe:.2f}")
            
    # Save training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(returns_history, color="navy", linewidth=1.8)
    plt.title("Cumulative Net Return")
    plt.xlabel("Episodes")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.subplot(1, 2, 2)
    plt.plot(sharpe_history, color="crimson", linewidth=1.8)
    plt.title("Annualized Sharpe Ratio")
    plt.xlabel("Episodes")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("training_performance.png", dpi=300)
    plt.close()
    print("\nTraining completed successfully! Saved plot to training_performance.png.")

if __name__ == "__main__":
    train_portfolio_agent(num_episodes=100)
