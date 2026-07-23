import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from portfolio_env import EMKOPortfolioEnv
from models import SMPGuidedPortfolioPolicy

def train_smp_guided_agent(num_episodes=100, num_assets=5, gamma=2.0, lambda_smp=0.5):
    """
    Trains SMP-Guided Portfolio Optimization Agent and compares performance
    against Kim & Omberg analytical baseline and Static Merton allocation.
    """
    env = EMKOPortfolioEnv(num_assets=num_assets, gamma=gamma, seed=42)
    policy_net = SMPGuidedPortfolioPolicy(lookback=10, feature_dim=2, gamma=gamma)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    
    smp_wealth_history = []
    ko_wealth_history = []
    merton_wealth_history = []
    
    print("Starting SMP-Guided Portfolio Optimization Training & Benchmark Evaluation...")
    
    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        done = False
        total_loss = 0.0
        
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            
            # Policy forward pass (extract weights, value, adjoint p_t, q_t, smp_guidance_logits)
            weights, val, p_t, q_t, smp_logits = policy_net(obs_tensor, wealth_t=env.wealth)
            action = weights.squeeze(0).detach().numpy()
            
            next_obs, reward, done, info = env.step(action)
            
            # CRRA Terminal Utility Target
            reward_tensor = torch.tensor([[reward]], dtype=torch.float32)
            advantage = reward_tensor - val
            
            # 1. Standard RL Loss
            val_loss = F.mse_loss(val, reward_tensor)
            policy_loss = -torch.mean(torch.log(weights + 1e-8) * advantage.detach())
            
            # 2. SMP Maximum Principle Guidance Loss: MSE between Policy & SMP Guidance Action
            smp_action_target = F.softmax(smp_logits, dim=-1).detach()
            smp_guidance_loss = F.mse_loss(weights, smp_action_target)
            
            # 3. BSDE Terminal Adjoint Boundary Loss: p_T = W_T^-gamma
            target_p_T = torch.tensor([[env.wealth ** (-gamma)]], dtype=torch.float32)
            bsde_loss = F.mse_loss(p_t, target_p_T) if done else torch.tensor(0.0)
            
            total_loss_val = policy_loss + 0.5 * val_loss + lambda_smp * smp_guidance_loss + 0.1 * bsde_loss
            
            optimizer.zero_grad()
            total_loss_val.backward()
            optimizer.step()
            
            obs = next_obs
            
        smp_wealth_history.append(info["wealth"])
        
        # Evaluate Analytical Kim & Omberg Benchmark
        ko_wealth = evaluate_analytical_ko(env_params={"num_assets": num_assets, "gamma": gamma, "seed": 42 + episode})
        ko_wealth_history.append(ko_wealth)
        
        # Evaluate Static Merton Benchmark
        merton_wealth = evaluate_static_merton(env_params={"num_assets": num_assets, "gamma": gamma, "seed": 42 + episode})
        merton_wealth_history.append(merton_wealth)
        
        if episode % 20 == 0 or episode == 1:
            print(f"Episode {episode:03d} | SMP-Guided Wealth: {info['wealth']:.4f} | K&O Wealth: {ko_wealth:.4f} | Merton Wealth: {merton_wealth:.4f}")
            
    # Save performance plot
    plt.figure(figsize=(10, 5))
    plt.plot(smp_wealth_history, label="Proposed SMP-Guided Portfolio RL Agent", color="purple", linewidth=2)
    plt.plot(ko_wealth_history, label="Kim & Omberg Analytical Benchmark", color="green", linestyle="--")
    plt.plot(merton_wealth_history, label="Static Merton Allocation", color="red", linestyle=":")
    plt.title("Continuous-Time Terminal Wealth: SMP-Guided Optimization vs Benchmarks")
    plt.xlabel("Episodes")
    plt.ylabel("Terminal Wealth W_T")
    plt.legend()
    plt.grid(True)
    plt.savefig("smp_wealth_performance.png", dpi=300)
    plt.close()
    print("Performance plot saved to 'smp_wealth_performance.png'.")
    
    return policy_net

def evaluate_analytical_ko(env_params):
    env = EMKOPortfolioEnv(**env_params)
    obs = env.reset()
    done = False
    while not done:
        action = env.get_kim_omberg_analytical_action()
        obs, reward, done, info = env.step(action)
    return info["wealth"]

def evaluate_static_merton(env_params):
    env = EMKOPortfolioEnv(**env_params)
    obs = env.reset()
    done = False
    static_action = np.ones(env_params["num_assets"]) / env_params["num_assets"]
    while not done:
        obs, reward, done, info = env.step(static_action)
    return info["wealth"]

if __name__ == "__main__":
    trained_policy = train_smp_guided_agent(num_episodes=100)
