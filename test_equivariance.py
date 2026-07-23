import torch
import numpy as np
from models import SMPGuidedPortfolioPolicy

def test_s_n_equivariance():
    print("=== Testing S_N Permutation Equivariance Precision ===")
    torch.manual_seed(42)
    np.random.seed(42)
    
    num_assets = 5
    lookback = 10
    feature_dim = 2
    batch_size = 2
    
    policy_net = SMPGuidedPortfolioPolicy(lookback=lookback, feature_dim=feature_dim)
    policy_net.eval()
    
    # Generate random asset observation: (batch_size, num_assets, lookback, feature_dim)
    obs = torch.randn(batch_size, num_assets, lookback, feature_dim)
    
    # Generate random permutation matrix P \in S_N
    perm_indices = np.random.permutation(num_assets)
    
    # Original forward pass
    with torch.no_grad():
        weights_orig, val_orig, p_orig, q_orig, smp_orig = policy_net(obs)
        
    # Permuted forward pass: P * obs
    obs_perm = obs[:, perm_indices, :, :]
    with torch.no_grad():
        weights_perm, val_perm, p_perm, q_perm, smp_perm = policy_net(obs_perm)
        
    # Expected permuted output: P * weights_orig
    weights_expected = weights_orig[:, perm_indices]
    
    # Calculate Max Absolute Error
    max_err = torch.max(torch.abs(weights_perm - weights_expected)).item()
    val_err = torch.max(torch.abs(val_orig - val_perm)).item()
    
    print(f"Max Equivariance Error on Policy Weights: {max_err:.9e}")
    print(f"Max Invariance Error on Value Prediction: {val_err:.9e}")
    
    assert max_err < 1e-6, f"Equivariance test failed! Error {max_err} >= 1e-6"
    assert val_err < 1e-6, f"Invariance test failed! Error {val_err} >= 1e-6"
    print("PASSED S_N Permutation Equivariance & Invariance Precision Tests!\n")

if __name__ == "__main__":
    test_s_n_equivariance()
