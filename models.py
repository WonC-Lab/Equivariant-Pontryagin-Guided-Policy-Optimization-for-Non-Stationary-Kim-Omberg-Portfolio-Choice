import torch
import torch.nn as nn
import torch.nn.functional as F

class PermutationEquivariantLayer(nn.Module):
    """
    Permutation Equivariant Layer for set inputs (Zaheer et al., Deep Sets).
    Guarantees S_N permutation equivariance across asset dimensions.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w1 = nn.Linear(in_features, out_features, bias=False)
        self.w2 = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x):
        mean_x = torch.mean(x, dim=1, keepdim=True)
        out = self.w1(x) + self.w2(mean_x)
        return F.relu(out)

class SMPGuidedPortfolioPolicy(nn.Module):
    """
    Stochastic Pontryagin's Maximum Principle (SMP) Guided Policy Architecture.
    
    1. Adjoint Estimator Network: Predicts (p_t, q_t) from BSDE dynamics.
       - p_t: Marginal utility of wealth (Costate, p_T = W_T^-gamma)
       - q_t: Stochastic volatility adjoint multiplier
    2. Policy Network: Combines Merton prior + SMP Guidance action + DeepSets correction.
    """
    def __init__(self, lookback=10, feature_dim=2, hidden_dim=64, gamma=2.0, sigma_S=0.20):
        super().__init__()
        self.gamma = gamma
        self.sigma_S = sigma_S
        in_dim = lookback * feature_dim
        
        self.eq1 = PermutationEquivariantLayer(in_dim, hidden_dim)
        self.eq2 = PermutationEquivariantLayer(hidden_dim, hidden_dim)
        
        # 1. Adjoint Variable Estimator Networks
        # p_t: scalar marginal utility p_t (batch_size, 1)
        self.p_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus() # Marginal utility p_t > 0
        )
        # q_t: stochastic volatility adjoint per asset (batch_size, num_assets)
        self.q_head = nn.Linear(hidden_dim, 1)
        
        # 2. Neural Policy Correction Head
        self.policy_correction = nn.Linear(hidden_dim, 1)
        
        # 3. Value Network V(s)
        self.val_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, wealth_t=1.0):
        # x: (batch_size, num_assets, lookback, feature_dim)
        batch_size, num_assets, lookback, feature_dim = x.shape
        x_flat = x.reshape(batch_size, num_assets, -1)
        
        h = self.eq1(x_flat)
        h = self.eq2(h)
        
        # Extract estimated risk premium x_t (feature index 1 of last step)
        x_t_est = x[:, :, -1, 1] # (batch_size, num_assets)
        
        # 1. Predict Adjoint Process (p_t, q_t)
        global_h = torch.mean(h, dim=1) # (batch_size, hidden_dim)
        p_t = self.p_head(global_h) # (batch_size, 1)
        q_t = self.q_head(h).squeeze(-1) # (batch_size, num_assets)
        
        # 2. Compute SMP Analytical Guidance Action pi_SMP = - (p_t / (gamma * W_t)) * (x_t / sigma_S^2) - q_t
        pi_myopic = x_t_est / (self.gamma * (self.sigma_S ** 2))
        smp_guidance_logits = pi_myopic - 0.1 * q_t
        
        # 3. Neural Policy Correction
        correction = self.policy_correction(h).squeeze(-1)
        
        # Simplex Softmax Allocation
        total_logits = smp_guidance_logits + correction
        weights = F.softmax(total_logits, dim=-1)
        
        # Value prediction
        val = self.val_head(global_h)
        
        return weights, val, p_t, q_t, smp_guidance_logits

if __name__ == "__main__":
    net = SMPGuidedPortfolioPolicy(lookback=10, feature_dim=2)
    sample_obs = torch.randn(2, 5, 10, 2)
    w, v, p, q, pi_smp = net(sample_obs)
    print("SMP-Guided Policy weights shape:", w.shape, "Simplex sum:", torch.sum(w, dim=-1))
    print("Adjoint state p_t (marginal utility):", p)
    print("Adjoint state q_t shape:", q.shape)
