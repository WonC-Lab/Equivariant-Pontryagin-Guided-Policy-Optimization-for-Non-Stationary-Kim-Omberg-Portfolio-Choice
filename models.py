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
    Full Equivariant Pontryagin-Guided Policy Optimization (E-PGDPO) Architecture.
    
    Combines:
    1. DeepSets S_N Equivariant Encoding
    2. Adjoint Costate Process Guidance (p_t, q_t)
    3. Neural Friction Soft-Thresholding Head
    """
    def __init__(self, lookback=10, feature_dim=2, hidden_dim=64, gamma=2.0, sigma_S=0.20):
        super().__init__()
        self.gamma = gamma
        self.sigma_S = sigma_S
        in_dim = lookback * feature_dim
        
        self.eq1 = PermutationEquivariantLayer(in_dim, hidden_dim)
        self.eq2 = PermutationEquivariantLayer(hidden_dim, hidden_dim)
        
        # 1. Adjoint Variable Estimator Networks
        self.p_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus() # Marginal utility p_t > 0
        )
        self.q_head = nn.Linear(hidden_dim, 1)
        
        # 2. Neural Friction Soft-Thresholding Head
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
        
        # Extract estimated risk premium X_t
        x_t_est = x[:, :, -1, 1] # (batch_size, num_assets)
        
        # Predict Adjoint Process (p_t, q_t)
        global_h = torch.mean(h, dim=1) # (batch_size, hidden_dim)
        p_t = self.p_head(global_h) # (batch_size, 1)
        q_t = self.q_head(h).squeeze(-1) # (batch_size, num_assets)
        
        # Analytical Pontryagin Guidance
        pi_myopic = x_t_est / (self.gamma * (self.sigma_S ** 2))
        smp_guidance_logits = pi_myopic - 0.1 * q_t
        
        # Friction soft-thresholding correction
        correction = self.policy_correction(h).squeeze(-1)
        
        # Portfolio Simplex Allocation
        total_logits = smp_guidance_logits + correction
        weights = F.softmax(total_logits, dim=-1)
        
        val = self.val_head(global_h)
        
        return weights, val, p_t, q_t, smp_guidance_logits


# =========================================================
# REAL ABLATION VARIANT MODELS (For Exp 4 & Exp 9 Validation)
# =========================================================

class UnconstrainedMLPNetwork(nn.Module):
    """
    Ablation Variant 1: Standard Unconstrained Dense MLP Network (No S_N Equivariance).
    Treats asset input as a single flattened vector.
    """
    def __init__(self, num_assets=5, lookback=10, feature_dim=2, hidden_dim=64):
        super().__init__()
        self.num_assets = num_assets
        self.lookback = lookback
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        in_dim = num_assets * lookback * feature_dim
        
        self.fc1 = nn.Linear(in_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, num_assets)
        self.val_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, wealth_t=1.0):
        batch_size, num_assets, lookback, feature_dim = x.shape
        in_dim = num_assets * lookback * feature_dim
        if in_dim != self.fc1.in_features:
            device = next(self.parameters()).device
            self.fc1 = nn.Linear(in_dim, self.hidden_dim * 2).to(device)
            self.fc2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim).to(device)
            self.policy_head = nn.Linear(self.hidden_dim, num_assets).to(device)
            self.val_head = nn.Linear(self.hidden_dim, 1).to(device)
            self.num_assets = num_assets

        x_flat = x.reshape(batch_size, -1)
        
        h = F.relu(self.fc1(x_flat))
        h = F.relu(self.fc2(h))
        
        logits = self.policy_head(h)
        weights = F.softmax(logits, dim=-1)
        val = self.val_head(h)
        
        return weights, val, torch.ones(batch_size, 1), torch.zeros(batch_size, num_assets), logits


class NoHedgingGuidanceNetwork(nn.Module):
    """
    Ablation Variant 2: E-PGDPO without Intertemporal Costate Hedging Guidance.
    Relies purely on Merton myopic allocation + neural head.
    """
    def __init__(self, lookback=10, feature_dim=2, hidden_dim=64, gamma=2.0, sigma_S=0.20):
        super().__init__()
        self.gamma = gamma
        self.sigma_S = sigma_S
        in_dim = lookback * feature_dim
        
        self.eq1 = PermutationEquivariantLayer(in_dim, hidden_dim)
        self.eq2 = PermutationEquivariantLayer(hidden_dim, hidden_dim)
        self.policy_correction = nn.Linear(hidden_dim, 1)
        self.val_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, wealth_t=1.0):
        batch_size, num_assets, _, _ = x.shape
        x_flat = x.reshape(batch_size, num_assets, -1)
        
        h = self.eq1(x_flat)
        h = self.eq2(h)
        
        x_t_est = x[:, :, -1, 1]
        pi_myopic = x_t_est / (self.gamma * (self.sigma_S ** 2))
        correction = self.policy_correction(h).squeeze(-1)
        
        weights = F.softmax(pi_myopic + correction, dim=-1)
        global_h = torch.mean(h, dim=1)
        val = self.val_head(global_h)
        
        return weights, val, torch.ones(batch_size, 1), torch.zeros(batch_size, num_assets), pi_myopic


class NoFrictionHeadNetwork(nn.Module):
    """
    Ablation Variant 3: Analytical Pontryagin Base Policy without Friction Soft-Thresholding.
    Uses pure Pontryagin analytical allocation without neural thresholding correction.
    """
    def __init__(self, lookback=10, feature_dim=2, hidden_dim=64, gamma=2.0, sigma_S=0.20):
        super().__init__()
        self.gamma = gamma
        self.sigma_S = sigma_S
        in_dim = lookback * feature_dim
        
        self.eq1 = PermutationEquivariantLayer(in_dim, hidden_dim)
        self.eq2 = PermutationEquivariantLayer(hidden_dim, hidden_dim)
        self.q_head = nn.Linear(hidden_dim, 1)
        self.val_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, wealth_t=1.0):
        batch_size, num_assets, _, _ = x.shape
        x_flat = x.reshape(batch_size, num_assets, -1)

        h = self.eq1(x_flat)
        h = self.eq2(h)

        x_t_est = x[:, :, -1, 1]
        q_t = self.q_head(h).squeeze(-1)

        pi_pure = x_t_est / (self.gamma * (self.sigma_S ** 2)) - 0.1 * q_t
        weights = F.softmax(pi_pure, dim=-1)

        global_h = torch.mean(h, dim=1)
        val = self.val_head(global_h)

        return weights, val, torch.ones(batch_size, 1), q_t, pi_pure


class NonEquivariantPGDPONetwork(nn.Module):
    """
    Standard PG-DPO adapted to Kim-Omberg environment (No S_N Permutation Equivariance).

    Differences from E-PGDPO (SMPGuidedPortfolioPolicy):
    - Uses standard MLP encoding instead of DeepSets PermutationEquivariantLayer.
    - Retains KO Pontryagin costate guidance (p_t, q_t) and friction correction.
    - Represents PG-DPO (Huh et al. 2024) extended to the KO non-stationary setting.

    This is the correct baseline for isolating the S_N equivariance contribution.
    """
    def __init__(self, num_assets=5, lookback=10, feature_dim=2, hidden_dim=64, gamma=2.0, sigma_S=0.20):
        super().__init__()
        self.gamma = gamma
        self.sigma_S = sigma_S
        self.num_assets = num_assets
        self.lookback = lookback
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        in_dim = num_assets * lookback * feature_dim

        # Standard MLP encoding (NO permutation equivariance)
        self.fc1 = nn.Linear(in_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)

        # Adjoint Variable Estimator (same as E-PGDPO)
        self.p_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
        self.q_head   = nn.Linear(hidden_dim, num_assets)
        self.policy_correction = nn.Linear(hidden_dim, num_assets)
        self.val_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def _resize_if_needed(self, num_assets, lookback, feature_dim):
        in_dim = num_assets * lookback * feature_dim
        if in_dim != self.fc1.in_features or num_assets != self.num_assets:
            device = next(self.parameters()).device
            self.fc1 = nn.Linear(in_dim, self.hidden_dim * 2).to(device)
            self.fc2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim).to(device)
            self.q_head = nn.Linear(self.hidden_dim, num_assets).to(device)
            self.policy_correction = nn.Linear(self.hidden_dim, num_assets).to(device)
            self.num_assets = num_assets

    def forward(self, x, wealth_t=1.0):
        batch_size, num_assets, lookback, feature_dim = x.shape
        self._resize_if_needed(num_assets, lookback, feature_dim)

        x_t_est = x[:, :, -1, 1]           # (batch, num_assets)
        x_flat  = x.reshape(batch_size, -1)  # flatten (NO equivariant structure)

        h = F.relu(self.fc1(x_flat))
        h = F.relu(self.fc2(h))

        p_t = self.p_head(h)                          # (batch, 1)
        q_t = self.q_head(h)                          # (batch, num_assets)

        # Same Pontryagin KO guidance as E-PGDPO but encoded without equivariance
        pi_myopic  = x_t_est / (self.gamma * (self.sigma_S ** 2))
        smp_logits = pi_myopic - 0.1 * q_t
        correction = self.policy_correction(h)         # (batch, num_assets)

        weights = F.softmax(smp_logits + correction, dim=-1)
        val     = self.val_head(h)

        return weights, val, p_t, q_t, smp_logits


if __name__ == "__main__":
    net = SMPGuidedPortfolioPolicy(lookback=10, feature_dim=2)
    sample_obs = torch.randn(2, 5, 10, 2)
    w, v, p, q, pi_smp = net(sample_obs)
    print("Full E-PGDPO policy weights shape:", w.shape, "Simplex sum:", torch.sum(w, dim=-1))
    
    mlp_net = UnconstrainedMLPNetwork(num_assets=5, lookback=10, feature_dim=2)
    w_mlp, _, _, _, _ = mlp_net(sample_obs)
    print("Unconstrained MLP policy weights shape:", w_mlp.shape)
