import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

# -------------------------------
# Replay Buffer Implementation
# -------------------------------
# Input the scaled transitions into the buffer
# Save the scalar and use as input arguments to the HIQLModel
class ReplayBuffer:
    def __init__(self, capacity, observation_shape, action_dim,
                 observation_scaler=None, action_scaler=None, reward_scaler=None):
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.action_dim = action_dim

        self.observation_scaler = observation_scaler
        self.action_scaler = action_scaler
        self.reward_scaler = reward_scaler

        self.observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=bool)

        self.size = 0
        self.ptr = 0
        print('Updated')

    # Use this function when adding the transitions to the buffer
    def add(self, obs, action, reward, next_obs, done):
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # Random sample of transitions
    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        batch = {
            "observations": torch.tensor(self.observations[indices]),
            "actions": torch.tensor(self.actions[indices]),
            "rewards": torch.tensor(self.rewards[indices]).unsqueeze(-1),
            "next_observations": torch.tensor(self.next_observations[indices]),
            "dones": torch.tensor(self.dones[indices], dtype=torch.float32).unsqueeze(-1)
        }
        return batch
    
# -------------------------------
# Neural Network Modules
# -------------------------------
# Simple MLP encoder used in policy, value, and Q networks.
class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU, dropout_rate=None):
        super(MLPEncoder, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            if dropout_rate is not None:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
# Policy Network: outputs mean and log_std for a Gaussian distribution (Continuous Policy)
class PolicyNetwork(nn.Module):
    def __init__(self, observation_dim, hidden_dims, action_dim):
        super(PolicyNetwork, self).__init__()
        self.encoder = MLPEncoder(observation_dim, hidden_dims, hidden_dims[-1])
        self.mean_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], action_dim),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x, discrete_action):
        x = torch.cat([x, discrete_action.float()], dim=-1)
        features = self.encoder(x)
        mean = self.mean_layer(features)
        std = torch.exp(self.log_std.clamp(-20, 2))
        return torch.distributions.Normal(mean, std)

    def sample(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob
    
from torch.distributions import Categorical
# Discrete policy: Outputs the categorical probability of the logits
class DiscretePolicyNetwork(nn.Module):
    def __init__(self, observation_dim, hidden_dims, action_dim):
        super(DiscretePolicyNetwork, self).__init__()
        self.encoder = MLPEncoder(observation_dim, hidden_dims, hidden_dims[-1])
        self._fc = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, x):
        return Categorical(logits=self._fc(self.encoder(x)))

    def sample(self, x):
        dist = self.forward(x)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob
    

# Value Network: estimates V(s)
class ValueNetwork(nn.Module):
    def __init__(self, observation_dim, hidden_dims):
        super(ValueNetwork, self).__init__()
        self.encoder = MLPEncoder(observation_dim, hidden_dims, 1)

    def forward(self, x):
        return self.encoder(x)
    
# Q Network: estimates Q(s, a)
class QNetwork(nn.Module):
    def __init__(self, observation_dim, hidden_dims, action_dim):
        super(QNetwork, self).__init__()
        self.encoder = MLPEncoder(observation_dim + action_dim, hidden_dims, 1)

    def forward(self, x, a):
        inp = torch.cat([x, a], dim=-1)
        return self.encoder(inp)

# -------------------------------
# H-IQL Model Implementation
# -------------------------------
# Recommended fintuning using gamma, tau, expectile, and weight temp
# See original paper of IQL to understand their affection.
# Scalers are optional but if used, make sure you input the same scalers 
# used to scale the transitions in the buffer.
# Set cos_lr to number of traning iterations.
# The class is built with the structure [Continuous Discrete], hence this structure
# must be followed 
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils

class HIQLModel:
    # Initializes the networks and hyperparameters
    def __init__(self, observation_dim, discrete_dim, continuous_dim, hidden_dims, gamma=0.99, 
                 tau=0.005, expectile=0.7, weight_temp=3.0, max_weight=100.0, cos_lr=200000, zeta=0.5,
                 device='cpu', observation_scaler=None, action_scaler=None, reward_scaler=None):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.expectile = expectile
        self.weight_temp = weight_temp
        self.max_weight = max_weight
        self.action_dim = discrete_dim + continuous_dim
        self.discrete_dim = discrete_dim
        self.continuous_dim = continuous_dim

        self.observation_scaler = observation_scaler
        self.action_scaler = action_scaler
        self.reward_scaler = reward_scaler  

        self.policy = PolicyNetwork(observation_dim + self.discrete_dim, hidden_dims, self.continuous_dim).to(device)
        self.policy_d = DiscretePolicyNetwork(observation_dim, hidden_dims, discrete_dim).to(device)
        self.value_net = ValueNetwork(observation_dim, hidden_dims).to(device)

        # Initialize two Q-networks for double Q-learning:
        self.q_net1 = QNetwork(observation_dim, hidden_dims, self.action_dim).to(device)
        self.q_net2 = QNetwork(observation_dim, hidden_dims, self.action_dim).to(device)

        # Target networks:
        self.target_q_net1 = QNetwork(observation_dim, hidden_dims, self.action_dim).to(device)
        self.target_q_net2 = QNetwork(observation_dim, hidden_dims, self.action_dim).to(device)
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.policy_d_optimizer = optim.Adam(self.policy_d.parameters(), lr=3e-4)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=3e-4)
        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=3e-4)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=3e-4)
        self.actor_lr_schedule = CosineAnnealingLR(self.policy_optimizer, cos_lr)

        self.max_grad_norm = 1.0  
        self.zeta = zeta 
        
    # Updates the q and v networks
    def update_critic_value(self, batch):
        observations = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_obs = batch["next_observations"].to(self.device)
        dones = batch["dones"].to(self.device)

        # --- Update Q-networks ---
        with torch.no_grad():
            # Use the value network for the target
            target_v = self.value_net(next_obs)
            q_target = rewards + self.gamma * (1.0 - dones) * target_v
        
        q_pred1 = self.q_net1(observations, actions)
        q_pred2 = self.q_net2(observations, actions)
        q_loss1 = F.mse_loss(q_pred1, q_target)
        q_loss2 = F.mse_loss(q_pred2, q_target)
        q_loss = q_loss1 + q_loss2

        self.q_optimizer1.zero_grad()
        self.q_optimizer2.zero_grad()
        q_loss.backward()

        utils.clip_grad_norm_(self.q_net1.parameters(), self.max_grad_norm)
        utils.clip_grad_norm_(self.q_net2.parameters(), self.max_grad_norm)
        self.q_optimizer1.step()
        self.q_optimizer2.step()

        # --- Update Value Network using Expectile Regression ---
        with torch.no_grad():
            q1_detached = self.target_q_net1(observations, actions)
            q2_detached = self.target_q_net2(observations, actions)
            q_detached = torch.min(q1_detached, q2_detached)
        v_pred = self.value_net(observations)
        diff = q_detached - v_pred

        # For diff >= 0, weight = expectile; for diff < 0, weight = (1 - expectile)
        weight_v = torch.abs(self.expectile - (diff < 0).float())
        value_loss = (weight_v * diff.pow(2)).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        
        utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
        self.value_optimizer.step()

        # --- Soft Update Target Q Networks ---
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "q_loss": q_loss.item(),
            "value_loss": value_loss.item()
        }

    # Updates the policy networks using adaptive zeta
    def update_actor(self, batch):
        observations = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        continuous_actions = actions[:, :self.continuous_dim]
        discrete_actions = actions[:, self.continuous_dim:].long()

        dist_d = self.policy_d(observations)
        log_prob_d = dist_d.log_prob(discrete_actions.argmax(dim=-1)).unsqueeze(-1)
        
        dist_c = self.policy(observations, discrete_actions)
        log_prob_c = dist_c.log_prob(continuous_actions).sum(dim=-1, keepdim=True)
         
        ### Introduce a hyperparameter that is weighted for the harder learned policy
        log_probs_d = -log_prob_d 
        log_probs_c = -log_prob_c  
         
        log_probs = (1 - self.zeta) * log_probs_d + self.zeta * log_probs_c

        with torch.no_grad():
            q1 = self.target_q_net1(observations, actions)
            q2 = self.target_q_net2(observations, actions)
            q_value = torch.min(q1, q2)
            v_val = self.value_net(observations)
            adv = q_value - v_val
            weights = torch.exp(self.weight_temp * adv).clamp(max=self.max_weight)

        actor_loss = torch.mean((weights * log_probs))

        self.policy_optimizer.zero_grad()
        self.policy_d_optimizer.zero_grad()

        actor_loss.backward()

        # Change zeta depending on the sum of gradient changes
        grad_norm_c = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm).item()
        grad_norm_d = torch.nn.utils.clip_grad_norm_(self.policy_d.parameters(), self.max_grad_norm).item()
        grad_sum = grad_norm_c + grad_norm_d

        self.zeta = max(0.01, min(0.99, 0.99 * self.zeta + 0.01 * (grad_norm_c / grad_sum)))
        
        self.policy_optimizer.step()
        self.policy_d_optimizer.step()
        
        return {"actor_loss": actor_loss.item(),
                "Zeta": self.zeta}

    # Use to predict action, given a state
    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)

        if self.observation_scaler is not None:
            # If only parts of the observation is supposed to be scaled, set it here
            cont = self.observation_scaler.transform(x[:, :self.observation_scaler.n_features_in_])
            cat = x[:, self.observation_scaler.n_features_in_:]
            x = np.concatenate([cont, cat], axis=1)

        torch_x = torch.tensor(x, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            dist_d = self.policy_d.forward(torch_x)
            discrete_idx = dist_d.probs.argmax(dim=-1).item()
            discrete_action_onehot = torch.zeros(1, self.discrete_dim, device=self.device)
            discrete_action_onehot[0, discrete_idx] = 1

            dist = self.policy.forward(torch_x, discrete_action_onehot)
            continuous_action = dist.mean  

        # Inverse transform action into original ranges 
        if self.action_scaler is not None:
            cont_act_np = continuous_action.detach().cpu().numpy()
            cont_act_np = self.action_scaler.inverse_transform(cont_act_np)
            cont_act = torch.tensor(cont_act_np, dtype=torch.float32, device=self.device)
        else:
            cont_act = continuous_action
        
        action = torch.cat([cont_act, discrete_action_onehot], dim=1)
        return action.cpu().numpy().squeeze()

    # Predict Q-value given a state and an action
    def predict_value(self, x: np.ndarray, action: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        if action.ndim == 1:
            action = np.expand_dims(action, axis=0)
        if self.observation_scaler is not None:
            cont_x = self.observation_scaler.transform(x[:, :self.continuous_dim])
            cat_x = x[:, self.continuous_dim:]
            x = np.concatenate([cont_x, cat_x], axis=1)
        if self.action_scaler is not None:
            actions_cont = self.action_scaler.transform(action[:, :self.continuous_dim])
            actions_cat = action[:, self.continuous_dim:]
            action = np.concatenate([actions_cont, actions_cat], axis=1)

        torch_x = torch.tensor(x, dtype=torch.float32, device=self.device)
        torch_action = torch.tensor(action, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            value = self.q_net1(torch_x, torch_action)  
        return value.cpu().detach().numpy().squeeze(-1)
    
    # Predict state value given a state
    def predict_state_value(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        if self.observation_scaler is not None:
            continuous_x = self.observation_scaler.transform(x[:, :self.continuous_dim])
            categorical_x = x[:, self.continuous_dim:]
            x = np.concatenate([continuous_x, categorical_x], axis=1)
         
        torch_x = torch.tensor(x, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            state_value = self.value_net(torch_x)

        return state_value.cpu().detach().numpy().squeeze(-1)


# -------------------------------
# Scalers
# -------------------------------
# You might as well use sklearn scalers 
class StandardScaler:
    def __init__(self, eps: float = 1e-8):
        self.mean = None
        self.std = None
        self.eps = eps

    def fit(self, data: np.ndarray):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        # Avoid division by zero.
        self.std[self.std < self.eps] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")
        return data * self.std + self.mean

    def transform_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")
        
        mean = torch.tensor(self.mean, dtype=tensor.dtype, device=tensor.device)
        std = torch.tensor(self.std, dtype=tensor.dtype, device=tensor.device)
        # Adjust dimensions if needed
        while mean.ndim < tensor.ndim:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return (tensor - mean) / (std + self.eps)

    def inverse_transform_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")
        mean = torch.tensor(self.mean, dtype=tensor.dtype, device=tensor.device)
        std = torch.tensor(self.std, dtype=tensor.dtype, device=tensor.device)
        while mean.ndim < tensor.ndim:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return tensor * std + mean

class MinMaxScaler:
    def __init__(self, target_min: float = -1.0, target_max: float = 1.0, eps: float = 1e-8):
        self.data_min = None
        self.data_max = None
        self.data_range = None
        self.target_min = target_min
        self.target_max = target_max
        self.eps = eps

    def fit(self, data: np.ndarray):
        self.data_min = np.min(data, axis=0)
        self.data_max = np.max(data, axis=0)
        self.data_range = self.data_max - self.data_min
        self.data_range[self.data_range < self.eps] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        scale = (self.target_max - self.target_min) / self.data_range
        return self.target_min + (data - self.data_min) * scale

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        scale = self.data_range / (self.target_max - self.target_min)
        return self.data_min + (data - self.target_min) * scale

    def transform_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        data_min = torch.tensor(self.data_min, dtype=tensor.dtype, device=tensor.device)
        data_range = torch.tensor(self.data_range, dtype=tensor.dtype, device=tensor.device)
        target_min = self.target_min
        target_max = self.target_max

        while data_min.ndim < tensor.ndim:
            data_min = data_min.unsqueeze(0)
            data_range = data_range.unsqueeze(0)

        return target_min + (tensor - data_min) * (target_max - target_min) / (data_range + self.eps)

    def inverse_transform_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        data_min = torch.tensor(self.data_min, dtype=tensor.dtype, device=tensor.device)
        data_range = torch.tensor(self.data_range, dtype=tensor.dtype, device=tensor.device)
        target_min = self.target_min
        target_max = self.target_max

        while data_min.ndim < tensor.ndim:
            data_min = data_min.unsqueeze(0)
            data_range = data_range.unsqueeze(0)

        return data_min + (tensor - target_min) * data_range / (target_max - target_min + self.eps)