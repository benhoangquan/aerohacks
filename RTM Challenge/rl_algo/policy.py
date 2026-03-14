import torch
import torch.nn as nn
import numpy as np
import math
from typing import List
from aerohacks.policy.base import Policy
from aerohacks.core.models import Observation, Plan, ActionStep, ActionType, Position2D, Constraint, TrafficTrack

# -------------------------------------------------------------------------
# 1. THE ARCHITECTURE: Inspired by 1000-layer GCRL Paper
# -------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Using SiLU (Swish) and LayerNorm as suggested for stability in deep RL
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
    def forward(self, x):
        return x + self.net(x)

class DeepPolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, depth=20):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Scaling depth: Start with 20, can scale to 1000 if training is stable
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(depth)])
        
        # Output: 15 values (5 steps * [delta_x, delta_y, alt_layer])
        self.output_layer = nn.Linear(hidden_dim, 15)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)

# -------------------------------------------------------------------------
# 2. THE VECTORIZER: Converting Objects to Tensors
# -------------------------------------------------------------------------
def vectorize_observation(obs: Observation, num_traffic: int = 3, num_constraints: int = 2) -> torch.Tensor:
    """
    Converts the complex Observation dataclass into a fixed-size float tensor.
    HACKATHON TIP: Use RELATIVE coordinates (Goal - Self) so the model 
    doesn't have to learn the whole map's absolute coordinates.
    """
    own = obs.ownship_state
    goal = obs.mission_goal.region.center()
    
    # [1] Ownship State (4 features)
    # Energy is normalized (assuming max is around 500-1000)
    own_vec = [own.alt_layer / 5.0, own.energy / 500.0, own.velocity.x / 15.0, own.velocity.y / 15.0]
    
    # [2] Goal State (3 features - Relative)
    goal_vec = [
        (goal.x - own.position.x) / 1000.0, 
        (goal.y - own.position.y) / 1000.0, 
        (obs.mission_goal.target_alt_layer or own.alt_layer) / 5.0
    ]
    
    # [3] Traffic (Closest N drones - 4 features each)
    traffic_vec = []
    sorted_traffic = sorted(obs.traffic_tracks, 
                            key=lambda t: math.hypot(t.position.x - own.position.x, t.position.y - own.position.y))
    for i in range(num_traffic):
        if i < len(sorted_traffic):
            t = sorted_traffic[i]
            traffic_vec += [
                (t.position.x - own.position.x) / 500.0,
                (t.position.y - own.position.y) / 500.0,
                (t.alt_layer - own.alt_layer) / 2.0,
                (t.velocity.x if t.velocity else 0.0) / 15.0
            ]
        else:
            traffic_vec += [0.0, 0.0, 0.0, 0.0] # Padding

    # [4] Constraints/No-Fly Zones (Closest N constraints - 3 features each)
    # This is a simplification: treating centers as point obstacles
    const_vec = []
    sorted_constraints = sorted(obs.active_constraints, 
                                key=lambda c: math.hypot(c.region.center().x - own.position.x, c.region.center().y - own.position.y))
    for i in range(num_constraints):
        if i < len(sorted_constraints):
            c = sorted_constraints[i]
            center = c.region.center()
            const_vec += [
                (center.x - own.position.x) / 500.0,
                (center.y - own.position.y) / 500.0,
                1.0 if c.phase.value == "RESTRICTED" else 0.5
            ]
        else:
            const_vec += [0.0, 0.0, 0.0]

    # Combined Input Size: 4 + 3 + (3*4) + (2*3) = 25 features
    return torch.tensor(own_vec + goal_vec + traffic_vec + const_vec, dtype=torch.float32)

# -------------------------------------------------------------------------
# 3. THE POLICY WRAPPER
# -------------------------------------------------------------------------
class MyPolicy(Policy):
    def __init__(self):
        self.input_dim = 25 # Must match vectorize_observation output
        self.model = DeepPolicyNet(input_dim=self.input_dim, hidden_dim=128, depth=20)
        
        # TO DO: After training, uncomment this!
        # self.model.load_state_dict(torch.load("policy_model.pth"))
        self.model.eval()

    def step(self, obs: Observation) -> Plan:
        # 1. Check if we have a trained model
        # For now, let's use the Heuristic to generate data!
        # return self.heuristic_step(obs)
        
        # 2. Real Inference
        with torch.no_grad():
            state_tensor = vectorize_observation(obs).unsqueeze(0)
            prediction = self.model(state_tensor).view(5, 3) # [5 steps, 3 features]
        
        steps = []
        current_pos = obs.ownship_state.position
        for i in range(5):
            # Model predicts RELATIVE waypoints to current pos for better stability
            dx, dy, alt = prediction[i].tolist()
            # Scaling back up (since model outputs small numbers)
            target_x = current_pos.x + (dx * 100.0) 
            target_y = current_pos.y + (dy * 100.0)
            target_alt = int(np.clip(alt * 5.0, 0, 5))
            
            steps.append(ActionStep(
                ActionType.WAYPOINT, 
                Position2D(target_x, target_y), 
                target_alt
            ))
        return Plan(steps=steps)

    def heuristic_step(self, obs: Observation) -> Plan:
        """
        YOUR GOAL: Implement a basic 'Safe' pathfinder here.
        This doesn't need to be perfect, just 'expert enough' to 
        teach the neural network the basics.
        """
        # (Reference your original straight-line-to-goal code here)
        
        
