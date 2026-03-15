from aerohacks.policy.base import Policy
from aerohacks.core.models import Observation, Plan, ActionStep, ActionType, Position2D
import math
import json
import os
from dataclasses import asdict

class MyPolicy(Policy):
    """
    HEURISTIC ALGORITHM:
    1. Fly straight to the goal.
    2. If the path is blocked by a RESTRICTED zone, static obstacle, or traffic, trace the edge by rotating the velocity vector.
    3. Always prefer the straight path to the goal if it becomes available.
    """
    
    def __init__(self):
        self.log_file = "policy_output.jsonl"
        # Clear log file at start of new simulation run
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def is_blocked(self, pos: Position2D, alt: int, obs: Observation) -> bool:
        # 0. Check Map Boundaries
        if obs.map_boundaries and not obs.map_boundaries.contains(pos):
            return True

        # 1. Check Static Obstacles (Impassable at all layers)
        for region in obs.static_obstacles:
            if region.contains(pos):
                return True

        # 2. Check Restricted Airspace (Active + Permanent)
        all_constraints = obs.active_constraints + obs.permanent_constraints
        for c in all_constraints:
            if alt in c.alt_layers:
                phase = c.phase.value if hasattr(c.phase, 'value') else str(c.phase)
                if phase == "RESTRICTED":
                    if c.region.contains(pos):
                        return True
        
        # 3. Check Traffic (Avoidance)
        for t in obs.traffic_tracks:
            # Avoid drones at our altitude or adjacent layers
            if abs(t.alt_layer - alt) <= 1:
                dist = math.hypot(pos.x - t.position.x, pos.y - t.position.y)
                # Conflict separation is 50m, Collision is 20m.
                # We use 65m as a safe "bumping" distance.
                if dist < 65.0:
                    return True
        return False

    def get_avoidance_waypoint(self, current_pos: Position2D, goal_pos: Position2D, alt: int, obs: Observation) -> Position2D:
        dx = goal_pos.x - current_pos.x
        dy = goal_pos.y - current_pos.y
        dist_to_goal = math.hypot(dx, dy)
        
        if dist_to_goal < 1.0:
            return goal_pos
            
        angle_to_goal = math.atan2(dy, dx)
        # Maximum speed is 15.0m per tick
        speed = min(15.0, dist_to_goal)
        
        # 1. Try straight path first
        straight_pos = Position2D(
            x=current_pos.x + math.cos(angle_to_goal) * speed,
            y=current_pos.y + math.sin(angle_to_goal) * speed
        )
        if not self.is_blocked(straight_pos, alt, obs):
            return straight_pos

        # 2. Trace edge by rotating the velocity vector
        # We try rotating left/right in increasing increments
        for deg in range(10, 300, 10):
            for sign in [1, -1]: # Try both directions
                angle = angle_to_goal + sign * math.radians(deg)
                test_pos = Position2D(
                    x=current_pos.x + math.cos(angle) * speed,
                    y=current_pos.y + math.sin(angle) * speed
                )
                if not self.is_blocked(test_pos, alt, obs):
                    return test_pos
        
        # 3. If everything is blocked, hold position
        return current_pos

    def step(self, obs: Observation) -> Plan:
        own_state = obs.ownship_state
        current_pos = own_state.position
        goal_pos = obs.mission_goal.region.center()
        current_alt = own_state.alt_layer
        
        # Target altitude for the mission
        target_alt = obs.mission_goal.target_alt_layer if obs.mission_goal.target_alt_layer is not None else current_alt

        steps = []
        next_pos = current_pos
        
        # Generate 5 future steps
        for _ in range(5):
            next_pos = self.get_avoidance_waypoint(next_pos, goal_pos, current_alt, obs)
            steps.append(
                ActionStep(
                    action_type=ActionType.WAYPOINT,
                    target_position=next_pos,
                    target_alt_layer=target_alt
                )
            )
            
        return Plan(steps=steps)
