from aerohacks.policy.base import Policy
from aerohacks.core.models import Observation, Plan, ActionStep, ActionType, Position2D, State
import math
import heapq as hq
from dataclasses import dataclass
from typing import List, Set, Tuple, Optional, Dict

@dataclass(frozen=True, order=True)
class Node:
    """Graph node with snapped coordinates for robust hashing."""
    x: float
    y: float
    alt: int

    @classmethod
    def from_pos(cls, pos: Position2D, alt: int):
        return cls(round(pos.x, 1), round(pos.y, 1), int(alt))

class SquareRegion:
    """Bounding box representation of a circle for faster intersection checks."""
    def __init__(self, center: Position2D, radius: float):
        self.min_x = center.x - radius
        self.max_x = center.x + radius
        self.min_y = center.y - radius
        self.max_y = center.y + radius
    def contains(self, pos: Position2D) -> bool:
        return self.min_x <= pos.x <= self.max_x and self.min_y <= pos.y <= self.max_y
    def center(self):
        return Position2D((self.min_x + self.max_x)/2, (self.min_y + self.max_y)/2)

class Costmap3D:
    def __init__(self):
        self.constraints = []
        self.static_obstacles = []
        self.map_boundaries = None
        self.traffic = []

    def update(self, obs: Observation):
        self.constraints = obs.active_constraints + obs.permanent_constraints
        self.static_obstacles = obs.static_obstacles
        self.map_boundaries = obs.map_boundaries
        self.traffic = [(t.position.x, t.position.y, t.alt_layer) for t in obs.traffic_tracks]

    def get_edge_cost(self, u: Node, v: Node) -> float:
        dist = math.hypot(u.x - v.x, u.y - v.y)
        dz = abs(u.alt - v.alt)
        total_dist = dist + dz * 30.0

        num_samples = 5 # Constant samples for performance
        for i in range(1, num_samples + 1):
            t = i / num_samples
            sx, sy = u.x + (v.x - u.x) * t, u.y + (v.y - u.y) * t
            salt = int(round(u.alt + (v.alt - u.alt) * t))
            sample_pos = Position2D(sx, sy)
            
            if self.map_boundaries and not self.map_boundaries.contains(sample_pos):
                return float('inf')

            for region in self.static_obstacles:
                if region.contains(sample_pos): return float('inf')

            for c in self.constraints:
                if salt in c.alt_layers:
                    if c.region.contains(sample_pos):
                        phase = c.phase.value if hasattr(c.phase, 'value') else str(c.phase)
                        if phase == "RESTRICTED": return float('inf')
                        total_dist += (2000.0 if phase == "CONTROLLED" else 200.0) / num_samples

        return total_dist

class DStarLite:
    def __init__(self, s_start: Node, s_goal: Node):
        self.s_start = s_start
        self.s_goal = s_goal
        self.s_last = s_start
        self.km = 0.0
        self.g = {}
        self.rhs = {s_goal: 0.0}
        self.pq = [] 
        self._counter = 0
        self.all_nodes: Set[Node] = set([s_start, s_goal])
        self.neighbors: Dict[Node, List[Node]] = {}
        self.costmap = Costmap3D()
        self._push(self.s_goal, self._calculate_key(self.s_goal))

    def _heuristic(self, a: Node, b: Node) -> float:
        return math.hypot(a.x - b.x, a.y - b.y) + abs(a.alt - b.alt) * 30.0

    def _calculate_key(self, s: Node) -> tuple[float, float]:
        g_s = self.g.get(s, float('inf'))
        rhs_s = self.rhs.get(s, float('inf'))
        m = min(g_s, rhs_s)
        return (m + self._heuristic(self.s_start, s) + self.km, m)

    def _push(self, s: Node, key: tuple[float, float]):
        self._counter += 1
        hq.heappush(self.pq, (key, self._counter, s))

    def _update_vertex(self, u: Node):
        if u != self.s_goal:
            best = float('inf')
            for v in self.neighbors.get(u, []):
                g_v = self.g.get(v, float('inf'))
                val = self.costmap.get_edge_cost(u, v) + g_v
                if val < best: best = val
            self.rhs[u] = best
        
        if self.g.get(u, float('inf')) != self.rhs.get(u, float('inf')):
            self._push(u, self._calculate_key(u))

    def compute_shortest_path(self):
        max_exp = 5000 # Increased for robustness
        while self.pq and (self.pq[0][0] < self._calculate_key(self.s_start) or 
                          self.rhs.get(self.s_start, float('inf')) != self.g.get(self.s_start, float('inf'))):
            if max_exp <= 0: break
            max_exp -= 1
            
            k_old, _, u = hq.heappop(self.pq)
            k_new = self._calculate_key(u)
            
            if k_old < k_new:
                self._push(u, k_new)
            elif self.g.get(u, float('inf')) > self.rhs.get(u, float('inf')):
                self.g[u] = self.rhs[u]
                for s in self.neighbors.get(u, []):
                    self._update_vertex(s)
            else:
                self.g[u] = float('inf')
                for s in self.neighbors.get(u, []) + [u]:
                    self._update_vertex(s)

    def update_graph(self, incoming_nodes: Set[Node]):
        new_nodes = incoming_nodes - self.all_nodes
        if not new_nodes: return

        self.all_nodes.update(new_nodes)
        nodes_list = list(self.all_nodes)
        
        for u in new_nodes:
            # Use 8 closest nodes to ensure connectivity across gaps
            dists = [(math.hypot(u.x - v.x, u.y - v.y), v) for v in nodes_list if u != v]
            dists.sort()
            self.neighbors[u] = [v for d, v in dists[:8]]
            
            for v in self.neighbors[u]:
                if v not in self.neighbors: self.neighbors[v] = []
                if u not in self.neighbors[v]:
                    v_dists = [(math.hypot(v.x - n.x, v.y - n.y), n) for n in (self.neighbors[v] + [u])]
                    v_dists.sort()
                    self.neighbors[v] = [n for d, n in v_dists[:8]]
                    self._update_vertex(v)

    def move_start(self, s_new: Node):
        if s_new != self.s_last:
            self.km += self._heuristic(self.s_last, s_new)
            self.s_last = s_new
            self.s_start = s_new

class MyPolicy(Policy):
    def __init__(self):
        self.planner = None
        self.static_nodes = set()
        self.initialized_static = False

    def _get_strategic_points(self, region, alt: int) -> List[Node]:
        if hasattr(region, 'vertices'):
            return [Node.from_pos(v, alt) for v in region.vertices]
        if hasattr(region, 'center_pos') and hasattr(region, 'radius'):
            cp, r = region.center_pos, region.radius
            return [Node(round(cp.x+dx, 1), round(cp.y+dy, 1), alt) 
                    for dx, dy in [(-r,-r), (-r,r), (r,-r), (r,r)]]
        return []

    def step(self, obs: Observation) -> Plan:
        own = obs.ownship_state
        goal_center = obs.mission_goal.region.center()
        own_alt = own.alt_layer
        goal_alt = obs.mission_goal.target_alt_layer or own_alt
        
        s_start = Node.from_pos(own.position, own_alt)
        s_goal = Node.from_pos(goal_center, goal_alt)

        # 1. Initialize Static Nodes for ALL layers (1-5)
        if not self.initialized_static:
            for alt in range(1, 6):
                if hasattr(obs.map_boundaries, 'vertices'):
                    for v in obs.map_boundaries.vertices:
                        self.static_nodes.add(Node.from_pos(v, alt))
                for c in obs.permanent_constraints:
                    if alt in c.alt_layers:
                        self.static_nodes.update(self._get_strategic_points(c.region, alt))
                for region in obs.static_obstacles:
                    self.static_nodes.update(self._get_strategic_points(region, alt))
                for site in obs.emergency_landing_sites:
                    self.static_nodes.update(self._get_strategic_points(site.region, alt))
            self.initialized_static = True

        # 2. Extract Dynamic Nodes
        dynamic_nodes = {s_start, s_goal}
        
        restricted_regions = []
        for c in (obs.active_constraints + obs.permanent_constraints):
            phase = c.phase.value if hasattr(c.phase, 'value') else str(c.phase)
            if phase == "RESTRICTED" and own_alt in c.alt_layers:
                if hasattr(c.region, 'center_pos'):
                    restricted_regions.append(SquareRegion(c.region.center_pos, c.region.radius))
                else:
                    restricted_regions.append(c.region)

        for c in obs.active_constraints:
            if own_alt in c.alt_layers:
                pts = self._get_strategic_points(c.region, own_alt)
                for p in pts:
                    is_redundant = False
                    for r in restricted_regions:
                        if r.contains(Position2D(p.x, p.y)):
                            rc = r.center()
                            if math.hypot(p.x - rc.x, p.y - rc.y) < 1.0:
                                is_redundant = True
                                break
                    if not is_redundant: dynamic_nodes.add(p)

        for t in obs.traffic_tracks:
            if abs(t.alt_layer - own_alt) <= 1:
                dynamic_nodes.add(Node.from_pos(t.position, t.alt_layer))

        # 3. Update Planner
        if self.planner is None:
            self.planner = DStarLite(s_start, s_goal)
        
        all_nodes = self.static_nodes | dynamic_nodes
        self.planner.update_graph(all_nodes)
        self.planner.move_start(s_start)
        self.planner.costmap.update(obs)
        
        for n in dynamic_nodes:
            self.planner._update_vertex(n)
        self.planner.compute_shortest_path()
        
        # 4. Extract Path
        path = [s_start]
        curr, visited = s_start, {s_start}
        for _ in range(20):
            best_n, best_v = None, float('inf')
            for n in self.planner.neighbors.get(curr, []):
                if n in visited: continue
                v = self.planner.costmap.get_edge_cost(curr, n) + self.planner.g.get(n, float('inf'))
                if v < best_v: best_v, best_n = v, n
            if best_n:
                path.append(best_n)
                visited.add(best_n)
                curr = best_n
                if curr == s_goal: break
            else: break

        # 5. Interpolate steps
        steps = []
        cp, ca = own.position, own_alt
        path_idx = 1
        for _ in range(5):
            if path_idx < len(path):
                target = path[path_idx]
                dx, dy = target.x - cp.x, target.y - cp.y
                d = math.hypot(dx, dy)
                step_d = min(15.0, d)
                next_pos = Position2D(cp.x + (dx/d)*step_d, cp.y + (dy/d)*step_d) if d > 0 else Position2D(target.x, target.y)
                ca = target.alt if step_d >= d else ca
                steps.append(ActionStep(ActionType.WAYPOINT, next_pos, ca))
                cp = next_pos
                if math.hypot(cp.x - target.x, cp.y - target.y) < 1.0: path_idx += 1
            else:
                steps.append(ActionStep(ActionType.WAYPOINT, cp, ca))
        return Plan(steps=steps)
