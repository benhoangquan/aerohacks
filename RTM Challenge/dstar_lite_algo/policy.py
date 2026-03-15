from aerohacks.policy.base import Policy
from aerohacks.core.models import Observation, Plan, ActionStep, ActionType, Position2D
import math
import heapq
from typing import List, Tuple, Dict, Set, Optional

class PriorityQueue:
    def __init__(self):
        self.elements = []
        self.entry_finder = {}
        self.REMOVED = '<removed-node>'
        self.counter = 0

    def insert(self, node, priority):
        if node in self.entry_finder:
            self.remove(node)
        self.counter += 1
        entry = [priority, self.counter, node]
        self.entry_finder[node] = entry
        heapq.heappush(self.elements, entry)

    def remove(self, node):
        entry = self.entry_finder.pop(node)
        entry[-1] = self.REMOVED

    def top(self):
        while self.elements:
            priority, count, node = self.elements[0]
            if node is not self.REMOVED:
                return priority, node
            heapq.heappop(self.elements)
        return ([float('inf'), float('inf')], None)

    def pop(self):
        while self.elements:
            priority, count, node = heapq.heappop(self.elements)
            if node is not self.REMOVED:
                del self.entry_finder[node]
                return priority, node
        return ([float('inf'), float('inf')], None)
        
    def has(self, node):
        return node in self.entry_finder

class DStarLite:
    def __init__(self, s_start, s_goal):
        self.s_start = s_start
        self.s_goal = s_goal
        self.k_m = 0.0
        self.U = PriorityQueue()
        self.rhs = {}
        self.g = {}
        self.edges = {}
        self.nodes = {}
        
        if self.s_goal is not None:
            self.U.insert(self.s_goal, self.CalculateKey(self.s_goal))
            self.rhs[self.s_goal] = 0.0

    def get_g(self, u): return self.g.get(u, float('inf'))
    def get_rhs(self, u): return self.rhs.get(u, float('inf'))

    def heuristic_pos(self, pos_u, pos_v):
        dx = pos_u[0] - pos_v[0]
        dy = pos_u[1] - pos_v[1]
        dz = pos_u[2] - pos_v[2]
        return math.hypot(dx, dy) + abs(dz)*20.0

    def heuristic(self, u, v):
        if u not in self.nodes or v not in self.nodes: return float('inf')
        return self.heuristic_pos(self.nodes[u], self.nodes[v])

    def CalculateKey(self, s):
        g_s = self.get_g(s)
        rhs_s = self.get_rhs(s)
        min_g_rhs = min(g_s, rhs_s)
        return [min_g_rhs + self.heuristic(self.s_start, s) + self.k_m, min_g_rhs]

    def UpdateVertex(self, u):
        if u != self.s_goal:
            min_rhs = float('inf')
            for v, cost in self.edges.get(u, {}).items():
                min_rhs = min(min_rhs, cost + self.get_g(v))
            self.rhs[u] = min_rhs
            
        if self.U.has(u):
            self.U.remove(u)
            
        if self.get_g(u) != self.get_rhs(u):
            self.U.insert(u, self.CalculateKey(u))

    def ComputeShortestPath(self):
        iterations = 0
        while iterations < 10000:
            iterations += 1
            top = self.U.top()
            if top[1] is None: break
            k_old, u = top
            
            k_start = self.CalculateKey(self.s_start)
            if k_old >= k_start and self.get_rhs(self.s_start) == self.get_g(self.s_start):
                break
                
            self.U.pop()
            k_new = self.CalculateKey(u)
            
            if k_old < k_new:
                self.U.insert(u, k_new)
            elif self.get_g(u) > self.get_rhs(u):
                self.g[u] = self.get_rhs(u)
                for v in self.edges.get(u, {}):
                    self.UpdateVertex(v)
            else:
                self.g[u] = float('inf')
                self.UpdateVertex(u)
                for v in self.edges.get(u, {}):
                    self.UpdateVertex(v)

class MyPolicy(Policy):
    def __init__(self):
        self.dstar = None
        
    def ccw(self, A, B, C):
        return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)

    def proper_intersect(self, A, B, C, D):
        if (abs(A.x - C.x) < 1e-3 and abs(A.y - C.y) < 1e-3) or \
           (abs(A.x - D.x) < 1e-3 and abs(A.y - D.y) < 1e-3) or \
           (abs(B.x - C.x) < 1e-3 and abs(B.y - C.y) < 1e-3) or \
           (abs(B.x - D.x) < 1e-3 and abs(B.y - D.y) < 1e-3):
            return False
        return self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D)

    def is_point_in_polygon(self, point, vertices):
        x, y = point.x, point.y
        inside = False
        j = len(vertices) - 1
        for i in range(len(vertices)):
            xi, yi = vertices[i].x, vertices[i].y
            xj, yj = vertices[j].x, vertices[j].y
            intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi)
            if intersect:
                inside = not inside
            j = i
        return inside

    def is_point_strictly_in_polygon(self, point, vertices):
        if not self.is_point_in_polygon(point, vertices): return False
        for i in range(len(vertices)):
            A = vertices[i-1]
            B = vertices[i]
            l2 = (A.x - B.x)**2 + (A.y - B.y)**2
            if l2 == 0: continue
            t = max(0.0, min(1.0, ((point.x - A.x)*(B.x - A.x) + (point.y - A.y)*(B.y - A.y)) / l2))
            proj_x = A.x + t * (B.x - A.x)
            proj_y = A.y + t * (B.y - A.y)
            if math.hypot(point.x - proj_x, point.y - proj_y) < 1.0:
                return False
        return True

    def is_edge_valid(self, A, B, poly, is_start_or_goal=False):
        mid = Position2D(x=(A.x+B.x)/2.0, y=(A.y+B.y)/2.0)
        n = len(poly)
        for i in range(n):
            C = poly[i]
            D = poly[(i+1)%n]
            if self.proper_intersect(A, B, C, D):
                return False
        # Do not block paths originating from start/goal if they happen to be inside the buffer
        if not is_start_or_goal:
            if self.is_point_strictly_in_polygon(mid, poly):
                return False
        return True

    def circle_to_polygon(self, center, radius, expand_r, num_points=8):
        # We must circumscribe the polygon so straight edges don't cut inside the circle.
        # The apothem (closest distance to edge) is radius + expand_r.
        apothem = radius * 1.2 # Scale by radius might give a more safe margin
        r = apothem / math.cos(math.pi / num_points)
        verts = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            verts.append(Position2D(x=center.x + r * math.cos(angle), y=center.y + r * math.sin(angle)))
        return verts

    def expand_polygon(self, vertices, expand_r):
        if len(vertices) < 3: return vertices
        exp = []
        n = len(vertices)
        
        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i].x * vertices[j].y - vertices[j].x * vertices[i].y
        is_ccw = area > 0

        for i in range(n):
            prev_v = vertices[i-1]
            curr_v = vertices[i]
            next_v = vertices[(i+1)%n]
            
            e1x = curr_v.x - prev_v.x
            e1y = curr_v.y - prev_v.y
            e2x = next_v.x - curr_v.x
            e2y = next_v.y - curr_v.y
            
            l1 = math.hypot(e1x, e1y) or 1e-9
            l2 = math.hypot(e2x, e2y) or 1e-9
            e1x, e1y = e1x/l1, e1y/l1
            e2x, e2y = e2x/l2, e2y/l2
            
            if is_ccw:
                n1x, n1y = e1y, -e1x
                n2x, n2y = e2y, -e2x
            else:
                n1x, n1y = -e1y, e1x
                n2x, n2y = -e2y, e2x
                
            nx = n1x + n2x
            ny = n1y + n2y
            nl = math.hypot(nx, ny) or 1e-9
            nx, ny = nx/nl, ny/nl
            
            cos_theta = nx * n1x + ny * n1y
            if cos_theta < 0.1: cos_theta = 0.1
            dist = expand_r / cos_theta
            
            exp.append(Position2D(x=curr_v.x + nx * dist, y=curr_v.y + ny * dist))
        return exp

    def get_poly(self, region, expand_r):
        if hasattr(region, 'radius'):
            return self.circle_to_polygon(region.center_pos, region.radius, expand_r)
        else:
            return self.expand_polygon(region.vertices, expand_r)

    def get_box(self, center, size):
        r = size / 2.0
        return [
            Position2D(x=center.x - r, y=center.y - r),
            Position2D(x=center.x + r, y=center.y - r),
            Position2D(x=center.x + r, y=center.y + r),
            Position2D(x=center.x - r, y=center.y + r)
        ]

    def get_region_center(self, region) -> Position2D:
        if hasattr(region, 'center_pos'):
            return region.center_pos
        elif hasattr(region, 'vertices') and len(region.vertices) > 0:
            x = sum(v.x for v in region.vertices) / len(region.vertices)
            y = sum(v.y for v in region.vertices) / len(region.vertices)
            return Position2D(x=x, y=y)
        return Position2D(0.0, 0.0)

    def is_inside_map(self, point, map_vertices):
        if not map_vertices: return True
        return self.is_point_in_polygon(point, map_vertices)

    def compute_all_edges(self, obs, all_nodes):
        edges = {}
        for u in all_nodes:
            edges[u] = {}
            
        restricted_polys = []
        penalty_polys = []
        
        map_verts = []
        if hasattr(obs, 'map_boundaries') and hasattr(obs.map_boundaries, 'vertices'):
            map_verts = self.expand_polygon(obs.map_boundaries.vertices, 10.0)
            
        for obs_reg in obs.static_obstacles:
             restricted_polys.append((self.get_poly(obs_reg, 22.0), None))
             
        for c in obs.active_constraints:
             phase = c.phase.name if hasattr(c.phase, 'name') else str(c.phase)
             poly = self.get_poly(c.region, 22.0)
             if 'RESTRICTED' in phase:
                 restricted_polys.append((poly, c.alt_layers))
             elif 'CONTROLLED' in phase:
                 penalty_polys.append((poly, 50.0, c.alt_layers))
             elif 'ADVISORY' in phase:
                 penalty_polys.append((poly, 10.0, c.alt_layers))
                 
        for track in obs.traffic_tracks:
             box = self.get_box(track.position, 60.0)
             restricted_polys.append((box, [track.alt_layer - 1, track.alt_layer, track.alt_layer + 1]))
             
        nodes_list = list(all_nodes.keys())
        for i in range(len(nodes_list)):
            u = nodes_list[i]
            pos_u = all_nodes[u]
            for j in range(i+1, len(nodes_list)):
                v = nodes_list[j]
                pos_v = all_nodes[v]
                
                if pos_u[2] == pos_v[2]:
                    # Horizontal
                    alt = pos_u[2]
                    valid = True
                    A = Position2D(x=pos_u[0], y=pos_u[1])
                    B = Position2D(x=pos_v[0], y=pos_v[1])
                    is_sg = (u == 'start' or u == 'goal' or v == 'start' or v == 'goal')
                    
                    if map_verts:
                        if not self.is_inside_map(A, map_verts) or not self.is_inside_map(B, map_verts):
                            valid = False
                        else:
                            mid = Position2D(x=(A.x+B.x)/2, y=(A.y+B.y)/2)
                            if not self.is_inside_map(mid, map_verts):
                                valid = False
                                
                    if valid:
                        for poly, alts in restricted_polys:
                            if alts is None or alt in alts:
                                if not self.is_edge_valid(A, B, poly, is_sg):
                                    valid = False
                                    break
                    if valid:
                        dist = math.hypot(A.x - B.x, A.y - B.y)
                        for poly, weight, alts in penalty_polys:
                            if alts is None or alt in alts:
                                mid = Position2D(x=(A.x+B.x)/2, y=(A.y+B.y)/2)
                                if self.is_point_in_polygon(mid, poly):
                                    dist += weight * 10.0
                        edges[u][v] = dist
                        edges[v][u] = dist
                else:
                    # Vertical
                    if abs(pos_u[0] - pos_v[0]) < 1e-3 and abs(pos_u[1] - pos_v[1]) < 1e-3:
                        if abs(pos_u[2] - pos_v[2]) == 1:
                            A = Position2D(x=pos_u[0], y=pos_u[1])
                            valid = True
                            for poly, alts in restricted_polys:
                                if alts is None or pos_u[2] in alts or pos_v[2] in alts:
                                    # Use a relaxed check if moving vertically while in a buffer
                                    is_sg = (u == 'start' or u == 'goal' or v == 'start' or v == 'goal')
                                    if not is_sg and self.is_point_strictly_in_polygon(A, poly):
                                        valid = False
                                        break
                            if valid:
                                edges[u][v] = 20.0
                                edges[v][u] = 20.0
                                
        return edges

    def step(self, obs: Observation) -> Plan:
        
        if self.dstar is None:
            self.dstar = DStarLite(None, None)
            
        goal_alt = obs.mission_goal.target_alt_layer if obs.mission_goal.target_alt_layer is not None else obs.ownship_state.alt_layer
        goal_pos = self.get_region_center(obs.mission_goal.region)
        
        current_nodes = {}
        current_nodes['start'] = (obs.ownship_state.position.x, obs.ownship_state.position.y, obs.ownship_state.alt_layer)
        current_nodes['goal'] = (goal_pos.x, goal_pos.y, goal_alt)
        
        def add_poly_nodes(prefix, poly, alts):
            idx = 0
            for alt in alts:
                for v in poly:
                    current_nodes[f'{prefix}_{idx}'] = (v.x, v.y, alt)
                    idx += 1
                    
        alts_to_consider = [obs.ownship_state.alt_layer, goal_alt]
        for c in obs.active_constraints:
            for a in c.alt_layers:
                if a not in alts_to_consider: alts_to_consider.append(a)
                
        for i, obs_reg in enumerate(obs.static_obstacles):
            add_poly_nodes(f'stat_{i}', self.get_poly(obs_reg, 22.0), alts_to_consider)
            
        for c in obs.active_constraints:
            c_id = getattr(c, 'id', str(id(c)))
            add_poly_nodes(f'const_{c_id}', self.get_poly(c.region, 22.0), c.alt_layers)
            
        for i, track in enumerate(obs.traffic_tracks):
            t_id = getattr(track, 'id', str(i))
            box = self.get_box(track.position, 60.0)
            add_poly_nodes(f'traf_{t_id}', box, [track.alt_layer])
            
        if self.dstar.s_start is None:
            self.dstar.s_start = 'start'
            self.dstar.s_goal = 'goal'
            self.dstar.nodes = current_nodes
            self.dstar.U.insert('goal', self.dstar.CalculateKey('goal'))
            self.dstar.rhs['goal'] = 0.0
        else:
            old_start_pos = self.dstar.nodes['start']
            new_start_pos = current_nodes['start']
            self.dstar.k_m += self.dstar.heuristic_pos(old_start_pos, new_start_pos)
            for u, pos in current_nodes.items():
                self.dstar.nodes[u] = pos
                
        new_edges = self.compute_all_edges(obs, current_nodes)
        
        changed_nodes = set()
        
        for u in list(self.dstar.edges.keys()):
            for v in list(self.dstar.edges[u].keys()):
                old_c = self.dstar.edges[u][v]
                new_c = new_edges.get(u, {}).get(v, float('inf'))
                if abs(old_c - new_c) > 1e-3:
                    self.dstar.edges[u][v] = new_c
                    changed_nodes.add(u)
                    
        for u in new_edges:
            for v, new_c in new_edges[u].items():
                old_c = self.dstar.edges.get(u, {}).get(v, float('inf'))
                if abs(old_c - new_c) > 1e-3:
                    if u not in self.dstar.edges: self.dstar.edges[u] = {}
                    self.dstar.edges[u][v] = new_c
                    changed_nodes.add(u)
                    
        for u in changed_nodes:
            self.dstar.UpdateVertex(u)
            
        self.dstar.ComputeShortestPath()
        
        path = ['start']
        curr = 'start'
        
        max_path = 50
        while curr != 'goal' and len(path) < max_path:
            best_v = None
            best_cost = float('inf')
            best_g = float('inf')
            for v, edge_cost in self.dstar.edges.get(curr, {}).items():
                cost = edge_cost + self.dstar.get_g(v)
                if cost < best_cost - 1e-4:
                    best_cost = cost
                    best_v = v
                    best_g = self.dstar.get_g(v)
                elif abs(cost - best_cost) <= 1e-4:
                    if self.dstar.get_g(v) < best_g:
                        best_v = v
                        best_g = self.dstar.get_g(v)
                        
            if best_v is None or best_cost == float('inf'):
                break 
            path.append(best_v)
            curr = best_v
            
        steps = []
        current_world_pos = obs.ownship_state.position
        current_alt = obs.ownship_state.alt_layer
        
        path_idx = 1
        speed = 15.0
        
        for i in range(5):
            if path_idx < len(path):
                target_node = path[path_idx]
                target_pos = self.dstar.nodes[target_node]
                target_world = Position2D(x=target_pos[0], y=target_pos[1])
                target_a = target_pos[2]
                
                dx = target_world.x - current_world_pos.x
                dy = target_world.y - current_world_pos.y
                dist = math.hypot(dx, dy)
                
                if dist > speed:
                    step_x = current_world_pos.x + (dx/dist) * speed
                    step_y = current_world_pos.y + (dy/dist) * speed
                    next_pos = Position2D(x=step_x, y=step_y)
                elif dist > 1.0:
                    next_pos = target_world
                else:
                    next_pos = target_world
                    
                next_alt = target_a
                if next_alt > current_alt: next_alt = current_alt + 1
                elif next_alt < current_alt: next_alt = current_alt - 1
                
                if dist <= 1.0 and current_alt == target_a:
                    path_idx += 1
                    
                steps.append(ActionStep(
                    action_type=ActionType.WAYPOINT,
                    target_position=next_pos,
                    target_alt_layer=next_alt
                ))
                current_world_pos = next_pos
                current_alt = next_alt
            else:
                steps.append(ActionStep(ActionType.HOLD))
                
        if obs.ownship_state.energy < 20.0:
            return Plan(steps=[ActionStep(ActionType.EMERGENCY_LAND)] * 5)
        
        # import pdb; pdb.set_trace()
            
        return Plan(steps=steps)
