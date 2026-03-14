from aerohacks.policy.base import Policy
from aerohacks.core.models import Observation, Plan, ActionStep, ActionType, Position2D, State
import math
import heapq as hq

class MyPolicy(Policy):
    """
    BASELINE ALGORITHM: This is your starting point.
    It simply flies in a straight line towards the goal, ignoring any NO-FLY zones or traffic.
    You will need to improve this to avoid penalties!
    """
    
    def step(self, obs: Observation) -> Plan:
        # 1) Build start and goal States for the planner
        own = obs.ownship_state
        goal_center = obs.mission_goal.region.center()
        target_alt = obs.mission_goal.target_alt_layer or own.alt_layer

        s_start = State(
            position=Position2D(x=own.position.x, y=own.position.y),
            alt_layer=own.alt_layer,
            energy=own.energy,
            velocity=own.velocity,
            heading=own.heading,
        )
        s_goal = State(
            position=Position2D(x=goal_center.x, y=goal_center.y),
            alt_layer=target_alt,
            energy=own.energy,
            velocity=own.velocity,
            heading=own.heading,
        )

        # 2) Plan with D*
        planner = D_star(s_start=s_start, s_goal=s_goal, step_length=15.0)
        planner.compute_shortest_path()
        path_states = planner.extract_path()  # States from (near) start toward goal

        steps: list[ActionStep] = []

        # Fallback: if planner fails to produce a path, take a straight-line step
        if not path_states:
            current_pos = own.position
            dx = goal_center.x - current_pos.x
            dy = goal_center.y - current_pos.y
            dist = math.hypot(dx, dy)
            if dist > 0.0:
                step_dist = min(15.0, dist)
                step_x = current_pos.x + dx / dist * step_dist
                step_y = current_pos.y + dy / dist * step_dist
                target_pos = Position2D(x=step_x, y=step_y)
            else:
                target_pos = Position2D(x=current_pos.x, y=current_pos.y)

            for _ in range(5):
                steps.append(
                    ActionStep(
                        action_type=ActionType.WAYPOINT,
                        target_position=target_pos,
                        target_alt_layer=target_alt,
                    )
                )
            return Plan(steps=steps)

        # 3) Turn first 5 path states into waypoints
        for i in range(5):
            if i < len(path_states):
                state_i = path_states[i]
                target_pos = state_i.position
                target_layer = state_i.alt_layer
                steps.append(
                    ActionStep(
                        action_type=ActionType.WAYPOINT,
                        target_position=Position2D(x=target_pos.x, y=target_pos.y),
                        target_alt_layer=target_layer,
                    )
                )
            else:
                # If path is shorter than 5, just hold at the last position/alt
                last_pos = steps[-1].target_position if steps else own.position
                last_alt = steps[-1].target_alt_layer if steps else own.alt_layer
                steps.append(
                    ActionStep(
                        action_type=ActionType.WAYPOINT,
                        target_position=Position2D(x=last_pos.x, y=last_pos.y),
                        target_alt_layer=last_alt,
                    )
                )

        return Plan(steps=steps)
    
    '''
    U = priority queue (open list), key = [k1(s), k2(s)]

    Initialize():
    U = empty
    for each s: g(s) = rhs(s) = ∞
    rhs(s_goal) = 0
    U.insert(s_goal, CalculateKey(s_goal))

    CalculateKey(s):
    k1 = min(g(s), rhs(s)) + h(s_start, s)   // note: h from s_start to s (backward search)
    k2 = min(g(s), rhs(s))
    return [k1, k2]

    UpdateVertex(u):
    if u ≠ s_goal:
        rhs(u) = min over s' in Succ(u) of [ c(u, s') + g(s') ]
    if u in U: U.remove(u)
    if g(u) ≠ rhs(u): U.insert(u, CalculateKey(u))

    ComputeShortestPath():
    while U.topKey() < CalculateKey(s_start)  OR  g(s_start) ≠ rhs(s_start):
        u = U.pop()
        if g(u) > rhs(u):
        g(u) = rhs(u)
        for each s in Pred(u): UpdateVertex(s)
        else:
        g(u) = ∞
        for each s in Pred(u) ∪ {u}: UpdateVertex(s)

    Main (first run or after graph/cost changes):
    s_start = CurrentNode(observation.ownship_state)
    s_goal = any node that satisfies Goal(s)   // e.g. center of goal region
    Initialize()
    ComputeShortestPath()
    path = PathFromStartToGoal(s_start)      // follow min g(s) from s_start to s_goal
    return first 5 waypoints of path → Plan(steps=[...])
    
    
    '''


class D_star:
    """
    Minimal D* Lite-style planner working on a 2D grid of States.

    Notes:
    - We only implement a static single-shot plan right now (no dynamic updates).
    - g and rhs use lazy-initialisation: missing entries are treated as +inf.
    - get_surround_states generates 8-connected neighbors at a fixed step length.
    """

    def __init__(self, s_start: State, s_goal: State, step_length: float = 15.0):
        self.s_start = s_start
        self.s_goal = s_goal
        self.step_length = step_length

        # priority queue holds (key, State) where key is a (k1, k2) tuple
        self.pq: list[tuple[tuple[float, float], State]] = []

        # D* Lite value functions (lazy: missing => +inf)
        self.g: dict[tuple[float, float, int], float] = {}
        self.rhs: dict[tuple[float, float, int], float] = {}

        # Initialize goal
        self._set_rhs(self.s_goal, 0.0)
        hq.heappush(self.pq, (self._calculate_key(self.s_goal), self.s_goal))

    # --- Helpers for lazy g / rhs access -------------------------------------------------
    def _node_key(self, s: State) -> tuple[float, float, int]:
        return (s.position.x, s.position.y, s.alt_layer)
    def _same_node(self, a: State, b: State) -> bool:
        return self._node_key(a) == self._node_key(b)
    def _get_g(self, s: State) -> float:
        return self.g.get(self._node_key(s), float("inf"))
    def _set_g(self, s: State, value: float) -> None:
        self.g[self._node_key(s)] = value
    def _get_rhs(self, s: State) -> float:
        return self.rhs.get(self._node_key(s), float("inf"))
    def _set_rhs(self, s: State, value: float) -> None:
        self.rhs[self._node_key(s)] = value

    # --- Core D* Lite pieces -------------------------------------------------------------

    def _heuristic(self, s_from: State, s_to: State) -> float:
        # Simple admissible heuristic: straight-line distance
        dx = s_from.position.x - s_to.position.x
        dy = s_from.position.y - s_to.position.y
        return math.hypot(dx, dy)

    def _calculate_key(self, s: State) -> tuple[float, float]:
        g_s = self._get_g(s)
        rhs_s = self._get_rhs(s)
        m = min(g_s, rhs_s)
        k1 = m + self._heuristic(self.s_start, s)
        k2 = m
        return (k1, k2)

    def _update_vertex(self, s: State) -> None:
        # Update rhs(s) from one-step lookahead (except at goal)
        if not self._same_node(s, self.s_goal):
            neighbors = self.get_surround_states(s, self.step_length)
            best = float("inf")
            for n in neighbors:
                cost = self._cost(s, n) + self._get_g(n)
                if cost < best:
                    best = cost
            self._set_rhs(s, best)

        # If state is inconsistent, (re)insert it into the queue
        if not math.isclose(self._get_g(s), self._get_rhs(s)):
            hq.heappush(self.pq, (self._calculate_key(s), s))

    def compute_shortest_path(self) -> None:
        """
        Basic D* Lite loop. For simplicity we run until either:
        - queue is empty, or
        - start is locally consistent and has no better key in the queue.
        """
        while self.pq:
            top_key, u = hq.heappop(self.pq)

            # Skip stale entries
            if top_key > self._calculate_key(u):
                continue

            # Termination condition (approximate textbook condition)
            if (top_key >= self._calculate_key(self.s_start)
                    and math.isclose(self._get_g(self.s_start), self._get_rhs(self.s_start))):
                break

            g_u = self._get_g(u)
            rhs_u = self._get_rhs(u)

            if g_u > rhs_u:
                self._set_g(u, rhs_u)
                # Predecessors are approximated as neighbors in this simple grid
                for p in self.get_surround_states(u, self.step_length):
                    self._update_vertex(p)
            else:
                self._set_g(u, float("inf"))
                for p in self.get_surround_states(u, self.step_length) + [u]:
                    self._update_vertex(p)

    def extract_path(self) -> list[State]:
        """
        Follow greedy descent from s_start to s_goal using g-values.
        Returns a list of States from start (excluded) toward goal (included),
        or an empty list if no path is known.
        """
        path: list[State] = []
        current = self.s_start

        # Safety cap to avoid infinite loops on inconsistent graphs
        for _ in range(500):
            if self._same_node(current, self.s_goal):
                break

            neighbors = self.get_surround_states(current, self.step_length)
            if not neighbors:
                break

            best_neighbor = None
            best_value = float("inf")
            for n in neighbors:
                value = self._cost(current, n) + self._get_g(n)
                if value < best_value:
                    best_value = value
                    best_neighbor = n

            if best_neighbor is None or best_value == float("inf"):
                # No way forward with finite cost
                break

            path.append(best_neighbor)
            current = best_neighbor

            if self._same_node(current, self.s_goal):
                break

        return path

    # --- Graph structure -----------------------------------------------------------------

    @staticmethod
    def get_surround_states(s: State, step: float) -> list[State]:
        """
        Generate an 8-connected neighborhood around s in the horizontal plane.
        Altitude layer is kept constant.
        """
        neighbors: list[State] = []
        x0 = s.position.x
        y0 = s.position.y
        alt = s.alt_layer

        # 8-connected grid (dx, dy) in {-1, 0, 1}^2 \ {(0, 0)}
        for dx in (-1.0, 0.0, 1.0):
            for dy in (-1.0, 0.0, 1.0):
                if dx == 0.0 and dy == 0.0:
                    continue
                nx = x0 + dx * step
                ny = y0 + dy * step
                neighbors.append(
                    State(
                        position=Position2D(x=nx, y=ny),
                        alt_layer=alt,
                        energy=s.energy,
                        velocity=s.velocity,
                        heading=s.heading,
                    )
                )

        return neighbors

    def _cost(self, s_start: State, s_end: State) -> float:
        """
        Edge cost between two States.
        Currently just Euclidean distance; this is where you will later plug in
        weighted battery and penalty terms.
        """
        dx = s_start.position.x - s_end.position.x
        dy = s_start.position.y - s_end.position.y
        dist = math.hypot(dx, dy)
        return dist
