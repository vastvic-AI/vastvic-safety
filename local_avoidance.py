from typing import List, Tuple, Dict
import numpy as np

Cell = Tuple[int, int]

# Simple Social-Force + Steering hybrid for grid candidates
# It ranks allowed neighbor moves using a combined score from goal attraction
# and repulsion from agents and walls.

def rank_moves(
    grid: np.ndarray,
    cost_map: np.ndarray,
    pos: Cell,
    goal: Tuple[float, float],
    allowed: List[Cell],
    neighbor_agents: List[Tuple[float, float]],
    wall_repulsion_radius: int = 2,
    agent_repulsion_radius: float = 2.0,
    w_goal: float = 1.0,
    w_agents: float = 0.7,
    w_walls: float = 0.5,
) -> List[Tuple[float, Cell]]:
    """Return a list of (score, move_cell) sorted descending by score."""
    if not allowed:
        return []
    H, W = grid.shape

    # Precompute repulsive field from walls near current pos
    wall_pts: List[Tuple[int, int]] = []
    r0, c0 = pos
    for dr in range(-wall_repulsion_radius, wall_repulsion_radius + 1):
        for dc in range(-wall_repulsion_radius, wall_repulsion_radius + 1):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < H and 0 <= cc < W and grid[rr, cc] == 1:
                wall_pts.append((rr, cc))

    def repulsion_from_walls(p: Tuple[float, float]) -> float:
        s = 0.0
        for rr, cc in wall_pts:
            d = ((p[0] - rr) ** 2 + (p[1] - cc) ** 2) ** 0.5
            if d <= 0.0:
                s += 10.0
            elif d < wall_repulsion_radius + 1e-6:
                s += 1.0 / d
        return s

    def repulsion_from_agents(p: Tuple[float, float]) -> float:
        s = 0.0
        for ar, ac in neighbor_agents:
            d = ((p[0] - ar) ** 2 + (p[1] - ac) ** 2) ** 0.5
            if d <= 0.0:
                s += 10.0
            elif d < agent_repulsion_radius:
                s += 1.0 / d
        return s

    # Normalize goal vector
    gvec = np.array([goal[0] - (r0 + 0.0), goal[1] - (c0 + 0.0)], dtype=float)
    gnorm = np.linalg.norm(gvec) + 1e-9
    gdir = gvec / gnorm

    ranked: List[Tuple[float, Cell]] = []
    for mv in allowed:
        mr, mc = mv
        v = np.array([mr - r0, mc - c0], dtype=float)
        vnorm = np.linalg.norm(v) + 1e-9
        vdir = v / vnorm
        # Attraction to goal: cosine with goal direction
        goal_score = float(np.dot(vdir, gdir))
        # Repulsion terms (higher repulsion -> lower score)
        p = (mr + 0.0, mc + 0.0)
        rep_w = repulsion_from_walls(p)
        rep_a = repulsion_from_agents(p)
        # Slight preference for lower-cost destination
        cell_cost = float(cost_map[mr, mc]) if np.isfinite(cost_map[mr, mc]) else 1e6
        cost_bias = -0.05 * (cell_cost - 1.0)
        score = w_goal * goal_score - w_walls * rep_w - w_agents * rep_a + cost_bias
        ranked.append((score, mv))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked
