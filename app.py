# Simple 2D Pathfinder-style Evacuation Simulator (Streamlit)
# -----------------------------
# Run: streamlit run app.py

import os
import json
import time
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import numpy as np
import math
from path_smoothing import smooth_path_los
from local_avoidance import rank_moves

def _compute_distance_field(grid: np.ndarray, exits: List[Tuple[int, int]]) -> np.ndarray:
    """Multi-source BFS distance (in grid steps) from every free cell to nearest exit.
    Walls (non-zero grid) are set to +inf. Unreachable free cells remain +inf.
    """
    H, W = grid.shape
    dist = np.full((H, W), np.inf, dtype=float)
    from collections import deque
    q = deque()
    for ex in exits:
        r, c = ex
        if 0 <= r < H and 0 <= c < W and grid[r, c] == 0:
            dist[r, c] = 0.0
            q.append((r, c))
        else:
            # Treat exit cell itself as passable target even if painted as exit (non-zero)
            if 0 <= r < H and 0 <= c < W:
                dist[r, c] = 0.0
                q.append((r, c))
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while q:
        r, c = q.popleft()
        base = dist[r, c]
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0:
                if dist[nr, nc] > base + 1.0:
                    dist[nr, nc] = base + 1.0
                    q.append((nr, nc))
    return dist

# -----------------------------
# Multi-pattern motion helpers
# -----------------------------
def _local_free_space(grid: np.ndarray, p: Tuple[float, float], radius: int = 3) -> Dict[str, float]:
    """Estimate free-space availability in principal directions around point p.
    Returns normalized scores for keys: 'up','down','left','right','ul','ur','dl','dr'."""
    H, W = grid.shape
    ir, ic = int(np.clip(p[0], 0, H - 1)), int(np.clip(p[1], 0, W - 1))
    acc = {k: 0 for k in ['up','down','left','right','ul','ur','dl','dr']}
    total = {k: 0 for k in acc}
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            rr, cc = ir + dr, ic + dc
            if rr < 0 or rr >= H or cc < 0 or cc >= W:
                continue
            key = None
            if dr < 0 and dc == 0: key = 'up'
            elif dr > 0 and dc == 0: key = 'down'
            elif dr == 0 and dc < 0: key = 'left'
            elif dr == 0 and dc > 0: key = 'right'
            elif dr < 0 and dc < 0: key = 'ul'
            elif dr < 0 and dc > 0: key = 'ur'
            elif dr > 0 and dc < 0: key = 'dl'
            elif dr > 0 and dc > 0: key = 'dr'
            if key is None:
                continue
            total[key] += 1
            if grid[rr, cc] == 0:
                acc[key] += 1
    out = {k: (acc[k] / max(1, total[k])) for k in acc}
    return out

def _choose_motion_mode(grid: np.ndarray, a_pos: Tuple[int, int]) -> str:
    """Pick a motion mode based on local free space; biased toward straight in corridors and curvy/zigzag/spiral in open areas."""
    p = (a_pos[0] + 0.5, a_pos[1] + 0.5)
    fs = _local_free_space(grid, p, radius=3)
    horiz = (fs['left'] + fs['right']) * 0.5
    vert = (fs['up'] + fs['down']) * 0.5
    diag = (fs['ul'] + fs['ur'] + fs['dl'] + fs['dr']) / 4.0
    openness = (horiz + vert + diag) / 3.0
    corridor = abs(horiz - vert) < 0.2 and (horiz + vert) / 2.0 < 0.6
    rng = random.random()
    if corridor:
        # Mostly corridors: prefer straight; small chance of left/right for lane changing
        if rng < 0.8:
            return 'straight'
        return 'left' if rng < 0.9 else 'right'
    else:
        # Open areas: allow more expressive motion
        if openness > 0.7:
            if rng < 0.34:
                return 'curvy'
            elif rng < 0.67:
                return 'zigzag'
            else:
                return 'spiral'
        else:
            # Mixed: mostly straight with some variability
            if rng < 0.6:
                return 'straight'
            elif rng < 0.8:
                return 'curvy'
            else:
                return 'zigzag'
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

try:
    import imageio
except Exception:
    imageio = None

# -----------------------------
# Utility and core algorithms
# -----------------------------
CELL_METERS = 1.0  # meters per grid cell (assumption)

# Agent Type Configuration
TYPE_CONFIG = {
    "Pedestrian": {"speed_mps": (1.0, 1.5), "panic_mult": 1.0, "group_radius": 0, "wheelchair": False, "color": (30, 144, 255)},
    "Elderly": {"speed_mps": (0.6, 1.0), "panic_mult": 1.2, "group_radius": 3, "wheelchair": False, "color": (255, 165, 0)},
    # Use a non-green color for children to avoid confusion with exit green
    "Child": {"speed_mps": (0.8, 1.6), "panic_mult": 1.5, "group_radius": 2, "wheelchair": False, "color": (255, 105, 180)},
    "Wheelchair": {"speed_mps": (0.6, 1.2), "panic_mult": 1.0, "group_radius": 2, "wheelchair": True, "color": (148, 0, 211)},
    "Staff": {"speed_mps": (1.2, 2.0), "panic_mult": 0.8, "group_radius": 0, "wheelchair": False, "color": (220, 20, 60)},
}
Cell = Tuple[int, int]  # (row, col)
TICK_SECONDS = 0.2  # used to convert ticks to seconds for analytics


def load_floorplan(img: Optional[Image.Image], grid_size: int = 100) -> Dict:
    """Process a floorplan image into grids and masks per the rules.

    Rules:
    - White ≈ free (R,G,B ≥ 230)
    - Black ≈ wall (R,G,B ≤ 25)
    - Green exit: G ≥ 180 and R ≤ 100
    - Red exit: R ≥ 180 and G ≤ 100
    - Yellow/Orange = hazard (cost > 1, not blocked)
    """
    if img is None:
        # Default empty room with two exits
        H = W = grid_size
        grid = np.zeros((H, W), dtype=np.uint8)
        # Add walls on border
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1
        exits = [(H // 2, 0), (H // 2, W - 1)]
        hazards = np.zeros_like(grid, dtype=np.uint8)
        return dict(grid=grid, exits=exits, hazards=hazards, rgb=None)

    # Convert to RGB and resize preserving solid colors
    rgb_img = img.convert("RGB")
    orig_w, orig_h = rgb_img.size
    scale = min(grid_size / orig_w, grid_size / orig_h)
    new_size = (max(1, int(orig_w * scale)), max(1, int(orig_h * scale)))
    rgb_img = rgb_img.resize(new_size, Image.Resampling.NEAREST)

    rgb = np.array(rgb_img)
    H, W, _ = rgb.shape

    grid = np.ones((H, W), dtype=np.uint8)  # 1 = wall, 0 = free
    hazards = np.zeros((H, W), dtype=np.uint8)
    exits: List[Cell] = []

    # Vectorized color masks with tolerant thresholds
    R = rgb[:, :, 0].astype(np.int16)
    G = rgb[:, :, 1].astype(np.int16)
    B = rgb[:, :, 2].astype(np.int16)

    is_black = (R <= 25) & (G <= 25) & (B <= 25)
    is_white = (R >= 230) & (G >= 230) & (B >= 230)

    # Strong green/red dominance for exits, robust to anti-aliasing
    green_dom = (G - np.maximum(R, B) >= 40)
    red_dom = (R - np.maximum(G, B) >= 40)
    is_green_exit = (G >= 180) & (R <= 120) & green_dom
    is_red_exit = (R >= 180) & (G <= 120) & red_dom
    # ±20 tolerance around pure colors to handle compression/anti-aliasing
    green_tol = (np.abs(R - 0) <= 20) & (np.abs(G - 255) <= 20) & (np.abs(B - 0) <= 20)
    red_tol = (np.abs(R - 255) <= 20) & (np.abs(G - 0) <= 20) & (np.abs(B - 0) <= 20)

    # Hazards: yellow/orange bands
    is_yellow = (R >= 200) & (G >= 200) & (B <= 140)
    is_orange = (R >= 200) & (G >= 120) & (G <= 200) & (B <= 140)
    is_hazard = is_yellow | is_orange

    # Build grid: 1=wall, 0=free
    grid[:, :] = 0
    grid[is_black] = 1
    # Exits are free cells
    exit_mask = (is_green_exit | is_red_exit | green_tol | red_tol)
    hazards[:, :] = 0
    hazards[is_hazard & (~is_black)] = 1

    # Compose exits list from colored exits; carve exits as walkable (free)
    # Ensure exit cells are free in the grid so A* can reach them even if drawn over black border
    grid[exit_mask] = 0
    er, ec = np.where(exit_mask & (~is_black))
    exits = list(zip(er.tolist(), ec.tolist()))

    # Fallback: infer exits from border openings if none detected by color
    if len(exits) == 0:
        border_coords: List[Cell] = []
        # left and right borders
        for r in range(H):
            if grid[r, 0] == 0:  # opening on left border
                border_coords.append((r, 0))
            if grid[r, W - 1] == 0:  # opening on right border
                border_coords.append((r, W - 1))
        # top and bottom borders
        for c in range(W):
            if grid[0, c] == 0:
                border_coords.append((0, c))
            if grid[H - 1, c] == 0:
                border_coords.append((H - 1, c))

        # Reduce to a few representative exit points per contiguous segment
        def cluster_line(points: List[Cell], axis: int) -> List[Cell]:
            if not points:
                return []
            # sort by the varying coordinate
            pts = sorted(points, key=lambda x: x[1 - axis])
            clusters: List[List[Cell]] = []
            cur: List[Cell] = [pts[0]]
            for p in pts[1:]:
                if abs(p[1 - axis] - cur[-1][1 - axis]) <= 1:
                    cur.append(p)
                else:
                    clusters.append(cur)
                    cur = [p]
            clusters.append(cur)
            # take center of each cluster
            centers: List[Cell] = []
            for cl in clusters:
                centers.append(cl[len(cl) // 2])
            return centers

        left_pts = [(r, 0) for r in range(H) if grid[r, 0] == 0]
        right_pts = [(r, W - 1) for r in range(H) if grid[r, W - 1] == 0]
        top_pts = [(0, c) for c in range(W) if grid[0, c] == 0]
        bottom_pts = [(H - 1, c) for c in range(W) if grid[H - 1, c] == 0]
        inferred = []
        inferred += cluster_line(left_pts, axis=0)
        inferred += cluster_line(right_pts, axis=0)
        inferred += cluster_line(top_pts, axis=1)
        inferred += cluster_line(bottom_pts, axis=1)
        # Deduplicate
        exits = sorted(list(set(inferred)))

    # Final safety fallback: if still no exits, create synthetic exits mid-left and mid-right
    if len(exits) == 0:
        synth = [(H // 2, 0), (H // 2, W - 1)]
        for (r, c) in synth:
            if 0 <= r < H and 0 <= c < W:
                grid[r, c] = 0
        exits = synth

    return dict(grid=grid, exits=exits, hazards=hazards, rgb=rgb)


def detect_exits(data: Dict) -> List[Cell]:
    return data.get("exits", [])


def astar(cost_map: np.ndarray, start: Cell, goal: Cell) -> List[Cell]:
    """A* pathfinding on a weighted grid with 8-direction movement.

    - Orthogonal step base cost = 1, diagonal = sqrt(2), multiplied by destination cell cost.
    - Octile heuristic for accurate guidance.
    - Prevent diagonal corner cutting (both adjacent orthogonals must be passable).
    """
    H, W = cost_map.shape

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < H and 0 <= c < W

    def passable(rc: Cell) -> bool:
        return np.isfinite(cost_map[rc])

    DIRS = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]

    def neighbors(rc: Cell) -> List[Cell]:
        r, c = rc
        result: List[Cell] = []
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                continue
            if not passable((nr, nc)):
                continue
            # prevent corner cutting
            if dr != 0 and dc != 0:
                if not (in_bounds(r + dr, c) and in_bounds(r, c + dc)):
                    continue
                if not (passable((r + dr, c)) and passable((r, c + dc))):
                    continue
            result.append((nr, nc))
        return result

    SQRT2 = math.sqrt(2.0)
    def h(a: Cell, b: Cell) -> float:
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return (dx + dy) + (SQRT2 - 2.0) * min(dx, dy)

    import heapq

    g: Dict[Cell, float] = {start: 0.0}
    came: Dict[Cell, Cell] = {}
    f0 = h(start, goal)
    open_heap: List[Tuple[float, float, Cell]] = [(f0, f0, start)]
    in_open = {start}

    while open_heap:
        _, _, cur = heapq.heappop(open_heap)
        in_open.discard(cur)
        if cur == goal:
            path: List[Cell] = []
            while cur in came:
                path.append(cur)
                cur = came[cur]
            path.reverse()
            return path
        cr, cc = cur
        for nb in neighbors(cur):
            nbr, nbc = nb
            base = SQRT2 if (abs(nbr - cr) == 1 and abs(nbc - cc) == 1) else 1.0
            cell_cost = float(cost_map[nb])
            if not np.isfinite(cell_cost):
                continue
            step = base * cell_cost
            tentative = g[cur] + step
            if tentative < g.get(nb, 1e18):
                came[nb] = cur
                g[nb] = tentative
                hn = h(nb, goal)
                fn = tentative + hn
                if nb not in in_open:
                    heapq.heappush(open_heap, (fn, hn, nb))
                    in_open.add(nb)
    return []


def astar_cached(cost_map: np.ndarray, start: Cell, goal: Cell, cache: Optional[Dict[Tuple[Cell, Cell], List[Cell]]] = None) -> List[Cell]:
    """LRU-style cached A* wrapper keyed by (start, goal).
    Cache is a dict provided by caller; if None, falls back to direct astar.
    """
    if cache is None:
        return astar(cost_map, start, goal)
    key = (start, goal)
    if key in cache:
        return cache[key]
    path = astar(cost_map, start, goal)
    cache[key] = path
    return path

def compute_path_cost(cost_map: np.ndarray, start: Cell, path: List[Cell]) -> float:
    """Compute cumulative movement cost along a path sequence starting at `start`.
    Costs mirror astar(): orthogonal=1, diagonal=sqrt(2) multiplied by destination cell cost.
    Returns inf for empty/unreachable path.
    """
    if not path:
        return float("inf")
    total = 0.0
    SQRT2 = math.sqrt(2.0)
    prev = start
    for nxt in path:
        pr, pc = prev
        nr, nc = nxt
        base = SQRT2 if (abs(nr - pr) == 1 and abs(nc - pc) == 1) else 1.0
        cell_cost = float(cost_map[nxt])
        if not np.isfinite(cell_cost):
            return float("inf")
        total += base * cell_cost
        prev = nxt
    return total

def spawn_agents(free_cells: List[Cell], exits: List[Cell], N: int, seed: Optional[int] = None) -> List[Cell]:
    """Spawn N agents on free cells not on exits.
    If seed is None, use entropy-based RNG to ensure per-run variability."""
    rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)
    candidates = [cell for cell in free_cells if cell not in exits]
    if len(candidates) == 0:
        return []
    if N > len(candidates):
        N = len(candidates)
    idx = rng.choice(len(candidates), size=N, replace=False)
    return [candidates[i] for i in idx]


@dataclass
class AgentState:
    pos: Cell
    exited: bool = False
    exit_time: Optional[int] = None
    exit_pos: Optional[Cell] = None
    trapped: bool = False
    path: List[Cell] = field(default_factory=list)
    path_len: int = 0
    congestion: int = 0
    group_id: int = 0
    agent_type: str = "Pedestrian"
    speed_cells: float = 1.0
    panic_mult: float = 1.0
    group_radius: int = 0
    is_wheelchair: bool = False
    pre_delay_ticks: int = 0
    # Stuck and replanning management
    no_progress_ticks: int = 0
    replan_attempts: int = 0
    last_replan_tick: int = -9999
    last_progress_metric: float = 1e18
    stuck_events: int = 0
    total_replans: int = 0


def build_cost_map(grid: np.ndarray, hazards: np.ndarray, hazard_weight: float = 3.0) -> np.ndarray:
    # 0 (free) -> 1 cost; 1 (wall) -> inf; hazards add weight
    base = np.where(grid == 0, 1.0, np.inf)
    if hazards is not None and hazards.shape == grid.shape:
        base = base + hazards.astype(float) * float(hazard_weight)
    return base


def _exit_is_wide(ex: Cell, grid: np.ndarray, min_width: int = 2) -> bool:
    """Heuristic: an exit is considered wide if there are at least `min_width`
    contiguous free cells (grid==0) orthogonal to the boundary direction.

    We check in both horizontal and vertical directions around the exit cell
    and take the maximum contiguous run length including the exit cell.
    """
    H, W = grid.shape
    r, c = ex
    if not (0 <= r < H and 0 <= c < W):
        return False
    if grid[r, c] != 0:
        return False

    # horizontal run
    run_h = 1
    cc = c - 1
    while cc >= 0 and grid[r, cc] == 0:
        run_h += 1
        cc -= 1
    cc = c + 1
    while cc < W and grid[r, cc] == 0:
        run_h += 1
        cc += 1

    # vertical run
    run_v = 1
    rr = r - 1
    while rr >= 0 and grid[rr, c] == 0:
        run_v += 1
        rr -= 1
    rr = r + 1
    while rr < H and grid[rr, c] == 0:
        run_v += 1
        rr += 1

    return max(run_h, run_v) >= max(1, int(min_width))

 

def _exit_width_cells(ex: Cell, grid: np.ndarray) -> int:
    """Return the contiguous free-cell span through an exit cell (cells)."""
    H, W = grid.shape
    r, c = ex
    if not (0 <= r < H and 0 <= c < W) or grid[r, c] != 0:
        return 0
    run_h = 1
    cc = c - 1
    while cc >= 0 and grid[r, cc] == 0:
        run_h += 1; cc -= 1
    cc = c + 1
    while cc < W and grid[r, cc] == 0:
        run_h += 1; cc += 1
    run_v = 1
    rr = r - 1
    while rr >= 0 and grid[rr, c] == 0:
        run_v += 1; rr -= 1
    rr = r + 1
    while rr < H and grid[rr, c] == 0:
        run_v += 1; rr += 1
    return max(run_h, run_v)

def compute_paths(cost_map: np.ndarray, starts: List[Cell], exits: List[Cell], cache: Optional[Dict[Tuple[Cell, Cell], List[Cell]]] = None) -> List[List[Cell]]:
    paths: List[List[Cell]] = []
    for s in starts:
        best_path: List[Cell] = []
        best_cost = 1e18
        for ex in exits:
            p = astar_cached(cost_map, s, ex, cache)
            if p:
                c = compute_path_cost(cost_map, s, p)
                if c < best_cost:
                    best_path = p
                    best_cost = c
        # Optional smoothing to reduce jaggedness (line-of-sight smoothing)
        best_path = smooth_path_los(best_path, np.where(np.isfinite(cost_map), 0, 1)) if best_path else best_path
        paths.append(best_path)
    return paths


# Fast multi-exit Dijkstra precompute for initialization
def precompute_distance_and_parent(cost_map: np.ndarray, exits: List[Cell]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute multi-source Dijkstra from all exits.
    Returns (dist, parent_r, parent_c) where parent points toward an exit.
    Cells with inf in cost_map remain unreachable with dist=inf and parent=-1.
    """
    import heapq
    H, W = cost_map.shape
    dist = np.full((H, W), np.inf, dtype=np.float32)
    parent_r = np.full((H, W), -1, dtype=np.int16)
    parent_c = np.full((H, W), -1, dtype=np.int16)
    pq: List[Tuple[float, Tuple[int,int]]] = []

    def inb(r: int, c: int) -> bool:
        return 0 <= r < H and 0 <= c < W

    def passable(r: int, c: int) -> bool:
        return inb(r, c) and np.isfinite(cost_map[r, c])

    # Initialize queue with all exits
    for ex in exits:
        r, c = ex
        if passable(r, c):
            dist[r, c] = 0.0
            heapq.heappush(pq, (0.0, (r, c)))

    # 8-neighbor Dijkstra with diagonal costs and corner-cut prevention
    SQRT2 = math.sqrt(2.0)
    DIRS = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]
    while pq:
        d, (r, c) = heapq.heappop(pq)
        if d != dist[r, c]:
            continue
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not passable(nr, nc):
                continue
            if dr != 0 and dc != 0:
                if not (passable(r + dr, c) and passable(r, c + dc)):
                    continue
            base = SQRT2 if (dr != 0 and dc != 0) else 1.0
            step = float(cost_map[nr, nc])
            nd = d + base * step
            if nd < float(dist[nr, nc]):
                dist[nr, nc] = nd
                parent_r[nr, nc] = r
                parent_c[nr, nc] = c
                heapq.heappush(pq, (nd, (nr, nc)))
    return dist, parent_r, parent_c

def build_paths_from_parent(starts: List[Cell], parent_r: np.ndarray, parent_c: np.ndarray) -> List[List[Cell]]:
    paths: List[List[Cell]] = []
    H, W = parent_r.shape
    for s in starts:
        r, c = s
        path: List[Cell] = []
        # Follow parent pointers until we reach a cell whose parent is -1 (exit or unreachable)
        seen = 0
        while 0 <= r < H and 0 <= c < W and parent_r[r, c] != -1 and parent_c[r, c] != -1 and seen < (H*W):
            pr, pc = int(parent_r[r, c]), int(parent_c[r, c])
            path.append((pr, pc))
            r, c = pr, pc
            seen += 1
        paths.append(path)
    return paths

# -----------------------------
# Insurance analytics helpers
# -----------------------------
def _live_stampede_risk(state: Dict) -> float:
    """Heuristic: combine congestion and peak density into 0-100% risk."""
    t = max(1, state.get("t", 1))
    N = len(state.get("agents", [])) or 1
    # Congestion ratio per agent per tick
    total_cong = sum(a.congestion for a in state.get("agents", []))
    cong_ratio = total_cong / (N * t)
    # Peak density per cell per tick
    visits = state.get("visits")
    if visits is None:
        peak_density = 0.0
    else:
        peak_density = float(np.max(visits)) / t
        # normalize by a nominal crowding threshold (~0.5 agent per cell per tick)
        peak_density = min(1.0, peak_density / 0.5)
    # Combine with weights
    risk = min(100.0, (cong_ratio * 70.0 + peak_density * 50.0))
    return float(risk)


def _categorize(value: float, thresholds: Tuple[float, float]) -> str:
    lo, hi = thresholds
    if value <= lo:
        return "Low"
    if value <= hi:
        return "Medium"
    return "High"


def compute_insurance_report(state: Dict, avg_speed: float, panic: float) -> Dict[str, object]:
    total_agents = len(state.get("agents", []))
    ticks = int(state.get("t", 0))
    evac_seconds = ticks * TICK_SECONDS
    evac_time_str = f"{int(evac_seconds//60)}:{int(evac_seconds%60):02d} ({ticks} ticks)"

    # Exit usage percent
    exit_counts = state.get("exit_counts", {})
    total_exited = sum(exit_counts.values()) or 1
    exit_usage = {str(k): round(v/total_exited*100, 2) for k, v in exit_counts.items()}

    # Per-exit travel time stats (seconds)
    per_exit_times: Dict[str, Dict[str, float]] = {}
    times_by_exit: Dict[str, List[float]] = {}
    for a in state.get("agents", []):
        if a.exit_time is not None and a.exit_pos is not None:
            key = str(a.exit_pos)
            times_by_exit.setdefault(key, []).append(a.exit_time * TICK_SECONDS)
    for k, vals in times_by_exit.items():
        arr = np.array(sorted(vals))
        if arr.size:
            p50 = float(np.percentile(arr, 50))
            p90 = float(np.percentile(arr, 90))
            per_exit_times[k] = {
                "count": int(arr.size),
                "mean_s": float(arr.mean()),
                "p50_s": p50,
                "p90_s": p90,
                "max_s": float(arr.max()),
            }
        else:
            per_exit_times[k] = {"count": 0, "mean_s": 0.0, "p50_s": 0.0, "p90_s": 0.0, "max_s": 0.0}

    # Group-level statistics
    group_stats: Dict[str, Dict[str, object]] = {}
    groups = {}
    for a in state.get("agents", []):
        gid = getattr(a, "group_id", 0)
        groups.setdefault(gid, []).append(a)
    for gid, members in groups.items():
        times = [m.exit_time * TICK_SECONDS for m in members if m.exit_time is not None]
        cong = [m.congestion for m in members]
        usage: Dict[str, int] = {}
        for m in members:
            if m.exit_pos is not None:
                usage[str(m.exit_pos)] = usage.get(str(m.exit_pos), 0) + 1
        usage_total = sum(usage.values()) or 1
        usage_pct = {k: round(v/usage_total*100, 2) for k, v in usage.items()}
        if times:
            arr = np.array(sorted(times))
            group_stats[str(gid)] = {
                "count": len(members),
                "evac_mean_s": float(arr.mean()),
                "evac_median_s": float(np.median(arr)),
                "congestion_mean": float(np.mean(cong)) if cong else 0.0,
                "exit_usage": usage_pct,
            }
        else:
            group_stats[str(gid)] = {
                "count": len(members),
                "evac_mean_s": 0.0,
                "evac_median_s": 0.0,
                "congestion_mean": float(np.mean(cong)) if cong else 0.0,
                "exit_usage": usage_pct,
            }

    # Per-type statistics
    per_type: Dict[str, Dict[str, float]] = {}
    by_type_times: Dict[str, List[float]] = {}
    by_type_speed: Dict[str, List[float]] = {}
    by_type_cong: Dict[str, List[int]] = {}
    for a in state.get("agents", []):
        tname = getattr(a, "agent_type", "Pedestrian")
        if a.exit_time is not None:
            by_type_times.setdefault(tname, []).append(a.exit_time * TICK_SECONDS)
        by_type_speed.setdefault(tname, []).append(getattr(a, "speed_cells", 1.0) * CELL_METERS / TICK_SECONDS)
        by_type_cong.setdefault(tname, []).append(a.congestion)
    for tname in by_type_speed.keys():
        times = np.array(sorted(by_type_times.get(tname, [])))
        per_type[tname] = {
            "evac_mean_s": float(times.mean()) if times.size else 0.0,
            "evac_median_s": float(np.median(times)) if times.size else 0.0,
            "avg_speed_mps": float(np.mean(by_type_speed.get(tname, [0.0]))) if by_type_speed.get(tname) else 0.0,
            "risk_proxy": float(np.mean(by_type_cong.get(tname, [0]))) if by_type_cong.get(tname) else 0.0,
        }

    # Risks
    stampede_risk = _live_stampede_risk(state)
    avg_congestion = (sum(a.congestion for a in state.get("agents", [])) / max(1, total_agents))

    # Categories
    evac_cat = _categorize(evac_seconds, (120, 300))  # <=2min low, <=5min medium, else high
    stampede_cat = _categorize(stampede_risk, (10, 20))
    exit_cong_cat = _categorize(avg_congestion, (5, 20))
    behavior = "Normal" if panic < 0.1 else ("Mixed" if panic <= 0.4 else "Panic")

    # Component scores (0-100)
    def score_cat(cat: str) -> int:
        return {"Low": 20, "Medium": 60, "High": 85}.get(cat, 50)
    behavior_score = {"Normal": 20, "Mixed": 50, "Panic": 80}[behavior]
    risk_score = int(0.30 * score_cat(evac_cat) + 0.30 * stampede_risk + 0.25 * score_cat(exit_cong_cat) + 0.15 * behavior_score)

    # Recommendations
    recs: List[str] = []
    if evac_cat == "High":
        recs.append("Add more emergency exits or reduce travel distance.")
    if exit_cong_cat in ("Medium", "High"):
        recs.append("Improve exit signage or increase exit width to reduce queueing.")
    if stampede_risk > 20:
        recs.append("Conduct regular evacuation drills to reduce panic and crowd pressure.")
    recs.append("Ensure clear aisles and remove obstacles from primary egress routes.")
    recommendations = " | ".join(recs)

    # Safety checklist
    total_evac_pass = evac_seconds <= 300  # < 5 min
    per_exit_p90_pass = all((v.get("p90_s", 0.0) <= 120.0) for v in per_exit_times.values()) if per_exit_times else True
    stampede_pass = stampede_risk < 20.0
    safety_checklist = {
        "total_evac_under_5min": bool(total_evac_pass),
        "per_exit_p90_under_120s": bool(per_exit_p90_pass),
        "stampede_risk_under_20pct": bool(stampede_pass),
        "overall_pass": bool(total_evac_pass and per_exit_p90_pass and stampede_pass),
    }

    return {
        "total_agents": total_agents,
        "evacuation_time": evac_time_str,
        "avg_speed": round(avg_speed, 2),
        "panic_level": int(panic * 100),
        "stampede_risk": round(stampede_risk, 1),
        "exit_usage": exit_usage,
        "per_exit_times": per_exit_times,
        "group_stats": group_stats,
        "risk_score": risk_score,
        "per_type": per_type,
        "evacuation_time_cat": evac_cat,
        "stampede_risk_cat": stampede_cat,
        "exit_congestion_cat": exit_cong_cat,
        "agent_behavior": behavior,
        "recommendations": recommendations,
        "safety_checklist": safety_checklist,
    }


def step_simulation(state: Dict, panic: float = 0.1) -> None:
    """Advance the simulation by 1 tick with collision avoidance and panic hesitation.

    state keys used/updated:
      - t: int
      - agents: List[AgentState]
      - exits: List[Cell]
      - positions: List[Optional[Cell]]
      - paths: List[List[Cell]]
      - done: List[bool]
      - exit_counts: Dict[Cell, int]
      - visits: np.ndarray
    """
    t: int = state["t"]
    agents: List[AgentState] = state["agents"]
    exits: List[Cell] = state["exits"]
    paths: List[List[Cell]] = state["paths"]
    positions: List[Optional[Cell]] = [a.pos if not a.exited else None for a in agents]

    grid = state["grid"]
    H, W = grid.shape

    # Lazy init adaptive maps and RNG in session state
    if "_rng" not in state:
        state["_rng"] = random.Random(time.time())
    if "adapt_map" not in state:
        state["adapt_map"] = np.zeros_like(state["cost_map"], dtype=float)
    if "_last_pos" not in state:
        state["_last_pos"] = {i: None for i, _ in enumerate(agents)}
    if "_motion_mode" not in state:
        state["_motion_mode"] = {i: None for i, _ in enumerate(agents)}
    if "_visited" not in state:
        state["_visited"] = {i: set() for i, _ in enumerate(agents)}
    # Small decay to gradually forget old congestion
    state["adapt_map"] *= 0.985

    # Mark density/visits
    for a in agents:
        if not a.exited:
            r, c = a.pos
            state["visits"][r, c] += 1

    # Per-tick exit flow counters (for capacity limits)
    state.setdefault("_exit_flow_tick", {})
    exit_flow_tick: Dict[Cell, int] = {}

    # Proposals with one-agent-per-cell rule (except exits which allow queued intake)
    proposals: Dict[int, Cell] = {}
    target_counts: Dict[Cell, int] = {}
    rng = state["_rng"]
    for i, a in enumerate(agents):
        if a.exited:
            continue
        # Pre-evacuation delay: do nothing until delay expires (only if enabled)
        if bool(state.get("enable_pre_delay", False)) and t < getattr(a, "pre_delay_ticks", 0):
            proposals[i] = a.pos
            target_counts[a.pos] = target_counts.get(a.pos, 0) + 1
            continue
        # Reached exit already handled in post-move via capacity
        H, W = grid.shape
        next_pos = a.pos
        # Panic hesitation: tiny stochastic dithering (disabled if exit is in clear LOS nearby)
        if random.random() < (panic * getattr(a, "panic_mult", 1.0)) * 0.1:
            # If any exit is visible with LOS and within a reasonable heuristic distance, skip dithering
            los_near = False
            if exits:
                SQRT2 = math.sqrt(2.0)
                def h_oct(p, q):
                    dx = abs(p[0] - q[0]); dy = abs(p[1] - q[1])
                    return (dx + dy) + (SQRT2 - 2.0) * min(dx, dy)
                los_grid_chk = np.where(np.isfinite(state["cost_map"]), 0, 1)
                for ex in exits:
                    if _los_clear(los_grid_chk, a.pos, ex) and h_oct(a.pos, ex) <= 25.0:
                        los_near = True
                        break
            if not los_near:
                # Noise bounded by not increasing octile distance to nearest exit
                r, c = a.pos
                SQRT2 = math.sqrt(2.0)
                def h_oct(p, q):
                    dx = abs(p[0] - q[0]); dy = abs(p[1] - q[1])
                    return (dx + dy) + (SQRT2 - 2.0) * min(dx, dy)
                if exits:
                    cur_d = min(h_oct(a.pos, ex) for ex in exits)
                else:
                    cur_d = 0.0
                candidates = [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
                best = a.pos
                best_d = cur_d
                random.shuffle(candidates)
                for rr, cc in candidates:
                    if 0 <= rr < H and 0 <= cc < W and np.isfinite(state["cost_map"][rr, cc]):
                        d = min(h_oct((rr, cc), ex) for ex in exits) if exits else 0.0
                        if d <= best_d + 1e-6:
                            best = (rr, cc)
                            best_d = d
                next_pos = best
        else:
            # Goal-driven movement with optional local avoidance
            r, c = a.pos
            # Determine current goal: prefer next path waypoint; else nearest exit by octile heuristic
            if a.path:
                goal_cell = a.path[0]
            else:
                SQRT2 = math.sqrt(2.0)
                def h_oct(p, q):
                    dx = abs(p[0] - q[0]); dy = abs(p[1] - q[1])
                    return (dx + dy) + (SQRT2 - 2.0) * min(dx, dy)
                goal_cell = min(exits, key=lambda ex: h_oct(a.pos, ex)) if exits else a.pos

            # Exit awareness: if we have a clear line-of-sight to any exit nearby, bias strongly toward it
            los_grid = np.where(np.isfinite(state["cost_map"]), 0, 1)
            los_target = None
            if exits:
                # Prefer closest exit by octile heuristic that has LOS and is reasonably near
                SQRT2 = math.sqrt(2.0)
                def h_oct(p, q):
                    dx = abs(p[0] - q[0]); dy = abs(p[1] - q[1])
                    return (dx + dy) + (SQRT2 - 2.0) * min(dx, dy)
                sorted_exits = sorted(exits, key=lambda ex: h_oct(a.pos, ex))
                for ex in sorted_exits[:3]:
                    if _los_clear(los_grid, a.pos, ex) and h_oct(a.pos, ex) <= 25.0:
                        los_target = ex
                        break
            if los_target is not None:
                goal_cell = los_target
                state["_motion_mode"][i] = 'straight'
                mode = 'straight'

            # Motion mode selection to introduce natural variability
            if state["_motion_mode"].get(i) is None or rng.random() < 0.1:
                state["_motion_mode"][i] = _choose_motion_mode(grid, a.pos)
            mode = state["_motion_mode"].get(i, "straight")

            # Bias the goal slightly based on mode (spiral/curvy/zigzag/left/right)
            gr, gc = goal_cell
            mr, mc = a.pos
            dr, dc = gr - mr, gc - mc
            mag = max(1.0, math.hypot(dr, dc))
            ndir = (dr / mag, dc / mag)
            # Per-mode lateral offsets
            lat = (-ndir[1], ndir[0])
            fwd = ndir
            bias_scale = 1.0
            if mode == 'curvy':
                gr += int(round(0.5 * lat[0])); gc += int(round(0.5 * lat[1]))
                bias_scale = 0.9
            elif mode == 'zigzag':
                sign = 1 if (t // 3) % 2 == 0 else -1
                gr += int(round(sign * lat[0])); gc += int(round(sign * lat[1]))
                bias_scale = 0.85
            elif mode == 'spiral':
                # small forward + lateral drift
                gr += int(round(0.5 * fwd[0] + 0.5 * lat[0])); gc += int(round(0.5 * fwd[1] + 0.5 * lat[1]))
                bias_scale = 0.8
            elif mode == 'left':
                gr += int(round(lat[0])); gc += int(round(lat[1]))
                bias_scale = 0.95
            elif mode == 'right':
                gr -= int(round(lat[0])); gc -= int(round(lat[1]))
                bias_scale = 0.95
            # Keep biased goal within bounds and walkable if possible
            br = int(np.clip(gr, 0, H - 1)); bc = int(np.clip(gc, 0, W - 1))
            if 0 <= br < H and 0 <= bc < W and np.isfinite(state["cost_map"][br, bc]):
                goal_cell = (br, bc)

            if state.get("enable_local_avoidance", True):
                # Build allowed 8-neighborhood moves (including staying)
                allowed: List[Cell] = []
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = r + dr, c + dc
                        if dr == 0 and dc == 0:
                            allowed.append((r, c))
                            continue
                        if not (0 <= nr < H and 0 <= nc < W):
                            continue
                        if not np.isfinite(state["cost_map"][nr, nc]):
                            continue
                        # Prevent diagonal corner-cutting
                        if dr != 0 and dc != 0:
                            or1 = (r + dr, c)
                            or2 = (r, c + dc)
                            if not (0 <= or1[0] < H and 0 <= or1[1] < W and np.isfinite(state["cost_map"][or1])):
                                continue
                            if not (0 <= or2[0] < H and 0 <= or2[1] < W and np.isfinite(state["cost_map"][or2])):
                                continue
                        allowed.append((nr, nc))

                # Nearby agent positions for repulsion
                neighbor_agents: List[Tuple[float, float]] = []
                for j, other in enumerate(agents):
                    if j == i or other.exited:
                        continue
                    pr, pc = other.pos
                    if abs(pr - r) <= 2 and abs(pc - c) <= 2:
                        neighbor_agents.append((float(pr), float(pc)))

                # Effective cost map mixes base cost, learned avoidance, and live density
                # Learned avoidance encourages detours from historically congested/conflicted areas
                visits = state.get("visits")
                t_eff = max(1, t)
                density_term = (visits / float(t_eff)) if visits is not None else 0.0
                effective_cost = state["cost_map"] + 0.75 * state["adapt_map"] + 0.5 * density_term

                # Discourage backtracking explicitly by inflating cost of the previous cell
                last_p = state["_last_pos"].get(i)
                if last_p is not None:
                    lr, lc = last_p
                    if 0 <= lr < H and 0 <= lc < W:
                        effective_cost[lr, lc] = effective_cost[lr, lc] + 2.0

                # Strongly discourage revisiting any previously visited cells by this agent
                visited = state["_visited"].get(i)
                if visited:
                    for vr, vc in list(visited):
                        if 0 <= vr < H and 0 <= vc < W and np.isfinite(effective_cost[vr, vc]):
                            effective_cost[vr, vc] = effective_cost[vr, vc] + 1.0

                # Weight goals/repulsions based on motion mode and panic
                panic_mult = getattr(a, "panic_mult", 1.0)
                w_goal = 1.0 * bias_scale
                w_agents = 0.8 * (1.0 + 0.3 * (panic * panic_mult))
                w_walls = 0.6

                ranked = rank_moves(grid=np.where(np.isfinite(state["cost_map"]), 0, 1),
                                     cost_map=effective_cost,
                                     pos=a.pos,
                                     goal=(float(goal_cell[0]), float(goal_cell[1])),
                                     allowed=allowed,
                                     neighbor_agents=neighbor_agents,
                                     wall_repulsion_radius=2,
                                     agent_repulsion_radius=2.0,
                                     w_goal=w_goal,
                                     w_agents=w_agents,
                                     w_walls=w_walls)

                # If LOS to exit, force greedy step that most reduces octile distance.
                # Otherwise, avoid backtracking/visited if possible, else soft selection for variability
                if ranked:
                    if los_target is not None:
                        # Greedy: choose move that strictly reduces octile distance to the LOS exit
                        SQRT2 = math.sqrt(2.0)
                        def h_oct2(p, q):
                            dx = abs(p[0] - q[0]); dy = abs(p[1] - q[1])
                            return (dx + dy) + (SQRT2 - 2.0) * min(dx, dy)
                        cur_h = h_oct2(a.pos, los_target)
                        best_mv = a.pos
                        best_h = cur_h
                        for _, mv in ranked[:5]:
                            nh = h_oct2(mv, los_target)
                            if nh < best_h - 1e-6:
                                best_h = nh
                                best_mv = mv
                        next_pos = best_mv
                    else:
                        # Prefer candidates that are not last_pos and not in visited if they don't worsen heuristic
                        last_p = state["_last_pos"].get(i)
                        visited = state["_visited"].get(i, set())
                        # Heuristic toward current goal
                        SQRT2 = math.sqrt(2.0)
                        def h_goal(p):
                            dx = abs(p[0] - goal_cell[0]); dy = abs(p[1] - goal_cell[1])
                            return (dx + dy) + (SQRT2 - 2.0) * min(dx, dy)
                        cur_h = h_goal(a.pos)
                        candidates = [mv for _, mv in ranked[:5]]
                        # Filter out exact backtrack if alternatives exist
                        alt = [mv for mv in candidates if mv != last_p and h_goal(mv) <= cur_h + 1e-6]
                        if alt:
                            candidates = alt
                        # Prefer unvisited
                        alt2 = [mv for mv in candidates if mv not in visited]
                        if alt2:
                            candidates = alt2
                        # If multiple, sample softly among top 3 of the filtered
                        sub_ranked = [item for item in ranked if item[1] in candidates]
                        top_k = sub_ranked[:min(3, len(sub_ranked))] if sub_ranked else ranked[:min(3, len(ranked))]
                        temp = max(0.15, 0.5 * (2.0 - w_goal))
                        scores = np.array([s for s, _ in top_k], dtype=float)
                        scores = scores - scores.max()
                        probs = np.exp(scores / temp)
                        probs = probs / (probs.sum() + 1e-9)
                        choice = rng.choices(range(len(top_k)), weights=probs.tolist(), k=1)[0]
                        next_pos = top_k[choice][1]
                else:
                    next_pos = a.pos
            else:
                # Strict path following with 8-dir corner-cut prevention
                next_pos = a.pos
                if a.path:
                    step = a.path[0]
                    if step and 0 <= step[0] < H and 0 <= step[1] < W and np.isfinite(state["cost_map"][step]):
                        dr = step[0] - r
                        dc = step[1] - c
                        if dr != 0 and dc != 0:
                            or1 = (r + dr, c)
                            or2 = (r, c + dc)
                            if (0 <= or1[0] < H and 0 <= or1[1] < W and np.isfinite(state["cost_map"][or1])) and \
                               (0 <= or2[0] < H and 0 <= or2[1] < W and np.isfinite(state["cost_map"][or2])):
                                next_pos = step
                        else:
                            next_pos = step

        proposals[i] = next_pos
        target_counts[next_pos] = target_counts.get(next_pos, 0) + 1

    # First, handle agents proposing to move into exits this tick under capacity
    exit_candidates: Dict[Cell, List[int]] = {}
    for idx, a in enumerate(agents):
        if a.exited:
            continue
        nxt = proposals.get(idx, a.pos)
        if nxt in exits:
            exit_candidates.setdefault(nxt, []).append(idx)

    cap_scale = float(state.get("exit_capacity_scale", 0.0))
    exit_flow_tick = {}
    if cap_scale > 0.0:
        for ex in exits:
            width = _exit_width_cells(ex, grid)
            capacity = max(1, int(round(width * cap_scale)))
            exit_flow_tick[ex] = 0
            if ex in exit_candidates:
                allowed = exit_candidates[ex][:capacity]
                for idx in allowed:
                    exit_flow_tick[ex] += 1
    else:
        for ex, idxs in exit_candidates.items():
            exit_flow_tick[ex] = len(idxs)

    # Resolve conflicts for non-exit cells: allow exactly one winner per cell
    new_positions: set = set()
    # Track which agents actually entered an exit cell under capacity this tick
    evac_ready: Dict[Cell, List[int]] = {}

    # 1) Process exits first (respect capacity)
    # Build list of agent indices targeting exits
    exit_targeters: Dict[Cell, List[int]] = {}
    for idx, a in enumerate(agents):
        if a.exited:
            continue
        nxt = proposals.get(idx, a.pos)
        if nxt in exits:
            exit_targeters.setdefault(nxt, []).append(idx)

    for ex, idxs in exit_targeters.items():
        for idx in idxs:
            a = agents[idx]
            # Only move into exit cell if capacity allows (evacuation processed below)
            can_enter = True
            if cap_scale > 0.0:
                can_enter = exit_flow_tick.get(ex, 0) > 0
                if can_enter:
                    exit_flow_tick[ex] -= 1
            if can_enter:
                a.pos = ex
                new_positions.add(ex)
                evac_ready.setdefault(ex, []).append(idx)
            else:
                # Could not enter due to capacity -> stay
                new_positions.add(a.pos)
                a.congestion += 1
                if random.random() < 0.1:
                    a.congestion += 1

    # 2) Process non-exit targets: group by cell, pick one winner randomly
    cell_to_agents: Dict[Cell, List[int]] = {}
    for idx, a in enumerate(agents):
        if a.exited:
            continue
        nxt = proposals.get(idx, a.pos)
        if nxt in exits:
            continue  # already handled
        cell_to_agents.setdefault(nxt, []).append(idx)

    for cell, idxs in cell_to_agents.items():
        if cell in new_positions:
            # Already occupied this tick (e.g., exit), all stay
            for idx in idxs:
                a = agents[idx]
                new_positions.add(a.pos)
                a.congestion += 1
                if random.random() < 0.1:
                    a.congestion += 1
                # Learn: increase avoidance at congested spot
                pr, pc = a.pos
                if 0 <= pr < H and 0 <= pc < W:
                    state["adapt_map"][pr, pc] += 0.5
            continue
        if len(idxs) == 1:
            # Single agent can move
            idx = idxs[0]
            a = agents[idx]
            if a.path and a.path[0] == cell:
                a.path.pop(0)
            a.pos = cell
            new_positions.add(cell)
            a.path_len += 1
            a.congestion += 1 if len(new_positions) > 2 else 0
        else:
            # Multiple contenders: pick one winner randomly (deterministic per tick)
            rng.shuffle(idxs)
            winner = idxs[0]
            for idx in idxs:
                a = agents[idx]
                if idx == winner:
                    if a.path and a.path[0] == cell:
                        a.path.pop(0)
                    a.pos = cell
                    new_positions.add(cell)
                    a.path_len += 1
                    a.congestion += 1 if len(new_positions) > 2 else 0
                else:
                    # Losers wait in place
                    new_positions.add(a.pos)
                    a.congestion += 1
                    # Learn from losing a conflict
                    pr, pc = a.pos
                    if 0 <= pr < H and 0 <= pc < W:
                        state["adapt_map"][pr, pc] += 0.3

    

    # Apply evacuations for agents in exit cells under capacity this tick
    for ex, idxs in evac_ready.items():
        for idx in idxs:
            a = agents[idx]
            if a.exited:
                continue
            a.exited = True
            a.exit_time = t
            a.exit_pos = ex
            state["exit_counts"][ex] = state["exit_counts"].get(ex, 0) + 1
            atype = getattr(a, "agent_type", "Pedestrian")
            state.setdefault("exit_counts_by_type", {})
            state["exit_counts_by_type"].setdefault(atype, {})
            state["exit_counts_by_type"][atype][ex] = state["exit_counts_by_type"][atype].get(ex, 0) + 1

    # Update last positions for backtrack discouragement
    for i, a in enumerate(agents):
        if not a.exited:
            state["_last_pos"][i] = a.pos
            # Remember visited to reduce backtracking
            state["_visited"].setdefault(i, set()).add(a.pos)

    # (Removed) Stuck detection, periodic replanning, and Dijkstra fallback per user request

    # Update time-series (throughput per exit and total evac curve) at current tick
    for ex in state["exits"]:
        key = str(ex)
        cur = state["exit_counts"].get(ex, 0)
        state["exit_counts_ts"].setdefault(key, [])
        state["exit_counts_ts"][key].append(cur)
    total_exited = sum(state["exit_counts"].values())
    state["evac_curve_ts"].append(total_exited)

    state["t"] = t + 1


def render_frame(grid: np.ndarray, exits: List[Cell], hazards: np.ndarray, agents: List[AgentState], agent_positions: Optional[List[Optional[Tuple[float, float]]]] = None) -> plt.Figure:
    H, W = grid.shape
    fig, ax = plt.subplots(figsize=(8, 8))

    # Base visualization
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    vis[grid == 1] = [0, 0, 0]  # walls
    vis[grid == 0] = [255, 255, 255]  # free
    vis[hazards == 1] = [255, 200, 0]  # hazard overlay
    for (r, c) in exits:
        vis[r, c] = [0, 200, 0]

    ax.imshow(vis, origin='upper', interpolation='nearest')

    # Agents colored by type. If agent_positions is provided, use it even if agent has exited this tick.
    by_type: Dict[str, List[Tuple[float, float]]] = {}
    if agent_positions is None:
        for a in agents:
            if a.exited:
                continue
            by_type.setdefault(getattr(a, "agent_type", "Pedestrian"), []).append((float(a.pos[0]), float(a.pos[1])))
    else:
        for a, p in zip(agents, agent_positions):
            if p is None:
                continue
            by_type.setdefault(getattr(a, "agent_type", "Pedestrian"), []).append((float(p[0]), float(p[1])))

    for t, pts in by_type.items():
        if not pts:
            continue
        rr = [p[0] for p in pts]
        cc = [p[1] for p in pts]
        col = TYPE_CONFIG.get(t, TYPE_CONFIG["Pedestrian"]).get("color", (30,144,255))
        # Normalize 0-255 RGB to 0-1 for matplotlib
        col_norm = tuple([c/255.0 for c in col])
        ax.scatter(cc, rr, c=[col_norm], s=10, edgecolors='k', linewidths=0.3, label=t)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Evacuation")
    fig.tight_layout()
    return fig

# -----------------------------
# Human-like motion helpers (continuous space)
# -----------------------------
def _bresenham_line(r0: int, c0: int, r1: int, c1: int):
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = (dr - dc)
    r, c = r0, c0
    while True:
        yield (r, c)
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc

def _los_clear(grid: np.ndarray, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    H, W = grid.shape
    for rr, cc in _bresenham_line(a[0], a[1], b[0], b[1]):
        if not (0 <= rr < H and 0 <= cc < W):
            return False
        if grid[rr, cc] != 0 and (rr, cc) != a and (rr, cc) != b:
            return False
    return True

def _smooth_path_los(path: List[Tuple[int, int]], grid: np.ndarray) -> List[Tuple[int, int]]:
    if not path:
        return path
    sm = [path[0]]
    j = 0
    while j < len(path) - 1:
        k = len(path) - 1
        found = False
        while k > j + 1:
            if _los_clear(grid, path[j], path[k]):
                sm.append(path[k])
                j = k
                found = True
                break
            k -= 1
        if not found:
            sm.append(path[j + 1])
            j += 1
    return sm

def _chaikin(points: List[Tuple[float, float]], iterations: int = 2) -> List[Tuple[float, float]]:
    if len(points) < 3 or iterations <= 0:
        return points
    pts = points[:]
    for _ in range(iterations):
        new_pts = [pts[0]]
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            Q = (0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1)
            R = (0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1)
            new_pts.extend([Q, R])
        new_pts.append(pts[-1])
        pts = new_pts
    return pts

 


def export_outputs(out_dir: str, state: Dict) -> Dict[str, Optional[str]]:
    os.makedirs(out_dir, exist_ok=True)
    paths: Dict[str, Optional[str]] = {
        "floorplan": None,
        "gif": None,
        "mp4": None,
        "evac_curve": None,
        "density": None,
        "trajectories": None,
        "csv": None,
        "exit_usage": None,
        "insurance_report": None,
        "insurance_report_json": None,
        "time_series_csv": None,
        "time_series_json": None,
        "pdf_report": None,
        "exit_usage_by_type": None,
    }

    grid = state["grid"]
    hazards = state["hazards"]
    exits = state["exits"]
    rgb = state.get("rgb")
    H, W = grid.shape

    # Floorplan PNG
    fp_path = os.path.join(out_dir, "floorplan.png")
    fig, ax = plt.subplots(figsize=(8, 8))
    if rgb is not None:
        ax.imshow(rgb, origin='upper')
    else:
        vis = np.zeros((H, W, 3), dtype=np.uint8)
        vis[grid == 1] = [0, 0, 0]
        vis[grid == 0] = [255, 255, 255]
        vis[hazards == 1] = [255, 200, 0]
        for (r, c) in exits:
            vis[r, c] = [0, 200, 0]
        ax.imshow(vis, origin='upper')
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(fp_path, dpi=150); plt.close(fig)
    paths["floorplan"] = fp_path

    # GIF of sampled frames
    if imageio is not None and state.get("frames"):
        gif_path = os.path.join(out_dir, "evac_demo.gif")
        try:
            imageio.mimsave(gif_path, state["frames"], fps=6)
            paths["gif"] = gif_path
        except Exception:
            paths["gif"] = None

    # MP4 export if frames captured for video
    if imageio is not None and state.get("frames"):
        mp4_path = os.path.join(out_dir, "evac_demo.mp4")
        fps_out = int(state.get("_video_fps", 30) or 30)
        try:
            with imageio.get_writer(mp4_path, fps=fps_out, codec="libx264", quality=8) as writer:
                for frame in state["frames"]:
                    writer.append_data(frame)
            paths["mp4"] = mp4_path
        except Exception:
            # Fallback try without specifying codec
            try:
                with imageio.get_writer(mp4_path, fps=fps_out) as writer:
                    for frame in state["frames"]:
                        writer.append_data(frame)
                paths["mp4"] = mp4_path
            except Exception:
                paths["mp4"] = None

    # Evacuation curve
    evac_curve = os.path.join(out_dir, "evac_curve.png")
    exited = [a.exit_time for a in state["agents"] if a.exit_time is not None]
    max_t = max(exited + [0])
    curve = []
    for t in range(max_t + 1):
        curve.append(sum(1 for et in exited if et <= t))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(curve)), curve, 'b-')
    ax.set_xlabel('Time (ticks)'); ax.set_ylabel('Cumulative evacuees'); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(evac_curve, dpi=150); plt.close(fig)
    paths["evac_curve"] = evac_curve

    # Density heatmap
    density_path = os.path.join(out_dir, 'density.png')
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(state["visits"], cmap='hot', origin='upper')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(density_path, dpi=150); plt.close(fig)
    paths["density"] = density_path

    # Trajectories: sample first 30 agents
    traj_path = os.path.join(out_dir, 'trajectories.png')
    fig, ax = plt.subplots(figsize=(8, 8))
    base = np.zeros((H, W, 3), dtype=np.uint8)
    base[grid == 1] = [0, 0, 0]
    base[grid == 0] = [255, 255, 255]
    base[hazards == 1] = [255, 200, 0]
    for (r, c) in exits:
        base[r, c] = [0, 200, 0]
    ax.imshow(base, origin='upper')
    sampled = state.get("traj", [])
    for traj in sampled[:30]:
        if len(traj) >= 2:
            arr = np.array(traj)
            ax.plot(arr[:, 1], arr[:, 0], '-', alpha=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(traj_path, dpi=150); plt.close(fig)
    paths["trajectories"] = traj_path

    # Exit usage image
    if state.get("exit_counts"):
        ex_keys = list(state["exit_counts"].keys())
        ex_vals = [state["exit_counts"][k] for k in ex_keys]
        total_exited = sum(ex_vals) if sum(ex_vals) > 0 else 1
        perc = [v/total_exited*100 for v in ex_vals]
        fig_ex, ax_ex = plt.subplots(figsize=(6, 3))
        ax_ex.bar([str(k) for k in ex_keys], perc, color='tab:green')
        ax_ex.set_ylabel('% of evacuees'); ax_ex.set_ylim(0, 100)
        ax_ex.set_title('Exit Usage')
        exit_usage_path = os.path.join(out_dir, 'exit_usage.png')
        fig_ex.tight_layout(); fig_ex.savefig(exit_usage_path, dpi=150); plt.close(fig_ex)
        paths["exit_usage"] = exit_usage_path

    # Exit usage by type (stacked bars per exit)
    if state.get("exit_counts_by_type"):
        by_type = state["exit_counts_by_type"]  # {type: {exit: count}}
        exits_all = sorted({str(k) for d in by_type.values() for k in d.keys()})
        types_all = list(by_type.keys())
        data = np.array([[by_type.get(t, {}).get(eval(e) if e.startswith('(') else e, 0) for e in exits_all] for t in types_all])
        # convert eval strings back to str labels
        labels = [e for e in exits_all]
        totals = data.sum(axis=0); totals[totals == 0] = 1
        perc = (data / totals) * 100.0
        fig, ax = plt.subplots(figsize=(6, 3))
        bottom = np.zeros(len(exits_all))
        for i, t in enumerate(types_all):
            ax.bar(labels, perc[i], bottom=bottom, label=t)
            bottom += perc[i]
        ax.set_ylabel('% of evacuees'); ax.set_ylim(0, 100)
        ax.set_title('Exit Usage by Agent Type')
        ax.legend(fontsize=8, ncol=min(3, len(types_all)))
        path_t = os.path.join(out_dir, 'exit_usage_by_type.png')
        fig.tight_layout(); fig.savefig(path_t, dpi=150); plt.close(fig)
        paths["exit_usage_by_type"] = path_t

    # Time-series exports (CSV/JSON)
    ts_csv = os.path.join(out_dir, 'time_series.csv')
    ts_json = os.path.join(out_dir, 'time_series.json')
    try:
        # Build a flat table: tick, total_cum, and each exit cum
        max_len = len(state.get("evac_curve_ts", []))
        cols = {"tick": list(range(max_len)), "total_cum": state.get("evac_curve_ts", [])}
        for k, series in state.get("exit_counts_ts", {}).items():
            # pad to max_len
            s = list(series) + [series[-1] if series else 0] * (max_len - len(series))
            cols[f"exit_{k}"] = s
        pd.DataFrame(cols).to_csv(ts_csv, index=False)
        with open(ts_json, 'w', encoding='utf-8') as f:
            json.dump({"evac_curve_ts": state.get("evac_curve_ts", []), "exit_counts_ts": state.get("exit_counts_ts", {})}, f, ensure_ascii=False, indent=2)
        paths["time_series_csv"] = ts_csv
        paths["time_series_json"] = ts_json
    except Exception:
        pass

    # CSV results
    csv_path = os.path.join(out_dir, 'evac_results.csv')
    rows = []
    for i, a in enumerate(state["agents"]):
        rows.append({
            "agent_id": i,
            "start": str(state["starts"][i]) if i < len(state["starts"]) else "",
            "exit_cell": str(a.exit_pos) if a.exit_pos is not None else "",
            "exit_time": a.exit_time if a.exit_time is not None else "",
            "path_len": a.path_len,
            "congestion": a.congestion,
            "trapped": bool(a.trapped),
            "status": ("evacuated" if a.exit_time is not None else ("trapped" if a.trapped else "active")),
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    paths["csv"] = csv_path

    # Insurance report CSV
    report = state.get("report")
    if report is not None:
        ins_csv = os.path.join(out_dir, 'insurance_report.csv')
        pd.DataFrame([report]).to_csv(ins_csv, index=False)
        paths["insurance_report"] = ins_csv
        # JSON as well
        ins_json = os.path.join(out_dir, 'insurance_report.json')
        try:
            with open(ins_json, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            paths["insurance_report_json"] = ins_json
        except Exception:
            paths["insurance_report_json"] = None

    # PDF report export
    try:
        pdf_path = os.path.join(out_dir, 'insurance_report.pdf')
        with PdfPages(pdf_path) as pdf:
            # Page 1: Summary text
            fig, ax = plt.subplots(figsize=(8.3, 11.7))  # A4 portrait inches approx
            ax.axis('off')
            lines = [
                "Evacuation Insurance Report",
                "",
            ]
            rep = state.get("report") or {}
            for k in ["total_agents", "evacuation_time", "avg_speed", "panic_level", "stampede_risk", "risk_score"]:
                if k in rep:
                    lines.append(f"{k}: {rep[k]}")
            if rep.get("safety_checklist"):
                lines.append("")
                lines.append("Safety Checklist:")
                for ck, val in rep["safety_checklist"].items():
                    lines.append(f" - {ck}: {'PASS' if val else 'FAIL'}")
            ax.text(0.05, 0.95, "\n".join(lines), va='top', fontsize=11)
            pdf.savefig(fig); plt.close(fig)

            # Page 2: Evac curve
            fig, ax = plt.subplots(figsize=(8.3, 5))
            ev = state.get("evac_curve_ts", [])
            ax.plot(range(len(ev)), ev, 'b-')
            ax.set_title('Cumulative Evacuation Curve')
            ax.set_xlabel('Tick'); ax.set_ylabel('Cumulative evacuees'); ax.grid(True, alpha=0.3)
            pdf.savefig(fig); plt.close(fig)

            # Page 3: Per-exit cumulative curves
            fig, ax = plt.subplots(figsize=(8.3, 5))
            for k, series in (state.get("exit_counts_ts", {}) or {}).items():
                ax.plot(range(len(series)), series, label=str(k))
            ax.set_title('Per-Exit Cumulative Evacuees')
            ax.set_xlabel('Tick'); ax.set_ylabel('Cumulative by exit'); ax.legend()
            ax.grid(True, alpha=0.3)
            pdf.savefig(fig); plt.close(fig)

            # Page 4: Density heatmap
            fig, ax = plt.subplots(figsize=(8.3, 8.3))
            im = ax.imshow(state["visits"], cmap='hot', origin='upper')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title('Density Heatmap')
            ax.set_xticks([]); ax.set_yticks([])
            pdf.savefig(fig); plt.close(fig)

            # Page 5: Exit usage (if available)
            if state.get("exit_counts"):
                ex_keys = list(state["exit_counts"].keys())
                ex_vals = [state["exit_counts"][k] for k in ex_keys]
                total_exited = sum(ex_vals) if sum(ex_vals) > 0 else 1
                perc = [v/total_exited*100 for v in ex_vals]
                fig, ax = plt.subplots(figsize=(8.3, 5))
                ax.bar([str(k) for k in ex_keys], perc, color='tab:green')
                ax.set_ylabel('% of evacuees'); ax.set_ylim(0, 100)
                ax.set_title('Exit Usage')
                pdf.savefig(fig); plt.close(fig)
        paths["pdf_report"] = pdf_path
    except Exception:
        paths["pdf_report"] = None

    return paths


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Evacuation Simulator", layout="wide")

# Sidebar controls
st.sidebar.header("Simulation Parameters")
N = int(st.sidebar.slider("Number of agents", 10, 500, 100, 10))
num_groups = int(st.sidebar.slider("Number of Groups", 1, 10, 3, 1))
max_group_size = int(st.sidebar.slider("Max Group Size", 1, 20, 5, 1))
st.sidebar.markdown("**Agent Types**")
available_types = ["Pedestrian", "Elderly", "Child", "Wheelchair", "Staff"]
selected_types = st.sidebar.multiselect("Include types", available_types, default=["Pedestrian", "Elderly", "Child", "Wheelchair"]) 
dist = {}
if selected_types:
    st.sidebar.caption("Distribute percentages; will be normalized to 100%.")
    total_pct = 0
    for t in selected_types:
        default = 50 if t == "Pedestrian" else 0
        dist[t] = st.sidebar.slider(f"% {t}", 0, 100, default, 5)
        total_pct += dist[t]
    # normalize
    if total_pct == 0:
        dist = {t: (100//len(selected_types)) for t in selected_types}
    else:
        dist = {t: (v / total_pct) for t, v in dist.items()}
else:
    dist = {"Pedestrian": 1.0}
if selected_types:
    # Show projected counts by type
    st.sidebar.caption("Projected counts by type (approx):")
    cols_cnt = st.sidebar.columns(2)
    for i, t in enumerate(selected_types):
        cnt = int(round(N * dist.get(t, 0)))
        (cols_cnt[i % 2]).markdown(f"- {t}: **{cnt}**")
panic = float(st.sidebar.slider("Panic level (%)", 0.0, 100.0, 10.0, 1.0)) / 100.0
avg_speed = float(st.sidebar.slider("Average Speed (m/s)", 0.2, 2.5, 1.2, 0.1))
max_time = int(st.sidebar.slider("Max time (ticks)", 50, 5000, 500, 50))
hazard_weight = float(st.sidebar.slider("Hazard cost weight", 1.0, 10.0, 3.0, 0.5))
auto_export = st.sidebar.toggle("Auto-export outputs on run end", value=False)

# Visualization options
show_fps = st.sidebar.toggle("Show FPS", value=False)
vis_realtime = st.sidebar.toggle("Real-time Visualization", value=True)
smooth_mode = st.sidebar.toggle("Smooth Animation (30 FPS)", value=True)
fps = int(st.sidebar.slider("FPS", 10, 60, 30, 5))
vis_interval = int(st.sidebar.slider("Visual update every N ticks", 1, 5, 3, 1))
max_runtime_s = int(st.sidebar.slider("Max Runtime (seconds)", 30, 180, 90, 5))
fast_init = st.sidebar.toggle("Fast init (Dijkstra)", value=True)
capture_gif_frames = st.sidebar.toggle("Capture frames for GIF", value=False)
capture_mp4_frames = st.sidebar.toggle("Capture frames for MP4", value=False)

# Advanced controls
st.sidebar.header("Advanced")
enable_local_avoidance = st.sidebar.toggle("Enable local avoidance", value=True)

# Pre-evacuation delays
st.sidebar.header("Pre-evacuation Delays")
enable_pre_delay = st.sidebar.toggle("Enable pre-evac delays", value=False)
pre_delay_mean_s = float(st.sidebar.slider("Mean delay (s)", 0.0, 120.0, 5.0, 1.0))
pre_delay_std_s = float(st.sidebar.slider("Std delay (s)", 0.0, 60.0, 2.0, 1.0))

# Exit capacity model
st.sidebar.header("Exit Capacity")
cap_scale = float(st.sidebar.slider("EXIT_CAPACITY (agents/tick per width cell)", 0.0, 3.0, 0.0, 0.1))

# Spiral routing controls (no direction toggle)
st.sidebar.header("Floor Plan")
uploaded = st.sidebar.file_uploader("Upload image", type=["png", "jpg", "jpeg"]) 

base_grid_size = int(st.sidebar.slider("Grid resolution (max dimension, cells)", 60, 400, 150, 10))
grid_size = int(base_grid_size)

# Buttons
colb1, colb2, colb3 = st.sidebar.columns(3)
with colb1:
    start_clicked = st.button("Start")
with colb2:
    stop_clicked = st.button("Stop")
with colb3:
    reset_clicked = st.button("Reset")

# Human-like motion removed; no related UI/state

"""Load floorplan and detect changes; automatically reinitialize sim state if changed."""
floorplan_changed = False
if (uploaded is not None and uploaded.name != st.session_state.get("_last_fp_name")) or (st.session_state.get("_grid_size") != grid_size):
    img = Image.open(uploaded) if uploaded is not None else None
    st.session_state["floorplan"] = load_floorplan(img, grid_size=grid_size)
    st.session_state["_last_fp_name"] = uploaded.name if uploaded is not None else None
    st.session_state["_grid_size"] = grid_size
    floorplan_changed = True
    # Ensure running loop stops so we can reinitialize cleanly on next block
    st.session_state["running"] = False
elif "floorplan" not in st.session_state:
    st.session_state["floorplan"] = load_floorplan(None, grid_size=grid_size)
    st.session_state["_grid_size"] = grid_size

fp = st.session_state["floorplan"]
exits = detect_exits(fp)
if len(exits) == 0:
    st.error("No exits found")

"""Initialize or reinitialize simulation state"""
if reset_clicked or floorplan_changed or "state_init" not in st.session_state:
    grid = fp["grid"]
    hazards = fp["hazards"]
    cost_map = build_cost_map(grid, hazards, hazard_weight)

    free = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1]) if np.isfinite(cost_map[r, c])]
    starts = spawn_agents(free, exits, N)
    # Build agent type list according to distribution
    dist = st.session_state.get("type_distribution", {"Pedestrian": 1.0})
    types = list(dist.keys())
    weights = np.array([dist[t] for t in types], dtype=float)
    if weights.sum() <= 0:
        weights = np.ones(len(types), dtype=float)
    weights = weights / weights.sum()
    rng = np.random.default_rng(123)
    assigned_types = rng.choice(types, size=len(starts), p=weights)

    # Assign groups sequentially and cap per group size; set per-type attrs
    agents = []
    g = 0
    counts = [0] * max(1, num_groups)
    for s, tname in zip(starts, assigned_types):
        # find next group with room
        attempts = 0
        while attempts < num_groups and counts[g] >= max_group_size:
            g = (g + 1) % num_groups
            attempts += 1
        cfg = TYPE_CONFIG.get(tname, TYPE_CONFIG["Pedestrian"])
        sp_lo, sp_hi = cfg.get("speed_mps", (1.0, 1.5))
        sp_mps = float(rng.uniform(sp_lo, sp_hi))
        sp_cells = (sp_mps * TICK_SECONDS) / max(1e-6, CELL_METERS)
        a = AgentState(
            pos=s,
            group_id=g,
            agent_type=tname,
            speed_cells=sp_cells,
            panic_mult=float(cfg.get("panic_mult", 1.0)),
            group_radius=int(cfg.get("group_radius", 0)),
            is_wheelchair=bool(cfg.get("wheelchair", False)),
        )
        # Sample pre-evac delay if enabled; else ensure zero delay
        if enable_pre_delay:
            d_s = float(max(0.0, rng.normal(pre_delay_mean_s, pre_delay_std_s)))
            a.pre_delay_ticks = int(round(d_s / TICK_SECONDS))
        else:
            a.pre_delay_ticks = 0
        agents.append(a)
        counts[g] = counts[g] + 1
        g = (g + 1) % num_groups

    # Initialize path cache for A* memoization
    path_cache: Dict[Tuple[Cell, Cell], List[Cell]] = {}
    if exits:
        if st.session_state.get("fast_init", True):
            # One-time multi-exit Dijkstra, then build per-agent shortest paths
            _, pr, pc = precompute_distance_and_parent(cost_map, exits)
            paths = build_paths_from_parent(starts, pr, pc)
        else:
            paths = compute_paths(cost_map, starts, exits, cache=path_cache)
    else:
        paths = [[] for _ in starts]
    for a, p in zip(agents, paths):
        a.path = p

    # Continuous human-like motion removed; no continuous state or distance fields needed

    # Assign motion modes per agent
    motion_mode: List[str] = []
    mode_phase: List[float] = []
    for a in agents:
        motion_mode.append(_choose_motion_mode(grid, a.pos))
        mode_phase.append(0.0)

    st.session_state.update({
        "running": False,
        "t": 0,
        "agents": agents,
        "paths": paths,
        "starts": starts,
        "grid": grid,
        "hazards": hazards,
        "exits": exits,
        "cost_map": cost_map,
        "exit_counts": {},
        "exit_counts_ts": {str(ex): [] for ex in exits},
        "evac_curve_ts": [],
        "visits": np.zeros_like(grid, dtype=np.int32),
        "frames": [],
        # store trajectories for all agents; we will only plot a subset in visuals
        "traj": [[] for _ in range(len(starts))],
        "state_init": True,
        "_fps_clock": time.time(),
        "_frames": 0,
        "report": None,
        "_export_done_at_t": None,
        "exit_counts_by_type": {},
        # A* cache
        "path_cache": path_cache,
        # motion pattern state
        "motion_mode": motion_mode,
        "mode_phase": mode_phase,
    })

# Apply UI changes to state where relevant
st.session_state["panic"] = panic
st.session_state["max_time"] = max_time
st.session_state["hazard_weight"] = hazard_weight
st.session_state["auto_export"] = auto_export
st.session_state["type_distribution"] = dist
st.session_state["enable_pre_delay"] = enable_pre_delay
st.session_state["pre_delay_mean_s"] = pre_delay_mean_s
st.session_state["pre_delay_std_s"] = pre_delay_std_s
st.session_state["exit_capacity_scale"] = cap_scale
st.session_state["vis_realtime"] = vis_realtime
st.session_state["smooth_mode"] = smooth_mode
st.session_state["fps"] = fps
st.session_state["vis_interval"] = vis_interval
st.session_state["max_runtime_s"] = max_runtime_s
st.session_state["fast_init"] = fast_init
st.session_state["capture_gif_frames"] = capture_gif_frames
st.session_state["capture_mp4_frames"] = capture_mp4_frames
st.session_state["enable_local_avoidance"] = enable_local_avoidance

# (Human motion parameter UI removed; no persistence needed here)

# If pre-evac delays are disabled now, clear any existing per-agent delays
if not st.session_state.get("enable_pre_delay", False) and "agents" in st.session_state:
    for _a in st.session_state["agents"]:
        _a.pre_delay_ticks = 0

# Compute a lightweight signature of the current floorplan to detect changes
def _fp_signature(fp: Dict, hazard_weight: float) -> Tuple:
    g = fp.get("grid")
    h = fp.get("hazards")
    ex = tuple(fp.get("exits", []))
    if g is None or h is None:
        return (None, None, None, None)
    return (g.shape, int(g.sum()), h.shape, float(hazard_weight), ex)

new_sig = _fp_signature(fp, hazard_weight)
prev_sig = st.session_state.get("fp_sig")
floorplan_changed = (prev_sig is not None) and (prev_sig != new_sig)
st.session_state["fp_sig"] = new_sig

if start_clicked and len(exits) > 0:
    st.session_state["running"] = True
    st.session_state["_run_start_time"] = time.time()
if stop_clicked:
    st.session_state["running"] = False
if start_clicked and len(exits) == 0:
    st.warning("Cannot start: no exits detected in the current floor plan.")

# Main layout
left, right = st.columns([2, 1])

with left:
    st.subheader("Crowd Simulation View")
    plot_placeholder = st.empty()
    if st.session_state.get("vis_realtime", True):
        fig = render_frame(st.session_state["grid"], st.session_state["exits"], st.session_state["hazards"], st.session_state["agents"], None)
        plot_placeholder.pyplot(fig, use_container_width=True)
    else:
        st.info("Fast compute mode: visualization disabled")

    # Progress bar
    prog_bar = st.progress(0)

    # While loop to run simulation; limit steps per rerun to keep UI responsive
    steps_this_rerun = 0
    max_steps_per_rerun = 50
    while st.session_state.get("running", False):
        if st.session_state["t"] >= st.session_state["max_time"]:
            # Mark remaining as trapped
            for a in st.session_state["agents"]:
                if a.exit_time is None and not a.exited:
                    a.trapped = True
            st.session_state["running"] = False
            break
        # Wall-clock cap
        if time.time() - st.session_state.get("_run_start_time", time.time()) >= st.session_state.get("max_runtime_s", 90):
            for a in st.session_state["agents"]:
                if a.exit_time is None and not a.exited:
                    a.trapped = True
            st.session_state["running"] = False
            st.toast("Stopped: reached max runtime", icon="⏱️")
            break
        # stop if all evacuated
        if all(a.exited for a in st.session_state["agents"]):
            st.session_state["running"] = False
            break

        # Positions before tick (for interpolation)
        prev_positions: List[Optional[Tuple[float, float]]] = []
        for a in st.session_state["agents"]:
            if a.exited:
                prev_positions.append(None)
            else:
                prev_positions.append((float(a.pos[0]) + 0.5, float(a.pos[1]) + 0.5))

        # Advance one tick
        step_simulation(st.session_state, panic=st.session_state["panic"])

        # sample trajectories for first K agents (store tick positions)
        for i in range(min(len(st.session_state["agents"]), len(st.session_state["traj"]))):
            st.session_state["traj"][i].append(st.session_state["agents"][i].pos)

        # Real-time visualization
        if st.session_state.get("vis_realtime", True):
            if st.session_state.get("smooth_mode", True):
                # Interpolate subframes between prev and current positions
                cur_positions: List[Optional[Tuple[float, float]]] = []
                for a in st.session_state["agents"]:
                    if a.exited and a.exit_time is not None and a.exit_time <= st.session_state["t"]:
                        cur_positions.append(None)
                    else:
                        cur_positions.append((float(a.pos[0]) + 0.5, float(a.pos[1]) + 0.5))
                subframes = max(1, int(round(st.session_state.get("fps", 30) * TICK_SECONDS)))
                dt = TICK_SECONDS / max(1, subframes)
                for s in range(1, subframes + 1):
                    alpha = s / float(subframes)
                    interp_pos: List[Optional[Tuple[float, float]]] = []
                    for p0, p1 in zip(prev_positions, cur_positions):
                        if p0 is None and p1 is None:
                            interp_pos.append(None)
                        elif p0 is None:
                            # appear at current
                            interp_pos.append(p1)
                        elif p1 is None:
                            # fade out at last pos
                            interp_pos.append(p0)
                        else:
                            r = (1 - alpha) * p0[0] + alpha * p1[0]
                            c = (1 - alpha) * p0[1] + alpha * p1[1]
                            interp_pos.append((r, c))
                    fig = render_frame(st.session_state["grid"], st.session_state["exits"], st.session_state["hazards"], st.session_state["agents"], interp_pos)
                    try:
                        plot_placeholder.pyplot(fig, use_container_width=True)
                    except Exception:
                        pass
                    # capture frames for GIF/MP4 if requested
                    if st.session_state.get("capture_gif_frames", False) or st.session_state.get("capture_mp4_frames", False):
                        fig.canvas.draw()
                        buf = np.asarray(fig.canvas.buffer_rgba())
                        frame = buf[:, :, :3].copy()
                        st.session_state["frames"].append(frame)
                        st.session_state["_video_fps"] = st.session_state.get("fps", 30)
                    plt.close(fig)
                    # lightweight sleep to aim for FPS without freezing UI
                    time.sleep(min(0.02, dt))
            else:
                # Tick-only rendering at configured interval
                if (st.session_state["t"] % max(1, st.session_state.get("vis_interval", 2)) == 0):
                    fig = render_frame(st.session_state["grid"], st.session_state["exits"], st.session_state["hazards"], st.session_state["agents"], None)
                    try:
                        plot_placeholder.pyplot(fig, use_container_width=True)
                    except Exception:
                        pass
                    if st.session_state.get("capture_gif_frames", False):
                        fig.canvas.draw()
                        buf = np.asarray(fig.canvas.buffer_rgba())
                        frame = buf[:, :, :3].copy()
                        st.session_state["frames"].append(frame)
                    plt.close(fig)

        # Update progress bar
        total = len(st.session_state["agents"]) if st.session_state.get("agents") else 0
        exited = sum(1 for a in st.session_state["agents"] if a.exit_time is not None)
        prog = int((exited / total) * 100) if total > 0 else 0
        try:
            prog_bar.progress(min(max(prog, 0), 100))
        except Exception:
            pass

        steps_this_rerun += 1
        # FPS tracking (approx)
        if show_fps:
            st.session_state["_frames"] += 1
            now = time.time()
            if now - st.session_state["_fps_clock"] >= 1.0:
                st.session_state["_fps"] = st.session_state["_frames"] / (now - st.session_state["_fps_clock"])
                st.session_state["_fps_clock"] = now
                st.session_state["_frames"] = 0
        if steps_this_rerun >= max_steps_per_rerun:
            st.rerun()
        # Reduce sleeps in fast mode
        time.sleep(0.005 if st.session_state.get("vis_realtime", True) else 0.0)

    # If simulation is not running and we have progressed, compute report once
    if (not st.session_state.get("running", False)) and st.session_state.get("t", 0) > 0:
        if st.session_state.get("report") is None:
            st.session_state["report"] = compute_insurance_report(st.session_state, avg_speed, panic)
        # Auto-export when run ends (once per t)
        if st.session_state.get("auto_export"):
            cur_t = st.session_state.get("t")
            if st.session_state.get("_export_done_at_t") != cur_t:
                out_dir = os.path.join(os.getcwd(), "evac_outputs")
                paths = export_outputs(out_dir, st.session_state)
                st.session_state["_export_done_at_t"] = cur_t
                st.toast("Auto-exported outputs to evac_outputs/", icon="✅")

with right:
    if True:
        st.subheader("Simulation Metrics")
        total = len(st.session_state["agents"]) if st.session_state.get("agents") else 0
        exited = sum(1 for a in st.session_state["agents"] if a.exit_time is not None)
        trapped_now = sum(1 for a in st.session_state["agents"] if a.trapped)
        t = st.session_state["t"]

        # Basic metrics
        st.metric("Agents", f"{total - exited}/{total}")
        st.metric("Time", f"{t} ticks ({int((t*TICK_SECONDS)//60)}:{int((t*TICK_SECONDS)%60):02d})")
        st.metric("Average Speed", f"{avg_speed:.2f} m/s")
        st.metric("Panic Level", f"{int(panic*100)}%")
        st.metric("Evacuated", f"{exited/total*100 if total>0 else 0:.1f}%")
        st.metric("Trapped", f"{trapped_now}")
        st.metric("Stampede Risk", f"{_live_stampede_risk(st.session_state):.1f}%")
        if show_fps and st.session_state.get("_fps"):
            st.caption(f"Approx FPS: {st.session_state['_fps']:.1f}")

        # Exit usage graph (show all exits, including unused, as previous behavior)
        all_exits = list(st.session_state.get("exits", []))
        if all_exits:
            counts_dict = st.session_state.get("exit_counts", {})
            ex_vals = [counts_dict.get(ex, 0) for ex in all_exits]
            total_exited = sum(ex_vals)
            total_norm = total_exited if total_exited > 0 else 1
            perc = [v / total_norm * 100 for v in ex_vals]
            fig_ex, ax_ex = plt.subplots(figsize=(4, 2))
            ax_ex.bar([str(ex) for ex in all_exits], perc, color='tab:green')
            ax_ex.set_ylabel('% of evacuees'); ax_ex.set_ylim(0, 100)
            ax_ex.set_title('Exit Usage')
            st.pyplot(fig_ex)
        else:
            st.write("No exits detected in floorplan.")

        # Grid stats
        grid = st.session_state["grid"]
        hazards = st.session_state["hazards"]
        exits_list = st.session_state["exits"]
        walls = int(np.sum(grid == 1))
        free = int(np.sum(grid == 0))
        hz = int(np.sum(hazards == 1))
        st.caption(f"Grid stats — walls: {walls}, free: {free}, hazards: {hz}, exits: {len(exits_list)}")

# Optional A* debug diagnostics
st.markdown("---")
with st.expander("A* Pathfinding Debug", expanded=False):
    debug_on = st.checkbox("Enable A* debug (recompute ideal paths for sample agents)", value=False)
    sample_n = int(st.number_input("Sample agents", min_value=1, max_value=200, value=10, step=1)) if debug_on else 0
    if debug_on:
        agents = st.session_state.get("agents", [])
        exits = st.session_state.get("exits", [])
        cost_map = st.session_state.get("cost_map")
        if not agents or not exits or cost_map is None:
            st.info("A* debug requires agents, exits, and a cost map.")
        else:
            # Recompute best A* path per agent across exits
            rows = []
            for idx, a in enumerate(agents[:sample_n]):
                start = a.pos
                best_p = []
                best_c = float('inf')
                best_ex = None
                for ex in exits:
                    p = astar_cached(cost_map, start, ex, cache=None)
                    if p:
                        c = compute_path_cost(cost_map, start, p)
                        if c < best_c:
                            best_c = c; best_p = p; best_ex = ex
                rows.append({
                    "agent_index": idx,
                    "start": str(start),
                    "nearest_exit": str(best_ex) if best_ex is not None else None,
                    "path_len": len(best_p) if best_p else 0,
                    "cost": float(best_c) if best_p else float('inf'),
                    "reachable": bool(bool(best_p))
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            # Optionally show first debug path on the grid
            if rows:
                show_overlay = st.checkbox("Overlay first debug path on grid", value=False)
                if show_overlay and rows[0]["reachable"]:
                    # Render overlay path using current grid
                    fig, ax = plt.subplots(figsize=(6, 6))
                    base = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
                    base[grid == 1] = [0, 0, 0]
                    base[grid == 0] = [255, 255, 255]
                    hz = st.session_state.get("hazards")
                    if hz is not None:
                        base[hz == 1] = [255, 200, 0]
                    for (r, c) in exits:
                        base[r, c] = [0, 200, 0]
                    ax.imshow(base, origin='upper')
                    # Recompute path for first agent and plot
                    first = agents[0]
                    # choose the same best exit as in table
                    best_p = []
                    best_c = float('inf')
                    best_ex = None
                    for ex in exits:
                        p = astar_cached(cost_map, first.pos, ex, cache=None)
                        if p:
                            c = compute_path_cost(cost_map, first.pos, p)
                            if c < best_c:
                                best_c = c; best_p = p; best_ex = ex
                    if best_p:
                        arr = np.array([(first.pos[0], first.pos[1])] + best_p)
                        ax.plot(arr[:,1], arr[:,0], 'b--', linewidth=2, label='A* path')
                        ax.scatter([first.pos[1]],[first.pos[0]], c='blue', s=40, label='Start')
                        ax.scatter([best_ex[1]],[best_ex[0]], c='green', s=40, label='Exit')
                        ax.legend()
                    ax.set_xticks([]); ax.set_yticks([])
                    st.pyplot(fig)

# Tabs for visuals
st.markdown("---")

t1, t2, t3, t4 = st.tabs(["Evacuation Curve", "Heatmap", "Trajectories", "Agent Details"])

with t1:
    exited = [a.exit_time for a in st.session_state["agents"] if a.exit_time is not None]
    if exited:
        max_t = max(exited + [0])
        curve = []
        for tt in range(max_t + 1):
            curve.append(sum(1 for et in exited if et <= tt))
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(range(len(curve)), curve)
        ax.set_xlabel('Time (ticks)'); ax.set_ylabel('Cumulative evacuees'); ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("No evacuations yet.")

# Insurance Analytics – Risk Assessment panel
st.markdown("---")
st.subheader("Insurance Analytics – Risk Assessment")
report = st.session_state.get("report")
if report is None:
    st.info("Run a simulation and stop it to generate the analytics report.")
else:
    c1, c2 = st.columns([2, 1])
    with c1:
        df = pd.DataFrame([{k: v for k, v in report.items() if k not in ("exit_usage", "recommendations")}])
        st.dataframe(df, use_container_width=True)
        if report.get("exit_usage"):
            keys = list(report["exit_usage"].keys())
            vals = list(report["exit_usage"].values())
            fig_ex, ax_ex = plt.subplots(figsize=(5, 3))
            ax_ex.bar(keys, vals, color='tab:green')
            ax_ex.set_ylabel('% of evacuees'); ax_ex.set_ylim(0, 100)
            ax_ex.set_title('Exit Usage (Report)')
            st.pyplot(fig_ex)
        # Per-exit travel times
        if report.get("per_exit_times"):
            st.markdown("**Per-Exit Travel Times (seconds)**")
            pet = pd.DataFrame.from_dict(report["per_exit_times"], orient="index").reset_index().rename(columns={"index": "exit_cell"})
            st.dataframe(pet, use_container_width=True)
    with c2:
        st.markdown("**Categories**")
        st.write(f"Evacuation Time: {report['evacuation_time_cat']}")
        st.write(f"Stampede Risk: {report['stampede_risk_cat']}")
        st.write(f"Exit Congestion: {report['exit_congestion_cat']}")
        st.write(f"Agent Behavior: {report['agent_behavior']}")
        st.metric("Risk Score", report['risk_score'])
    st.markdown("**Recommendations**")
    st.write(report["recommendations"])
    # Safety checklist
    if report.get("safety_checklist"):
        st.markdown("**Safety Checklist**")
        ck = report["safety_checklist"]
        cols_ck = st.columns(4)
        cols_ck[0].metric("Total evac < 5min", "PASS" if ck.get("total_evac_under_5min") else "FAIL")
        cols_ck[1].metric("Per-exit p90 < 120s", "PASS" if ck.get("per_exit_p90_under_120s") else "FAIL")
        cols_ck[2].metric("Stampede < 20%", "PASS" if ck.get("stampede_risk_under_20pct") else "FAIL")
        cols_ck[3].metric("Overall", "PASS" if ck.get("overall_pass") else "FAIL")
    # Group-level stats table
    if report.get("group_stats"):
        st.markdown("**Group-Level Statistics**")
        gs = report["group_stats"]
        # Flatten exit usage dicts for display
        rows = []
        for gid, data in gs.items():
            base = {"group_id": gid, "count": data.get("count", 0), "evac_mean_s": data.get("evac_mean_s", 0.0), "evac_median_s": data.get("evac_median_s", 0.0), "congestion_mean": data.get("congestion_mean", 0.0)}
            usage = data.get("exit_usage", {})
            # include as JSON string for compactness
            base["exit_usage_pct"] = json.dumps(usage)
            rows.append(base)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# Time-series and Throughput Downloads
st.markdown("---")
st.subheader("Throughput and Time-Series")
cols = st.columns([2,1])
with cols[0]:
    # Plot per-exit cumulative evacuees
    if st.session_state.get("exit_counts_ts"):
        fig, ax = plt.subplots(figsize=(8, 3))
        for k, series in st.session_state["exit_counts_ts"].items():
            ax.plot(range(len(series)), series, label=str(k))
        ax.set_title('Per-Exit Cumulative Evacuees')
        ax.set_xlabel('Tick'); ax.set_ylabel('Cumulative by exit'); ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    # Total curve
    if st.session_state.get("evac_curve_ts"):
        fig, ax = plt.subplots(figsize=(8, 3))
        ev = st.session_state["evac_curve_ts"]
        ax.plot(range(len(ev)), ev, 'b-')
        ax.set_title('Cumulative Evacuation Curve')
        ax.set_xlabel('Tick'); ax.set_ylabel('Cumulative evacuees'); ax.grid(True, alpha=0.3)
        st.pyplot(fig)
with cols[1]:
    # Download buttons for CSV/JSON
    try:
        # prepare CSV
        max_len = len(st.session_state.get("evac_curve_ts", []))
        cols_df = {"tick": list(range(max_len)), "total_cum": st.session_state.get("evac_curve_ts", [])}
        for k, series in st.session_state.get("exit_counts_ts", {}).items():
            s = list(series) + [series[-1] if series else 0] * (max_len - len(series))
            cols_df[f"exit_{k}"] = s
        csv_bytes = pd.DataFrame(cols_df).to_csv(index=False).encode('utf-8')
        json_bytes = json.dumps({"evac_curve_ts": st.session_state.get("evac_curve_ts", []), "exit_counts_ts": st.session_state.get("exit_counts_ts", {})}, ensure_ascii=False, indent=2).encode('utf-8')
        st.download_button("Download Time-Series CSV", data=csv_bytes, file_name="time_series.csv", mime="text/csv")
        st.download_button("Download Time-Series JSON", data=json_bytes, file_name="time_series.json", mime="application/json")
    except Exception:
        pass

with t2:
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(st.session_state["visits"], cmap='YlOrRd', origin='upper')
    plt.colorbar(im, ax=ax)
    ax.set_xticks([]); ax.set_yticks([])
    st.pyplot(fig)

with t3:
    fig, ax = plt.subplots(figsize=(6, 6))
    grid = st.session_state["grid"]; exits = st.session_state["exits"]; hazards = st.session_state["hazards"]
    base = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
    base[grid == 1] = [0, 0, 0]
    base[grid == 0] = [255, 255, 255]
    base[hazards == 1] = [255, 200, 0]
    for (r, c) in exits:
        base[r, c] = [0, 200, 0]
    ax.imshow(base, origin='upper')
    for traj in st.session_state.get("traj", [])[:30]:
        if len(traj) >= 2:
            arr = np.array(traj)
            ax.plot(arr[:, 1], arr[:, 0], '-', alpha=0.6)
    ax.set_xticks([]); ax.set_yticks([])
    st.pyplot(fig)

with t4:
    total_agents = len(st.session_state["agents"]) if st.session_state.get("agents") else 0
    idx = st.selectbox("Select Agent", list(range(total_agents))) if total_agents > 0 else None
    if idx is not None:
        a = st.session_state["agents"][idx]
        st.write(f"Exited: {a.exited}")
        st.write(f"Exit time: {a.exit_time}")
        st.write(f"Path length: {a.path_len}")
        st.write(f"Congestion: {a.congestion}")
        # small path preview
        if st.session_state.get("traj") and idx < len(st.session_state["traj"]):
            traj = st.session_state["traj"][idx]
            if len(traj) >= 2:
                fig, ax = plt.subplots(figsize=(5, 5))
                arr = np.array(traj)
                ax.plot(arr[:, 1], arr[:, 0], '-')
                ax.set_xticks([]); ax.set_yticks([])
                st.pyplot(fig)

# Export outputs button
st.markdown("---")
exp_col1, exp_col2 = st.columns([1, 3])
with exp_col1:
    if st.button("Export Outputs"):
        out_dir = os.path.join(os.getcwd(), "evac_outputs")
        paths = export_outputs(out_dir, st.session_state)
        st.success("Exported.")
        st.json(paths)

# -----------------------------
# Monte Carlo Runner (in-file)
# -----------------------------
st.markdown("---")
st.subheader("Monte Carlo Runner")
mc_cols = st.columns(3)
with mc_cols[0]:
    n_runs = int(st.number_input("Runs", min_value=1, max_value=200, value=10, step=1))
with mc_cols[1]:
    seed0 = int(st.number_input("Base Seed", min_value=0, value=1234, step=1))
with mc_cols[2]:
    run_mc = st.button("Run Monte Carlo")

def _simulate_once(seed: int) -> Dict[str, object]:
    # Build a fresh state without UI rendering
    rng_local = np.random.default_rng(seed)
    img = Image.open(uploaded) if uploaded is not None else None
    fp_local = load_floorplan(img, grid_size=grid_size)
    grid_local = fp_local["grid"]
    hazards_local = fp_local["hazards"]
    exits_local = detect_exits(fp_local)
    cost_local = build_cost_map(grid_local, hazards_local, st.session_state.get("hazard_weight", 3.0))
    free_local = [(r, c) for r in range(grid_local.shape[0]) for c in range(grid_local.shape[1]) if np.isfinite(cost_local[r, c])]
    starts_local = spawn_agents(free_local, exits_local, N, seed=seed)
    # Assign types
    dist_local = st.session_state.get("type_distribution", {"Pedestrian": 1.0})
    types_local = list(dist_local.keys())
    w = np.array([dist_local[t] for t in types_local], dtype=float)
    if w.sum() <= 0: w = np.ones(len(types_local), dtype=float)
    w = w / w.sum()
    assigned = rng_local.choice(types_local, size=len(starts_local), p=w)
    agents_local: List[AgentState] = []
    g_id = 0
    counts_local = [0] * max(1, num_groups)
    for s, tname in zip(starts_local, assigned):
        while counts_local[g_id] >= max_group_size:
            g_id = (g_id + 1) % num_groups
        cfg = TYPE_CONFIG.get(tname, TYPE_CONFIG["Pedestrian"])
        sp_lo, sp_hi = cfg.get("speed_mps", (1.0, 1.5))
        sp_mps = float(rng_local.uniform(sp_lo, sp_hi))
        sp_cells = (sp_mps * TICK_SECONDS) / max(1e-6, CELL_METERS)
        a = AgentState(pos=s, group_id=g_id, agent_type=tname, speed_cells=sp_cells, panic_mult=float(cfg.get("panic_mult", 1.0)), group_radius=int(cfg.get("group_radius", 0)), is_wheelchair=bool(cfg.get("wheelchair", False)))
        if st.session_state.get("enable_pre_delay"):
            d_s = float(max(0.0, rng_local.normal(st.session_state.get("pre_delay_mean_s", 5.0), st.session_state.get("pre_delay_std_s", 2.0))))
            a.pre_delay_ticks = int(round(d_s / TICK_SECONDS))
        else:
            a.pre_delay_ticks = 0
        agents_local.append(a)
        counts_local[g_id] += 1
        g_id = (g_id + 1) % num_groups
    paths_local = compute_paths(cost_local, starts_local, exits_local) if exits_local else [[] for _ in starts_local]
    for a, p in zip(agents_local, paths_local): a.path = p
    state_local: Dict[str, object] = dict(
        running=False, t=0, agents=agents_local, paths=paths_local, starts=starts_local,
        grid=grid_local, hazards=hazards_local, exits=exits_local, cost_map=cost_local,
        exit_counts={}, exit_counts_ts={str(ex): [] for ex in exits_local}, evac_curve_ts=[],
        visits=np.zeros_like(grid_local, dtype=np.int32), frames=[], traj=[[] for _ in range(len(starts_local))],
        report=None, _export_done_at_t=None, exit_counts_by_type={},
        panic=st.session_state.get("panic", 0.1), exit_capacity_scale=st.session_state.get("exit_capacity_scale", 0.0)
    )
    # Run until done or max_time
    while state_local["t"] < st.session_state.get("max_time", 500):
        if not state_local["exits"] or all(a.exited for a in state_local["agents"]):
            break
        step_simulation(state_local, panic=st.session_state.get("panic", 0.1))
    # Summaries
    exit_times = [a.exit_time for a in state_local["agents"] if getattr(a, "exit_time", None) is not None]
    total_ticks = max(exit_times) if exit_times else state_local["t"]
    arr = np.array([et for et in exit_times]) if exit_times else np.array([])
    p50 = float(np.percentile(arr, 50)) if arr.size else None
    p90 = float(np.percentile(arr, 90)) if arr.size else None
    return dict(seed=seed, total_ticks=int(total_ticks), total_seconds=float(total_ticks*TICK_SECONDS), p50_ticks=p50, p90_ticks=p90, evacuated=len(exit_times), total=len(state_local["agents"]))

if run_mc:
    results = []
    for i in range(n_runs):
        results.append(_simulate_once(seed0 + i))
    df_mc = pd.DataFrame(results)
    st.dataframe(df_mc, use_container_width=True)
    # Aggregate percentiles across runs
    st.write("Summary across runs:")
    if not df_mc.empty:
        st.write({
            "total_seconds_p50": float(np.percentile(df_mc["total_seconds"], 50)),
            "total_seconds_p90": float(np.percentile(df_mc["total_seconds"], 90)),
            "evacuated_mean": float(df_mc["evacuated"].mean()),
        })
        st.download_button("Download MC CSV", data=df_mc.to_csv(index=False).encode("utf-8"), file_name="mc_results.csv", mime="text/csv")
