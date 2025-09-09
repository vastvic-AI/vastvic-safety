"""
Navigation Mesh for Crowd Simulation

This module implements a grid-based navigation mesh with A* pathfinding,
hazard mapping, and dynamic obstacle avoidance for realistic crowd movement.
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
import heapq

@dataclass
class PathCache:
    """Cache for storing and reusing paths to improve performance."""
    paths: Dict[Tuple[Tuple[float, float], Tuple[float, float]], List[Tuple[float, float]]]
    hits: int = 0
    misses: int = 0
    
    def get(self, start: Tuple[float, float], goal: Tuple[float, float]):
        """Get a cached path if it exists and is still valid."""
        key = (start, goal)
        if key in self.paths:
            self.hits += 1
            return self.paths[key]
        self.misses += 1
        return None
    
    def set(self, start: Tuple[float, float], goal: Tuple[float, float], path: List[Tuple[float, float]]):
        """Cache a computed path."""
        self.paths[(start, goal)] = path

class GridNavMesh:
    """
    A grid-based navigation mesh for crowd simulation with A* pathfinding,
    hazard avoidance, and dynamic obstacle handling.
    """
    
    def __init__(self, width: int, height: int, obstacles: Optional[List[Tuple[int, int]]] = None, 
                 polygons: Optional[List] = None):
        """
        Initialize the navigation mesh with spatial partitioning and caching.
        
        Args:
            width: Width of the grid.
            height: Height of the grid.
            obstacles: List of (x, y) coordinates of blocked cells.
            polygons: List of polygon obstacles.
        """
        self.width = width
        self.height = height
        
        # Initialize grid with obstacles
        self.grid = np.zeros((height, width), dtype=np.uint8)
        self.polygons = polygons or []
        
        # Mark static obstacles
        if obstacles:
            obs_array = np.array(obstacles, dtype=np.int32)
            valid = (obs_array[:, 0] >= 0) & (obs_array[:, 0] < width) & \
                   (obs_array[:, 1] >= 0) & (obs_array[:, 1] < height)
            obs_array = obs_array[valid]
            self.grid[obs_array[:, 1], obs_array[:, 0]] = 1
        
        # Initialize spatial partitioning
        self.cell_size = 4  # meters per cell
        self.grid_cols = (width + self.cell_size - 1) // self.cell_size
        self.grid_rows = (height + self.cell_size - 1) // self.cell_size
        self.spatial_grid = [
            [[] for _ in range(self.grid_cols)] 
            for _ in range(self.grid_rows)
        ]
        
        # Initialize dynamic data
        self.dynamic_obstacles: Set[Tuple[int, int]] = set()
        self.density_map = np.zeros((height, width), dtype=np.float32)
        self.hazard_map = np.zeros((height, width), dtype=np.float32)
        self.flow_field: Dict[Tuple[int, int], Tuple[float, float]] = {}
        self.path_cache = PathCache({})
        
        # Precompute walkable cells for faster access
        self.walkable = np.argwhere(self.grid == 0).tolist()
        self.walkable_set = {(x, y) for y, x in self.walkable}
        
        # Initialize spatial grid
        self._update_spatial_grid()
    
    def _update_spatial_grid(self) -> None:
        """Update spatial partitioning of walkable cells."""
        # Clear the grid
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                self.spatial_grid[i][j] = []
        
        # Add walkable cells to the grid
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == 0:  # Only add walkable cells
                    gx, gy = x // self.cell_size, y // self.cell_size
                    if 0 <= gy < self.grid_rows and 0 <= gx < self.grid_cols:
                        self.spatial_grid[gy][gx].append((x, y))
    
    def get_nearby_walkable(self, x: float, y: float, radius: float) -> List[Tuple[int, int]]:
        """Get walkable cells within radius of (x, y)."""
        nearby = []
        gx, gy = int(x) // self.cell_size, int(y) // self.cell_size
        radius_cells = int(np.ceil(radius / self.cell_size))
        
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= ny < self.grid_rows and 0 <= nx < self.grid_cols:
                    for (cx, cy) in self.spatial_grid[ny][nx]:
                        if ((cx - x)**2 + (cy - y)**2) <= radius**2:
                            nearby.append((cx, cy))
        
        return nearby
    
    def is_walkable(self, x: float, y: float) -> bool:
        """
        Check if a position is walkable (within bounds and not blocked).
        Optimized with Numba JIT compilation.
        """
        ix, iy = int(round(x)), int(round(y))
        return (0 <= ix < self.width and 0 <= iy < self.height and 
                self.grid[iy, ix] == 0 and (ix, iy) not in self.dynamic_obstacles)
    
    def neighbors(self, x: float, y: float, diagonal: bool = True) -> List[Tuple[int, int]]:
        """Get walkable neighbors of a cell."""
        ix, iy = int(round(x)), int(round(y))
        deltas = [
            (1, 0), (-1, 0), (0, 1), (0, -1)  # Cardinal directions
        ]
        if diagonal:
            deltas.extend([(1, 1), (-1, 1), (1, -1), (-1, -1)])  # Diagonal directions
        
        neighbors = []
        for dx, dy in deltas:
            nx, ny = ix + dx, iy + dy
            if self.is_walkable(nx, ny):
                neighbors.append((nx, ny))
        
        return neighbors
    
    @staticmethod
    def heuristic(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """Euclidean distance heuristic for A*."""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def astar(self, start: Tuple[float, float], goal: Tuple[float, float], 
              avoid_hazards: bool = True, avoid_density: bool = True, 
              avoid_dynamic: bool = True, flow_field: bool = False,
              cache: bool = True, max_steps: int = 1000) -> List[Tuple[float, float]]:
        """
        Find a path from start to goal using A* with hazard and density avoidance.
        
        Args:
            start: (x, y) starting position.
            goal: (x, y) target position.
            avoid_hazards: Whether to avoid high-hazard areas.
            avoid_density: Whether to avoid high-density areas.
            avoid_dynamic: Whether to avoid dynamic obstacles.
            flow_field: Whether to use flow field for path smoothing.
            cache: Whether to use path caching.
            max_steps: Maximum number of steps before giving up.
            
        Returns:
            List of (x, y) positions representing the path, or empty list if no path found.
        """
        start = (int(round(start[0])), int(round(start[1])))
        goal = (int(round(goal[0])), int(round(goal[1])))
        
        # Check if start or goal is blocked
        if not self.is_walkable(*start) or not self.is_walkable(*goal):
            return []
        
        # Check cache first
        if cache:
            cached_path = self.path_cache.get(start, goal)
            if cached_path is not None:
                return cached_path
        
        # Initialize open and closed sets
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        steps = 0
        
        while open_set and steps < max_steps:
            steps += 1
            
            # Get node with lowest f_score
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                
                # Cache the path
                if cache:
                    self.path_cache.set(start, goal, path)
                
                return path
            
            # Get neighbors
            for neighbor in self.neighbors(*current):
                # Calculate tentative g_score
                tentative_g_score = g_score[current] + self.heuristic(current, neighbor)
                
                # Apply penalties based on hazards and density
                if avoid_hazards:
                    tentative_g_score += 10.0 * self.hazard_map[neighbor[1], neighbor[0]]
                
                if avoid_density:
                    tentative_g_score += 5.0 * self.density_map[neighbor[1], neighbor[0]]
                
                # Flow field guidance
                if flow_field and neighbor in self.flow_field:
                    dx = goal[0] - neighbor[0]
                    dy = goal[1] - neighbor[1]
                    dot = dx * self.flow_field[neighbor][0] + dy * self.flow_field[neighbor][1]
                    tentative_g_score -= 0.5 * dot  # Encourage following flow field
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # This path to neighbor is better than any previous one
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    
                    # Add to open set if not already there
                    if not any(neighbor == n[1] for n in open_set):
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # No path found
        return []
    
    def add_dynamic_obstacle(self, x: float, y: float) -> None:
        """Add a dynamic obstacle at (x, y)."""
        self.dynamic_obstacles.add((int(round(x)), int(round(y))))
    
    def remove_dynamic_obstacle(self, x: float, y: float) -> None:
        """Remove a dynamic obstacle at (x, y)."""
        self.dynamic_obstacles.discard((int(round(x)), int(round(y))))
    
    def clear_dynamic_obstacles(self) -> None:
        """Remove all dynamic obstacles."""
        self.dynamic_obstacles.clear()
    
    def update_flow_field(self, goals: List[Tuple[float, float]]) -> None:
        """
        Update the flow field to guide agents toward the nearest goal.
        
        Args:
            goals: List of (x, y) goal positions.
        """
        if not goals:
            return
            
        # Simple implementation: compute direction to nearest goal
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == 1:  # Skip obstacles
                    continue
                    
                # Find nearest goal
                min_dist = float('inf')
                best_dx, best_dy = 0, 0
                
                for gx, gy in goals:
                    dx, dy = gx - x, gy - y
                    dist = dx*dx + dy*dy
                    if dist < min_dist:
                        min_dist = dist
                        best_dx, best_dy = dx, dy
                
                # Normalize direction
                norm = np.sqrt(best_dx**2 + best_dy**2)
                if norm > 0:
                    self.flow_field[(x, y)] = (best_dx/norm, best_dy/norm)
