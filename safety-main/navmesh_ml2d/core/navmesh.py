"""
Navigation Mesh for Crowd Simulation

This module implements a grid-based navigation mesh with A* pathfinding,
hazard mapping, and dynamic obstacle avoidance for realistic crowd movement.
"""

import numpy as np
import heapq
import numba
from typing import List, Tuple, Dict, Set, Optional, Any, DefaultDict
from dataclasses import dataclass, field
from collections import defaultdict
import time
import math

@dataclass
class PathCache:
    """Cache for storing and reusing paths to improve performance."""
    paths: Dict[Tuple[Tuple[float, float], Tuple[float, float]], List[Tuple[float, float]]]
    hits: int = 0
    misses: int = 0
    
    def get(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """Get a cached path if it exists and is still valid."""
        key = (start, goal)
        if key in self.paths:
            self.hits += 1
            return self.paths[key]
        self.misses += 1
        return None
    
    def set(self, start: Tuple[float, float], goal: Tuple[float, float], path: List[Tuple[float, float]]) -> None:
        """Cache a computed path."""
        self.paths[(start, goal)] = path

class GridNavMesh:
    """
    A grid-based navigation mesh for crowd simulation with A* pathfinding,
    hazard avoidance, and dynamic obstacle handling.
    
    Attributes:
        width (int): Width of the grid.
        height (int): Height of the grid.
        grid (np.ndarray): 2D array representing walkable (0) and blocked (1) cells.
        polygons (List): Optional polygon obstacles for more complex environments.
        dynamic_obstacles (Set[Tuple[int, int]]): Set of temporarily blocked cells.
        density_map (np.ndarray): 2D array of crowd density values.
        hazard_map (np.ndarray): 2D array of hazard intensities (0-1).
        flow_field (Dict[Tuple[int, int], Tuple[float, float]]): Precomputed flow field for pathfinding.
        path_cache (PathCache): Cache for storing computed paths.
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
        self.spatial_grid: List[List[List[Tuple[int, int]]]] = [
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
        
    def _update_spatial_grid(self):
        """Update spatial partitioning of walkable cells."""
        # Clear grid
        for row in self.spatial_grid:
            for cell in row:
                cell.clear()
        
        # Populate grid with walkable cells
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == 0:
                    gx, gy = x // self.cell_size, y // self.cell_size
                    if gx < self.grid_cols and gy < self.grid_rows:
                        self.spatial_grid[gy][gx].append((x, y))
    
    def get_nearby_walkable(self, x: float, y: float, radius: float) -> List[Tuple[int, int]]:
        """Get walkable cells within radius of (x, y)."""
        x, y = int(round(x)), int(round(y))
        radius_cells = int(math.ceil(radius / self.cell_size))
        
        min_gx = max(0, (x - radius) // self.cell_size)
        max_gx = min(self.grid_cols - 1, (x + radius) // self.cell_size)
        min_gy = max(0, (y - radius) // self.cell_size)
        max_gy = min(self.grid_rows - 1, (y + radius) // self.cell_size)
        
        result = []
        radius_sq = radius * radius
        
        for gy in range(min_gy, max_gy + 1):
            for gx in range(min_gx, max_gx + 1):
                for (cx, cy) in self.spatial_grid[gy][gx]:
                    dx, dy = cx - x, cy - y
                    if dx*dx + dy*dy <= radius_sq:
                        result.append((cx, cy))
                        
        return result
    
    @numba.njit(cache=True)
    def is_walkable(self, x: float, y: float) -> bool:
        """
        Check if a position is walkable (within bounds and not blocked).
        Optimized with Numba JIT compilation.
        """
        ix, iy = int(round(x)), int(round(y))
        if not (0 <= ix < self.width and 0 <= iy < self.height):
            return False
            
        # Check static obstacles
        if self.grid[iy, ix] != 0:
            return False
            
        # Check dynamic obstacles (slower, so check last)
        return (ix, iy) not in self.dynamic_obstacles
    
    def neighbors(self, x: float, y: float, diagonal: bool = True) -> List[Tuple[float, float]]:
        """Get walkable neighbors of a cell."""
        ix, iy = int(round(x)), int(round(y))
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if diagonal:
            deltas += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        neighbors = []
        for dx, dy in deltas:
            nx, ny = ix + dx, iy + dy
            if 0 <= nx < self.width and 0 <= ny < self.height and self.grid[ny, nx] == 0:
                neighbors.append((nx, ny))
        return neighbors
    
    def heuristic(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
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
        # Check if start or goal is blocked
        if not self.is_walkable(*start) or not self.is_walkable(*goal):
            return []
        
        # Check cache first
        if cache:
            cached_path = self.path_cache.get(start, goal)
            if cached_path and all(self.is_walkable(x, y) for x, y in cached_path):
                return cached_path
        
        # Initialize A* search
        open_set = []
        heapq.heappush(open_set, (0, 0, start, [start], None))
        visited = set()
        steps = 0
        
        while open_set and steps < max_steps:
            _, cost, current, path, _ = heapq.heappop(open_set)
            
            if current == goal:
                if cache:
                    self.path_cache.set(start, goal, path)
                return path
                
            if current in visited:
                continue
                
            visited.add(current)
            steps += 1
            
            # Get neighbors with flow field guidance if enabled
            neighbors = self.get_neighbors_with_flow(current, goal) if flow_field else self.neighbors(*current)
            
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                    
                # Calculate movement cost
                move_cost = 1.0
                nx, ny = neighbor
                
                # Apply penalties
                if avoid_hazards:
                    move_cost += self.hazard_map[ny, nx] * 100
                if avoid_density:
                    move_cost += self.density_map[ny, nx] * 2
                if avoid_dynamic and (nx, ny) in self.dynamic_obstacles:
                    move_cost += 1000
                
                # Calculate total cost and heuristic
                new_cost = cost + move_cost
                priority = new_cost + self.heuristic(neighbor, goal)
                
                # Add to open set if not already there
                heapq.heappush(open_set, (priority, new_cost, neighbor, path + [neighbor], current))
        
        return []  # No path found
    
    def add_hazard(self, x: float, y: float, radius: float, intensity: float = 1.0) -> None:
        """
        Add a hazard to the hazard map.
        
        Args:
            x: X-coordinate of the hazard center.
            y: Y-coordinate of the hazard center.
            radius: Radius of the hazard area.
            intensity: Maximum intensity at the center (0-1).
        """
        ix, iy = int(round(x)), int(round(y))
        radius = int(round(radius))
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = ix + dx, iy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist <= radius:
                        falloff = 1 - (dist / radius)
                        self.hazard_map[ny, nx] = min(1.0, self.hazard_map[ny, nx] + intensity * falloff)
    
    def add_dynamic_obstacle(self, x: float, y: float) -> None:
        """Add a dynamic obstacle at the specified position."""
        self.dynamic_obstacles.add((int(round(x)), int(round(y))))
    
    def remove_dynamic_obstacle(self, x: float, y: float) -> None:
        """Remove a dynamic obstacle from the specified position."""
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
