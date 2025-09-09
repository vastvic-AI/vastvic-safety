"""Pathfinding utilities for the evacuation simulation."""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Set
import heapq
import math
from collections import defaultdict, deque

from .config import ENV_CONFIG, AGENT_CONFIG

class AStar:
    """A* pathfinding algorithm implementation."""
    
    def __init__(self, grid_size: float = 1.0):
        """Initialize the A* pathfinder.
        
        Args:
            grid_size: Size of each grid cell for pathfinding
        """
        self.grid_size = grid_size
        self.obstacle_grid = None
        self.width = 0
        self.height = 0
    
    def init_grid(self, width: int, height: int, obstacles: List[Tuple[float, float]]):
        """Initialize the grid with obstacles.
        
        Args:
            width: Width of the environment
            height: Height of the environment
            obstacles: List of (x, y) obstacle positions
        """
        self.width = int(np.ceil(width / self.grid_size))
        self.height = int(np.ceil(height / self.grid_size))
        
        # Initialize obstacle grid
        self.obstacle_grid = np.zeros((self.width, self.height), dtype=bool)
        
        # Mark obstacles
        for x, y in obstacles:
            grid_x = int(x / self.grid_size)
            grid_y = int(y / self.grid_size)
            
            if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                self.obstacle_grid[grid_x, grid_y] = True
    
    def is_valid_position(self, x: float, y: float) -> bool:
        """Check if a position is valid (not an obstacle)."""
        grid_x = int(x / self.grid_size)
        grid_y = int(y / self.grid_size)
        
        if not (0 <= grid_x < self.width and 0 <= grid_y < self.height):
            return False
            
        return not self.obstacle_grid[grid_x, grid_y]
    
    def find_path(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        """Find a path from start to goal using A* algorithm.
        
        Args:
            start: Starting position as [x, y]
            goal: Goal position as [x, y]
            
        Returns:
            List of waypoints from start to goal, or empty list if no path found
        """
        if self.obstacle_grid is None:
            return []
            
        # Convert to grid coordinates
        start_x, start_y = int(start[0] / self.grid_size), int(start[1] / self.grid_size)
        goal_x, goal_y = int(goal[0] / self.grid_size), int(goal[1] / self.grid_size)
        
        # Check if start or goal is invalid
        if not (0 <= start_x < self.width and 0 <= start_y < self.height and 
                not self.obstacle_grid[start_x, start_y]):
            return []
            
        if not (0 <= goal_x < self.width and 0 <= goal_y < self.height and 
                not self.obstacle_grid[goal_x, goal_y]):
            return []
        
        # Directions: 8-way movement
        directions = [
            (1, 0), (1, 1), (0, 1), (-1, 1),
            (-1, 0), (-1, -1), (0, -1), (1, -1)
        ]
        
        # Priority queue: (f_score, g_score, x, y)
        open_set = []
        heapq.heappush(open_set, (0, 0, start_x, start_y))
        
        # For node n, came_from[n] is the node immediately preceding it on the cheapest path
        came_from = {}
        
        # For node n, g_score[n] is the cost of the cheapest path from start to n
        g_score = defaultdict(lambda: float('inf'))
        g_score[(start_x, start_y)] = 0
        
        # For node n, f_score[n] = g_score[n] + h(n)
        f_score = defaultdict(lambda: float('inf'))
        f_score[(start_x, start_y)] = self._heuristic(start, goal)
        
        open_set_hash = {(start_x, start_y)}
        
        while open_set:
            current_f, current_g, current_x, current_y = heapq.heappop(open_set)
            
            # Check if we've reached the goal
            if (current_x, current_y) == (goal_x, goal_y):
                return self._reconstruct_path(came_from, (current_x, current_y))
            
            open_set_hash.remove((current_x, current_y))
            
            # Check all neighbors
            for dx, dy in directions:
                neighbor_x, neighbor_y = current_x + dx, current_y + dy
                
                # Skip if out of bounds or obstacle
                if not (0 <= neighbor_x < self.width and 0 <= neighbor_y < self.height):
                    continue
                    
                if self.obstacle_grid[neighbor_x, neighbor_y]:
                    continue
                
                # Calculate tentative g_score
                # Use 1.0 for straight moves, sqrt(2) for diagonal
                move_cost = 1.0 if dx == 0 or dy == 0 else math.sqrt(2)
                tentative_g = current_g + move_cost
                
                # If this path to neighbor is better
                if tentative_g < g_score[(neighbor_x, neighbor_y)]:
                    came_from[(neighbor_x, neighbor_y)] = (current_x, current_y)
                    g_score[(neighbor_x, neighbor_y)] = tentative_g
                    
                    neighbor_pos = (neighbor_x * self.grid_size + self.grid_size/2, 
                                  neighbor_y * self.grid_size + self.grid_size/2)
                    h = self._heuristic(np.array(neighbor_pos), goal)
                    f = tentative_g + h
                    
                    if (neighbor_x, neighbor_y) not in open_set_hash:
                        heapq.heappush(open_set, (f, tentative_g, neighbor_x, neighbor_y))
                        open_set_hash.add((neighbor_x, neighbor_y))
        
        # No path found
        return []
    
    def _heuristic(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate the heuristic (Euclidean distance)."""
        return np.linalg.norm(b - a)
    
    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[np.ndarray]:
        """Reconstruct the path from start to goal."""
        total_path = [np.array([(current[0] + 0.5) * self.grid_size, 
                              (current[1] + 0.5) * self.grid_size])]
        
        while current in came_from:
            current = came_from[current]
            total_path.append(np.array([(current[0] + 0.5) * self.grid_size, 
                                      (current[1] + 0.5) * self.grid_size]))
        
        # Reverse to get path from start to goal
        return total_path[::-1]


def find_path_around_obstacles(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: List[Tuple[float, float]],
    agent_radius: float = 0.5,
    grid_size: float = 1.0
) -> List[np.ndarray]:
    """Find a path from start to goal avoiding obstacles.
    
    Args:
        start: Starting position as [x, y]
        goal: Goal position as [x, y]
        obstacles: List of (x, y) obstacle positions
        agent_radius: Radius of the agent (for obstacle expansion)
        grid_size: Size of grid cells for pathfinding
        
    Returns:
        List of waypoints from start to goal, or empty list if no path found
    """
    # Expand obstacles by agent radius
    expanded_obstacles = []
    expansion_cells = int(np.ceil(agent_radius / grid_size))
    
    for ox, oy in obstacles:
        # Add the obstacle itself
        expanded_obstacles.append((ox, oy))
        
        # Add cells around the obstacle within agent_radius
        for dx in range(-expansion_cells, expansion_cells + 1):
            for dy in range(-expansion_cells, expansion_cells + 1):
                if dx == 0 and dy == 0:
                    continue
                    
                dist = np.linalg.norm([dx * grid_size, dy * grid_size])
                if dist <= agent_radius:
                    expanded_obstacles.append((ox + dx * grid_size, oy + dy * grid_size))
    
    # Use A* to find path
    astar = AStar(grid_size)
    astar.init_grid(
        width=ENV_CONFIG['width'],
        height=ENV_CONFIG['height'],
        obstacles=expanded_obstacles
    )
    
    return astar.find_path(start, goal)


def smooth_path(
    path: List[np.ndarray], 
    obstacles: List[Tuple[float, float]],
    iterations: int = 10
) -> List[np.ndarray]:
    """Smooth a path using simple interpolation.
    
    Args:
        path: Original path as list of points
        obstacles: List of obstacle positions
        iterations: Number of smoothing iterations
        
    Returns:
        Smoothed path
    """
    if len(path) <= 2:
        return path
    
    # Convert obstacles to a set for faster lookups
    obstacle_set = set((int(x), int(y)) for x, y in obstacles)
    
    for _ in range(iterations):
        new_path = [path[0]]  # Keep the start point
        
        for i in range(1, len(path) - 1):
            prev = path[i-1]
            curr = path[i]
            next_p = path[i+1]
            
            # Calculate new point as average of neighbors
            new_x = (prev[0] + next_p[0]) / 2
            new_y = (prev[1] + next_p[1]) / 2
            new_point = np.array([new_x, new_y])
            
            # Check if new point is valid (not in obstacle)
            grid_x, grid_y = int(new_x), int(new_y)
            if (grid_x, grid_y) not in obstacle_set:
                new_path.append(new_point)
            else:
                new_path.append(curr)
        
        new_path.append(path[-1])  # Keep the end point
        path = new_path
    
    return path


def get_visibility_graph(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: List[Tuple[float, float]],
    agent_radius: float = 0.5
) -> Dict[Tuple[float, float], List[Tuple[float, float]]]:
    """Create a visibility graph for pathfinding.
    
    Args:
        start: Starting position
        goal: Goal position
        obstacles: List of obstacle positions
        agent_radius: Radius of the agent
        
    Returns:
        Dictionary mapping points to their visible neighbors
    """
    # Add start and goal to the graph
    points = [tuple(start), tuple(goal)]
    
    # Add obstacle corners
    for x, y in obstacles:
        # Add points around the obstacle for better pathfinding
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                point = (x + dx * agent_radius, y + dy * agent_radius)
                points.append(point)
    
    # Remove duplicates
    points = list(dict.fromkeys(points))
    
    # Build visibility graph
    graph = {point: [] for point in points}
    
    # Check visibility between all pairs of points
    for i, p1 in enumerate(points):
        for p2 in points[i+1:]:
            if is_visible(p1, p2, obstacles, agent_radius):
                graph[p1].append(p2)
                graph[p2].append(p1)
    
    return graph


def is_visible(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    obstacles: List[Tuple[float, float]],
    agent_radius: float = 0.5
) -> bool:
    """Check if there's a direct line of sight between two points."""
    x1, y1 = p1
    x2, y2 = p2
    
    # Check line of sight to each obstacle
    for ox, oy in obstacles:
        # Skip if obstacle is at one of the points
        if (ox == x1 and oy == y1) or (ox == x2 and oy == y2):
            return False
        
        # Check distance from line to obstacle
        dist = point_to_line_distance((ox, oy), p1, p2)
        if dist < agent_radius:
            return False
    
    return True


def point_to_line_distance(
    point: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float]
) -> float:
    """Calculate the shortest distance from a point to a line segment."""
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Line length squared
    l2 = (x2 - x1)**2 + (y2 - y1)**2
    if l2 == 0:
        return math.hypot(px - x1, py - y1)  # line is a point
    
    # Consider the line extending the segment, parameterized as line_start + t (line_end - line_start)
    # We find projection of point onto the line
    # It falls where t = [(point-line_start) . (line_end-line_start)] / |line_end-line_start|^2
    t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2
    
    # Limit t to the segment
    t = max(0, min(1, t))
    
    # Projection falls on the segment
    projection_x = x1 + t * (x2 - x1)
    projection_y = y1 + t * (y2 - y1)
    
    return math.hypot(px - projection_x, py - projection_y)
