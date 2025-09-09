"""
Advanced Agent Model with Social Force, Panic, and Group Dynamics

This module implements a realistic agent model for crowd simulation, including:
- Social force model for collision avoidance and group cohesion
- Panic propagation and stress response
- Group behavior and leader-following dynamics
- Adaptive movement based on density and hazards
"""

import numpy as np
import random
import time
import heapq
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numba
from collections import deque, namedtuple
from typing import Dict, List, Optional, Tuple, Any, Deque, Set, Callable

# Experience replay buffer for RL
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.memory: Deque[Transition] = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, min(len(self.memory), batch_size))
    
    def __len__(self) -> int:
        return len(self.memory)

class DQN(nn.Module):
    """Deep Q-Network for agent decision making."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class Agent:
    """An agent in the crowd simulation with realistic movement and behavior."""
    
    def __init__(self, agent_id: int, start: Tuple[float, float], goal: Tuple[float, float], 
                 profile: Optional[Dict[str, Any]] = None):
        """Initialize an agent.
        
        Args:
            agent_id: Unique identifier.
            start: Starting position (x, y).
            goal: Target position (x, y).
            profile: Agent characteristics (speed, panic, group_id, etc.).
        """
        self.agent_id = agent_id
        self.start = np.array(start, dtype=np.float32)
        self.goal = np.array(goal, dtype=np.float32)
        
        # Default profile with realistic parameters
        self.profile = {
            'speed': 1.4,           # Base speed (m/s)
            'panic': 0.0,           # Panic level (0-1)
            'group_id': None,       # Group identifier
            'size': 0.4,            # Physical radius (m)
            'priority': 1.0,        # Priority for path planning
            'panic_threshold': 0.3, # Threshold for panic behavior
            'panic_decay': 0.01,    # Rate of panic decay per second
            'mass': 70.0,           # Mass (kg) for force calculations
            'relaxation_time': 0.5, # Time to reach desired velocity (s)
            'mcts_simulations': 50, # Number of MCTS simulations
            'learning_rate': 1e-4,  # Learning rate for DQN
            'gamma': 0.99,          # Discount factor
            'epsilon_start': 1.0,   # Exploration rate
            'epsilon_end': 0.01,
            'epsilon_decay': 0.999,
            'batch_size': 64,       # Batch size for training
            'memory_capacity': 10000, # Replay buffer size
            **({} if profile is None else profile)
        }
        
        # Initialize state
        self.state = {
            'pos': np.array(start, dtype=np.float32),  # Current position (x, y)
            'vel': np.zeros(2, dtype=np.float32),      # Current velocity (vx, vy)
            'desired_vel': np.zeros(2, dtype=np.float32),  # Desired velocity
            'panic': float(self.profile['panic']),     # Current panic level (0-1)
            'group_id': self.profile['group_id'],      # Group identifier
            'status': 'normal',                       # Current status (normal, panicked, etc.)
            'path': [],                               # Planned path (if any)
            'waited': 0,                              # Time spent waiting
            'collisions': 0,                          # Number of collisions
            'exited': False,                          # Whether agent has exited
            'exit_time': None,                        # Time of exit (if exited)
            'last_update': 0.0,                       # Last update time (s)
            'path_history': []                        # History of paths taken
        }
        
        # Pathfinding attributes
        self.current_path = []                       # Current A* path to follow
        self.steps_since_replan = 0                  # Steps since last path replan
        self.path_history = []                       # History of paths for visualization
        self.path_update_interval = 20               # How often to update path (steps)
        self.lookahead_distance = 5                  # How far ahead to look in the path
        self.waypoint_threshold = 1.0                # Distance threshold to reach a waypoint

    def _get_neighbors(self, pos, navmesh):
        """Get walkable neighbors of a position."""
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), 
                      (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            neighbor = (x + dx, y + dy)
            
            # Check bounds and walkability
            if (0 <= neighbor[0] < navmesh.width and 
                0 <= neighbor[1] < navmesh.height and 
                navmesh.is_walkable(*neighbor)):
                neighbors.append(neighbor)
                
        return neighbors

    def _calculate_g_score(self, current, neighbor, navmesh):
        """Calculate the g-score between current and neighbor nodes."""
        # Simple Euclidean distance for now
        dx = current[0] - neighbor[0]
        dy = current[1] - neighbor[1]
        return math.sqrt(dx*dx + dy*dy)
    
    def _reconstruct_path(self, came_from, current):
        """Reconstruct the path from start to goal using the came_from dictionary."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def a_star_search(self, navmesh, start: Tuple[int, int], goal: Tuple[int, int], 
                     max_steps: int = 10000) -> List[Tuple[int, int]]:
        """
        Optimized A* pathfinding using Numba JIT compilation.
        
        Args:
            navmesh: Navigation mesh with obstacle information
            start: Starting position (x, y)
            goal: Target position (x, y)
            max_steps: Maximum number of steps before giving up
            
        Returns:
            List of (x, y) positions representing the path from start to goal
        """
        # Convert to integer coordinates
        start = (int(round(start[0])), int(round(start[1])))
        goal = (int(round(goal[0])), int(round(goal[1])))
        
        # Check if start or goal is blocked
        if not (0 <= start[0] < navmesh.width and 0 <= start[1] < navmesh.height and
                0 <= goal[0] < navmesh.width and 0 <= goal[1] < navmesh.height):
            return []
            
        if not navmesh.is_walkable(start[0], start[1]) or not navmesh.is_walkable(goal[0], goal[1]):
            return []
        
        try:
            # Convert navmesh grid to numpy array if not already
            if not hasattr(navmesh, '_grid_np'):
                navmesh._grid_np = np.asarray(navmesh.grid, dtype=np.uint8)
            
            # Call JIT-compiled function
            path, success = self._astar_jit(navmesh._grid_np, start[0], start[1], goal[0], goal[1], max_steps)
            
            return path if success else []
            
        except Exception as e:
            print(f"A* search error: {e}")
            return []

    @staticmethod
    @numba.jit(nopython=True)
    def _astar_jit(grid: np.ndarray, start_x: int, start_y: int, goal_x: int, goal_y: int, 
                  max_steps: int = 10000):
        """Numba-accelerated A* pathfinding core."""
        # Directions: 8-way movement
        directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
        
        # Initialize data structures
        open_set = [(0, 0, start_x, start_y)]  # (f_score, count, x, y)
        came_from = {}
        g_score = {(start_x, start_y): 0}
        f_score = {(start_x, start_y): np.hypot(start_x - goal_x, start_y - goal_y)}
        
        count = 1  # For tie-breaking in priority queue
        
        while open_set and len(g_score) < max_steps:
            # Get the node with lowest f_score
            current_f, _, current_x, current_y = heapq.heappop(open_set)
            current = (current_x, current_y)
            
            # Check if we've reached the goal
            if current == (goal_x, goal_y):
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path, True
            
            # Check all neighbors
            for dx, dy in directions:
                neighbor = (current_x + dx, current_y + dy)
                
                # Check bounds and walkability
                if (neighbor[0] < 0 or neighbor[0] >= grid.shape[1] or 
                    neighbor[1] < 0 or neighbor[1] >= grid.shape[0] or 
                    grid[neighbor[1], neighbor[0]] != 0):
                    continue
                
                # Calculate tentative g score
                tentative_g_score = g_score[current] + np.hypot(dx, dy)
                
                # Check if this is a better path
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    h_score = np.hypot(neighbor[0] - goal_x, neighbor[1] - goal_y)
                    f_score[neighbor] = tentative_g_score + h_score
                    
                    # Add to open set if not already there
                    heapq.heappush(open_set, (f_score[neighbor], count, neighbor[0], neighbor[1]))
                    count += 1
        
        return [], False  # No path found
