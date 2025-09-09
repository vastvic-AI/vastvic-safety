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

class MCTSNode:
    """Node for Monte Carlo Tree Search."""
    def __init__(self, state: np.ndarray, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.reward = 0.0
    
    def select_child(self, exploration_weight: float = 1.0) -> 'MCTSNode':
        """Select child node using UCB1 formula."""
        log_n = np.log(self.visits + 1e-6)
        def ucb(c):
            return c.value / (c.visits + 1e-6) + exploration_weight * np.sqrt(log_n / (c.visits + 1e-6))
        return max(self.children, key=ucb)
    
    def expand(self, actions: List[np.ndarray]) -> 'MCTSNode':
        """Expand a new child node."""
        for action in actions:
            if not any(np.array_equal(action, c.action) for c in self.children):
                new_state = self.state.copy()
                new_state[:2] += action  # Update position
                child = MCTSNode(new_state, self, action)
                self.children.append(child)
                return child
        return self  # No new actions to expand
    
    def update(self, reward: float):
        """Update node statistics."""
        self.visits += 1
        self.value += (reward - self.value) / self.visits
        if self.parent:
            self.parent.update(reward)

class Agent:
    """
    An agent in the crowd simulation with realistic movement and behavior.
    
    Attributes:
        agent_id (int): Unique identifier for the agent.
        start (np.ndarray): Starting position (x, y).
        goal (np.ndarray): Target position (x, y).
        profile (dict): Agent characteristics (speed, panic, group_id, etc.).
        state (dict): Dynamic state (position, panic, status, etc.).
    """
    
    def __init__(self, agent_id: int, start: Tuple[float, float], goal: Tuple[float, float], 
                 profile: Optional[Dict[str, Any]] = None):
        """
        Initialize an agent.
        
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
            'acceleration': 3.5,    # Max acceleration (m/s^2) for responsiveness
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
        
        # Initialize ML models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = 10  # [pos(2), vel(2), goal(2), panic, density, hazard, group_size]
        self.action_dim = 8  # 8 possible directions
        self.action_space = np.array([
            [1, 0], [0, 1], [-1, 0], [0, -1],  # Cardinal directions
            [0.707, 0.707], [0.707, -0.707], [-0.707, 0.707], [-0.707, -0.707]  # Diagonals
        ], dtype=np.float32)
        
        # DQN networks
        self.policy_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Training setup
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.profile['learning_rate'])
        self.memory = ReplayBuffer(self.profile['memory_capacity'])
        self.steps_done = 0
        self.epsilon = self.profile['epsilon_start']
        
        # MCTS
        self.mcts_root = None
        self.current_simulation = 0
        
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
        self.waypoint_threshold = 0.5                # Smaller threshold to avoid stop-go at waypoints
    
    def move(self, density_map, navmesh, agents: Optional[List['Agent']] = None, dt: float = 0.1) -> None:
        """
        Move the agent based on the current environment and other agents.
        
        Args:
            density_map: 2D array of local crowd density.
            navmesh: Navigation mesh for pathfinding.
            agents: List of other agents in the simulation.
            dt: Time step (s).
        """
        if self.state['exited']:
            return
            
        # If we don't have a path, find one to the goal
        if not self.current_path:
            self.current_path = self.a_star_search(
                navmesh,
                (int(self.state['pos'][0]), int(self.state['pos'][1])),
                (int(self.goal[0]), int(self.goal[1]))
            )
            
        # Update the agent's state
        self.update(navmesh, density_map, np.zeros_like(density_map), time.time(), agents, dt)
    
    def update(self, navmesh, density_map, hazard_map, t: float, 
              agents: Optional[List['Agent']] = None, dt: float = 0.1) -> None:
        """
        Update the agent's state based on the environment and other agents.
        
        Args:
            navmesh: Navigation mesh for pathfinding.
            density_map: 2D array of local crowd density.
            hazard_map: 2D array of hazard intensities.
            t: Current simulation time.
            agents: List of other agents in the simulation.
            dt: Time step (s).
        """
        if self.state['exited']:
            return
            
        # Create default maps if not provided
        if density_map is None:
            density_map = np.zeros((navmesh.height, navmesh.width), dtype=np.float32)
        if hazard_map is None:
            hazard_map = np.zeros((navmesh.height, navmesh.width), dtype=np.float32)
            
        # Update panic level based on local conditions
        self._update_panic(density_map, hazard_map, agents, dt)
        
        # If we don't have a path, find one to the goal
        if not hasattr(self, 'current_path') or not self.current_path:
            self.current_path = self.a_star_search(navmesh, 
                tuple(map(int, self.state['pos'])), 
                tuple(map(int, self.goal)))
        
        # Calculate desired velocity (goal-seeking + social forces)
        self._update_desired_velocity(navmesh, agents, dt)
        
        # Update position based on velocity
        self._update_position(dt)
        
        # Check if we've reached the goal (exit)
        if np.linalg.norm(self.state['pos'] - self.goal) < 1.5:  # Within 1.5 units of goal
            self.state['exited'] = True
            self.state['exit_time'] = t
            self.state['pos'] = np.array([-100, -100])  # Move off-screen
            return
            
        # Update state
        self.state['last_update'] = t
    
    def _update_panic(self, density_map: np.ndarray, hazard_map: np.ndarray, 
                     agents: Optional[List['Agent']], dt: float) -> None:
        """Update the agent's panic level based on local conditions."""
        x, y = map(int, self.state['pos'])
        h, w = hazard_map.shape
        
        # Panic increases in hazardous or crowded areas
        panic_increase = 0.0
        
        # Check hazards
        if 0 <= y < h and 0 <= x < w and hazard_map[y, x] > 0:
            panic_increase += 0.15 * hazard_map[y, x]
            self.state['status'] = 'avoiding hazard'
        
        # Check crowd density
        if 0 <= y < h and 0 <= x < w and density_map[y, x] > 3.0:
            panic_increase += 0.05 * min(density_map[y, x] / 10.0, 1.0)
            self.state['status'] = 'crowded'
        else:
            self.state['status'] = 'normal'
        
        # Panic spreads from nearby agents
        if agents:
            for other in agents:
                if other is self or not hasattr(other, 'state'):
                    continue
                
                dist = np.linalg.norm(self.state['pos'] - other.state['pos'])
                if 0 < dist < 2.5 and other.state.get('panic', 0.0) > self.state['panic']:
                    # Panic spreads more quickly from panicked agents
                    panic_increase += 0.03 * (other.state['panic'] - self.state['panic'])
        
        # Update panic with decay
        self.state['panic'] = np.clip(
            self.state['panic'] * (1.0 - self.profile['panic_decay'] * dt) + panic_increase * dt,
            0.0, 1.0
        )
    
    def _get_state(self, navmesh, agents: List['Agent'], density_map: np.ndarray, 
                   hazard_map: np.ndarray) -> np.ndarray:
        """Get the current state representation for RL."""
        # Get local density and hazard
        pos = self.state['pos'].astype(int)
        density = density_map[min(pos[1], density_map.shape[0]-1), min(pos[0], density_map.shape[1]-1)]
        hazard = hazard_map[min(pos[1], hazard_map.shape[0]-1), min(pos[0], hazard_map.shape[1]-1)]
        
        # Get group info
        group_size = 1
        if self.state['group_id'] is not None and agents is not None:
            group_size = sum(1 for a in agents if a.state.get('group_id') == self.state['group_id'])
        
        return np.concatenate([
            self.state['pos'] / np.array([navmesh.width, navmesh.height]),
            self.state['vel'] / self.profile['speed'],
            (self.goal - self.state['pos']) / np.array([navmesh.width, navmesh.height]),
            [self.state['panic']],
            [density],
            [hazard],
            [group_size / 10.0]  # Normalize group size
        ], dtype=np.float32)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[np.ndarray, int]:
        """Select an action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            action_idx = random.randrange(self.action_dim)
            return self.action_space[action_idx], action_idx
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            action_idx = q_values.argmax().item()
            return self.action_space[action_idx], action_idx
    
    def optimize_model(self):
        """Perform one step of optimization on the policy network."""
        if len(self.memory) < self.profile['batch_size']:
            return
        
        transitions = self.memory.sample(self.profile['batch_size'])
        batch = Transition(*zip(*transitions))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.profile['batch_size'], device=self.device)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                    device=self.device, dtype=torch.bool)
        
        if non_final_mask.any():
            non_final_next_states = torch.FloatTensor(
                np.array([s for s in batch.next_state if s is not None])
            ).to(self.device)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        # Compute expected Q values
        expected_state_action_values = (next_state_values * self.profile['gamma']) + reward_batch
        
        # Compute loss
        loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    def monte_carlo_path_planning(self, navmesh, agents: List['Agent'], depth: int = 3) -> np.ndarray:
        """Perform MCTS to find the best action."""
        if self.mcts_root is None or self.current_simulation == 0:
            self.mcts_root = MCTSNode(self._get_state(navmesh, agents, None, None))
        
        for _ in range(self.profile['mcts_simulations']):
            node = self.mcts_root
            
            # Selection
            while node.children:
                node = node.select_child()
            
            # Expansion
            if node.visits > 0 and depth > 0:
                node = node.expand(self.action_space)
                
                # Simulation
                reward = self._simulate(node.state, depth-1, navmesh, agents)
                node.update(reward)
            
            self.current_simulation += 1
        
        # Select best action
        if self.mcts_root.children:
            best_child = max(self.mcts_root.children, key=lambda c: c.visits)
            return best_child.action
        return np.zeros(2)
    
    def _simulate(self, state: np.ndarray, depth: int, navmesh, agents: List['Agent']) -> float:
        """Simulate a random rollout from the given state."""
        if depth == 0:
            return 0.0
            
        # Simple reward: distance to goal (negative because we want to minimize it)
        current_pos = state[:2] * np.array([navmesh.width, navmesh.height])
        goal_pos = state[4:6] * np.array([navmesh.width, navmesh.height])
        dist_to_goal = np.linalg.norm(goal_pos - current_pos)
        
        # Penalize collisions and hazards
        collision_penalty = 0.0
        if not navmesh.is_walkable(int(current_pos[0]), int(current_pos[1])):
            collision_penalty = -1.0
        
        hazard_penalty = -state[7]  # Hazard intensity from state
        
        return -dist_to_goal + collision_penalty + hazard_penalty
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Calculate the heuristic distance between two points (Euclidean distance)."""
        return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

    @staticmethod
    @numba.njit(cache=True)
    def _astar_jit(grid: np.ndarray, start_x: int, start_y: int, goal_x: int, goal_y: int, 
                  max_steps: int = 10000) -> tuple:
        """Numba-accelerated A* pathfinding core."""
        # Directions: (dx, dy, cost)
        directions = np.array([
            [0, 1, 1.0],    # N
            [1, 0, 1.0],    # E
            [0, -1, 1.0],   # S
            [-1, 0, 1.0],   # W
            [1, 1, 1.4142], # NE
            [1, -1, 1.4142],# SE
            [-1, 1, 1.4142],# NW
            [-1, -1, 1.4142]# SW
        ], dtype=np.float32)
        
        height, width = grid.shape
        
        # Initialize data structures
        open_set = [(0.0, start_x, start_y)]
        came_from = {}
        g_score = np.full((height, width), np.inf)
        g_score[start_y, start_x] = 0
        f_score = np.full((height, width), np.inf)
        f_score[start_y, start_x] = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
        
        open_set_hash = {(start_x, start_y)}
        
        while open_set and len(open_set_hash) > 0 and max_steps > 0:
            max_steps -= 1
            
            # Get node with lowest f_score
            current_f, current_x, current_y = heapq.heappop(open_set)
            current_pos = (current_x, current_y)
            
            if current_pos not in open_set_hash:
                continue
                
            if current_x == goal_x and current_y == goal_y:
                # Reconstruct path
                path = []
                while current_pos in came_from:
                    path.append((current_pos[0], current_pos[1]))
                    current_pos = came_from[current_pos]
                path.append((start_x, start_y))
                return path[::-1], True
                
            open_set_hash.remove(current_pos)
            
            for dx, dy, cost in directions:
                nx, ny = int(current_x + dx), int(current_y + dy)
                
                if not (0 <= nx < width and 0 <= ny < height and grid[ny, nx] == 0):
                    continue
                    
                tentative_g = g_score[current_y, current_x] + cost
                
                if tentative_g < g_score[ny, nx]:
                    came_from[(nx, ny)] = (current_x, current_y)
                    g_score[ny, nx] = tentative_g
                    h = np.sqrt((goal_x - nx)**2 + (goal_y - ny)**2)
                    f_score[ny, nx] = tentative_g + h
                    
                    if (nx, ny) not in open_set_hash:
                        heapq.heappush(open_set, (f_score[ny, nx], nx, ny))
                        open_set_hash.add((nx, ny))
        
        return [], False

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
        
        # Early exit checks
        if start == goal:
            return [start]
            
        # Check bounds and walkability
        if (not (0 <= start[0] < navmesh.width and 0 <= start[1] < navmesh.height) or
            not (0 <= goal[0] < navmesh.width and 0 <= goal[1] < navmesh.height)):
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
                navmesh.is_walkable(neighbor[0], neighbor[1])):
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

    def _is_path_blocked(self, navmesh) -> bool:
        """
        Check if the current path is blocked by obstacles or other agents.
        
        Args:
            navmesh: Navigation mesh for collision detection
            
        Returns:
            bool: True if the path is blocked, False otherwise
        """
        if not hasattr(self, 'current_path') or not self.current_path:
            return True
            
        # Check the next few waypoints in the path
        check_ahead = min(5, len(self.current_path))
        for i in range(check_ahead):
            x, y = map(int, self.current_path[i])
            if not (0 <= x < navmesh.width and 0 <= y < navmesh.height):
                return True
            if not navmesh.is_walkable(x, y):
                return True
                
        return False
        
    def _is_better_path_available(self, navmesh, current_pos, goal_pos) -> bool:
        """
        Check if a better path exists than the current one.
        
        Args:
            navmesh: Navigation mesh for pathfinding
            current_pos: Current position of the agent
            goal_pos: Target position
            
        Returns:
            bool: True if a better path is available
        """
        if not hasattr(self, 'current_path') or not self.current_path:
            return True
            
        # Check if we're stuck (not making progress toward goal)
        if len(self.current_path) > 10:  # If path is long
            # Calculate progress toward goal
            start_dist = np.linalg.norm(np.array(current_pos) - np.array(goal_pos))
            path_dist = sum(np.linalg.norm(np.array(self.current_path[i]) - np.array(self.current_path[i-1])) 
                          for i in range(1, min(10, len(self.current_path))))
            
            # If we're not making good progress toward the goal
            if path_dist > start_dist * 1.5:  # Allow some detour
                return True
                
        return False
        
    def _calculate_reward(self, state: np.ndarray, next_state: np.ndarray, navmesh) -> float:
        """Calculate reward for the RL agent."""
        # Position components
        goal_pos = state[4:6] * np.array([navmesh.width, navmesh.height])
        current_pos = state[:2] * np.array([navmesh.width, navmesh.height])
        next_pos = next_state[:2] * np.array([navmesh.width, navmesh.height])
        
        # Distance to goal (negative because we want to minimize it)
        prev_dist = np.linalg.norm(goal_pos - current_pos)
        new_dist = np.linalg.norm(goal_pos - next_pos)
        dist_reward = prev_dist - new_dist  # Positive if moving toward goal
        
        # Penalize collisions
        collision_penalty = -1.0 if not navmesh.is_walkable(int(next_pos[0]), int(next_pos[1])) else 0.0
        
        # Penalize high density areas
        density_penalty = -next_state[7]  # Density component from state
        
        return dist_reward + collision_penalty + 0.5 * density_penalty
    
    def _calculate_social_force(self, agents: List['Agent'], navmesh) -> np.ndarray:
        """Calculate social force from other agents and obstacles, including group behaviors."""
        my_pos = self.state['pos']
        my_group = self.state.get('group_id')
        
        # Initialize forces
        repulsion_force = np.zeros(2)
        cohesion_force = np.zeros(2)
        alignment_force = np.zeros(2)
        separation_force = np.zeros(2)
        
        # Group tracking
        group_members = []
        
        # First pass: collect group members and calculate repulsion
        for agent in agents:
            if agent is self or agent.state.get('exited', False) or not hasattr(agent, 'state'):
                continue
                
            # Calculate vector to other agent
            diff = my_pos - agent.state['pos']
            dist = np.linalg.norm(diff)
            
            # Only consider nearby agents (within 5m for repulsion, 10m for group behaviors)
            if dist > 10.0 or dist < 1e-6:
                continue
                
            # Check if in same group
            is_group_member = (my_group is not None and 
                             agent.state.get('group_id') == my_group)
            
            if is_group_member and dist > 0:
                group_members.append((agent, dist))
            
            # Calculate repulsion (stronger from non-group members)
            if dist < 3.0:  # Only repulse when very close
                repulsion_dir = diff / (dist + 1e-6)
                strength = 2.0 * np.exp(-dist / 1.5)
                
                # Stronger repulsion from non-group members or panicked agents
                if not is_group_member:
                    strength *= 1.5
                if 'panic' in agent.state and agent.state['panic'] > 0.5:
                    strength *= (1.0 + agent.state['panic'])
                    
                repulsion_force += repulsion_dir * strength
        
        # Group behaviors (only if we have group members)
        if group_members and my_group is not None:
            group_center = np.zeros(2)
            group_velocity = np.zeros(2)
            group_count = len(group_members)
            
            # Calculate group center and average velocity
            for agent, dist in group_members:
                group_center += agent.state['pos']
                group_velocity += agent.state.get('vel', np.zeros(2))
                
                # Separation: avoid crowding local group members
                if dist < 2.5:  # Only separate when very close to group members
                    separation_dir = (my_pos - agent.state['pos']) / (dist + 1e-6)
                    separation_force += separation_dir * (1.0 / (dist + 0.1))
            
            # Calculate group behaviors
            if group_count > 0:
                group_center /= group_count
                group_velocity /= group_count
                
                # Cohesion: move toward group center
                if np.linalg.norm(group_center - my_pos) > 1.0:  # Only if not already at center
                    cohesion_dir = (group_center - my_pos)
                    cohesion_dir = cohesion_dir / (np.linalg.norm(cohesion_dir) + 1e-6)
                    cohesion_force = cohesion_dir * self.profile.get('cohesion_strength', 0.3)
                
                # Alignment: match group velocity
                if np.linalg.norm(group_velocity) > 0.1:  # Only if group is moving
                    alignment_force = group_velocity * self.profile.get('alignment_strength', 0.4)
                
                # Scale separation force
                separation_force = separation_force * self.profile.get('separation_strength', 0.2)
        
        # Repulsion from obstacles
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            test_pos = (int(my_pos[0] + dx), int(my_pos[1] + dy))
            if not (0 <= test_pos[0] < navmesh.width and 0 <= test_pos[1] < navmesh.height):
                continue
                
            if not navmesh.is_walkable(*test_pos):
                # Push away from obstacle
                repulsion_force += np.array([-dx, -dy], dtype=np.float32) * 2.5
        
        # Combine all forces with weights
        social_force = (
            repulsion_force * 1.0 +
            cohesion_force * (1.0 - self.state.get('panic', 0.0)) +  # Less cohesion when panicked
            alignment_force * (1.0 - self.state.get('panic', 0.0)) +  # Less alignment when panicked
            separation_force
        )
        
        # Add goal force (weighted by panic level - more goal-oriented when panicked)
        if hasattr(self, 'goal'):
            to_goal = self.goal - my_pos
            goal_dist = np.linalg.norm(to_goal)
            if goal_dist > 1.0:  # Only apply if not at goal
                goal_force = (to_goal / (goal_dist + 1e-6)) * (0.5 + self.state.get('panic', 0.5))
                social_force += goal_force
        
        # Normalize to max speed
        speed = np.linalg.norm(social_force)
        if speed > 0:
            social_force = social_force / speed * min(speed, self.profile['speed'] * (1.0 + self.state.get('panic', 0.0)))
            
        return social_force
    
    def _update_desired_velocity(self, navmesh, agents: Optional[List['Agent']], dt: float) -> None:
        """Compute desired velocity by blending path-following with social forces.
        Ensures the attribute exists to avoid runtime errors and provides natural motion.
        """
        # Ensure path exists
        if not hasattr(self, 'current_path') or not self.current_path:
            self.current_path = self.a_star_search(
                navmesh,
                (int(self.state['pos'][0]), int(self.state['pos'][1])),
                (int(self.goal[0]), int(self.goal[1]))
            )
        
        # Path-following component
        path_vel = np.zeros(2, dtype=np.float32)
        if self.current_path:
            target = np.array(self.current_path[0], dtype=np.float32)
            to_target = target - self.state['pos']
            dist = np.linalg.norm(to_target)
            if dist > 1e-6:
                path_vel = (to_target / dist) * self.profile['speed']
        
        # Social force component
        social = self._calculate_social_force(agents if agents is not None else [], navmesh)
        
        # Blend based on panic (more goal-directed when panicked)
        panic = float(self.state.get('panic', 0.0))
        w_path = 0.6 + 0.3 * panic
        w_social = 1.0 - w_path
        desired = w_path * path_vel + w_social * social
        
        # Small noise to avoid perfect alignment
        desired += np.random.normal(0, 0.02, 2)
        
        # Clamp to max speed
        sp = np.linalg.norm(desired)
        if sp > 0:
            desired = desired / sp * min(self.profile['speed'] * (1.0 + 0.2 * panic), sp)
        
        self.state['desired_vel'] = desired.astype(np.float32)
    
    def _update_position(self, dt: float) -> None:
        """
        Update the agent's position using A* path following with smooth movement.
        Implements realistic acceleration/deceleration and path following.
        
        Args:
            dt: Time step for movement update
        """
        if not hasattr(self, 'current_path') or not self.current_path:
            return
        
        # Get current and target positions
        current_pos = np.array(self.state['pos'])
        
        # Find the next waypoint to follow (look ahead in the path)
        lookahead = min(3, len(self.current_path))
        next_idx = max(0, lookahead - 1)
        target_pos = np.array(self.current_path[next_idx])
        
        # Calculate direction to target
        to_target = target_pos - current_pos
        dist_to_target = np.linalg.norm(to_target)
        
        # If we're close enough to the current waypoint, move to the next one
        if dist_to_target < self.waypoint_threshold:
            if len(self.current_path) > 1:
                self.current_path.pop(0)
                target_pos = np.array(self.current_path[0])
                to_target = target_pos - current_pos
                dist_to_target = np.linalg.norm(to_target)
            else:
                # Reached the end of the path - check if we're at the exit
                distance_to_goal = np.linalg.norm(current_pos - self.goal)
                if distance_to_goal < 1.5:  # Within exit threshold
                    self.state['exited'] = True
                    self.state['pos'] = np.array([-100, -100])  # Move off-screen
                self.current_path = []
                self.state['vel'] = np.zeros(2)
                return
        
        # Calculate desired velocity (prefer value computed in _update_desired_velocity)
        desired_vel = self.state.get('desired_vel', np.zeros(2))
        if np.linalg.norm(desired_vel) <= 1e-6:
            # Fallback: basic path-following
            if dist_to_target > 0:
                desired_vel = (to_target / dist_to_target) * self.profile['speed']
                if len(self.current_path) > 1:
                    desired_vel += np.random.normal(0, 0.05, 2)
                spd = np.linalg.norm(desired_vel)
                if spd > 0:
                    desired_vel = desired_vel / spd * self.profile['speed']
            else:
                desired_vel = np.zeros(2)
        
        # Update velocity with inertia (smoother movement)
        if 'vel' not in self.state:
            self.state['vel'] = np.zeros(2)
        
        # Apply acceleration/deceleration based on distance to target
        accel = self.profile.get('acceleration', 3.5)  # m/s² (more responsive)
        max_speed = self.profile['speed']
        
        # Slow down when approaching target
        stopping_distance = (np.linalg.norm(self.state['vel']) ** 2) / (2 * accel)
        
        # If we're close to the exit, speed up slightly
        distance_to_goal = np.linalg.norm(current_pos - self.goal)
        if distance_to_goal < 5.0:  # Within 5 units of goal
            max_speed *= 1.2  # 20% speed boost when near exit
        if dist_to_target < stopping_distance:
            target_speed = max(0.1, max_speed * (dist_to_target / (stopping_distance + 1e-6)))
            desired_vel = desired_vel * (target_speed / max_speed)
        
        # Update velocity with acceleration limits
        vel_diff = desired_vel - self.state['vel']
        if np.linalg.norm(vel_diff) > 0:
            accel_vec = (vel_diff / np.linalg.norm(vel_diff)) * min(accel * dt, np.linalg.norm(vel_diff))
            self.state['vel'] += accel_vec
        
        # Update position
        self.state['pos'] += self.state['vel'] * dt
        
        # Record path for visualization
        if 'path' not in self.state:
            self.state['path'] = []
        self.state['path'].append(tuple(self.state['pos']))
        
        # Keep path history manageable
        if len(self.state['path']) > 1000:
            self.state['path'] = self.state['path'][-1000:]
        
        # Stuck detection and auto-replan
        last_pos = getattr(self, '_last_pos_for_stuck', None)
        moved = 0.0 if last_pos is None else float(np.linalg.norm(self.state['pos'] - last_pos))
        self._last_pos_for_stuck = self.state['pos'].copy()
        if not hasattr(self, '_stuck_steps'):
            self._stuck_steps = 0
        if moved < 1e-3 and dist_to_target > 0.5:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0
        
        if self._stuck_steps >= 6:
            # Replan path and add a small nudge to break deadlocks
            self.current_path = self.a_star_search(
                navmesh,
                (int(self.state['pos'][0]), int(self.state['pos'][1])),
                (int(self.goal[0]), int(self.goal[1]))
            )
            self.state['vel'] += np.random.normal(0, 0.05, 2)
            self._stuck_steps = 0
    
    def get_speed(self) -> float:
        """Get the current speed of the agent."""
        return np.linalg.norm(self.state['vel'])
    
    def get_direction(self) -> np.ndarray:
        """Get the normalized direction of movement."""
        vel_norm = np.linalg.norm(self.state['vel'])
        return self.state['vel'] / (vel_norm + 1e-6) if vel_norm > 0 else np.zeros(2)
    
    def is_panicked(self) -> bool:
        """Check if the agent is in a panicked state."""
        return self.state['panic'] > self.profile['panic_threshold']
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get a dictionary of agent metrics for analysis.
        
        Returns:
            Dictionary containing various metrics about the agent's state.
        """
        speed = self.get_speed()
        is_panicked = self.is_panicked()
        
        # Calculate distance to goal
        goal_dist = np.linalg.norm(np.array(self.goal) - self.state['pos']) if not self.state['exited'] else 0.0
        
        return {
            'agent_id': self.agent_id,
            'position': tuple(self.state['pos'].astype(float)),
            'velocity': tuple(self.state['vel'].astype(float)),
            'speed': float(speed),
            'panic': float(self.state['panic']),
            'is_panicked': bool(is_panicked),
            'status': str(self.state['status']),
            'exited': bool(self.state['exited']),
            'exit_time': float(self.state.get('exit_time', 0.0)) if self.state.get('exit_time') is not None else None,
            'collisions': int(self.state.get('collisions', 0)),
            'waited': float(self.state.get('waited', 0.0)),
            'goal_distance': float(goal_dist),
            'group_id': self.state.get('group_id'),
            'density': 0.0,  # Will be updated by the simulation
            'hazard': 0.0,   # Will be updated by the simulation
            'timestamp': time.time(),
            'step': getattr(self, 'current_step', 0)
        }
    def reset(self):
        """Reset the agent's state for a new episode."""
        self.state.update({
            'pos': np.array(self.start, dtype=np.float32),
            'vel': np.zeros(2, dtype=np.float32),
            'desired_vel': np.zeros(2, dtype=np.float32),
            'panic': float(self.profile['panic']),
            'status': 'normal',
            'path': [],
            'waited': 0,
            'collisions': 0,
            'exited': False,
            'exit_time': None,
            'last_update': 0.0
        })
        self.mcts_root = None
        self.current_simulation = 0
        
    def save_model(self, path: str):
        """Save the agent's neural network weights."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'epsilon': self.epsilon
        }, path)
    
    def load_model(self, path: str):
        """Load the agent's neural network weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.epsilon = checkpoint['epsilon']
        
        # Update target network
        self.target_net.load_state_dict(self.policy_net.state_dict())

