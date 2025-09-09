"""
Crowd Simulation Engine with Stampede Simulation

This module implements the main crowd simulation loop, including:
- Agent state updates with stampede behavior
- Density and hazard map management
- Multi-threaded agent updates for performance
- Statistics collection and analysis
- Stampede detection and propagation
"""

import numpy as np
import numba
from typing import List, Dict, Tuple, Optional, Any, Set, DefaultDict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time
import logging
from dataclasses import dataclass
from enum import Enum, auto
from collections import defaultdict
import multiprocessing as mp
from functools import partial

class StampedeState(Enum):
    """Possible states of stampede detection."""
    NONE = auto()           # No stampede detected
    DETECTED = auto()       # Stampede conditions met
    ACTIVE = auto()         # Stampede in progress
    DISSIPATING = auto()    # Stampede is ending

@dataclass
class StampedeMetrics:
    """Metrics for tracking stampede behavior."""
    start_time: float = 0.0
    duration: float = 0.0
    max_agents: int = 0
    max_panic: float = 0.0
    max_velocity: float = 0.0
    total_injuries: int = 0
    avg_density: float = 0.0
    
    def reset(self):
        """Reset metrics for a new stampede."""
        self.__init__()

class CrowdSim:
    """
    Main crowd simulation class that manages agents, environment, and simulation steps.
    
    Attributes:
        agents (List[Agent]): List of agents in the simulation.
        exits (List[Tuple[float, float]]): List of exit positions.
        navmesh (GridNavMesh): Navigation mesh for pathfinding.
        step (int): Current simulation step.
        time_elapsed (float): Total simulation time in seconds.
        dt (float): Time step per simulation update (seconds).
        agent_paths (Dict[int, List[Tuple[float, float]]]): Trajectory history for each agent.
        stats (Dict[str, Any]): Simulation statistics.
    """
    
    def __init__(self, agents: List[Any], exits: List[Tuple[float, float]], navmesh: Any, dt: float = 0.1):
        """
        Initialize the crowd simulation with stampede support.
        
        Args:
            agents: List of Agent instances.
            exits: List of (x, y) exit positions.
            navmesh: Navigation mesh for pathfinding.
            dt: Time step per update (seconds).
        """
        self.agents = agents
        self.exits = exits
        self.navmesh = navmesh
        self.step = 0
        self.time_elapsed = 0.0
        self.dt = dt
        
        # Initialize agent paths and state
        self.agent_paths = {i: [tuple(agent.pos)] for i, agent in enumerate(agents)}
        self.width = navmesh.width
        
        # Stampede detection and management
        self.stampede_state = StampedeState.NONE
        self.stampede_metrics = StampedeMetrics()
        self.stampede_center = np.zeros(2)
        self.stampede_direction = np.zeros(2)
        self.stampede_agents: Set[int] = set()  # IDs of agents in stampede
        self.stampede_detection_threshold = 0.65  # Confidence threshold (0-1)
        self.stampede_detection_window = 1.0  # Seconds to confirm stampede
        self.stampede_detection_timer = 0.0
        self.last_stampede_time = -100.0  # Time of last stampede
        self.stampede_cooldown = 10.0  # Minimum seconds between stampedes
        
        # Initialize hazard and density maps (same size as navmesh)
        self.hazard_map = np.zeros((navmesh.height, navmesh.width), dtype=np.float32)
        self.density_map = np.zeros((navmesh.height, navmesh.width), dtype=np.float32)
        
        # Logging
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('CrowdSim')
        self.height = navmesh.height
        
        # Initialize statistics
        self.stats = {
            'num_agents': len(agents),
            'num_exited': 0,
            'num_injured': 0,
            'panic_levels': [],  # Track individual agent panic levels
            'avg_panic': 0.0,
            'max_density': 0.0,
            'avg_speed': 0.0,
            'evacuation_time': None,
            'start_time': time.time(),
            'stampedes': 0,
            'total_stampede_time': 0.0,
            'max_stampede_size': 0,
            'stampede_injuries': 0,
            'stampede_detected': False
        }
        
        # Performance tracking
        self._performance = {
            'update_times': [],
            'pathfinding_times': []
        }
        
    def detect_stampede(self) -> bool:
        """
        Detect if a stampede is occurring based on crowd metrics.
        
        Returns:
            bool: True if a stampede is detected, False otherwise
        """
        # Skip detection if we're already in a stampede or in cooldown
        if (self.stampede_state in [StampedeState.ACTIVE, StampedeState.DETECTED] or 
            (self.time_elapsed - self.last_stampede_time) < self.stampede_cooldown):
            return self.stampede_state in [StampedeState.ACTIVE, StampedeState.DETECTED]
            
        # Calculate crowd metrics
        positions = []
        velocities = []
        panic_levels = []
        
        for agent in self.agents:
            if hasattr(agent, 'state') and 'pos' in agent.state and not agent.state.get('exited', False):
                positions.append(agent.state['pos'])
                if hasattr(agent, 'velocity'):
                    velocities.append(agent.velocity)
                panic_levels.append(agent.panic if hasattr(agent, 'panic') else 0.0)
        
        if len(positions) < 10:  # Need minimum agents for stampede
            return False
            
        positions = np.array(positions)
        velocities = np.array(velocities) if velocities else np.zeros((len(positions), 2))
        panic_levels = np.array(panic_levels)
        
        # Calculate metrics
        avg_panic = np.mean(panic_levels)
        high_panic_ratio = np.mean(panic_levels > 0.7)
        
        # Calculate velocity coherence
        if len(velocities) > 0:
            speeds = np.linalg.norm(velocities, axis=1)
            avg_speed = np.mean(speeds)
            
            # Normalize velocities for direction calculation
            norms = np.linalg.norm(velocities, axis=1, keepdims=True)
            valid = norms.squeeze() > 0
            if np.any(valid):
                dirs = np.zeros_like(velocities)
                dirs[valid] = velocities[valid] / norms[valid]
                direction_alignment = np.mean(np.dot(dirs, dirs.T))
            else:
                direction_alignment = 0.0
        else:
            avg_speed = 0.0
            direction_alignment = 0.0
            
        # Calculate local densities
        from sklearn.neighbors import NearestNeighbors
        if len(positions) > 5:
            nbrs = NearestNeighbors(n_neighbors=5).fit(positions)
            distances, _ = nbrs.kneighbors(positions)
            local_density = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-6)  # Exclude self
            avg_local_density = np.mean(local_density)
        else:
            avg_local_density = 0.0
            
        # Stampede detection conditions
        stampede_conditions = [
            avg_panic > 0.6,                    # High average panic
            high_panic_ratio > 0.5,             # Majority in high panic
            avg_speed > 1.5,                    # Faster than normal walking
            direction_alignment > 0.6,           # Moving in similar directions
            avg_local_density > 2.0,             # High local density
            (self.time_elapsed - self.last_stampede_time) > self.stampede_cooldown  # Not in cooldown
        ]
        
        confidence = sum(stampede_conditions) / len(stampede_conditions)
        
        # Update stampede state machine
        if confidence >= self.stampede_detection_threshold:
            if self.stampede_state == StampedeState.NONE:
                self.stampede_detection_timer += self.dt
                if self.stampede_detection_timer >= self.stampede_detection_window:
                    self._start_stampede(positions, velocities, panic_levels)
                    return True
            return self.stampede_state in [StampedeState.ACTIVE, StampedeState.DETECTED]
        else:
            self.stampede_detection_timer = max(0, self.stampede_detection_timer - self.dt)
            return False
            
    def _start_stampede(self, positions: np.ndarray, velocities: np.ndarray, 
                       panic_levels: np.ndarray) -> None:
        """Initialize a new stampede event."""
        self.stampede_state = StampedeState.DETECTED
        self.stampede_metrics.reset()
        self.stampede_metrics.start_time = self.time_elapsed
        
        # Calculate stampede center and direction
        self.stampede_center = np.mean(positions, axis=0)
        
        # Calculate dominant direction using PCA on velocities
        if len(velocities) > 0:
            cov_matrix = np.cov(velocities.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            self.stampede_direction = eigenvectors[:, np.argmax(eigenvalues)]
            # Ensure direction points away from center (for circular crowds)
            if np.dot(self.stampede_direction, self.stampede_center) < 0:
                self.stampede_direction *= -1
        else:
            self.stampede_direction = np.array([1.0, 0.0])  # Default direction
            
        # Initialize stampede metrics
        self.stampede_metrics.max_panic = np.max(panic_levels)
        self.stampede_metrics.avg_density = len(positions) / (self.width * self.height)
        
        self.logger.warning(
            f"STAMPEDE DETECTED at t={self.time_elapsed:.1f}s: "
            f"{len(positions)} agents, panic={np.mean(panic_levels):.2f}"
        )
        
        # Mark stampede as active after a short delay
        self.stampede_state = StampedeState.ACTIVE
        self.stats['stampedes'] += 1
        self.stats['stampede_detected'] = True
        
    def _update_stampede(self) -> None:
        """Update stampede state and metrics."""
        if self.stampede_state != StampedeState.ACTIVE:
            return
            
        # Update stampede center and direction based on current agents
        positions = []
        velocities = []
        panic_levels = []
        
        for agent in self.agents:
            if hasattr(agent, 'state') and 'pos' in agent.state and not agent.state.get('exited', False):
                positions.append(agent.state['pos'])
                if hasattr(agent, 'velocity'):
                    velocities.append(agent.velocity)
                panic_levels.append(agent.panic if hasattr(agent, 'panic') else 0.0)
                
        if not positions:
            self._end_stampede()
            return
            
        positions = np.array(positions)
        velocities = np.array(velocities) if velocities else np.zeros((len(positions), 2))
        
        # Update stampede center and direction (with momentum)
        new_center = np.mean(positions, axis=0)
        self.stampede_center = 0.3 * new_center + 0.7 * self.stampede_center
        
        if len(velocities) > 0:
            avg_velocity = np.mean(velocities, axis=0)
            if np.linalg.norm(avg_velocity) > 0.1:  # Only update if there's significant movement
                new_direction = avg_velocity / np.linalg.norm(avg_velocity)
                self.stampede_direction = 0.2 * new_direction + 0.8 * self.stampede_direction
                self.stampede_direction /= np.linalg.norm(self.stampede_direction)
        
        # Update metrics
        self.stampede_metrics.duration = self.time_elapsed - self.stampede_metrics.start_time
        self.stampede_metrics.max_agents = max(self.stampede_metrics.max_agents, len(positions))
        self.stampede_metrics.max_panic = max(self.stampede_metrics.max_panic, np.max(panic_levels) if panic_levels else 0)
        
        # Update stampede agents set
        self.stampede_agents = {i for i, agent in enumerate(self.agents) 
                              if hasattr(agent, 'in_stampede') and agent.in_stampede}
        
        # Check for stampede end conditions
        if len(positions) < 5 or np.mean(panic_levels) < 0.3:
            self._end_stampede()
            
    def _end_stampede(self) -> None:
        """End the current stampede and log metrics."""
        if self.stampede_state != StampedeState.ACTIVE:
            return
            
        self.stampede_state = StampedeState.DISSIPATING
        self.last_stampede_time = self.time_elapsed
        self.stats['total_stampede_time'] += self.stampede_metrics.duration
        self.stats['max_stampede_size'] = max(
            self.stats['max_stampede_size'], 
            self.stampede_metrics.max_agents
        )
        
        self.logger.info(
            f"STAMPEDE ENDED after {self.stampede_metrics.duration:.1f}s: "
            f"Max agents={self.stampede_metrics.max_agents}, "
            f"Max panic={self.stampede_metrics.max_panic:.2f}"
        )
        
        # Reset after a short delay
        self.stampede_state = StampedeState.NONE
        self.stampede_agents.clear()
        self.stats['stampede_detected'] = False
        
    def update_agent_stampede_behavior(self, agent_idx: int, agent: Any) -> None:
        """Update agent behavior based on stampede state."""
        if self.stampede_state != StampedeState.ACTIVE:
            if hasattr(agent, 'in_stampede') and agent.in_stampede:
                # Agent was in stampede but it ended
                agent.in_stampede = False
                agent.panic = max(0.5, agent.panic * 0.8)  # Reduce but maintain some panic
            return
            
        # Check if agent should join the stampede
        if not agent.in_stampede:
            # Agents with high panic are more likely to join
            panic_threshold = 0.6 - (0.2 * (1.0 - self.stampede_metrics.max_panic))
            if (agent.panic > panic_threshold and 
                np.random.random() < 0.3):  # 30% chance to join per step
                agent.in_stampede = True
                self.stampede_agents.add(agent_idx)
                
        # Update agent's stampede behavior
        if agent.in_stampede:
            # Calculate direction to stampede center and movement direction
            to_center = self.stampede_center - np.array(agent.state['pos'])
            dist_to_center = np.linalg.norm(to_center)
            
            if dist_to_center > 0:
                to_center /= dist_to_center
                
            # Blend between moving toward center and moving in stampede direction
            center_weight = 0.3 * (1.0 - np.exp(-dist_to_center / 5.0))
            stampede_force = (center_weight * to_center + 
                            (1.0 - center_weight) * self.stampede_direction)
            
            # Add some noise and normalize
            noise = np.random.normal(0, 0.1, 2)
            stampede_force = 0.9 * stampede_force + 0.1 * noise
            stampede_force = stampede_force / (np.linalg.norm(stampede_force) + 1e-6)
            
            # Update agent's desired velocity
            agent.desired_velocity = stampede_force * agent.max_speed * 1.3  # 30% speed boost
            
            # Increase panic over time in stampede
            agent.panic = min(1.0, agent.panic + 0.02 * self.dt)
            
            # Reduce personal space and disable pathfinding
            agent.personal_space_radius = agent.radius * 1.5
            agent.use_pathfinding = False
        
        # Performance tracking
        self._last_update_time = time.time()
    
    def t(self) -> float:
        """Get the current simulation time in seconds."""
        return self.time_elapsed
    
    def update_density_map(self, cell_size: float = 1.0) -> np.ndarray:
        """
        Update the density map based on agent positions.
        
        Args:
            cell_size: Size of each grid cell in meters.
            
        Returns:
            density_map: 2D array of agent density.
        """
        density_map = np.zeros((self.height, self.width), dtype=np.float32)
        
        for agent in self.agents:
            if agent.state['exited']:
                continue
                
            x, y = map(int, agent.state['pos'])
            if 0 <= y < self.height and 0 <= x < self.width:
                # Gaussian kernel for smoother density
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < self.height and 0 <= nx < self.width:
                            # Weight by distance from center
                            weight = np.exp(-0.5 * (dx*dx + dy*dy))
                            density_map[ny, nx] += weight
        
        # Update max density
        self.stats['max_density'] = max(self.stats['max_density'], np.max(density_map))
        return density_map
    
    def update(self, dt: float = None) -> None:
        """
        Update the simulation state by one time step with stampede detection and management.
        
        Args:
            dt: Time step in seconds. If None, uses the simulation's default dt.
        """
        if dt is None:
            dt = self.dt
            
        self.time_elapsed += dt
        self.step += 1
        
        # Update density map for this step
        self.density_map = self.update_density_map()
        
        # Detect stampede conditions
        stampede_detected = self.detect_stampede()
        
        # Update stampede state if active
        if self.stampede_state == StampedeState.ACTIVE:
            self._update_stampede()
        
        # Update all agents
        for i, agent in enumerate(self.agents):
            if not agent.state.get('exited', False):
                # Update agent's stampede behavior if needed
                if stampede_detected or self.stampede_state == StampedeState.ACTIVE:
                    self.update_agent_stampede_behavior(i, agent)
                
                # Update agent state
                agent.update(self.navmesh, self.density_map, self.hazard_map, 
                           self.time_elapsed, self.agents, dt)
                
                # Record agent path
                self.agent_paths[i].append(tuple(agent.state['pos']))
                
                # Check for exit condition
                for exit_pos in self.exits:
                    if np.linalg.norm(agent.state['pos'] - np.array(exit_pos)) < 0.5:
                        agent.state['exited'] = True
                        agent.state['exit_time'] = self.time_elapsed
                        self.stats['num_exited'] += 1
                        
                        # If agent was in stampede, remove from stampede set
                        if i in self.stampede_agents:
                            self.stampede_agents.remove(i)
                        break
                        
                # Check for injuries during stampede
                if self.stampede_state == StampedeState.ACTIVE and hasattr(agent, 'in_stampede') and agent.in_stampede:
                    # Higher panic and density increase injury chance
                    local_density = self.density_map[int(agent.state['pos'][1]), int(agent.state['pos'][0])]
                    injury_risk = 0.1 * agent.panic * (1.0 + local_density)
                    if np.random.random() < injury_risk * dt:
                        agent.state['injured'] = True
                        self.stats['num_injured'] += 1
                        self.stats['stampede_injuries'] += 1
                        
                        # Injured agents slow down and can't continue in stampede
                        if hasattr(agent, 'in_stampede'):
                            agent.in_stampede = False
                            agent.max_speed *= 0.5  # Halve max speed when injured
        
        # Update statistics
        self._update_statistics()
        
        # Update performance metrics
        now = time.time()
        self._performance['update_times'].append(now - self._last_update_time)
        self._last_update_time = now
        
    def _update_statistics(self) -> None:
        """Update simulation statistics."""
        # Calculate agent statistics
        active_agents = [a for a in self.agents if not a.state.get('exited', False)]
        if not active_agents:
            if self.stats['evacuation_time'] is None:
                self.stats['evacuation_time'] = self.time_elapsed
            return
            
        # Update panic statistics
        panic_levels = [a.panic for a in active_agents if hasattr(a, 'panic')]
        self.stats['avg_panic'] = np.mean(panic_levels) if panic_levels else 0.0
        
        # Update speed statistics
        speeds = []
        for agent in active_agents:
            if hasattr(agent, 'velocity'):
                speed = np.linalg.norm(agent.velocity)
                speeds.append(speed)
                
                # Update max velocity in stampede metrics if in stampede
                if (hasattr(agent, 'in_stampede') and agent.in_stampede and 
                    self.stampede_state == StampedeState.ACTIVE):
                    self.stampede_metrics.max_velocity = max(
                        self.stampede_metrics.max_velocity, speed)
        
        self.stats['avg_speed'] = np.mean(speeds) if speeds else 0.0
        
        # Update stampede statistics
        if self.stampede_state == StampedeState.ACTIVE:
            self.stampede_metrics.duration = self.time_elapsed - self.stampede_metrics.start_time
            self.stampede_metrics.max_agents = max(
                self.stampede_metrics.max_agents, 
                len(self.stampede_agents)
            )
            
            # Update stampede center and direction based on stampeding agents
            if self.stampede_agents:
                stampeding_positions = [
                    self.agents[i].state['pos'] 
                    for i in self.stampede_agents 
                    if i < len(self.agents) and not self.agents[i].state.get('exited', False)
                ]
                if stampeding_positions:
                    self.stampede_center = np.mean(stampeding_positions, axis=0)
                    
                    # Update direction based on velocity of stampeding agents
                    stampeding_velocities = [
                        self.agents[i].velocity 
                        for i in self.stampede_agents 
                        if (i < len(self.agents) and 
                            hasattr(self.agents[i], 'velocity') and 
                            not self.agents[i].state.get('exited', False))
                    ]
                    if stampeding_velocities:
                        avg_velocity = np.mean(stampeding_velocities, axis=0)
                        if np.linalg.norm(avg_velocity) > 0.1:  # Only update if significant movement
                            self.stampede_direction = 0.2 * (avg_velocity / np.linalg.norm(avg_velocity)) + \
                                                   0.8 * self.stampede_direction
                            self.stampede_direction /= np.linalg.norm(self.stampede_direction)
        
    def _update_agent(self, agent: Any, density_map: np.ndarray, hazard_map: np.ndarray) -> Dict[str, Any]:
        """
        Update a single agent's state (thread-safe).
        
        Args:
            agent: The agent to update.
            density_map: Current density map.
            hazard_map: Current hazard map.
            
        Returns:
            metrics: Agent metrics after update.
        """
        if agent.state.get('exited', False):
            return agent.get_metrics()
            
        try:
            # Get agent position for density/hazard sampling
            x, y = int(round(agent.state['pos'][0])), int(round(agent.state['pos'][1]))
            
            # Ensure position is within map bounds
            h, w = density_map.shape
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            
            # Sample density and hazard values
            local_density = float(density_map[y, x])
            local_hazard = float(hazard_map[y, x]) if hasattr(self.navmesh, 'hazard_map') else 0.0
            
            # Update agent with current maps
            agent.update(
                navmesh=self.navmesh,
                density_map=density_map,
                hazard_map=hazard_map,
                t=self.time_elapsed,
                agents=self.agents,
                dt=self.dt
            )
            
            # Record path
            agent_id = getattr(agent, 'agent_id', id(agent) % 1000)
            if 'path' not in agent.state:
                agent.state['path'] = []
            agent.state['path'].append(tuple(agent.state['pos'].copy()))
            self.agent_paths[agent_id] = agent.state['path']
            
            # Check if agent reached exit
            for exit_pos in self.exits:
                if np.linalg.norm(agent.state['pos'] - np.array(exit_pos)) < 1.0:
                    agent.state['exited'] = True
                    agent.state['exit_time'] = self.time_elapsed
                    self.stats['num_exited'] += 1
                    break
                    
            # Get metrics and update with local density/hazard values
            metrics = agent.get_metrics()
            metrics.update({
                'density': local_density,
                'hazard': local_hazard,
                'step': self.step
            })
            return metrics
                    
        except Exception as e:
            logging.error(f"Error updating agent {getattr(agent, 'agent_id', 'unknown')}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return basic metrics even if update failed
            metrics = agent.get_metrics()
            metrics.update({
                'density': 0.0,
                'hazard': 0.0,
                'step': self.step,
                'error': str(e)
            })
            return metrics
    
    def step_sim(self) -> Dict[str, Any]:
        """
        Advance the simulation by one time step.
        
        Returns:
            Dictionary containing simulation metrics and agent states.
        """
        start_time = time.time()
        
        try:
            # Update density and hazard maps
            density_map = self.update_density_map(cell_size=1.0)  # Default cell size of 1.0 meters
            hazard_map = self.navmesh.hazard_map if hasattr(self.navmesh, 'hazard_map') else np.zeros_like(density_map)
            
            # Update all agents (in parallel for large crowds)
            metrics_list = []
            if len(self.agents) > 50:  # Use threading for large crowds
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(self._update_agent, agent, density_map, hazard_map)
                        for agent in self.agents
                        if not agent.state.get('exited', False)  # Skip exited agents
                    ]
                    metrics_list = [f.result() for f in futures]
            else:
                metrics_list = [
                    self._update_agent(agent, density_map, hazard_map)
                    for agent in self.agents
                    if not agent.state.get('exited', False)  # Skip exited agents
                ]
            
            # Update simulation state
            self.step += 1
            self.time_elapsed += self.dt
            
            # Track panic levels and other metrics
            if metrics_list:
                # Filter out None values and get valid metrics
                valid_metrics = [m for m in metrics_list if m is not None]
                
                if valid_metrics:
                    # Update panic statistics
                    panic_levels = [m.get('panic', 0.0) for m in valid_metrics]
                    self.stats['panic_levels'] = panic_levels
                    self.stats['avg_panic'] = float(np.mean(panic_levels))
                    
                    # Update speed statistics
                    speeds = [m.get('speed', 0.0) for m in valid_metrics]
                    self.stats['avg_speed'] = float(np.mean(speeds))
                    self.stats['max_speed'] = float(np.max(speeds)) if speeds else 0.0
                    
                    # Update density statistics
                    densities = [m.get('density', 0.0) for m in valid_metrics]
                    self.stats['avg_density'] = float(np.mean(densities)) if densities else 0.0
                    self.stats['max_density'] = float(np.max(densities)) if densities else 0.0
                    
                    # Update hazard statistics
                    hazards = [m.get('hazard', 0.0) for m in valid_metrics]
                    self.stats['avg_hazard'] = float(np.mean(hazards)) if hazards else 0.0
                    self.stats['max_hazard'] = float(np.max(hazards)) if hazards else 0.0
            
            # Calculate performance metrics
            update_time = time.time() - start_time
            self._performance['update_times'].append(update_time)
            
            # Keep performance history from growing too large
            if len(self._performance['update_times']) > 1000:
                self._performance['update_times'] = self._performance['update_times'][-1000:]
            
            # Update stampede state
            self._update_stampede()
            
            # Calculate evacuation metrics
            num_active = len([a for a in self.agents if not a.state.get('exited', False)])
            num_exited = len([a for a in self.agents if a.state.get('exited', False)])
            
            # Update evacuation time if all agents have exited
            if num_exited > 0 and num_active == 0 and self.stats['evacuation_time'] is None:
                self.stats['evacuation_time'] = self.time_elapsed
            
            # Prepare return metrics
            metrics = {
                'step': self.step,
                'time_elapsed': self.time_elapsed,
                'num_agents': len(self.agents),
                'num_active': num_active,
                'num_exited': num_exited,
                'evacuation_progress': num_exited / len(self.agents) if self.agents else 0.0,
                'avg_panic': self.stats.get('avg_panic', 0.0),
                'avg_speed': self.stats.get('avg_speed', 0.0),
                'max_density': self.stats.get('max_density', 0.0),
                'avg_density': self.stats.get('avg_density', 0.0),
                'max_hazard': self.stats.get('max_hazard', 0.0),
                'avg_hazard': self.stats.get('avg_hazard', 0.0),
                'stampede_active': self.stampede_state == StampedeState.ACTIVE,
                'stampede_size': len(self.stampede_agents),
                'update_time': update_time,
                'fps': 1.0 / update_time if update_time > 0 else 0.0
            }
            
            # Add per-agent metrics if not too many agents
            if len(self.agents) <= 100:  # Only include for small crowds
                metrics['agents'] = metrics_list
                
            return metrics
                
        except Exception as e:
            logging.error(f"Error in simulation step: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return basic metrics even if update failed
            return {
                'step': self.step,
                'time_elapsed': self.time_elapsed,
                'error': str(e),
                'num_agents': len(self.agents),
                'num_active': len([a for a in self.agents if not a.state.get('exited', False)]),
                'num_exited': len([a for a in self.agents if a.state.get('exited', False)])
            }
