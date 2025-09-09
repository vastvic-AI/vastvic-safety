import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from .agent import EvacuationAgent
from navmesh_ml2d.core.navmesh_fixed import GridNavMesh
import random

@dataclass
class EvacuationEnvironment:
    """Environment class for evacuation simulation with hazard and insurance features."""
    
    # Environment parameters
    width: float = 100.0  # meters
    height: float = 100.0  # meters
    cell_size: float = 1.0  # meters per grid cell
    
    # Simulation state
    time: float = 0.0
    agents: List[EvacuationAgent] = field(default_factory=list)
    obstacles: List[Tuple[float, float]] = field(default_factory=list)
    exits: List[Tuple[float, float]] = field(default_factory=list)
    hazards: Dict[str, Any] = field(default_factory=dict)
    
    # Navigation
    navmesh: Optional[GridNavMesh] = None
    
    # Insurance and risk assessment
    insurance_policies: Dict[str, Any] = field(default_factory=dict)
    total_risk_score: float = 0.0
    
    def __post_init__(self):
        # Initialize navigation mesh
        grid_width = int(self.width / self.cell_size)
        grid_height = int(self.height / self.cell_size)
        
        # Convert obstacles to grid coordinates
        grid_obstacles = []
        for x, y in self.obstacles:
            gx = int(x / self.cell_size)
            gy = int(y / self.cell_size)
            if 0 <= gx < grid_width and 0 <= gy < grid_height:
                grid_obstacles.append((gx, gy))
        
        self.navmesh = GridNavMesh(grid_width, grid_height, grid_obstacles)
        
        # Initialize hazards
        self.hazards = {
            'fire': {'positions': [], 'intensities': [], 'spread_rate': 0.1},
            'smoke': {'positions': [], 'intensities': [], 'spread_rate': 0.2},
            'blockage': {'positions': [], 'intensities': []}
        }
        
        # Initialize insurance policies
        self.initialize_insurance_policies()
    
    def initialize_insurance_policies(self):
        """Initialize insurance policies for the environment."""
        self.insurance_policies = {
            'property_damage': 1000000,  # Total coverage for property damage
            'business_interruption': 500000,
            'liability': 2000000,
            'evacuation_costs': 100000,
            'active_claims': [],
            'premium_rate': 0.01  # 1% of total insured value
        }
    
    def add_agent(self, agent: EvacuationAgent):
        """Add an agent to the environment."""
        self.agents.append(agent)
    
    def add_obstacle(self, x: float, y: float):
        """Add an obstacle at the specified position."""
        self.obstacles.append((x, y))
        
        # Update navmesh if it exists
        if self.navmesh:
            gx = int(x / self.cell_size)
            gy = int(y / self.cell_size)
            if 0 <= gx < self.navmesh.width and 0 <= gy < self.navmesh.height:
                self.navmesh.grid[gy, gx] = 1
    
    def add_exit(self, x: float, y: float):
        """Add an exit point."""
        self.exits.append((x, y))
    
    def add_hazard(self, hazard_type: str, position: Tuple[float, float], intensity: float = 1.0):
        """Add a hazard to the environment."""
        if hazard_type in self.hazards:
            self.hazards[hazard_type]['positions'].append(position)
            self.hazards[hazard_type]['intensities'].append(intensity)
    
    def update_hazards(self, time_step: float):
        """Update hazard states over time."""
        # Update fire spread
        self._update_fire_spread(time_step)
        
        # Update smoke spread
        self._update_smoke_spread(time_step)
        
        # Update hazard intensities
        self._update_hazard_intensities(time_step)
    
    def _update_fire_spread(self, time_step: float):
        """Simulate fire spread to nearby areas."""
        new_fire_positions = []
        new_intensities = []
        
        for i, (x, y) in enumerate(self.hazards['fire']['positions']):
            intensity = self.hazards['fire']['intensities'][i]
            
            # Fire can spread to adjacent cells
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                        
                    new_x = x + dx * self.cell_size
                    new_y = y + dy * self.cell_size
                    
                    # Check if position is valid and not already on fire
                    if (self._is_position_valid((new_x, new_y)) and 
                        not self._is_position_hazard('fire', (new_x, new_y))):
                        
                        # Probability of spreading decreases with distance
                        spread_prob = self.hazards['fire']['spread_rate'] * time_step * intensity
                        if random.random() < spread_prob:
                            new_fire_positions.append((new_x, new_y))
                            new_intensities.append(intensity * 0.8)  # Slightly less intense
        
        # Add new fire positions
        self.hazards['fire']['positions'].extend(new_fire_positions)
        self.hazards['fire']['intensities'].extend(new_intensities)
    
    def _update_smoke_spread(self, time_step: float):
        """Simulate smoke spread from fire sources."""
        # Smoke spreads from fire sources
        for (fx, fy), intensity in zip(self.hazards['fire']['positions'], 
                                     self.hazards['fire']['intensities']):
            if intensity > 0.3:  # Only significant fires produce smoke
                # Add smoke at fire location
                if not self._is_position_hazard('smoke', (fx, fy)):
                    self.hazards['smoke']['positions'].append((fx, fy))
                    self.hazards['smoke']['intensities'].append(intensity * 0.7)
    
    def _update_hazard_intensities(self, time_step: float):
        """Update intensities of all hazards over time."""
        for hazard_type in ['fire', 'smoke']:
            # Decrease intensity over time
            for i in range(len(self.hazards[hazard_type]['intensities'])):
                # Fire burns out, smoke dissipates
                decay_rate = 0.05 if hazard_type == 'fire' else 0.1
                self.hazards[hazard_type]['intensities'][i] *= (1.0 - decay_rate * time_step)
            
            # Remove hazards that have dissipated
            active_positions = []
            active_intensities = []
            
            for pos, intensity in zip(self.hazards[hazard_type]['positions'],
                                   self.hazards[hazard_type]['intensities']):
                if intensity > 0.01:  # Threshold for removal
                    active_positions.append(pos)
                    active_intensities.append(intensity)
            
            self.hazards[hazard_type]['positions'] = active_positions
            self.hazards[hazard_type]['intensities'] = active_intensities
    
    def get_danger_level(self, position: Tuple[float, float]) -> float:
        """Get the danger level at a specific position."""
        max_danger = 0.0
        
        # Check all hazard types
        for hazard_type in self.hazards:
            for hazard_pos, intensity in zip(self.hazards[hazard_type]['positions'],
                                          self.hazards[hazard_type]['intensities']):
                dist = np.linalg.norm(np.array(position) - np.array(hazard_pos))
                danger = intensity * max(0, 1 - dist / 10.0)  # Danger decreases with distance
                max_danger = max(max_danger, danger)
        
        return min(max_danger, 1.0)
    
    def get_congestion(self, position: Tuple[float, float], radius: float = 5.0) -> float:
        """Get congestion level (0-1) around a position."""
        if not self.agents:
            return 0.0
            
        nearby_agents = 0
        max_agents = 10  # Maximum expected agents in the radius
        
        for agent in self.agents:
            dist = np.linalg.norm(np.array(position) - agent.position)
            if dist <= radius:
                nearby_agents += 1
        
        return min(nearby_agents / max_agents, 1.0)
    
    def find_path(self, start: Tuple[float, float], end: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Find a path from start to end, avoiding obstacles and hazards."""
        if not self.navmesh:
            return [end]  # Fallback: direct path
            
        # Convert to grid coordinates
        start_grid = (int(start[0] / self.cell_size), int(start[1] / self.cell_size))
        end_grid = (int(end[0] / self.cell_size), int(end[1] / self.cell_size))
        
        # Find path using A*
        path = self.navmesh.astar(start_grid, end_grid)
        
        # Convert back to world coordinates
        world_path = [(x * self.cell_size, y * self.cell_size) for x, y in path]
        
        return world_path
    
    def step(self, time_step: float):
        """Advance the simulation by one time step."""
        self.time += time_step
        
        # Update hazards
        self.update_hazards(time_step)
        
        # Update agents
        for agent in self.agents:
            agent.step(time_step, self)
        
        # Update insurance and risk assessment
        self.update_risk_assessment()
    
    def update_risk_assessment(self):
        """Update overall risk assessment for insurance purposes."""
        # Calculate total risk score (0-1)
        hazard_risk = sum(sum(self.hazards[ht]['intensities']) for ht in self.hazards)
        hazard_risk = min(hazard_risk / 10.0, 1.0)  # Normalize
        
        congestion_risk = 0.0
        if self.agents:
            # Average agent panic level
            congestion_risk = sum(agent.panic_level for agent in self.agents) / len(self.agents)
        
        # Combine risk factors
        self.total_risk_score = 0.6 * hazard_risk + 0.4 * congestion_risk
        
        # Update insurance premiums based on risk
        self.insurance_policies['premium_rate'] = 0.01 * (1.0 + self.total_risk_score * 2)
    
    def calculate_insurance_claims(self) -> float:
        """Calculate total insurance claims from all agents and property damage."""
        total_claims = 0.0
        
        # Agent claims
        for agent in self.agents:
            total_claims += agent.calculate_insurance_claim()
        
        # Property damage claims
        property_damage = 0.0
        for hazard_type in self.hazards:
            # Estimate property damage based on hazard intensity
            property_damage += sum(self.hazards[hazard_type]['intensities']) * 1000
        
        total_claims += min(property_damage, self.insurance_policies['property_damage'])
        
        return total_claims
    
    def _is_position_valid(self, position: Tuple[float, float]) -> bool:
        """Check if a position is within environment bounds."""
        x, y = position
        return 0 <= x <= self.width and 0 <= y <= self.height
    
    def _is_position_hazard(self, hazard_type: str, position: Tuple[float, float]) -> bool:
        """Check if a position is affected by a specific hazard."""
        if hazard_type not in self.hazards:
            return False
            
        for hazard_pos in self.hazards[hazard_type]['positions']:
            if np.allclose(position, hazard_pos):
                return True
        return False
