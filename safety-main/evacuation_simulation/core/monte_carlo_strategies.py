import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import random
from collections import defaultdict, deque
import sys
from pathlib import Path
from enum import Enum, auto

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Use absolute imports
from evacuation_simulation.core.agent import EvacuationAgent
from evacuation_simulation.core.environment import EvacuationEnvironment

class EvacuationStrategy(Enum):
    SHORTEST_PATH = auto()
    LEAST_RISK = auto()
    CROWD_AVOIDANCE = auto()
    COMBINED = auto()

@dataclass
class PathOption:
    path: List[Tuple[float, float]]
    risk_score: float = 0.0
    crowd_density: float = 0.0
    distance: float = 0.0
    
    @property
    def utility(self) -> float:
        """Calculate path utility based on risk, distance, and crowd density."""
        # Normalize and weight factors (weights can be adjusted)
        risk_factor = 0.4 * (1 - min(self.risk_score, 1.0))
        distance_factor = 0.4 * (1 - min(self.distance / 200.0, 1.0))
        crowd_factor = 0.2 * (1 - min(self.crowd_density, 1.0))
        return risk_factor + distance_factor + crowd_factor

@dataclass
class EvacuationScenario:
    """Represents a single evacuation scenario in the Monte Carlo simulation."""
    scenario_id: int
    agents: List[EvacuationAgent]
    hazards: List[Dict]
    success_rate: float = 0.0
    avg_evacuation_time: float = 0.0
    risk_distribution: Dict[str, float] = None
    
    def calculate_metrics(self):
        """Calculate key metrics for this scenario."""
        if not self.agents:
            return
            
        # Calculate success rate (percentage of agents who reached an exit)
        evacuated = sum(1 for agent in self.agents if hasattr(agent, 'evacuated') and agent.evacuated)
        self.success_rate = (evacuated / len(self.agents)) * 100
        
        # Calculate average evacuation time
        evacuation_times = [agent.evacuation_time for agent in self.agents 
                          if hasattr(agent, 'evacuated') and agent.evacuated]
        self.avg_evacuation_time = np.mean(evacuation_times) if evacuation_times else float('inf')
        
        # Calculate risk distribution
        self.risk_distribution = {
            'low': 0,
            'medium': 0,
            'high': 0
        }
        
        for agent in self.agents:
            if hasattr(agent, 'risk_factors'):
                risk = sum(agent.risk_factors.values()) / len(agent.risk_factors)
                if risk < 0.3:
                    self.risk_distribution['low'] += 1
                elif risk < 0.7:
                    self.risk_distribution['medium'] += 1
                else:
                    self.risk_distribution['high'] += 1


class MonteCarloEvacuation:
    """Implements Monte Carlo simulation for evacuation scenarios with Thunderhead strategies."""
    
    def __init__(self, 
                num_simulations: int = 100,
                strategy: EvacuationStrategy = EvacuationStrategy.COMBINED,
                sample_paths: int = 50,
                risk_weight: float = 0.4,
                distance_weight: float = 0.4,
                crowd_weight: float = 0.2):
        self.num_simulations = num_simulations
        self.strategy = strategy
        self.sample_paths = sample_paths
        self.risk_weight = risk_weight
        self.distance_weight = distance_weight
        self.crowd_weight = crowd_weight
        self.scenarios: List[EvacuationScenario] = []
        self.convergence_data = []
        self.path_cache = {}  # Cache for computed paths
        
    def run_simulation(self, env: EvacuationEnvironment, max_steps: int = 1000, 
                      step_callback=None) -> List[EvacuationScenario]:
        """Run multiple evacuation scenarios using Monte Carlo simulation.
        
        Args:
            env: The environment to run the simulation in
            max_steps: Maximum number of steps per scenario
            step_callback: Optional callback function that receives (env, step) 
                         and returns True to continue or False to stop
                          
        Returns:
            List of EvacuationScenario objects
        """
        self.scenarios = []
        self.convergence_data = []
        
        for sim_idx in range(self.num_simulations):
            # Create a copy of the environment for this scenario
            scenario_env = self._create_scenario_environment(env)
            
            # Run the scenario
            scenario = self._run_single_scenario(scenario_env, max_steps, step_callback)
            scenario.scenario_id = sim_idx
            
            # Calculate metrics
            scenario.calculate_metrics()
            self.scenarios.append(scenario)
            
            # Update convergence data
            self._update_convergence_data(scenario)
            
            # Check if we should stop
            if step_callback and not step_callback(scenario_env, sim_idx):
                break
                
        return self.scenarios
    
    def _create_scenario_environment(self, base_env: EvacuationEnvironment) -> EvacuationEnvironment:
        """Create a new environment for a single scenario with randomized parameters."""
        # Create a copy of the base environment
        env = EvacuationEnvironment(width=base_env.width, height=base_env.height)
        
        # Copy obstacles and exits
        env.obstacles = base_env.obstacles.copy()
        env.exits = base_env.exits.copy()
        
        # Randomize agent positions and behaviors
        for agent in base_env.agents:
            # Create a copy of the agent with some randomization
            new_agent = EvacuationAgent(
                agent_id=agent.agent_id,
                position=self._get_random_position(env),
                agent_type=agent.agent_type,
                panic_level=agent.panic_level * random.uniform(0.8, 1.2),
                health=agent.health * random.uniform(0.9, 1.1)
            )
            
            # Copy insurance and risk factors
            if hasattr(agent, 'insurance_coverage'):
                new_agent.insurance_coverage = agent.insurance_coverage.copy()
            if hasattr(agent, 'risk_factors'):
                new_agent.risk_factors = agent.risk_factors.copy()
                
            env.add_agent(new_agent)
        
        # Randomize hazards
        for hazard_type, hazard_data in base_env.hazards.items():
            for pos, intensity in zip(hazard_data['positions'], hazard_data['intensities']):
                new_pos = (
                    pos[0] * random.uniform(0.8, 1.2),
                    pos[1] * random.uniform(0.8, 1.2)
                )
                new_intensity = intensity * random.uniform(0.7, 1.3)
                env.add_hazard(hazard_type, new_pos, new_intensity)
            
        return env
    
    def _run_single_scenario(self, env: EvacuationEnvironment, max_steps: int, 
                           step_callback=None) -> EvacuationScenario:
        """Run a single evacuation scenario.
        
        Args:
            env: The environment to run in
            max_steps: Maximum number of steps
            step_callback: Optional callback function that receives (env, step)
            
        Returns:
            EvacuationScenario with results
        """
        scenario = EvacuationScenario(
            scenario_id=0,  # Will be set by run_simulation
            agents=env.agents.copy(),
            hazards=[]  # Will be populated from env
        )
        
        # Convert hazards to list of dicts for the scenario
        for hazard_type, hazard_data in env.hazards.items():
            for pos, intensity in zip(hazard_data['positions'], hazard_data['intensities']):
                scenario.hazards.append({
                    'type': hazard_type,
                    'position': pos,
                    'intensity': intensity,
                    'radius': hazard_data.get('radius', 5.0)
                })
        
        # Run the simulation steps
        for step in range(max_steps):
            # Update agent behaviors and move them
            for agent in env.agents:
                if not hasattr(agent, 'evacuated') or not agent.evacuated:
                    # Agent decides next action
                    action = agent.decide_action(env)
                    agent.act(action, env)
            
            # Update environment
            env.step(time_step=0.1)
            
            # Call the step callback if provided
            if step_callback:
                if not step_callback(env, step):
                    break
            
            # Check if all agents have evacuated
            if all(hasattr(agent, 'evacuated') and agent.evacuated for agent in env.agents):
                break
                
        return scenario
    
    def _get_random_position(self, env: EvacuationEnvironment, min_dist: float = 5.0) -> Tuple[float, float]:
        """Get a random position in the environment, ensuring minimum distance from hazards and exits."""
        max_attempts = 100
        for _ in range(max_attempts):
            x = random.uniform(0, env.width)
            y = random.uniform(0, env.height)
            pos = np.array([x, y])
            
            # Check distance from hazards
            too_close = any(
                np.linalg.norm(pos - np.array(hazard_pos)) < min_dist
                for hazard_type, hazard_data in env.hazards.items()
                for hazard_pos in hazard_data['positions']
            )
            
            # Check distance from exits
            too_close |= any(
                np.linalg.norm(pos - np.array(exit_pos)) < min_dist
                for exit_pos in env.exits
            )
            
            if not too_close:
                return pos
                
        # If no valid position found after max attempts, return any position
        return np.array([random.uniform(0, env.width), random.uniform(0, env.height)])
    
    def _calculate_path_risk(self, path: List[Tuple[float, float]], env: EvacuationEnvironment) -> float:
        """Calculate total risk along a path."""
        if not path:
            return 1.0
            
        total_risk = 0.0
        for point in path:
            point_risk = 0.0
            for hazard_type, hazard_data in env.hazards.items():
                for hazard_pos, intensity in zip(hazard_data['positions'], hazard_data['intensities']):
                    dist = np.linalg.norm(np.array(point) - np.array(hazard_pos))
                    hazard_radius = 10.0  # Default hazard radius
                    if dist < hazard_radius:
                        point_risk += intensity * (1 - dist/hazard_radius)
            total_risk += min(point_risk, 1.0)
            
        return total_risk / len(path)
    
    def _estimate_crowd_density(self, path: List[Tuple[float, float]], agents: List[EvacuationAgent]) -> float:
        """Estimate crowd density along a path."""
        if not path or not agents:
            return 0.0
            
        density = 0.0
        detection_radius = 5.0
        
        for point in path:
            nearby_agents = sum(
                1 for agent in agents 
                if hasattr(agent, 'position') and 
                np.linalg.norm(np.array(point) - np.array(agent.position)) < detection_radius
            )
            density += min(nearby_agents / 10.0, 1.0)  # Normalize
            
        return density / len(path)
    
    def _find_paths_to_exits(self, start: Tuple[float, float], env: EvacuationEnvironment) -> List[PathOption]:
        """Find multiple paths from start to all exits using different strategies."""
        if not env.exits:
            return []
            
        paths = []
        
        # Try direct path to each exit
        for exit_pos in env.exits:
            # Simple straight-line path
            path = [start, exit_pos]
            risk = self._calculate_path_risk(path, env)
            paths.append(PathOption(
                path=path,
                risk_score=risk,
                distance=np.linalg.norm(np.array(start) - np.array(exit_pos)),
                crowd_density=0.0  # Will be updated later
            ))
            
        # Add some randomized paths
        for _ in range(min(self.sample_paths, 10)):  # Limit number of samples for performance
            intermediate_points = [
                (random.uniform(0, env.width), random.uniform(0, env.height))
                for _ in range(random.randint(1, 3))  # 1-3 intermediate points
            ]
            path = [start] + intermediate_points + [random.choice(env.exits)]
            risk = self._calculate_path_risk(path, env)
            paths.append(PathOption(
                path=path,
                risk_score=risk,
                distance=sum(
                    np.linalg.norm(np.array(path[i]) - np.array(path[i-1]))
                    for i in range(1, len(path))
                ),
                crowd_density=0.0
            ))
            
        return paths
    
    def _update_convergence_data(self, scenario: EvacuationScenario):
        """Update convergence metrics after each scenario."""
        if not self.convergence_data:
            self.convergence_data = [{
                'scenario': scenario.scenario_id,
                'success_rate': scenario.success_rate,
                'avg_evacuation_time': scenario.avg_evacuation_time,
                'risk_distribution': scenario.risk_distribution
            }]
        else:
            # Calculate running averages
            prev_avg = self.convergence_data[-1]
            new_avg = {
                'scenario': scenario.scenario_id,
                'success_rate': (prev_avg['success_rate'] * (len(self.convergence_data) - 1) + scenario.success_rate) / len(self.convergence_data),
                'avg_evacuation_time': (prev_avg['avg_evacuation_time'] * (len(self.convergence_data) - 1) + scenario.avg_evacuation_time) / len(self.convergence_data),
                'risk_distribution': {
                    'low': (prev_avg['risk_distribution']['low'] * (len(self.convergence_data) - 1) + scenario.risk_distribution['low']) / len(self.convergence_data),
                    'medium': (prev_avg['risk_distribution']['medium'] * (len(self.convergence_data) - 1) + scenario.risk_distribution['medium']) / len(self.convergence_data),
                    'high': (prev_avg['risk_distribution']['high'] * (len(self.convergence_data) - 1) + scenario.risk_distribution['high']) / len(self.convergence_data)
                }
            }
            self.convergence_data.append(new_avg)
    
    def get_optimal_paths(self) -> Dict[int, List[Tuple[float, float]]]:
        """Analyze scenarios to find optimal paths based on success rates."""
        if not self.scenarios:
            return {}
            
        # Find the most successful scenario
        best_scenario = max(self.scenarios, key=lambda x: x.success_rate)
        
        # Extract paths from the best scenario
        paths = {}
        for agent in best_scenario.agents:
            if hasattr(agent, 'path') and agent.path:
                paths[agent.agent_id] = agent.path
                
        return paths
    
    def get_risk_analysis(self) -> Dict[str, float]:
        """Analyze risk factors across all scenarios."""
        if not self.scenarios:
            return {}
            
        risk_analysis = {
            'total_simulations': len(self.scenarios),
            'avg_success_rate': np.mean([s.success_rate for s in self.scenarios]),
            'avg_evacuation_time': np.mean([s.avg_evacuation_time for s in self.scenarios if s.avg_evacuation_time < float('inf')] or [0]),
            'risk_distribution': {
                'low': np.mean([s.risk_distribution['low'] for s in self.scenarios]),
                'medium': np.mean([s.risk_distribution['medium'] for s in self.scenarios]),
                'high': np.mean([s.risk_distribution['high'] for s in self.scenarios])
            },
            'agent_type_risks': defaultdict(list),
            'hazard_impacts': defaultdict(list)
        }
        
        # Analyze agent type risks
        for scenario in self.scenarios:
            agents_by_type = defaultdict(list)
            for agent in scenario.agents:
                agents_by_type[agent.agent_type].append(agent)
                
            for agent_type, agents in agents_by_type.items():
                success_rate = (sum(1 for a in agents if hasattr(a, 'evacuated') and a.evacuated) / len(agents)) * 100
                risk_analysis['agent_type_risks'][agent_type].append(success_rate)
        
        # Calculate average success rates by agent type
        for agent_type in risk_analysis['agent_type_risks']:
            risk_analysis['agent_type_risks'][agent_type] = np.mean(risk_analysis['agent_type_risks'][agent_type])
            
        return risk_analysis
