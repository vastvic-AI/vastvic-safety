import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import random

@dataclass
class EvacuationAgent:
    """Enhanced agent class with evacuation-specific behaviors and insurance attributes."""
    
    # Basic attributes
    agent_id: int
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    
    # Agent characteristics
    agent_type: str = "adult"  # adult, child, elderly, mobility_impaired
    panic_level: float = 0.0  # 0.0 (calm) to 1.0 (panicked)
    health: float = 1.0  # 0.0 (incapacitated) to 1.0 (healthy)
    
    # Group/family information
    group_id: Optional[int] = None
    group_members: List[int] = field(default_factory=list)  # IDs of group members
    
    # Navigation
    target_exit: Optional[int] = None
    path: List[Tuple[float, float]] = field(default_factory=list)
    
    # Insurance and risk assessment
    insurance_coverage: Dict[str, Any] = field(default_factory=dict)
    risk_factors: Dict[str, float] = field(default_factory=dict)
    
    # Movement parameters
    max_speed: float = 1.33  # m/s (average walking speed)
    acceleration: float = 0.5  # m/s²
    radius: float = 0.3  # m (physical size)
    
    def __post_init__(self):
        # Initialize insurance and risk factors
        self.initialize_insurance()
        self.assess_risk_factors()
        
        # Initialize risk factors based on agent type
        self.risk_factors = self._initialize_risk_factors()
    
    def assess_risk_factors(self):
        """Assess and update risk factors for the agent based on current state."""
        if not hasattr(self, 'risk_factors') or not self.risk_factors:
            self.risk_factors = self._initialize_risk_factors()
        
        # Update dynamic risk factors
        self.risk_factors['panic_risk'] = max(0.1, min(1.0, self.panic_level))
        
        # Health-based adjustments
        health_risk = 1.0 - self.health
        self.risk_factors['health_risk'] = max(0.1, min(1.0, health_risk))
        
        # Environmental risk (can be updated by environment)
        if not hasattr(self, 'environmental_risk'):
            self.environmental_risk = 0.0
        
        # Calculate overall risk score (weighted average of all risk factors)
        weights = {
            'panic_risk': 0.25,
            'mobility_risk': 0.25,
            'awareness_risk': 0.15,
            'health_risk': 0.2,
            'evacuation_risk': 0.15
        }
        
        # Calculate weighted risk
        weighted_sum = 0.0
        total_weight = 0.0
        
        for factor, weight in weights.items():
            if factor in self.risk_factors:
                weighted_sum += self.risk_factors[factor] * weight
                total_weight += weight
        
        # Add environmental risk (weighted separately)
        env_weight = 0.3
        self.overall_risk = (weighted_sum + self.environmental_risk * env_weight) / (total_weight + env_weight)
        self.overall_risk = max(0.0, min(1.0, self.overall_risk))  # Ensure 0-1 range
        
        # Update insurance factors based on new risk assessment
        self._update_insurance_factors()
        
        return self.risk_factors
    
    def _update_insurance_factors(self):
        """Update insurance-related factors based on current risk assessment."""
        if not hasattr(self, 'insurance_coverage'):
            return
            
        if not hasattr(self, 'insurance_factors'):
            self.insurance_factors = {}
        
        # Calculate insurance risk factors
        self.insurance_factors.update({
            'evacuation_risk': self.risk_factors.get('evacuation_risk', 0.5) * 0.7 + self.panic_level * 0.3,
            'injury_risk': self.risk_factors.get('health_risk', 0.3) * 0.8 + self.overall_risk * 0.2,
            'property_damage_risk': 0.3,  # Base property risk
            'liability_risk': 0.2,        # Base liability risk
            'business_interruption_risk': 0.4  # Base business risk
        })
        
        # Adjust based on agent type
        if self.agent_type in ['child', 'elderly', 'mobility_impaired']:
            self.insurance_factors['special_needs_risk'] = 0.7
        
        # Ensure all values are between 0 and 1
        for factor in self.insurance_factors:
            self.insurance_factors[factor] = min(1.0, max(0.0, self.insurance_factors[factor]))
        
        # Update insurance premium if coverage exists
        if hasattr(self, 'insurance_coverage') and self.insurance_coverage.get('coverage_active', False):
            self.insurance_coverage['premium'] = self._calculate_premium()
    
    def _initialize_risk_factors(self):
        """Initialize risk factors based on agent type and characteristics."""
        risk_factors = {}
        
        # Base risk factors by agent type
        if self.agent_type == 'child':
            risk_factors.update({
                'panic_risk': 0.7,      # Higher panic risk
                'mobility_risk': 0.3,   # Lower mobility risk
                'awareness_risk': 0.6,  # Lower situational awareness
                'health_risk': 0.2,     # Generally healthy
                'evacuation_risk': 0.4  # May need assistance
            })
        elif self.agent_type == 'elderly':
            risk_factors.update({
                'panic_risk': 0.4,
                'mobility_risk': 0.7,
                'awareness_risk': 0.5,
                'health_risk': 0.8,     # Higher health risks
                'evacuation_risk': 0.6  # May need assistance
            })
        elif self.agent_type == 'mobility_impaired':
            risk_factors.update({
                'panic_risk': 0.6,
                'mobility_risk': 0.9,   # Highest mobility risk
                'awareness_risk': 0.4,
                'health_risk': 0.5,
                'evacuation_risk': 0.8  # High need for assistance
            })
        else:  # adult
            risk_factors.update({
                'panic_risk': 0.3,
                'mobility_risk': 0.1,   # Most mobile
                'awareness_risk': 0.3,
                'health_risk': 0.2,
                'evacuation_risk': 0.2  # Most capable
            })
        
        # Add some random variation (±10%)
        for factor in risk_factors:
            risk_factors[factor] = min(1.0, max(0.1, risk_factors[factor] * (0.9 + 0.2 * random.random())))
            
        return risk_factors
    
    def initialize_insurance(self):
        """Initialize insurance coverage based on agent type and demographics."""
        # Base coverage amounts by agent type
        coverages = {
            'adult': {
                'personal_accident': 100000,  # $100,000
                'medical_expenses': 50000,    # $50,000
                'evacuation_costs': 25000,    # $25,000
                'temporary_accommodation': 10000,  # $10,000
                'coverage_active': np.random.random() > 0.3,  # 70% chance of having insurance
                'deductible': 1000,
                'coverage_limits': {
                    'per_incident': 1000000,
                    'annual': 2000000
                }
            },
            'child': {
                'personal_accident': 150000,
                'medical_expenses': 75000,
                'evacuation_costs': 30000,
                'temporary_accommodation': 15000,
                'coverage_active': True,  # Children are typically covered under family plans
                'deductible': 500,
                'coverage_limits': {
                    'per_incident': 1000000,
                    'annual': 2000000
                }
            },
            'elderly': {
                'personal_accident': 75000,
                'medical_expenses': 100000,  # Higher medical coverage for elderly
                'evacuation_costs': 20000,
                'temporary_accommodation': 15000,
                'coverage_active': np.random.random() > 0.2,  # 80% chance of having insurance
                'deductible': 2000,
                'coverage_limits': {
                    'per_incident': 1500000,
                    'annual': 3000000
                }
            },
            'mobility_impaired': {
                'personal_accident': 125000,
                'medical_expenses': 100000,
                'evacuation_costs': 50000,
                'temporary_accommodation': 25000,
                'coverage_active': np.random.random() > 0.5,  # 50% chance of having insurance
                'deductible': 1500,
                'coverage_limits': {
                    'per_incident': 1250000,
                    'annual': 2500000
                }
            }
        }
        
        # Get coverage based on agent type, default to adult if type not found
        self.insurance_coverage = coverages.get(self.agent_type, coverages['adult']).copy()
        
        # Add policy details
        self.insurance_coverage.update({
            'policy_number': f"POL-{np.random.randint(10000, 99999)}",
            'provider': random.choice(['SafeGuard Ins', 'Global Shield', 'Family First', 'SecureLife']),
            'premium': self._calculate_premium(),
            'claims_history': [],
            'policy_start_date': '2025-01-01',  # Placeholder
            'policy_end_date': '2026-01-01'     # Placeholder
        })
        
        # Initialize risk factors if not already set
        if not hasattr(self, 'risk_factors') or not self.risk_factors:
            self.risk_factors = self._initialize_risk_factors()
    
    def _calculate_premium(self) -> float:
        """Calculate insurance premium based on risk factors."""
        # Base premium by agent type
        base_rates = {
            'adult': 1000,
            'child': 800,
            'elderly': 1500,
            'mobility_impaired': 2000
        }
        
        # Get base rate
        base_premium = base_rates.get(self.agent_type, base_rates['adult'])
        
        # Adjust based on risk factors
        risk_adjustment = 1.0
        if hasattr(self, 'risk_factors') and self.risk_factors:
            # Calculate average risk factor (0-1)
            avg_risk = sum(self.risk_factors.values()) / len(self.risk_factors)
            # Adjust premium by ±50% based on risk
            risk_adjustment = 0.5 + avg_risk
        
        # Apply random variation (±10%)
        random_factor = 0.9 + 0.2 * random.random()
        
        return round(base_premium * risk_adjustment * random_factor, 2)
    
    def decide_action(self, env):
        """
        Decide the next action for the agent based on the current environment.
        
        Args:
            env: The current environment state
            
        Returns:
            dict: Action to take, containing 'movement' and 'target'
        """
        # If agent has already evacuated, do nothing
        if hasattr(self, 'evacuated') and self.evacuated:
            return {'movement': 'stay', 'target': None}
            
        # If no path, find a path to the nearest exit
        if not self.path:
            self._find_path_to_exit(env)
            
        # If we have a path, move along it
        if self.path:
            next_pos = self.path[0]
            direction = next_pos - self.position
            distance = np.linalg.norm(direction)
            
            # If we're close to the next point, move to it and remove from path
            if distance < 0.5 and len(self.path) > 1:
                self.path.pop(0)
                next_pos = self.path[0]
                direction = next_pos - self.position
                distance = np.linalg.norm(direction)
                
            # Move towards the next point
            if distance > 0:
                direction = direction / distance  # Normalize
                return {
                    'movement': 'move',
                    'direction': direction,
                    'speed': min(self.max_speed, distance/0.1)  # Don't overshoot
                }
                
        # Default: stay in place
        return {'movement': 'stay', 'target': None}
        
    def act(self, action, env):
        """
        Execute the chosen action.
        
        Args:
            action: Dictionary containing action details
            env: The environment
        """
        if action['movement'] == 'move':
            # Update velocity based on desired direction and speed
            desired_velocity = action['direction'] * action['speed']
            self.velocity = desired_velocity  # Simple movement model
            
            # Update position based on velocity (with time step of 0.1s)
            new_position = self.position + self.velocity * 0.1
            
            # Check if we've reached an exit
            for exit_pos in env.exits:
                if np.linalg.norm(new_position - np.array(exit_pos)) < 1.0:  # Within 1m of exit
                    self.evacuated = True
                    self.evacuation_time = env.current_time
                    return
            
            # Update position if it's valid
            if 0 <= new_position[0] <= env.width and 0 <= new_position[1] <= env.height:
                self.position = new_position
        
        # Update health based on hazards
        self._update_health(env)
    
    def _update_health(self, env):
        """Update agent health based on nearby hazards."""
        for hazard_type, hazard_data in env.hazards.items():
            for pos, intensity in zip(hazard_data['positions'], hazard_data['intensities']):
                distance = np.linalg.norm(self.position - np.array(pos))
                if distance < hazard_data.get('radius', 5.0):
                    # Reduce health based on hazard intensity and distance
                    damage = intensity * (1 - distance/hazard_data.get('radius', 5.0)) * 0.01
                    self.health = max(0, self.health - damage)
    
    def _find_path_to_exit(self, env):
        """Find a path to the nearest exit."""
        if not env.exits:
            return
            
        # Simple path: move directly to the nearest exit
        nearest_exit = min(
            env.exits,
            key=lambda exit_pos: np.linalg.norm(self.position - np.array(exit_pos))
        )
        
        # Add some intermediate points for smoother movement
        self.path = [
            self.position + 0.25 * (np.array(nearest_exit) - self.position),
            self.position + 0.5 * (np.array(nearest_exit) - self.position),
            self.position + 0.75 * (np.array(nearest_exit) - self.position),
            np.array(nearest_exit)
        ]
    
    def update_panic(self, danger_level: float, time_step: float):
        """Update panic level based on danger and environment."""
        # Base panic increase based on danger
        panic_increase = danger_level * 0.1 * time_step
        
        # Adjust based on agent type and current panic
        if self.agent_type in ["child", "elderly"]:
            panic_increase *= 1.3  # More susceptible to panic
            
        # Apply panic increase with some randomness
        self.panic_level = np.clip(
            self.panic_level + panic_increase * (0.8 + 0.4 * random.random()),
            0.0, 1.0
        )
        
        # Panic affects movement speed
        self.max_speed = 1.33 * (1.0 + 0.5 * self.panic_level)  # Up to 50% faster when panicked
    
    def update_health(self, hazard_intensity: float, time_step: float):
        """Update health status based on hazards."""
        if hazard_intensity > 0:
            # Health decreases based on hazard intensity and exposure time
            health_decrease = hazard_intensity * 0.1 * time_step
            self.health = np.clip(self.health - health_decrease, 0.0, 1.0)
            
            # Update insurance risk factors
            if health_decrease > 0.1:
                self.insurance_coverage["medical_incident"] = True
    
    def calculate_insurance_claim(self):
        """Calculate potential insurance claim based on agent status."""
        if not self.insurance_coverage.get("coverage_active", False):
            return 0.0
            
        claim_amount = 0.0
        
        # Base claim for being in a dangerous situation
        claim_amount += 1000 * (1.0 - self.health)
        
        # Additional claims for specific incidents
        if self.insurance_coverage.get("medical_incident", False):
            claim_amount += min(
                self.insurance_coverage["medical_expenses"] * (1.0 - self.health),
                self.insurance_coverage["medical_expenses"]
            )
            
        if self.health < 0.3:  # Severe injury
            claim_amount += self.insurance_coverage["personal_accident"] * 0.3
            
        return claim_amount
    
    def step(self, time_step: float, environment):
        """Update agent state for one time step."""
        # Update panic and health
        danger_level = environment.get_danger_level(self.position)
        self.update_panic(danger_level, time_step)
        self.update_health(danger_level, time_step)
        
        # Update movement
        self.update_movement(time_step, environment)
    
    def update_movement(self, time_step: float, environment):
        """Update agent's position based on current state and environment."""
        if not self.path:
            # If no path, find a new target exit if needed
            if self.target_exit is None:
                self.select_best_exit(environment)
            else:
                # Try to find a new path to the target exit
                if environment.exits and self.target_exit < len(environment.exits):
                    self.path = environment.find_path(self.position, environment.exits[self.target_exit])
            return
            
        # Get next waypoint
        target = np.array(self.path[0])
        direction = target - self.position
        distance = np.linalg.norm(direction)
        
        if distance < 0.1:  # Reached waypoint
            self.path.pop(0)
            return
            
        # Normalize direction and apply speed
        if distance > 0:
            direction = direction / distance
            
        # Adjust speed based on panic and health
        current_speed = self.max_speed * (0.7 + 0.3 * self.health)
        
        # Update velocity with acceleration
        target_velocity = direction * current_speed
        self.velocity = self.velocity * 0.8 + target_velocity * 0.2
        
        # Update position
        self.position += self.velocity * time_step
    
    def select_best_exit(self, environment):
        """Select the best exit based on distance, congestion, and safety."""
        if not environment.exits:
            return None
            
        best_exit = None
        best_score = -float('inf')
        best_path = []
        
        for exit_id, exit_pos in enumerate(environment.exits):
            # First try to find a path to this exit
            path = environment.find_path(self.position, exit_pos)
            if not path:  # Skip if no path found
                continue
                
            # Calculate path length (approximate)
            distance = 0
            prev_point = self.position
            for point in path[1:]:  # Skip the first point (current position)
                distance += np.linalg.norm(np.array(point) - np.array(prev_point))
                prev_point = point
            
            # Get congestion near exit (higher is worse)
            congestion = environment.get_congestion(exit_pos, radius=5.0)
            
            # Get safety level (inverse of danger, higher is better)
            danger = environment.get_danger_level(exit_pos)
            safety = 1.0 - danger
            
            # Calculate score (weighted combination of factors)
            # Higher score is better
            distance_score = 1.0 / (distance + 1e-6)  # Prefer shorter paths
            safety_score = safety  # Prefer safer paths
            congestion_score = 1.0 - congestion  # Prefer less congested paths
            
            # Weighted sum of factors
            score = (distance_score * 0.5 + 
                   safety_score * 0.3 + 
                   congestion_score * 0.2)
            
            if score > best_score:
                best_score = score
                best_exit = exit_id
                best_path = path
        
        self.target_exit = best_exit
        self.path = best_path  # Store the actual path found
        
        # If we couldn't find a path to any exit, try moving randomly
        if not self.path and best_exit is not None:
            # Try to find any reachable cell in the general direction of the exit
            exit_pos = environment.exits[best_exit]
            direction = np.array(exit_pos) - self.position
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
                # Try to move in the general direction of the exit
                target_pos = self.position + direction * 5.0  # Move 5 units toward exit
                # Find path to this intermediate target
                self.path = environment.find_path(self.position, target_pos.tolist())
        
        return best_exit
