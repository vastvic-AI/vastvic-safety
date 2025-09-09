"""Configuration settings for the evacuation simulation."""

from typing import Dict, Any, List, Tuple
import numpy as np

# Environment configuration
ENV_CONFIG = {
    'width': 100,                    # Width of the environment
    'height': 100,                   # Height of the environment
    'cell_size': 1.0,                # Size of each grid cell
    'time_step': 0.1,                # Simulation time step (seconds)
    'max_steps': 1000,               # Maximum number of simulation steps
    'obstacle_margin': 0.5,          # Margin around obstacles for collision detection
    'exit_width': 4.0,               # Width of exit areas
    'random_seed': 42,               # Random seed for reproducibility
}

# Agent configuration
AGENT_CONFIG = {
    'radius': 0.5,                   # Agent radius
    'max_speed': 1.4,                # Maximum speed (m/s) - normal walking speed
    'max_acceleration': 0.5,         # Maximum acceleration (m/s²)
    'preferred_speed': 1.2,          # Preferred walking speed (m/s)
    'neighbor_dist': 10.0,           # Maximum distance to consider neighbors
    'max_neighbors': 10,             # Maximum number of neighbors to consider
    'time_horizon': 5.0,             # Time horizon for collision avoidance (s)
    'time_horizon_obst': 2.0,        # Time horizon for obstacle avoidance (s)
    'agent_radius': 0.5,             # Radius of other agents for collision avoidance
    'obstacle_radius': 0.25,         # Radius of obstacles for collision avoidance
    'goal_weight': 2.0,              # Weight for goal attraction force
    'obstacle_weight': 1.0,          # Weight for obstacle avoidance force
    'social_weight': 2.0,            # Weight for social force between agents
    'panic_weight': 1.5,             # Weight for panic behavior
    'panic_threshold': 0.7,          # Panic level threshold for aggressive behavior
    'panic_increase_rate': 0.05,     # Rate at which panic increases in dangerous situations
    'panic_decrease_rate': 0.01,     # Rate at which panic decreases in safe conditions
    'health_decrease_rate': 0.1,     # Rate at which health decreases in hazards
    'health_increase_rate': 0.02,    # Rate at which health increases when safe
    'group_cohesion_weight': 0.5,    # Weight for group cohesion force
    'group_separation_weight': 0.7,  # Weight for group separation force
    'group_alignment_weight': 0.3,   # Weight for group alignment force
    'group_radius': 5.0,             # Radius for group detection
    'max_group_size': 5,             # Maximum size of a group
}

# Agent type configurations
AGENT_TYPES = {
    'adult': {
        'max_speed': 1.4,            # Maximum speed (m/s)
        'panic_threshold': 0.7,      # Higher threshold for panic
        'panic_increase_rate': 0.03, # Slower panic increase
        'health': 1.0,               # Full health
        'radius': 0.5,               # Standard size
        'mass': 70.0,                # kg
        'insurance_coverage': 0.8,    # 80% coverage
        'risk_tolerance': 0.3,       # Lower risk tolerance
    },
    'child': {
        'max_speed': 1.0,
        'panic_threshold': 0.5,      # Lower threshold for panic
        'panic_increase_rate': 0.08, # Faster panic increase
        'health': 1.0,
        'radius': 0.4,               # Smaller size
        'mass': 30.0,                # kg
        'insurance_coverage': 0.9,    # 90% coverage (often covered by parents)
        'risk_tolerance': 0.1,       # Very low risk tolerance
    },
    'elderly': {
        'max_speed': 0.8,
        'panic_threshold': 0.6,
        'panic_increase_rate': 0.06,
        'health': 0.8,               # Slightly reduced health
        'radius': 0.5,
        'mass': 65.0,                # kg
        'insurance_coverage': 0.95,   # 95% coverage (often good coverage)
        'risk_tolerance': 0.2,       # Low risk tolerance
    },
    'mobility_impaired': {
        'max_speed': 0.5,
        'panic_threshold': 0.4,      # Very low threshold for panic
        'panic_increase_rate': 0.1,  # Very fast panic increase
        'health': 0.6,               # Reduced health
        'radius': 0.6,               # Potentially larger for wheelchair
        'mass': 80.0,                # kg (including mobility device)
        'insurance_coverage': 0.99,   # 99% coverage (often fully covered)
        'risk_tolerance': 0.0,       # No risk tolerance
    }
}

# Hazard configurations
HAZARD_CONFIG = {
    'fire': {
        'radius': 5.0,               # Radius of effect
        'intensity': 0.8,            # Base intensity
        'spread_rate': 0.05,         # Rate of spread
        'decay_rate': 0.01,          # Rate of intensity decay
        'damage_rate': 0.2,          # Damage rate to agents per second
        'panic_effect': 0.3,         # Panic increase rate when in hazard
    },
    'smoke': {
        'radius': 8.0,
        'intensity': 0.5,
        'spread_rate': 0.1,
        'decay_rate': 0.005,
        'damage_rate': 0.05,         # Lower damage than fire
        'panic_effect': 0.2,
        'visibility_reduction': 0.7,  # Percentage reduction in visibility
    },
    'blockage': {
        'radius': 3.0,
        'intensity': 1.0,            # Blockage is binary (blocked or not)
        'spread_rate': 0.0,          # Doesn't spread
        'decay_rate': 0.0,           # Doesn't decay
        'damage_rate': 0.0,          # No direct damage
        'panic_effect': 0.1,         # Slight panic increase
    }
}

# Insurance configuration
INSURANCE_CONFIG = {
    'base_premium': 1000.0,          # Base premium amount
    'claim_multipliers': {
        'injury': 5000.0,            # Per injury claim
        'fatal': 100000.0,           # Per fatality claim
        'property_damage': 1000.0,   # Per unit of property damage
    },
    'risk_factors': {
        'hazard_exposure': 0.5,      # Weight for hazard exposure in risk score
        'congestion': 0.3,           # Weight for congestion in risk score
        'panic_level': 0.2,          # Weight for average panic in risk score
    },
    'coverage_limits': {
        'medical': 1000000.0,        # Maximum medical coverage per person
        'property': 500000.0,        # Maximum property damage coverage
        'liability': 2000000.0,      # Maximum liability coverage
    },
    'deductibles': {
        'medical': 1000.0,           # Medical deductible
        'property': 2500.0,          # Property damage deductible
        'liability': 5000.0,         # Liability deductible
    }
}

def get_agent_config(agent_type: str) -> Dict[str, Any]:
    """Get configuration for a specific agent type with defaults.
    
    Args:
        agent_type: Type of agent ('adult', 'child', etc.)
        
    Returns:
        Dictionary of configuration parameters for the agent type
    """
    # Start with base agent config
    config = AGENT_CONFIG.copy()
    
    # Update with agent type specific config
    if agent_type in AGENT_TYPES:
        config.update(AGENT_TYPES[agent_type])
    else:
        # Default to adult if type not found
        config.update(AGENT_TYPES['adult'])
    
    return config

def get_hazard_config(hazard_type: str) -> Dict[str, Any]:
    """Get configuration for a specific hazard type.
    
    Args:
        hazard_type: Type of hazard ('fire', 'smoke', 'blockage')
        
    Returns:
        Dictionary of configuration parameters for the hazard type
    """
    if hazard_type in HAZARD_CONFIG:
        return HAZARD_CONFIG[hazard_type].copy()
    else:
        raise ValueError(f"Unknown hazard type: {hazard_type}")

def get_insurance_claim_amount(claim_type: str) -> float:
    """Get the base claim amount for an insurance claim type.
    
    Args:
        claim_type: Type of claim ('injury', 'fatal', 'property_damage')
        
    Returns:
        Base claim amount
    """
    return INSURANCE_CONFIG['claim_multipliers'].get(claim_type, 0.0)

def get_insurance_premium(risk_score: float) -> float:
    """Calculate insurance premium based on risk score.
    
    Args:
        risk_score: Risk score (0.0 to 1.0)
        
    Returns:
        Insurance premium amount
    """
    base = INSURANCE_CONFIG['base_premium']
    # Premium increases with risk score (0% to 200% of base)
    return base * (1.0 + 2.0 * risk_score)

def get_evacuation_route(env, agent, exit_id=None):
    """Get an evacuation route for an agent.
    
    Args:
        env: The environment
        agent: The agent to get a route for
        exit_id: Optional specific exit to target
        
    Returns:
        List of waypoints to the exit
    """
    # Simple implementation - can be replaced with A* or other pathfinding
    if exit_id is None or exit_id >= len(env.exits):
        # Find nearest exit
        exit_id = np.argmin([
            np.linalg.norm(agent.position - np.array(exit_pos))
            for exit_pos in env.exits
        ])
    
    target_exit = np.array(env.exits[exit_id])
    
    # Simple direct path (could be improved with pathfinding)
    return [target_exit]
