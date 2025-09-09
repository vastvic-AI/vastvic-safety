"""Metrics collection and analysis for the evacuation simulation."""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class SimulationMetrics:
    """Class for collecting and analyzing simulation metrics."""
    
    def __init__(self):
        """Initialize the metrics collector."""
        self.metrics = {
            'timestep': [],
            'timestamp': [],
            'agents_total': [],
            'agents_evacuated': [],
            'agents_injured': [],
            'agents_panicked': [],
            'risk_score': [],
            'insurance_claims': [],
            'active_hazards': [],
            'evacuation_progress': [],
            'average_panic_level': [],
            'average_health': [],
            'congestion_level': [],
            'hazard_exposure': []
        }
        self.start_time = None
        self.agent_metrics = {}
    
    def start_timer(self):
        """Start the simulation timer."""
        self.start_time = datetime.now()
    
    def record_step(self, env, timestep: int):
        """Record metrics for the current simulation step.
        
        Args:
            env: The simulation environment
            timestep: Current simulation step
        """
        # Calculate metrics
        agents_evacuated = sum(1 for agent in env.agents if agent.evacuated)
        agents_injured = sum(1 for agent in env.agents if agent.health < 1.0)
        agents_panicked = sum(1 for agent in env.agents if agent.panic_level > 0.5)
        
        # Calculate average panic and health
        if env.agents:
            avg_panic = sum(agent.panic_level for agent in env.agents) / len(env.agents)
            avg_health = sum(agent.health for agent in env.agents) / len(env.agents)
        else:
            avg_panic = 0.0
            avg_health = 1.0
        
        # Calculate hazard exposure
        hazard_exposure = self._calculate_hazard_exposure(env)
        
        # Record metrics
        self.metrics['timestep'].append(timestep)
        self.metrics['timestamp'].append(datetime.now() - self.start_time)
        self.metrics['agents_total'].append(len(env.agents))
        self.metrics['agents_evacuated'].append(agents_evacuated)
        self.metrics['agents_injured'].append(agents_injured)
        self.metrics['agents_panicked'].append(agents_panicked)
        self.metrics['risk_score'].append(env.total_risk_score * 100)  # Scale to 0-100
        
        # Calculate insurance claims
        claims = env.calculate_insurance_claims()
        self.metrics['insurance_claims'].append(claims)
        
        # Count active hazards
        active_hazards = sum(len(positions) for hazard in env.hazards.values() 
                            for positions in [hazard['positions']] if positions)
        self.metrics['active_hazards'].append(active_hazards)
        
        # Calculate evacuation progress
        if env.initial_agent_count > 0:
            progress = (agents_evacuated / env.initial_agent_count) * 100
        else:
            progress = 0.0
        self.metrics['evacuation_progress'].append(progress)
        
        self.metrics['average_panic_level'].append(avg_panic)
        self.metrics['average_health'].append(avg_health)
        self.metrics['congestion_level'].append(env.calculate_congestion())
        self.metrics['hazard_exposure'].append(hazard_exposure)
        
        # Update agent-specific metrics
        self._update_agent_metrics(env, timestep)
    
    def _calculate_hazard_exposure(self, env) -> float:
        """Calculate the average hazard exposure for all agents."""
        if not env.agents:
            return 0.0
            
        total_exposure = 0.0
        for agent in env.agents:
            # Calculate hazard influence on this agent
            hazard_influence = 0.0
            for hazard_type, hazard_data in env.hazards.items():
                for pos, intensity in zip(hazard_data['positions'], 
                                       hazard_data['intensities']):
                    dist = np.linalg.norm(agent.position - np.array(pos))
                    if dist < hazard_data['radius']:
                        influence = (1 - dist / hazard_data['radius']) * intensity
                        hazard_influence = max(hazard_influence, influence)
            total_exposure += hazard_influence
        
        return total_exposure / len(env.agents)
    
    def _update_agent_metrics(self, env, timestep: int):
        """Update metrics for individual agents."""
        for agent in env.agents:
            if agent.agent_id not in self.agent_metrics:
                self.agent_metrics[agent.agent_id] = {
                    'timesteps': [],
                    'positions': [],
                    'velocities': [],
                    'panic_levels': [],
                    'health_levels': [],
                    'evacuation_time': None,
                    'injury_time': None,
                    'agent_type': agent.agent_type
                }
            
            # Record agent state
            self.agent_metrics[agent.agent_id]['timesteps'].append(timestep)
            self.agent_metrics[agent.agent_id]['positions'].append(agent.position.copy())
            self.agent_metrics[agent.agent_id]['velocities'].append(
                np.linalg.norm(agent.velocity))
            self.agent_metrics[agent.agent_id]['panic_levels'].append(agent.panic_level)
            self.agent_metrics[agent.agent_id]['health_levels'].append(agent.health)
            
            # Record evacuation time if applicable
            if agent.evacuated and self.agent_metrics[agent.agent_id]['evacuation_time'] is None:
                self.agent_metrics[agent.agent_id]['evacuation_time'] = timestep
            
            # Record injury time if applicable
            if agent.health < 1.0 and self.agent_metrics[agent.agent_id]['injury_time'] is None:
                self.agent_metrics[agent.agent_id]['injury_time'] = timestep
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Get all metrics as a pandas DataFrame."""
        return pd.DataFrame(self.metrics)
    
    def get_agent_metrics_dataframe(self) -> pd.DataFrame:
        """Get agent-specific metrics as a pandas DataFrame."""
        rows = []
        for agent_id, metrics in self.agent_metrics.items():
            if not metrics['timesteps']:
                continue
                
            row = {
                'agent_id': agent_id,
                'agent_type': metrics['agent_type'],
                'evacuated': metrics['evacuation_time'] is not None,
                'evacuation_time': metrics['evacuation_time'],
                'injured': metrics['injury_time'] is not None,
                'injury_time': metrics['injury_time'],
                'max_panic': max(metrics['panic_levels']) if metrics['panic_levels'] else 0,
                'min_health': min(metrics['health_levels']) if metrics['health_levels'] else 1.0,
                'average_speed': np.mean(metrics['velocities']) if metrics['velocities'] else 0,
                'total_distance': self._calculate_total_distance(metrics['positions'])
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _calculate_total_distance(self, positions: List[np.ndarray]) -> float:
        """Calculate total distance traveled from a list of positions."""
        if len(positions) < 2:
            return 0.0
            
        total_distance = 0.0
        for i in range(1, len(positions)):
            total_distance += np.linalg.norm(positions[i] - positions[i-1])
            
        return total_distance
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of the simulation."""
        if not self.metrics['timestep']:
            return {}
            
        df = self.get_metrics_dataframe()
        agent_df = self.get_agent_metrics_dataframe()
        
        if df.empty or agent_df.empty:
            return {}
        
        # Calculate statistics
        total_sim_time = self.metrics['timestamp'][-1]
        
        # Agent statistics
        agent_stats = {
            'total_agents': len(agent_df),
            'evacuated_agents': agent_df['evacuated'].sum(),
            'injured_agents': agent_df['injured'].sum(),
            'evacuation_rate': agent_df['evacuated'].mean() * 100,  # Percentage
            'injury_rate': agent_df['injured'].mean() * 100,  # Percentage
            'avg_evacuation_time': agent_df[agent_df['evacuation_time'].notna()]['evacuation_time'].mean(),
            'avg_injury_time': agent_df[agent_df['injury_time'].notna()]['injury_time'].mean(),
            'avg_max_panic': agent_df['max_panic'].mean(),
            'avg_min_health': agent_df['min_health'].mean(),
            'avg_distance_traveled': agent_df['total_distance'].mean()
        }
        
        # Simulation statistics
        sim_stats = {
            'total_simulation_time': total_sim_time.total_seconds(),
            'final_evacuation_progress': self.metrics['evacuation_progress'][-1],
            'max_risk_score': max(self.metrics['risk_score']),
            'total_insurance_claims': sum(self.metrics['insurance_claims']),
            'max_active_hazards': max(self.metrics['active_hazards']),
            'max_congestion': max(self.metrics['congestion_level']),
            'max_hazard_exposure': max(self.metrics['hazard_exposure'])
        }
        
        return {
            'agent_statistics': agent_stats,
            'simulation_statistics': sim_stats,
            'timestamp': datetime.now().isoformat(),
            'simulation_duration': total_sim_time.total_seconds()
        }
    
    def save_metrics(self, filename: str):
        """Save metrics to a CSV file.
        
        Args:
            filename: Output filename (should end with .csv)
        """
        df = self.get_metrics_dataframe()
        df.to_csv(filename, index=False)
        
        # Save agent metrics to a separate file
        if '.' in filename:
            agent_filename = filename.replace('.csv', '_agents.csv')
        else:
            agent_filename = f"{filename}_agents.csv"
            
        agent_df = self.get_agent_metrics_dataframe()
        agent_df.to_csv(agent_filename, index=False)
        
        # Save summary statistics
        summary = self.get_summary_statistics()
        if summary:
            import json
            summary_filename = filename.replace('.csv', '_summary.json') if '.csv' in filename else f"{filename}_summary.json"
            with open(summary_filename, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
