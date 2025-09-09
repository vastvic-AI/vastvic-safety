"""
Monte Carlo Crowd Egress Simulation (Locally Quickest, India-specific)
- 50m x 50m open area, 4 exits
- 80% adults, 20% children, profile diversity
- Each agent dynamically chooses best exit based on Locally Quickest algorithm
- India-specific speed-density curve
- Monte Carlo analysis (100 runs)
- Outputs: CSVs and charts for occupancy, flow, exit times
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from navmesh import GridNavMesh

# Configure plotting
plt.style.use('seaborn-v0_8')  # Use seaborn style if available, otherwise use default
sns.set_theme(style="whitegrid")  # Use whitegrid style from seaborn
plt.rcParams['figure.figsize'] = (10, 6)

# --- Parameters ---
AREA_SIZE = 50  # meters
GRID_RES = 0.5  # meters per cell
GRID_SIZE = int(AREA_SIZE / GRID_RES)
N_EXITS = 4
EXIT_WIDTH = int(2 / GRID_RES)  # 2m wide exits
N_AGENTS = 100
class SimulationConfig:
    """Configuration for Monte Carlo simulation runs"""
    def __init__(self):
        self.n_runs = 100
        self.max_steps = 1000
        self.confidence_level = 0.95
        self.output_dir = Path('simulation_results')
        self.scenarios = {
            'baseline': {'n_agents': 100, 'exit_block_prob': 0.0},
            'high_density': {'n_agents': 200, 'exit_block_prob': 0.1},
            'exit_blocked': {'n_agents': 100, 'exit_block_prob': 0.5},
        }

@dataclass
class SimulationMetrics:
    """Container for simulation metrics"""
    run_id: int
    scenario: str
    total_evacuation_time: float
    agents_evacuated: int
    avg_evac_time: float
    max_density: float
    exit_utilization: Dict[int, float]
    
    def to_dict(self):
        return asdict(self)

ADULT_FRAC = 0.8
ADULT_SPEED_MEAN = 1.4
ADULT_SPEED_STD = 0.1
ADULT_WIDTH = 0.5 / GRID_RES
CHILD_SPEED_MEAN = 1.0
CHILD_SPEED_STD = 0.1
CHILD_WIDTH = 0.3 / GRID_RES

np.random.seed(42)

# --- India-specific speed-density curve ---
def india_speed_density(density):
    if density < 2:
        return 1.0
    elif density < 5:
        return 0.7
    else:
        return 0.3

# --- Exit locations (sides) ---
def get_exits():
    margin = 0
    exits = [
        (GRID_SIZE//2, margin),                # Top center
        (GRID_SIZE//2, GRID_SIZE-1-margin),    # Bottom center
        (margin, GRID_SIZE//2),                # Left center
        (GRID_SIZE-1-margin, GRID_SIZE//2),    # Right center
    ]
    return exits

# --- Agent profile generator ---
def generate_profiles(n_agents):
    n_adults = int(n_agents * ADULT_FRAC)
    n_children = n_agents - n_adults
    profiles = []
    for _ in range(n_adults):
        speed = np.random.normal(ADULT_SPEED_MEAN, ADULT_SPEED_STD)
        width = ADULT_WIDTH
        profiles.append({'type': 'adult', 'speed': speed, 'width': width})
    for _ in range(n_children):
        speed = np.random.normal(CHILD_SPEED_MEAN, CHILD_SPEED_STD)
        width = CHILD_WIDTH
        profiles.append({'type': 'child', 'speed': speed, 'width': width})
    np.random.shuffle(profiles)
    return profiles

# --- Agent class ---
class MC_Agent:
    def __init__(self, idx, pos, profile):
        self.idx = idx
        self.pos = np.array(pos, dtype=np.float32)
        self.profile = profile
        self.speed = profile['speed']
        self.width = profile['width']
        self.goal = None
        self.path = []
        self.exited = False
        self.exit_time = None
        self.history = []

    def choose_exit(self, exits, density_map, queue_map, navmesh):
        best_time = float('inf')
        best_exit = None
        for exit_pos in exits:
            dist = np.linalg.norm(self.pos - np.array(exit_pos))
            travel_time = dist * GRID_RES / self.speed
            queue_len = queue_map.get(tuple(exit_pos), 0)
            local_density = density_map[int(exit_pos[1]), int(exit_pos[0])]
            speed_factor = india_speed_density(local_density)
            queue_time = queue_len * (self.width * GRID_RES) / (EXIT_WIDTH * speed_factor)
            total_time = travel_time + queue_time
            if total_time < best_time:
                best_time = total_time
                best_exit = exit_pos
        self.goal = best_exit
        # Plan path
        self.path = navmesh.astar(tuple(map(int, self.pos)), tuple(map(int, self.goal)))
        if not self.path or len(self.path) < 2:
            self.path = [tuple(map(int, self.pos)), tuple(map(int, self.goal))]

    def smooth_path(self, path, interp_points=5):
        # Linear interpolation between waypoints for smoothness
        if not path or len(path) < 2:
            return path
        smooth = []
        for i in range(len(path) - 1):
            p0 = np.array(path[i], dtype=np.float32)
            p1 = np.array(path[i+1], dtype=np.float32)
            for alpha in np.linspace(0, 1, interp_points, endpoint=False):
                smooth.append(tuple(p0 + (p1 - p0) * alpha))
        smooth.append(tuple(path[-1]))
        return smooth

    def move(self, density_map, navmesh, agents=None, lookahead=3, jitter=0.08):
        if self.exited:
            return
        # Always follow the original A* path (no smoothing/lookahead for navigation)
        if self.path and len(self.path) > 1:
            next_pos = np.array(self.path[1], dtype=np.float32)
            direction = next_pos - self.pos
            # Social force: repulsion from nearby agents
            if agents is not None:
                repulse = np.zeros(2)
                for other in agents:
                    if other is self or not hasattr(other, 'pos'):
                        continue
                    op = other.pos if isinstance(other.pos, np.ndarray) else np.array(other.pos)
                    dist = np.linalg.norm(self.pos - op)
                    if 0 < dist < 2.0:
                        repulse += (self.pos - op) / (dist**2 + 1e-4)
                direction = direction + 1.5 * repulse
            # Add jitter
            direction += np.random.uniform(-jitter, jitter, size=2)
            norm = np.linalg.norm(direction)
            if norm > 0:
                step = min(self.speed, norm)
                # Slow down in high density
                local_density = density_map[int(self.pos[1]), int(self.pos[0])]
                speed_factor = india_speed_density(local_density)
                step *= speed_factor
                prev_pos = np.copy(self.pos)
                self.pos += (direction / norm) * step
                # Constrain to walkable area
                if not navmesh.is_walkable(self.pos[0], self.pos[1]):
                    self.pos = prev_pos  # revert to last valid position
                # Only proceed to next waypoint when close enough
                if np.linalg.norm(self.pos - next_pos) < 0.2:
                    self.path = self.path[1:]
        self.history.append(tuple(self.pos))

    def check_exit(self, exits, t):
        for exit_pos in exits:
            if np.linalg.norm(self.pos - np.array(exit_pos)) < 1.0:
                self.exited = True
                self.exit_time = t
                return True
        return False

# --- Main Monte Carlo Simulation ---
def run_mc_sim(run_idx: int, scenario: str, config: SimulationConfig) -> SimulationMetrics:
    """Run a single Monte Carlo simulation with given scenario
    
    Args:
        run_idx: Unique identifier for this run
        scenario: Scenario name from config
        config: Simulation configuration
        
    Returns:
        SimulationMetrics: Collected metrics for this run
    """
    params = config.scenarios[scenario]
    n_agents = params['n_agents']
    max_steps = config.max_steps
    
    # Set random seed for reproducibility
    np.random.seed(run_idx)
    
    # Initialize navigation and environment
    navmesh = GridNavMesh(GRID_SIZE, GRID_SIZE)
    exits = get_exits()
    
    # Block exits based on scenario
    if np.random.random() < params['exit_block_prob']:
        exits = exits[:-1]  # Block one exit
    
    # Generate agent profiles with realistic parameters
    profiles = generate_profiles(n_agents)
    agents = []
    
    # Position agents with minimum spacing
    positions = set()
    min_distance = 2.0  # meters
    grid_spacing = max(1, int(min_distance / GRID_RES))
    
    for i, profile in enumerate(profiles):
        tries = 0
        while tries < 100:
            x = np.random.randint(5, GRID_SIZE-5)
            y = np.random.randint(5, GRID_SIZE-5)
            if all(np.linalg.norm(np.array([x, y])-np.array(e)) > 5 for e in exits):
                break
            tries += 1
            if tries > 100:
                break
        agents.append(MC_Agent(i, (x, y), profile))
    # Initialize metrics collection
    metrics = {
        'step_agents': [],
        'exit_flow': {i: [] for i in range(len(exits))},
        'density_maps': [],
        'exit_times': [],
        'stampede_events': []
    }
    
    # Initialize stampede detection if available
    try:
        from stampede_detection import StampedeDetector
        stampede_detector = StampedeDetector(area_width=AREA_SIZE, area_height=AREA_SIZE)
    except ImportError:
        stampede_detector = None
    # Main simulation loop
    for t in range(max_steps):
        # Update density map
        density_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for agent in agents:
            if not agent.exited:
                x, y = int(agent.pos[0]), int(agent.pos[1])
                if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                    density_map[y, x] += 1
        
        # Track metrics for this timestep
        metrics['step_agents'].append(len([a for a in agents if not a.exited]))
        metrics['density_maps'].append(density_map.copy())
        
        # Initialize queue map for exit flows
        queue_map = defaultdict(int)
        exit_flow = defaultdict(int)
        
        # Calculate queue lengths at exits
        for agent in agents:
            if not agent.exited:
                for exit_pos in exits:
                    if np.linalg.norm(agent.pos - np.array(exit_pos)) < EXIT_WIDTH:
                        queue_map[tuple(exit_pos)] += 1
        
        # Update agent states and handle exits
        for agent in agents:
            if not agent.exited:
                # Choose exit based on current conditions
                agent.choose_exit(exits, density_map, queue_map, navmesh)
                
                # Move agent with collision avoidance
                agent.move(density_map, navmesh, agents)
                
                # Check if agent reached an exit
                for exit_idx, exit_pos in enumerate(exits):
                    if np.linalg.norm(agent.pos - np.array(exit_pos)) < 1.0:
                        agent.exited = True
                        agent.exit_time = t
                        exit_flow[exit_idx] += 1
                        metrics['exit_times'].append({
                            'agent_id': agent.idx,
                            'exit_time': t,
                            'exit': exit_idx,
                            'scenario': scenario
                        })
                        break
        
        # Record exit flows for this timestep
        for exit_idx in range(len(exits)):
            metrics['exit_flow'][exit_idx].append(exit_flow.get(exit_idx, 0))
        
        # Detect stampede conditions if detector is available
        if stampede_detector:
            stampede_events = stampede_detector.detect([
                a for a in agents if not a.exited
            ], t)
            if stampede_events:
                metrics['stampede_events'].extend(stampede_events)
        
        # Check for simulation completion conditions
        if all(agent.exited for agent in agents):
            break
            
        # Early termination if no progress
        if t > 100 and len(metrics['exit_times']) == 0:
            break
    # Create output directory for this run
    run_dir = config.output_dir / f'run_{run_idx}'
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed metrics
    metrics_df = pd.DataFrame({
        'step': range(len(metrics['step_agents'])),
        'agents_remaining': metrics['step_agents']
    })
    
    # Add exit flows
    for exit_idx in range(len(exits)):
        metrics_df[f'exit_{exit_idx}_flow'] = metrics['exit_flow'][exit_idx]
    
    # Save metrics to CSV
    metrics_df.to_csv(run_dir / 'metrics.csv', index=False)
    
    # Save exit times
    if metrics['exit_times']:
        exit_times_df = pd.DataFrame(metrics['exit_times'])
        exit_times_df.to_csv(run_dir / 'exit_times.csv', index=False)
    
    # Save density maps (as numpy arrays)
    if metrics['density_maps']:
        np.save(run_dir / 'density_maps.npy', np.array(metrics['density_maps']))
    
    # Generate and save visualizations
    plt.figure(figsize=(12, 6))
    plt.plot(metrics['step_agents'])
    plt.title('Agents Remaining Over Time')
    plt.xlabel('Step')
    plt.ylabel('Number of Agents')
    plt.tight_layout()
    plt.savefig(run_dir / 'agents_remaining.png')
    plt.close()
    
    # Plot exit flows
    plt.figure(figsize=(12, 6))
    for exit_idx in range(len(exits)):
        plt.plot(metrics['exit_flow'][exit_idx], label=f'Exit {exit_idx}')
    plt.title('Exit Flow Rates')
    plt.xlabel('Step')
    plt.ylabel('Agents Exiting')
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / 'exit_flows.png')
    plt.close()

    # Save agent paths for visualization
    with open(run_dir / 'paths.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['agent', 'x', 'y', 'step', 'exited'])
        for agent in agents:
            for step, pos in enumerate(agent.history):
                writer.writerow([agent.idx, pos[0], pos[1], step, agent.exited and agent.exit_time == step])
    # Calculate final metrics
    total_agents = len(agents)
    evacuated_agents = len([a for a in agents if a.exited])
    
    # Calculate evacuation times
    if metrics['exit_times']:
        exit_times = [et['exit_time'] for et in metrics['exit_times']]
        avg_evac_time = sum(exit_times) / len(exit_times)
        max_evac_time = max(exit_times) if exit_times else 0
    else:
        avg_evac_time = float('inf')
        max_evac_time = 0
    
    # Calculate exit utilization
    exit_counts = {i: sum(1 for t in metrics['exit_times'] if t['exit'] == i) 
                  for i in range(len(exits))}
    exit_utilization = {i: count / evacuated_agents if evacuated_agents > 0 else 0 
                       for i, count in exit_counts.items()}
    
    # Calculate max density across all steps
    max_density = max([np.max(dm) for dm in metrics['density_maps']] or [0])
    
    # Create and return metrics object
    return SimulationMetrics(
        run_id=run_idx,
        scenario=scenario,
        total_evacuation_time=max_evac_time,
        agents_evacuated=evacuated_agents,
        avg_evac_time=avg_evac_time,
        max_density=float(max_density),
        exit_utilization=exit_utilization
    )

def analyze_results(metrics: List[SimulationMetrics], config: SimulationConfig):
    """Analyze and visualize simulation results"""
    # Create output directory
    config.output_dir.mkdir(exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame([m.to_dict() for m in metrics])
    
    # Calculate statistics
    stats = df.groupby('scenario').agg({
        'total_evacuation_time': ['mean', 'std', 'min', 'max'],
        'avg_evac_time': ['mean', 'std'],
        'agents_evacuated': 'mean'
    }).round(2)
    
    # Save results
    stats.to_csv(config.output_dir / 'simulation_statistics.csv')
    
    # Plot evacuation times
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='scenario', y='total_evacuation_time', data=df)
    plt.title('Evacuation Time by Scenario')
    plt.tight_layout()
    plt.savefig(config.output_dir / 'evac_times.png')
    plt.close()
    
    # Generate report
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'total_runs': len(metrics),
        'scenarios': list(df['scenario'].unique()),
        'statistics': stats.to_dict()
    }
    
    with open(config.output_dir / 'report.json', 'w') as f:
        json.dump(report, f, indent=2)

def main():
    config = SimulationConfig()
    all_metrics = []
    
    # Run simulations for each scenario
    for scenario in config.scenarios:
        print(f"\n=== Running scenario: {scenario} ===")
        for run in range(config.n_runs):
            if run % 10 == 0:
                print(f"Run {run+1}/{config.n_runs}")
            metrics = run_mc_sim(run, scenario, config)
            all_metrics.append(metrics)
    
    # Analyze and save results
    analyze_results(all_metrics, config)
    print("\nSimulation complete! Results saved to:", config.output_dir.absolute())

if __name__ == '__main__':
    main()
