#!/usr/bin/env python3
"""
Performance Benchmarking for Crowd Simulation

This script benchmarks the performance of the optimized crowd simulation
components, including pathfinding, spatial queries, and agent updates.
"""

import time
import numpy as np
import numba
from pathlib import Path
import argparse
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent))

from navmesh_ml2d.core.navmesh import GridNavMesh
from navmesh_ml2d.core.agent import Agent

class BenchmarkResult:
    def __init__(self, name: str, timings: List[float], params: Dict[str, Any]):
        self.name = name
        self.timings = np.array(timings)
        self.params = params
        self.mean = np.mean(self.timings)
        self.std = np.std(self.timings)
        self.min = np.min(self.timings)
        self.max = np.max(self.timings)
        self.p50 = np.percentile(self.timings, 50)
        self.p95 = np.percentile(self.timings, 95)
        self.p99 = np.percentile(self.timings, 99)

    def __str__(self):
        return (
            f"{self.name} (n={len(self.timings)}): "
            f"mean={self.mean*1000:.2f}ms ± {self.std*1000:.2f}ms, "
            f"min={self.min*1000:.2f}ms, max={self.max*1000:.2f}ms, "
            f"p50={self.p50*1000:.2f}ms, p95={self.p95*1000:.2f}ms, p99={self.p99*1000:.2f}ms"
        )

def benchmark_pathfinding(
    navmesh: GridNavMesh, 
    num_runs: int = 100,
    map_size: Tuple[int, int] = (100, 100),
    obstacle_density: float = 0.2
) -> BenchmarkResult:
    """Benchmark A* pathfinding performance."""
    width, height = map_size
    np.random.seed(42)
    
    # Generate random start and goal points
    starts = []
    goals = []
    for _ in range(num_runs):
        while True:
            start = (np.random.randint(0, width), np.random.randint(0, height))
            goal = (np.random.randint(0, width), np.random.randint(0, height))
            if (start != goal and 
                navmesh.is_walkable(*start) and 
                navmesh.is_walkable(*goal)):
                starts.append(start)
                goals.append(goal)
                break
    
    # Warmup
    for start, goal in zip(starts[:10], goals[:10]):
        navmesh.find_path(start, goal)
    
    # Benchmark
    timings = []
    for start, goal in tqdm(zip(starts, goals), total=num_runs, desc="Pathfinding"):
        start_time = time.perf_counter()
        navmesh.find_path(start, goal)
        timings.append(time.perf_counter() - start_time)
    
    return BenchmarkResult(
        "A* Pathfinding",
        timings,
        {"num_runs": num_runs, "map_size": map_size, "obstacle_density": obstacle_density}
    )

def benchmark_spatial_queries(
    navmesh: GridNavMesh,
    num_agents: int = 1000,
    num_queries: int = 1000,
    query_radius: float = 5.0
) -> BenchmarkResult:
    """Benchmark spatial query performance."""
    # Generate random agent positions
    walkable = np.argwhere(navmesh.grid == 0)
    agent_indices = np.random.choice(len(walkable), num_agents, replace=False)
    agents = [tuple(walkable[i]) for i in agent_indices]
    
    # Generate random query points
    query_points = []
    for _ in range(num_queries):
        idx = np.random.randint(0, len(walkable))
        query_points.append(tuple(walkable[idx]))
    
    # Warmup
    for point in query_points[:10]:
        navmesh.get_nearby_walkable(*point, query_radius)
    
    # Benchmark
    timings = []
    for x, y in tqdm(query_points, desc="Spatial Queries"):
        start_time = time.perf_counter()
        navmesh.get_nearby_walkable(x, y, query_radius)
        timings.append(time.perf_counter() - start_time)
    
    return BenchmarkResult(
        "Spatial Queries",
        timings,
        {"num_agents": num_agents, "num_queries": num_queries, "query_radius": query_radius}
    )

def benchmark_agent_updates(
    navmesh: GridNavMesh,
    num_agents: int = 1000,
    num_steps: int = 100
) -> BenchmarkResult:
    """Benchmark agent update performance."""
    # Create agents at random walkable positions
    walkable = np.argwhere(navmesh.grid == 0)
    agent_indices = np.random.choice(len(walkable), num_agents, replace=False)
    agents = [
        Agent(
            agent_id=i,
            pos=tuple(walkable[idx].tolist()),
            radius=0.3,
            max_speed=1.4,
            pref_speed=1.0,
            goal_pos=(navmesh.width-1, navmesh.height-1)
        )
        for i, idx in enumerate(agent_indices)
    ]
    
    # Warmup
    for agent in agents[:10]:
        agent.update(nearby_agents=[], density_map=navmesh.density_map, 
                    hazard_map=navmesh.hazard_map, exits=[(navmesh.width-1, navmesh.height-1)],
                    navmesh=navmesh, dt=0.1)
    
    # Benchmark
    timings = []
    for _ in tqdm(range(num_steps), desc="Agent Updates"):
        step_times = []
        for agent in agents:
            start_time = time.perf_counter()
            agent.update(
                nearby_agents=[a for a in agents if a is not agent and 
                             np.linalg.norm(np.array(agent.pos) - np.array(a.pos)) < 5.0],
                density_map=navmesh.density_map,
                hazard_map=navmesh.hazard_map,
                exits=[(navmesh.width-1, navmesh.height-1)],
                navmesh=navmesh,
                dt=0.1
            )
            step_times.append(time.perf_counter() - start_time)
        timings.extend(step_times)
    
    return BenchmarkResult(
        "Agent Updates",
        timings,
        {"num_agents": num_agents, "num_steps": num_steps}
    )

def plot_results(results: List[BenchmarkResult], output_dir: Path):
    """Plot benchmark results."""
    sns.set_theme(style="whitegrid")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot timing distributions
    plt.figure(figsize=(12, 6))
    data = []
    for result in results:
        for timing in result.timings * 1000:  # Convert to ms
            data.append({"Benchmark": result.name, "Time (ms)": timing})
    
    if data:
        df = pd.DataFrame(data)
        ax = sns.boxplot(x="Benchmark", y="Time (ms)", data=df, showfliers=False)
        ax.set_title("Benchmark Results (log scale)")
        ax.set_yscale("log")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "benchmark_results.png", dpi=300, bbox_inches="tight")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Benchmark crowd simulation components")
    parser.add_argument("--map-size", type=int, nargs=2, default=[100, 100],
                       help="Map dimensions (width height)")
    parser.add_argument("--obstacle-density", type=float, default=0.2,
                       help="Density of obstacles (0-1)")
    parser.add_argument("--num-agents", type=int, default=1000,
                       help="Number of agents for simulation")
    parser.add_argument("--num-runs", type=int, default=100,
                       help="Number of benchmark runs")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                       help="Output directory for results")
    args = parser.parse_args()
    
    # Create navmesh with obstacles
    width, height = args.map_size
    num_obstacles = int(width * height * args.obstacle_density)
    obstacles = [
        (np.random.randint(0, width), np.random.randint(0, height))
        for _ in range(num_obstacles)
    ]
    
    navmesh = GridNavMesh(width, height, obstacles)
    
    # Run benchmarks
    results = []
    
    # Pathfinding benchmark
    path_result = benchmark_pathfinding(
        navmesh, 
        num_runs=args.num_runs,
        map_size=(width, height),
        obstacle_density=args.obstacle_density
    )
    results.append(path_result)
    print(f"\n{path_result}")
    
    # Spatial queries benchmark
    spatial_result = benchmark_spatial_queries(
        navmesh,
        num_agents=args.num_agents,
        num_queries=args.num_runs
    )
    results.append(spatial_result)
    print(f"\n{spatial_result}")
    
    # Agent updates benchmark
    agent_result = benchmark_agent_updates(
        navmesh,
        num_agents=args.num_agents,
        num_steps=10
    )
    results.append(agent_result)
    print(f"\n{agent_result}")
    
    # Plot results
    output_dir = Path(args.output_dir)
    plot_results(results, output_dir)
    
    # Save raw results
    import json
    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump([{
            "name": r.name,
            "mean_ms": r.mean * 1000,
            "std_ms": r.std * 1000,
            "min_ms": r.min * 1000,
            "max_ms": r.max * 1000,
            "p50_ms": r.p50 * 1000,
            "p95_ms": r.p95 * 1000,
            "p99_ms": r.p99 * 1000,
            "params": r.params
        } for r in results], f, indent=2)

if __name__ == "__main__":
    import pandas as pd  # Import here to avoid circular imports
    main()
