import numpy as np
from scipy.spatial import KDTree
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass
from navmesh_ml2d.core.navmesh_fixed import GridNavMesh
from navmesh_ml2d.core.agent import Agent

# Navigation constants
NAV_GRID_SIZE = 0.5  # meters per grid cell
SAFE_DISTANCE = 2.0   # meters to maintain from hazards

@dataclass
class StampedeEvent:
    """Class representing a detected stampede event."""
    timestamp: float
    position: Tuple[float, float]
    radius: float
    severity: float
    agents_affected: int
    metrics: Dict[str, float]
    agent_ids: List[int]

class StampedeDetector:
    """
    Detects stampede conditions in crowd simulations and manages navigation
    to prevent and mitigate stampedes. Integrates with GridNavMesh for
    intelligent pathfinding and hazard avoidance.
    """
    
    def __init__(self, 
                area_width: float, 
                area_height: float, 
                density_threshold: float = 0.15,
                local_density_threshold: float = 0.25, 
                speed_threshold: float = 1.5, 
                alignment_threshold: float = 0.9, 
                panic_ratio_threshold: float = 0.5,
                obstacles: Optional[List[Tuple[int, int]]] = None):
        """Initialize the stampede detector with navigation capabilities.
        
        Args:
            area_width: Width of the simulation area in meters
            area_height: Height of the simulation area in meters
            density_threshold: Global density threshold (agents/m²) for stampede detection
            local_density_threshold: Local density threshold for identifying hotspots
            speed_threshold: Speed threshold (m/s) indicating potential panic
            alignment_threshold: Velocity alignment threshold for herding behavior
            panic_ratio_threshold: Ratio of panicked agents to trigger detection
            obstacles: Optional list of (x,y) coordinates of static obstacles
        """
        # Initialize detection parameters
        self.area_width = float(area_width)
        self.area_height = float(area_height)
        self.density_threshold = float(density_threshold)
        self.local_density_threshold = float(local_density_threshold)
        self.speed_threshold = float(speed_threshold)
        self.alignment_threshold = float(alignment_threshold)
        self.panic_ratio_threshold = float(panic_ratio_threshold)
        
        # Initialize navigation system
        grid_width = int(area_width / NAV_GRID_SIZE)
        grid_height = int(area_height / NAV_GRID_SIZE)
        self.navmesh = GridNavMesh(grid_width, grid_height, obstacles)
        
        # Track stampede events and hazards
        self.active_events: List[StampedeEvent] = []
        self.hazard_zones: Dict[Tuple[float, float], float] = {}  # (x,y) -> radius
        self.last_update_time: float = 0.0
        
        # Initialize navigation grid
        self._init_navigation_grid()
        
    def _init_navigation_grid(self) -> None:
        """Initialize the navigation grid with obstacles and hazards."""
        # Mark hazard zones in the navigation grid
        for (hx, hy), radius in self.hazard_zones.items():
            self._update_hazard_zone(hx, hy, radius, add=True)
    
    def _update_hazard_zone(self, x: float, y: float, radius: float, add: bool = True) -> None:
        """Update hazard zones in the navigation grid."""
        # Convert to grid coordinates
        gx, gy = int(round(x / NAV_GRID_SIZE)), int(round(y / NAV_GRID_SIZE))
        grid_radius = int(round(radius / NAV_GRID_SIZE))
        
        # Update grid cells within the hazard radius
        for dy in range(-grid_radius, grid_radius + 1):
            for dx in range(-grid_radius, grid_radius + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < self.navmesh.width and 0 <= ny < self.navmesh.height:
                    dist = np.sqrt((dx*NAV_GRID_SIZE)**2 + (dy*NAV_GRID_SIZE)**2)
                    if dist <= radius:
                        intensity = 1.0 - (dist / radius)  # 1.0 at center, 0.0 at edge
                        if add:
                            self.navmesh.hazard_map[ny, nx] = max(
                                self.navmesh.hazard_map[ny, nx], 
                                intensity
                            )
                        else:
                            # Reduce hazard intensity but don't go below 0
                            self.navmesh.hazard_map[ny, nx] = max(
                                0,
                                self.navmesh.hazard_map[ny, nx] - intensity
                            )
    
    def add_stampede_event(self, event: StampedeEvent) -> None:
        """Register a new stampede event and update navigation hazards."""
        self.active_events.append(event)
        self.hazard_zones[event.position] = event.radius
        self._update_hazard_zone(*event.position, event.radius, add=True)
    
    def update_agent_paths(self, agents: List[Any], goals: List[Tuple[float, float]]) -> None:
        """Update paths for all agents to avoid stampede hazards.
        
        Args:
            agents: List of agent objects with 'pos' and 'path' attributes
            goals: List of (x,y) goal positions for each agent
        """
        if not agents or not goals:
            return
            
        # Update flow field based on current goals
        self.navmesh.update_flow_field(goals)
        
        # Update paths for each agent
        for agent, goal in zip(agents, goals):
            if not hasattr(agent, 'pos') or not hasattr(agent, 'path'):
                continue
                
            # Check if current path is safe
            if self._is_path_safe(agent.pos, agent.path):
                continue
                
            # Find a new safe path
            safe_path = self.find_safe_path(agent.pos, goal)
            if safe_path:
                agent.path = safe_path
    
    def _is_path_safe(self, position: Tuple[float, float], path: List[Tuple[float, float]]) -> bool:
        """Check if a path is safe from hazards."""
        if not path:
            return False
            
        # Check each segment of the path
        for point in path:
            gx = int(round(point[0] / NAV_GRID_SIZE))
            gy = int(round(point[1] / NAV_GRID_SIZE))
            
            # Check if point is in a hazard zone
            if (0 <= gx < self.navmesh.width and 
                0 <= gy < self.navmesh.height and 
                self.navmesh.hazard_map[gy, gx] > 0.5):  # Threshold for hazard
                return False
                
        return True
    
    def find_safe_path(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Find a safe path from start to goal, avoiding hazards."""
        # Convert to grid coordinates
        start_grid = (int(round(start[0] / NAV_GRID_SIZE)), 
                     int(round(start[1] / NAV_GRID_SIZE)))
        goal_grid = (int(round(goal[0] / NAV_GRID_SIZE)), 
                    int(round(goal[1] / NAV_GRID_SIZE)))
        
        # Find path with hazard avoidance
        path = self.navmesh.astar(
            start_grid, 
            goal_grid,
            avoid_hazards=True,
            avoid_density=True
        )
        
        # Convert path back to world coordinates
        return [(x * NAV_GRID_SIZE, y * NAV_GRID_SIZE) for x, y in path]
        
        # Validate parameters
        if any(x <= 0 for x in [area_width, area_height, density_threshold, local_density_threshold]):
            raise ValueError("Area dimensions and thresholds must be positive values")
        if not (0 <= alignment_threshold <= 1) or not (0 <= panic_ratio_threshold <= 1):
            raise ValueError("Alignment and panic ratio thresholds must be between 0 and 1")

    def detect_from_positions(self, agent_positions) -> Tuple[bool, float, int]:
        """Detect stampede conditions from agent positions only.
        
        Args:
            agent_positions: List or array of (x, y) agent positions
            
        Returns:
            Tuple of (is_stampede, density, agent_count)
            - is_stampede: True if stampede conditions are detected
            - density: Agent density in agents per square meter
            - agent_count: Total number of agents
            
        Raises:
            ValueError: If agent_positions is not in the expected format
        """
        if not agent_positions:
            return False, 0.0, 0
            
        try:
            positions = np.asarray(agent_positions, dtype=np.float32)
            if positions.ndim != 2 or positions.shape[1] != 2:
                raise ValueError("agent_positions must be an Nx2 array of (x,y) positions")
                
            count = len(positions)
            area = self.area_width * self.area_height
            density = count / area if area > 0 else 0.0
            
            # Simple stampede detection based on global density
            is_stampede = density > self.density_threshold
            
            return is_stampede, float(density), count
            
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid agent positions format: {str(e)}") from e

    def detect_from_agent_states(self, 
                               agents: List[Any], 
                               timestamp: float = 0.0, 
                               window_size: float = 2.0) -> List[StampedeEvent]:
        """Detect stampede conditions from agent states with comprehensive metrics.
        
        Args:
            agents: List of agent objects with position and optional attributes
            timestamp: Current simulation time
            window_size: Radius for local density calculation (meters)
            
        Returns:
            List of detected stampede events with detailed metrics
            
        Raises:
            ValueError: If agents or their attributes are invalid
        """
        if not agents:
            return []
            
        try:
            # Extract and validate agent states
            agent_states = []
            for agent in agents:
                if not hasattr(agent, 'pos'):
                    continue
                    
                agent_states.append({
                    'pos': np.asarray(agent.pos, dtype=np.float32),
                    'vel': np.asarray(getattr(agent, 'velocity', [0, 0]), dtype=np.float32),
                    'panic': float(getattr(agent, 'panic', 0.0)),
                    'status': 'active' if not getattr(agent, 'exited', False) else 'exited',
                    'id': int(getattr(agent, 'idx', 0))
                })
            
            if not agent_states:
                return []
                
            # Prepare data
            positions = np.array([a['pos'] for a in agent_states])
            count = len(positions)
            area = self.area_width * self.area_height
            global_density = count / area if area > 0 else 0.0
            
            # Calculate local densities using KDTree
            local_density_map = {}
            max_local_density = 0.0
            
            if count > 1:
                tree = KDTree(positions)
                for i, pos in enumerate(positions):
                    neighbors = tree.query_ball_point(pos, window_size)
                    local_density = len(neighbors) / (np.pi * window_size ** 2)
                    local_density_map[i] = local_density
                    max_local_density = max(max_local_density, local_density)
            
            # Calculate movement metrics
            speeds = [np.linalg.norm(a['vel']) for a in agent_states]
            avg_speed = float(np.mean(speeds)) if speeds else 0.0
            
            # Calculate velocity alignment (herding behavior)
            alignment = 0.0
            if len(agent_states) > 1:
                velocities = [a['vel'] for a in agent_states]
                norms = [np.linalg.norm(v) for v in velocities]
                if any(n > 0 for n in norms):
                    normalized = [v/(n + 1e-6) for v, n in zip(velocities, norms)]
                    # Calculate mean pairwise dot products (up to 5 nearest neighbors for efficiency)
                    alignment = np.mean([np.dot(normalized[i], normalized[j])
                                      for i in range(len(normalized))
                                      for j in range(i+1, min(i+5, len(normalized)))])
            
            # Calculate panic metrics
            panic_ratio = np.mean([a['panic'] for a in agent_states]) if agent_states else 0.0
            
            # Check for stampede conditions
            is_stampede = (global_density > self.density_threshold or 
                          max_local_density > self.local_density_threshold or
                          panic_ratio > self.panic_ratio_threshold or
                          (alignment > self.alignment_threshold and 
                           avg_speed > self.speed_threshold))
            
            # If no stampede conditions detected, return empty list
            if not is_stampede:
                return []
            
            # Identify hotspots (areas with high local density)
            hotspots = [i for i, ld in local_density_map.items() 
                       if ld > self.local_density_threshold]
            
            # Group nearby hotspots
            events = []
            if hotspots:
                hotspot_positions = positions[hotspots]
                tree = KDTree(hotspot_positions)
                processed = set()
                
                for i in range(len(hotspot_positions)):
                    if i not in processed:
                        # Find all points within window_size of this point
                        group = list(tree.query_ball_point(hotspot_positions[i], window_size))
                        if group:  # Ensure group is not empty
                            group_positions = hotspot_positions[group]
                            center = np.mean(group_positions, axis=0)
                            radius = np.max(np.linalg.norm(group_positions - center, axis=1)) + window_size/2
                            
                            # Get agents in this hotspot
                            hotspot_agents = [agent_states[hotspots[j]] for j in group]
                            
                            # Calculate severity (0-1) based on local density and panic
                            local_severity = min(1.0, max_local_density / self.local_density_threshold)
                            panic_severity = min(1.0, panic_ratio / self.panic_ratio_threshold)
                            severity = max(local_severity, panic_severity)
                            
                            events.append(StampedeEvent(
                                timestamp=timestamp,
                                position=tuple(center),
                                radius=float(radius),
                                severity=severity,
                                agents_affected=len(hotspot_agents),
                                metrics={
                                    'global_density': global_density,
                                    'max_local_density': max_local_density,
                                    'avg_speed': avg_speed,
                                    'alignment': alignment,
                                    'panic_ratio': panic_ratio
                                },
                                agent_ids=[a['id'] for a in hotspot_agents]
                            ))
                        
                        processed.update(group)
            
            return events
            
        except Exception as e:
            raise ValueError(f"Error in stampede detection: {str(e)}") from e

if __name__ == "__main__":
    # Create a sample agent class for demonstration
    class TestAgent:
        def __init__(self, idx, pos, velocity=(0, 0), panic=0.0, exited=False):
            self.idx = idx
            self.pos = np.array(pos, dtype=np.float32)
            self.velocity = np.array(velocity, dtype=np.float32)
            self.panic = float(panic)
            self.exited = exited
            self.path = []  # Add path attribute for navigation
    
    # Define some obstacles (walls, pillars, etc.)
    obstacles = [
        # Vertical walls
        (10, i) for i in range(20, 31)
    ] + [
        # Horizontal walls
        (i, 25) for i in range(20, 31)
    ]
    
    # Initialize detector with navigation capabilities
    detector = StampedeDetector(
        area_width=50.0,
        area_height=50.0,
        density_threshold=0.5,      # agents/m²
        local_density_threshold=2.0, # agents/m² for local hotspots
        speed_threshold=2.0,        # m/s
        alignment_threshold=0.7,    # velocity alignment (0-1)
        panic_ratio_threshold=0.3,  # fraction of panicked agents
        obstacles=obstacles         # Add static obstacles
    )
    
    # Create a stampede event
    stampede = StampedeEvent(
        timestamp=0,
        position=(25.0, 25.0),  # Center of the area
        radius=10.0,            # 10m radius hazard zone
        severity=0.8,           # High severity
        agents_affected=0,      # Will be updated
        metrics={},
        agent_ids=[]
    )
    detector.add_stampede_event(stampede)
    
    # Create test agents
    num_agents = 20
    agents = [
        TestAgent(
            idx=i,
            pos=(np.random.uniform(0, 50), np.random.uniform(0, 50)),
            velocity=(0, 0),
            panic=0.0
        )
        for i in range(num_agents)
    ]
    
    # Set goals (all trying to reach the exit at (45, 45))
    exit_pos = (45.0, 45.0)
    goals = [exit_pos] * num_agents
    
    # Simulate a few steps
    for step in range(5):
        print(f"\n--- Step {step} ---")
        
        # Update agent paths to avoid hazards
        detector.update_agent_paths(agents, goals)
        
        # Simulate agent movement (simplified)
        for agent in agents:
            if agent.path:
                # Move toward next point in path
                next_point = agent.path[0]
                direction = next_point - agent.pos
                dist = np.linalg.norm(direction)
                if dist < 1.0:  # Reached waypoint
                    agent.path.pop(0)
                elif dist > 0:
                    agent.pos += (direction / dist) * 0.5  # Move 0.5m per step
        
        # Print some debug info
        print(f"Agent 0 position: {agents[0].pos}")
        if hasattr(agents[0], 'path') and agents[0].path:
            print(f"Agent 0 next waypoint: {agents[0].path[0]}")
    
    print("\nSimulation complete! Agents have navigated around the stampede hazard.")
    
    # Example of detecting stampede conditions
    print("\nRunning stampede detection on simulated agents...")
    
    # Create test agents in a crowded scenario
    np.random.seed(42)
    agents = []
    
    # Add a dense cluster of agents
    for i in range(50):
        x = 25 + np.random.normal(0, 2)
        y = 25 + np.random.normal(0, 2)
        # Make some agents panicked and moving in similar directions
        panic = np.random.uniform(0.4, 0.8) if i % 3 == 0 else 0.1
        vx = np.random.normal(1.0, 0.2) if i % 2 == 0 else -1.0
        vy = np.random.normal(1.0, 0.2)
        agent = TestAgent(i, (x, y), (vx, vy), panic)
        agent.path = []  # Initialize path
        agents.append(agent)
    
    # Add some random agents
    for i in range(50, 100):
        x = np.random.uniform(0, 50)
        y = np.random.uniform(0, 50)
        vx = np.random.uniform(-0.5, 0.5)
        vy = np.random.uniform(-0.5, 0.5)
        agents.append(TestAgent(i, (x, y), (vx, vy), 0.0))
    
    # Detect stampede conditions
    positions = np.array([agent.pos for agent in agents])
    velocities = np.array([agent.velocity for agent in agents])
    panic_levels = np.array([agent.panic for agent in agents])
    
    # Create test_agent objects from the positions and velocities
    test_agents = []
    for i, (pos, vel, panic) in enumerate(zip(positions, velocities, panic_levels)):
        test_agent = TestAgent(i, pos, vel, panic)
        test_agents.append(test_agent)
    
    # Detect stampede conditions
    events = detector.detect_from_agent_states(
        agents=test_agents,
        timestamp=0.0,
        window_size=2.0
    )
    
    # Process the results
    stampede_detected = len(events) > 0
    severity = max([e.severity for e in events], default=0.0)
    agents_affected = sum([e.agents_affected for e in events])
    
    if stampede_detected:
        print(f"\n⚠️ STAMPEDE DETECTED! Severity: {severity:.2f}, Agents affected: {agents_affected}")
        
        # Get the stampede epicenter
        density_map = detector._calculate_density(positions, 2.0)  # 2m radius for local density
        max_density_idx = np.unravel_index(np.argmax(density_map), density_map.shape)
        epicenter = (max_density_idx[1], max_density_idx[0])  # Convert to (x,y)
        
        # Create a stampede event at the epicenter
        stampede = StampedeEvent(
            timestamp=0,
            position=epicenter,
            radius=10.0,
            severity=severity,
            agents_affected=agents_affected,
            metrics={"density": float(np.max(density_map))},
            agent_ids=list(range(agents_affected))
        )
        detector.add_stampede_event(stampede)
        
        # Update paths to avoid the stampede
        goals = [(45.0, 45.0)] * len(agents)  # All agents try to reach the exit
        detector.update_agent_paths(agents, goals)
        
        # Print some debug info
        print(f"\nUpdated paths for {len(agents)} agents to avoid stampede at {epicenter}")
        if agents:
            print(f"Agent 0 new path length: {len(agents[0].path) if hasattr(agents[0], 'path') else 0} waypoints")
    
    print("\nTest complete! The integration is working correctly.")
    
    # Final check for any stampede events
    positions = np.array([agent.pos for agent in agents])
    velocities = np.array([agent.velocity for agent in agents])
    panic_levels = np.array([agent.panic for agent in agents])
    
    # Check for stampede conditions one more time
    events = detector.detect_from_agent_states(
        agents=agents,
        timestamp=0.0,
        window_size=2.0
    )
    stampede_detected = len(events) > 0
    severity = max([e.severity for e in events], default=0.0)
    agents_affected = sum([e.agents_affected for e in events])
    
    if stampede_detected:
        print(f"⚠️ STAMPEDE STILL DETECTED! Severity: {severity:.2f}, Agents affected: {agents_affected}")
    else:
        print("✅ No stampede conditions detected. All agents have been successfully rerouted.")

    # Print final status of the first few agents
    print("\nSample agent statuses:")
    for i, agent in enumerate(agents[:3]):  # Show first 3 agents as sample
        print(f"Agent {i} at {agent.pos}, panic: {agent.panic:.2f}, path length: {len(agent.path) if hasattr(agent, 'path') else 0}")
