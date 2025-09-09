import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from collections import deque
import heapq

class AStarPathfinder:
    def __init__(self, grid):
        self.grid = grid
        self.rows, self.cols = grid.shape
        
    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(self, pos):
        row, col = pos
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-connected grid
            r, c = row + dr, col + dc
            if 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r, c] == 0:
                neighbors.append((r, c))
        return neighbors
    
    def find_path(self, start, goal):
        if self.grid[goal] != 0:  # Goal is not walkable
            return []
            
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
                
            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    if neighbor not in [i[1] for i in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        
        return []  # No path found

class Agent:
    def __init__(self, pos, agent_id):
        self.pos = pos
        self.id = agent_id
        self.path = []
        self.exited = False
        self.exit_time = None
        
    def update_path(self, pathfinder, exits):
        if self.exited or self.path:
            return
            
        # Find closest exit
        min_dist = float('inf')
        best_path = []
        
        for exit_pos in exits:
            path = pathfinder.find_path(self.pos, exit_pos)
            if path and len(path) < min_dist:
                min_dist = len(path)
                best_path = path
                
        self.path = best_path
        
    def move(self, occupied_cells):
        if self.exited or not self.path:
            return False
            
        next_pos = self.path[0]
        if next_pos in occupied_cells:
            return False  # Cell is occupied, wait for next tick
            
        self.pos = next_pos
        self.path = self.path[1:]
        return True

class CrowdSimulation:
    def __init__(self, width=40, height=30, num_agents=70):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))  # 0 = walkable, 1 = wall, 2 = exit
        self.agents = []
        self.exits = []
        self.time_step = 0
        self.escaped_agents = []
        self.density_map = np.zeros((height, width))
        
        # Create environment
        self._create_environment()
        
        # Create agents
        self._create_agents(num_agents)
        
        # Initialize pathfinder: only walls are blocked, exits must be walkable
        # grid == 1 -> wall (blocked=1); 0 (free) and 2 (exit) -> walkable (0)
        self.pathfinder = AStarPathfinder((self.grid == 1).astype(int))
        
    def _create_environment(self):
        # Create a simple environment with a corridor and two exits
        # 1. Create outer walls
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1
        
        # 2. Add horizontal corridor
        corridor_y = self.height // 2
        self.grid[corridor_y, :] = 0  # Make corridor walkable
        
        # 3. Add vertical walls with gaps
        wall_thickness = 2
        gap_size = 5
        
        # Left wall with gap
        self.grid[wall_thickness:-wall_thickness, :wall_thickness] = 1
        self.grid[corridor_y-gap_size//2:corridor_y+gap_size//2+1, :wall_thickness] = 0
        
        # Right wall with gap
        self.grid[wall_thickness:-wall_thickness, -wall_thickness:] = 1
        self.grid[corridor_y-gap_size//2:corridor_y+gap_size//2+1, -wall_thickness:] = 0
        
        # 4. Add exits (green cells)
        exit1 = (corridor_y, 0)
        exit2 = (corridor_y, self.width-1)
        self.grid[exit1] = 2
        self.grid[exit2] = 2
        self.exits = [exit1, exit2]
        
    def _create_agents(self, num_agents):
        # Find all walkable cells that are not exits
        walkable = np.argwhere((self.grid == 0))
        np.random.shuffle(walkable)
        
        # Create agents at random walkable positions
        for i in range(min(num_agents, len(walkable))):
            pos = tuple(walkable[i])
            self.agents.append(Agent(pos, i))
    
    def apply_floorplan(self, obstacle_grid, exits):
        """
        Apply a new floorplan to the simulation.
        obstacle_grid: 2D numpy array where 1 indicates a wall/obstacle, 0 indicates free space.
        exits: list of (row, col) tuples indicating exit cells. These will be set as walkable/exits (value 2).
        """
        # Validate inputs
        if obstacle_grid is None or not hasattr(obstacle_grid, 'shape'):
            raise ValueError("Invalid obstacle_grid provided to apply_floorplan")
        H, W = obstacle_grid.shape
        # Resize sim if dimensions changed
        self.height, self.width = H, W
        # Build grid: start from obstacle map and mark exits as 2
        self.grid = obstacle_grid.astype(int).copy()
        self.exits = []
        if exits:
            for (r, c) in exits:
                if 0 <= r < H and 0 <= c < W:
                    self.grid[r, c] = 2
                    self.exits.append((r, c))
        # Rebuild pathfinder: only walls (1) are blocked; free (0) and exits (2) are traversable
        self.pathfinder = AStarPathfinder((self.grid == 1).astype(int))
        # Reset state
        self.time_step = 0
        self.escaped_agents = []
        self.density_map = np.zeros((H, W))
        # Reseed agents on free cells (not walls). Allow placing on 0 or 2 (exits) but prefer 0
        num_agents = len(self.agents) if self.agents else 0
        self.agents = []
        walkable = np.argwhere((self.grid != 1))  # 0 or 2
        if len(walkable) == 0:
            return  # nothing to place
        np.random.shuffle(walkable)
        for i in range(min(num_agents, len(walkable))):
            pos = tuple(walkable[i])
            self.agents.append(Agent(pos, i))
    
    def update(self):
        # Update density map
        for agent in self.agents:
            if not agent.exited:
                r, c = agent.pos
                self.density_map[r, c] += 1
        
        # Update agent paths
        for agent in self.agents:
            if not agent.exited and not agent.path:
                agent.update_path(self.pathfinder, self.exits)
        
        # Move agents with collision avoidance
        occupied_cells = set()
        
        # First pass: collect positions that will be occupied
        for agent in self.agents:
            if not agent.exited and agent.path:
                next_pos = agent.path[0]
                occupied_cells.add(next_pos)
        
        # Second pass: actually move agents
        agents_to_remove = []
        
        for agent in self.agents:
            if agent.exited:
                continue
                
            # Check if agent has reached an exit
            if agent.pos in [tuple(exit_pos) for exit_pos in self.exits]:
                agent.exited = True
                agent.exit_time = self.time_step
                self.escaped_agents.append(agent)
                agents_to_remove.append(agent)
                continue
                
            # Move agent if possible
            agent.move(occupied_cells)
            
            # Add new position to occupied cells
            if not agent.exited:
                occupied_cells.add(agent.pos)
        
        # Remove escaped agents
        for agent in agents_to_remove:
            if agent in self.agents:
                self.agents.remove(agent)
        
        self.time_step += 1
        
    def get_evacuation_curve(self):
        times = [0]  # Start with 0 agents evacuated at time 0
        counts = [0]
        
        for t in range(1, self.time_step + 1):
            count = sum(1 for agent in self.escaped_agents if agent.exit_time <= t)
            times.append(t)
            counts.append(count)
            
        return times, counts
    
    def plot(self, ax):
        # Clear axis
        ax.clear()
        
        # Create colormap: white for empty, black for walls, green for exits
        cmap = plt.cm.colors.ListedColormap(['white', 'black', 'green'])
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        
        # Plot the grid
        ax.imshow(self.grid, cmap=cmap, norm=norm, interpolation='none')
        
        # Plot agents
        for agent in self.agents:
            if not agent.exited:
                circle = plt.Circle((agent.pos[1], agent.pos[0]), 0.3, color='red')
                ax.add_patch(circle)
        
        # Plot path for first agent (for visualization)
        if self.agents and not self.agents[0].exited and hasattr(self.agents[0], 'path') and self.agents[0].path:
            path = [self.agents[0].pos] + self.agents[0].path
            path_y = [p[0] for p in path]
            path_x = [p[1] for p in path]
            ax.plot(path_x, path_y, 'r-', alpha=0.5)
        
        # Set title and remove axis ticks
        ax.set_title(f'Time Step: {self.time_step} | Agents: {len(self.agents)} | Escaped: {len(self.escaped_agents)}')
        ax.set_xticks([])
        ax.set_yticks([])
        
        return ax

if __name__ == "__main__":
    # Create simulation
    sim = CrowdSimulation(width=60, height=40, num_agents=70)
    
    # Set up the figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(wspace=0.3)
    
    # Animation function
    def update(frame):
        # Update simulation
        sim.update()
        
        # Plot main simulation
        sim.plot(ax1)
        
        # Plot evacuation curve
        times, counts = sim.get_evacuation_curve()
        ax2.clear()
        ax2.plot(times, counts, 'b-')
        ax2.set_title('Evacuation Progress')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Agents Evacuated')
        ax2.grid(True)
        
        # Plot density map
        ax3.clear()
        density = sim.density_map / (sim.density_map.max() + 1e-6)  # Normalize
        ax3.imshow(density, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
        ax3.set_title('Density Map')
        ax3.set_xticks([])
        ax3.set_yticks([])
        
        # Stop animation if all agents have exited
        if not sim.agents:
            ani.event_source.stop()
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=200, interval=200, repeat=False)
    plt.show()
