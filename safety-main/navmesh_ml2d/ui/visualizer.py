import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

class CrowdVisualizer:
    def __init__(self, env):
        self.env = env
        self.width = env.width
        self.height = env.height
        self.n_agents = len(env.agents)
        self.cum_density = np.zeros((self.height, self.width), dtype=np.float32)
        self.cum_wait = np.zeros((self.height, self.width), dtype=np.float32)
        self.cum_panic = np.zeros((self.height, self.width), dtype=np.float32)
        self.agent_trails = [[] for _ in env.agents]
        # Store initial positions as start points
        self.start_positions = [tuple(agent.pos) for agent in env.agents]
        # Track collisions
        self.collision_points = []
        self.exit_times = []
        self.panic_over_time = []
        self.congestion_over_time = []
        self.fig = None
        self.ax_main = None
        self.ax_density = None
        self.ax_hazard = None
        self.ax_exit_hist = None
        self.ax_panic = None
        self.ax_cong = None

    def draw(self, selected_agents=None, obstacle_img=None, color_mode="Status", exits=None, stampede_risk=None):
        # Create a new figure with a reasonable size for Streamlit
        if self.fig is not None:
            plt.close(self.fig)
        self.fig = plt.figure(figsize=(12, 8))
        gs = self.fig.add_gridspec(2, 2)
        
        # Set a non-interactive backend to avoid warnings
        plt.ioff()
        
        # Create main axes
        self.ax_main = self.fig.add_subplot(gs[:, 0])  # Main view takes left half
        
        # Create right-side axes
        self.ax_density = self.fig.add_subplot(gs[0, 1])  # Top right
        self.ax_panic = self.fig.add_subplot(gs[1, 1])  # Bottom right
        self.ax_cong = self.ax_panic.twinx()  # Secondary y-axis for congestion
        
        # Initialize other axes to None since we're not using them in this layout
        self.ax_hazard = None
        self.ax_exit_hist = None
        # --- Main animation: agents, grid, hazards, trails ---
        self.ax_main.clear()
        if obstacle_img is not None:
            self.ax_main.imshow(obstacle_img, cmap='gray', alpha=0.7, extent=[0, self.width, 0, self.height], origin='lower')
        else:
            self.ax_main.imshow(self.env.navmesh.grid, cmap='Greys', vmin=0, vmax=2, alpha=0.3)
        grid = self.env.navmesh.grid
        # Draw grid lines with coordinates
        for x in range(self.width + 1):
            self.ax_main.axvline(x - 0.5, color='lightgray', linestyle='-', alpha=0.3)
        for y in range(self.height + 1):
            self.ax_main.axhline(y - 0.5, color='lightgray', linestyle='-', alpha=0.3)
            
        # Add coordinate labels
        if self.width <= 20:  # Only show coordinates if grid isn't too large
            for x in range(self.width):
                self.ax_main.text(x, -0.7, str(x), ha='center', va='top', color='gray', fontsize=8)
            for y in range(self.height):
                self.ax_main.text(-0.7, y, str(y), ha='right', va='center', color='gray', fontsize=8)
        
        # Draw obstacles as black squares
        obstacle_coords = np.argwhere(grid == 1)
        if len(obstacle_coords) > 0:
            self.ax_main.scatter(obstacle_coords[:, 1], obstacle_coords[:, 0], c='black', marker='s', s=60, label='Obstacle', zorder=2)
            
        # Draw start positions
        if hasattr(self, 'start_positions') and self.start_positions:
            starts = np.array(self.start_positions)
            self.ax_main.scatter(starts[:, 0], starts[:, 1], c='lime', marker='*', s=100, 
                               edgecolors='black', label='Start', zorder=3)
        # Draw exits as large yellow squares with black borders and labels
        exits = getattr(self.env, 'exits', None)
        if exits is not None:
            for idx, (x, y) in enumerate(exits):
                self.ax_main.scatter(x, y, c='yellow', marker='s', s=120, edgecolors='black', label='Exit' if idx == 0 else "", zorder=3)
                self.ax_main.text(x, y, str(idx+1), color='black', fontsize=12, ha='center', va='center', fontweight='bold', zorder=4)
        for y in range(self.height):
            for x in range(self.width):
                if grid[y, x] == 0:
                    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height and grid[ny, nx] == 0:
                            self.ax_main.plot([x, nx], [y, ny], color='lightgray', alpha=0.1, linewidth=0.5, zorder=0)
        # Draw hazards if hazard_map exists
        if hasattr(self.env, 'hazard_map') and hasattr(self.env.hazard_map, 'shape'):
            hazard = np.where(self.env.hazard_map > 0)
            if len(hazard[0]) > 0:  # Only plot if there are hazards
                self.ax_main.scatter(hazard[1], hazard[0], c='red', marker='x', s=100, label='Hazard', zorder=5)
        # Add legend with unique labels
        handles, labels = self.ax_main.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax_main.legend(by_label.values(), by_label.keys(), loc='upper right')
        # Draw non-selected agent trails
        for i, agent in enumerate(self.env.agents):
            if selected_agents and agent.idx in selected_agents:
                continue
            self.agent_trails[i].append(tuple(agent.pos))
            trail = np.array(self.agent_trails[i])
            if len(trail) > 1:
                self.ax_main.plot(trail[:,0], trail[:,1], color='gray', alpha=0.3, linewidth=1, zorder=1)
        # Draw selected agent trails on top
        if selected_agents:
            for i, agent in enumerate(self.env.agents):
                if agent.idx in selected_agents:
                    trail = np.array(self.agent_trails[i])
                    if len(trail) > 1:
                        self.ax_main.plot(trail[:,0], trail[:,1], color='red', alpha=1.0, linewidth=3, zorder=10)
        # Draw non-selected agent paths
        for agent in self.env.agents:
            if selected_agents and agent.idx in selected_agents:
                continue
            path = getattr(agent, 'path', None)
            if path and len(path) > 1:
                path = np.array(path)
                self.ax_main.plot(path[:,0], path[:,1], color='cyan', alpha=0.5, linewidth=1.5, linestyle='dashed', zorder=2)
        # Draw selected agent paths on top
        if selected_agents:
            for agent in self.env.agents:
                if agent.idx in selected_agents:
                    path = getattr(agent, 'path', None)
                    if path and len(path) > 1:
                        path = np.array(path)
                        self.ax_main.plot(path[:,0], path[:,1], color='magenta', alpha=1.0, linewidth=3, linestyle='dashed', zorder=11)
        # Track agent positions for collision detection
        agent_positions = {}
        
        # First pass: collect positions and detect collisions
        for agent in self.env.agents:
            if getattr(agent, 'exited', False):
                continue
            pos = tuple(map(int, agent.pos))
            if pos in agent_positions:
                agent_positions[pos].append(agent)
                if pos not in [cp[0] for cp in self.collision_points]:
                    self.collision_points.append((pos, self.env.t))
            else:
                agent_positions[pos] = [agent]
        
        # Draw collision indicators
        current_time = self.env.t
        self.collision_points = [(pos, t) for pos, t in self.collision_points 
                               if current_time - t < 10]  # Show collisions for 10 steps
        
        for pos, t in self.collision_points:
            self.ax_main.add_patch(plt.Circle(pos, 0.5, color='red', alpha=0.3, zorder=15))
            self.ax_main.text(pos[0], pos[1], '!', color='red', ha='center', va='center', 
                            fontweight='bold', fontsize=14, zorder=16)
        
        # Draw non-selected agents
        group_ids = [getattr(agent, 'group_id', None) for agent in self.env.agents]
        unique_groups = sorted(set(g for g in group_ids if g is not None))
        group_cmap = plt.cm.get_cmap('tab10', max(1, len(unique_groups)))
        
        # Second pass: draw agents
        for agent in self.env.agents:
            if getattr(agent, 'exited', False):
                continue
            if selected_agents and agent.idx in selected_agents:
                continue
            if color_mode == "Group" and getattr(agent, 'group_id', None) is not None:
                idx = unique_groups.index(getattr(agent, 'group_id', None)) if getattr(agent, 'group_id', None) in unique_groups else 0
                color = group_cmap(idx)
            elif color_mode == "Panic":
                panic = getattr(agent, 'panic', 0.0)
                color = (panic, 0, 1-panic)
            else:
                color = 'blue' if getattr(agent, 'status', 'normal') == 'normal' else 'orange' if getattr(agent, 'status', 'normal') == 'avoiding hazard' else 'red'
            self.ax_main.scatter(agent.pos[0], agent.pos[1], c=[color], s=60, edgecolors='k', zorder=3)
        # Draw selected agents on top with behavior info
        if selected_agents:
            for agent in self.env.agents:
                if getattr(agent, 'exited', False):
                    continue
                if agent.idx in selected_agents:
                    # Draw the agent
                    self.ax_main.scatter(agent.pos[0], agent.pos[1], c=['red'], s=120, 
                                       edgecolors='yellow', linewidths=2, zorder=20)
                    
                    # Add behavior info as text
                    info = []
                    if hasattr(agent, 'panic'):
                        info.append(f"Panic: {agent.panic:.2f}")
                    if hasattr(agent, 'speed'):
                        info.append(f"Speed: {agent.speed:.2f}")
                    if hasattr(agent, 'status'):
                        info.append(f"Status: {agent.status}")
                        
                    # Position the text above the agent
                    text_y_offset = 0.5
                    for i, line in enumerate(info):
                        self.ax_main.text(agent.pos[0], agent.pos[1] + text_y_offset + i*0.3, line,
                                        color='black', fontsize=8, ha='center', va='bottom',
                                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'),
                                        zorder=25)
        # Add scale bar if not showing coordinates
        if self.width > 20:
            scale_length = max(1, self.width // 10)  # Scale bar is 1/10th of width
            scale_x, scale_y = 0.5, 0.05 * self.height
            self.ax_main.plot([0, scale_length], [scale_y, scale_y], 'k-', linewidth=2)
            self.ax_main.text(scale_length/2, scale_y + 0.5, f"{scale_length} units", 
                            ha='center', va='bottom', fontsize=8)
            
        # Add legend with more information
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Agent'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='lime', markersize=10, label='Start'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='yellow', markersize=8, label='Exit'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=8, label='Obstacle'),
            plt.Line2D([0], [0], marker='x', color='red', markersize=8, label='Hazard', linestyle='None'),
            plt.Line2D([0], [0], marker='o', color='red', alpha=0.3, markersize=15, label='Collision', linestyle='None')
        ]
        
        self.ax_main.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # Update title with more stats
        exited = sum(getattr(a, 'exited', False) for a in self.env.agents)
        avg_panic = np.mean([getattr(a, 'panic', 0) for a in self.env.agents])
        collisions = len([p for p, t in self.collision_points if t == self.env.t])
        
        title = (f"Step {self.env.t} | Exited: {exited}/{self.n_agents} | "
                f"Avg Panic: {avg_panic:.2f} | Collisions: {collisions}")
        self.ax_main.set_title(title, fontsize=10)
        self.ax_main.set_xlim(-0.5, self.width-0.5)
        self.ax_main.set_ylim(-0.5, self.height-0.5)
        self.ax_main.set_xticks([])
        self.ax_main.set_yticks([])
        self.ax_main.legend()
        # --- Density (occupancy) heatmap ---
        if not hasattr(self.env, 'density_map') or not hasattr(self.env.density_map, 'shape'):
            # Initialize density_map if it doesn't exist
            self.env.density_map = np.zeros((self.height, self.width), dtype=np.float32)
        
        self.cum_density += self.env.density_map
        density_cmap = plt.cm.viridis
        norm = mcolors.Normalize(vmin=0, vmax=max(1, np.max(self.cum_density)))
        self.ax_density.clear()
        self.ax_density.imshow(self.cum_density, cmap=density_cmap, norm=norm, alpha=0.8)
        self.ax_density.set_title("Cumulative Density Heatmap")
        self.ax_density.set_xticks([])
        self.ax_density.set_yticks([])
        # --- Hazard heatmap overlay ---
        # Only show hazard map if the axis exists
        if hasattr(self, 'ax_hazard') and self.ax_hazard is not None:
            self.ax_hazard.clear()
            if not hasattr(self.env, 'hazard_map') or not hasattr(self.env.hazard_map, 'shape'):
                # Initialize hazard_map if it doesn't exist
                self.env.hazard_map = np.zeros((self.height, self.width), dtype=np.float32)
                
            self.ax_hazard.imshow(self.env.hazard_map, cmap='Reds', vmin=0, vmax=1, alpha=0.8)
            self.ax_hazard.set_title("Hazard (Fire/Smoke) Heatmap")
            self.ax_hazard.set_xticks([])
            self.ax_hazard.set_yticks([])
        # --- Per-cell average wait and panic ---
        for agent in self.env.agents:
            x, y = map(int, agent.pos)
            self.cum_wait[y, x] += getattr(agent, 'waited', 0)
            self.cum_panic[y, x] += getattr(agent, 'panic', 0.0)
        # --- Metrics over time ---
        exited = [getattr(agent, 'exit_time', None) for agent in self.env.agents if getattr(agent, 'exited', False)]
        self.exit_times = exited
        avg_panic = np.mean([getattr(agent, 'panic', 0.0) for agent in self.env.agents])
        self.panic_over_time.append(avg_panic)
        avg_cong = np.mean(self.env.density_map)
        self.congestion_over_time.append(avg_cong)
        # --- Exit time histogram ---
        # Only show exit histogram if the axis exists
        if hasattr(self, 'ax_exit_hist') and self.ax_exit_hist is not None:
            self.ax_exit_hist.clear()
            if len(self.exit_times) > 0:
                self.ax_exit_hist.hist(self.exit_times, bins=20, color='green', alpha=0.7)
            self.ax_exit_hist.set_title("Exit Times Histogram")
            self.ax_exit_hist.set_xlabel("Step")
            self.ax_exit_hist.set_ylabel("# Exited")
        # --- Panic and congestion over time ---
        self.ax_panic.clear()
        self.ax_panic.plot(self.panic_over_time, color='red', label='Avg Panic')
        self.ax_panic.set_ylabel('Avg Panic')
        self.ax_panic.set_xlabel('Step')
        self.ax_panic.set_title('Panic & Congestion Over Time')
        self.ax_cong.clear()
        self.ax_cong.plot(self.congestion_over_time, color='blue', label='Avg Congestion')
        self.ax_cong.set_ylabel('Avg Congestion')
        self.ax_panic.legend(loc='upper left')
        self.ax_cong.legend(loc='upper right')
        # Adjust layout to prevent overlap
        try:
            self.fig.tight_layout()
        except Exception as e:
            # If tight_layout fails, just continue
            pass
