import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import sys

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class MonteCarloVisualizer:
    """Visualizes Monte Carlo simulation results for evacuation scenarios with Thunderhead strategies."""
    
    def __init__(self, figsize=(12, 10)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        # Visualization elements
        self.obstacles = []
        self.hazards = []
        self.exits = []
        self.agents = []
        self.paths = []
        self.heatmap = None
        self.floor_plan = None
        
        # Color palette
        self.color_palette = {
            'low_risk': '#4CAF50',    # Green
            'medium_risk': '#FFC107',  # Amber
            'high_risk': '#F44336',    # Red
            'success': '#2196F3',      # Blue
            'agent': {
                'adult': '#3F51B5',
                'child': '#E91E63',
                'elderly': '#9C27B0',
                'mobility_impaired': '#FF9800',
                'default': '#2196F3'
            },
            'hazard': {
                'fire': '#FF5722',
                'smoke': '#795548',
                'debris': '#9E9E9E',
                'water': '#00BCD4',
                'default': '#FF9800'
            },
            'exit': '#4CAF50',
            'obstacle': '#9E9E9E',
            'path': {
                'shortest': '#2196F3',
                'safest': '#4CAF50',
                'least_crowded': '#9C27B0',
                'combined': '#FF9800',
                'default': '#607D8B'
            },
            'text': '#FFFFFF'
        }
    
    def set_floor_plan(self, image_path: str = None):
        """Set the floor plan background image if available."""
        if image_path and os.path.exists(image_path):
            try:
                img = plt.imread(image_path)
                self.floor_plan = self.ax.imshow(
                    img, 
                    extent=[0, 100, 0, 100], 
                    alpha=0.5,
                    aspect='auto',
                    zorder=0
                )
            except Exception as e:
                print(f"Error loading floor plan: {e}")
    
    def update_environment(self, env_data: Dict):
        """Update the environment visualization with current state."""
        # Clear previous visualization elements
        for element in self.obstacles + self.hazards + self.exits + self.agents + self.paths:
            if hasattr(element, 'remove'):
                element.remove()
        
        self.obstacles = []
        self.hazards = []
        self.exits = []
        self.agents = []
        self.paths = []
        
        # Update obstacles
        for obstacle in env_data.get('obstacles', []):
            rect = plt.Rectangle(
                (obstacle[0] - 0.5, obstacle[1] - 0.5), 1, 1,
                color=self.color_palette['obstacle'],
                alpha=0.7,
                zorder=5
            )
            self.obstacles.append(rect)
            self.ax.add_patch(rect)
        
        # Update hazards
        for hazard in env_data.get('hazards', []):
            hazard_type = hazard.get('type', 'default')
            x, y = hazard['position']
            radius = hazard.get('radius', 5.0)
            intensity = hazard.get('intensity', 0.7)
            
            # Hazard area
            circle = plt.Circle(
                (x, y), radius,
                color=self.color_palette['hazard'].get(hazard_type, self.color_palette['hazard']['default']),
                alpha=0.2 * intensity,
                zorder=10
            )
            self.hazards.append(circle)
            self.ax.add_patch(circle)
            
            # Hazard label
            text = self.ax.text(
                x, y, hazard_type[0].upper(),
                ha='center', va='center',
                color='white', weight='bold',
                fontsize=8, zorder=15
            )
            self.hazards.append(text)
        
        # Update exits
        for i, exit_pos in enumerate(env_data.get('exits', []), 1):
            exit_rect = plt.Rectangle(
                (exit_pos[0] - 1, exit_pos[1] - 1), 2, 2,
                color=self.color_palette['exit'],
                alpha=0.7,
                zorder=20
            )
            self.exits.append(exit_rect)
            self.ax.add_patch(exit_rect)
            
            # Exit label
            text = self.ax.text(
                exit_pos[0], exit_pos[1], f"Exit {i}",
                ha='center', va='center',
                color='white', weight='bold',
                fontsize=10, zorder=25
            )
            self.exits.append(text)
    
    def update_agents(self, agents_data: List[Dict], show_paths: bool = True):
        """Update agent visualization with current state."""
        for agent in agents_data:
            if agent.get('status') != 'active':
                continue
                
            x, y = agent['position']
            agent_type = agent.get('type', 'default')
            
            # Agent marker
            marker = plt.Circle(
                (x, y), 0.5,
                color=self.color_palette['agent'].get(agent_type, self.color_palette['agent']['default']),
                alpha=0.8,
                zorder=30
            )
            self.agents.append(marker)
            self.ax.add_patch(marker)
            
            # Agent path
            if show_paths and 'path' in agent and len(agent['path']) > 1:
                path = np.array(agent['path'])
                path_line, = self.ax.plot(
                    path[:, 0], path[:, 1],
                    '--', 
                    color=self.color_palette['path'].get(agent.get('strategy', 'default'), 
                                                       self.color_palette['path']['default']),
                    alpha=0.6,
                    linewidth=1.5,
                    zorder=5
                )
                self.paths.append(path_line)
    
    def plot_risk_heatmap(self, risk_data: np.ndarray):
        """Plot a heatmap of risk distribution in the environment."""
        if self.heatmap:
            self.heatmap.remove()
            
        if risk_data is not None and risk_data.size > 0:
            self.heatmap = self.ax.imshow(
                risk_data.T, 
                origin='lower',
                extent=[0, 100, 0, 100],
                cmap='YlOrRd',
                alpha=0.3,
                zorder=1
            )
            plt.colorbar(self.heatmap, ax=self.ax, label='Risk Level')
    
    def finalize_plot(self, title: str = 'Evacuation Simulation'):
        """Finalize the plot with labels and legend."""
        self.ax.set_title(title, fontsize=14, pad=20)
        self.ax.set_xlabel('X Position (m)', fontsize=10)
        self.ax.set_ylabel('Y Position (m)', fontsize=10)
        
        # Create custom legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Agents',
                      markerfacecolor=self.color_palette['agent']['default'], markersize=8),
            plt.Line2D([0], [0], marker='s', color='w', label='Exits',
                      markerfacecolor=self.color_palette['exit'], markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Hazards',
                      markerfacecolor=self.color_palette['hazard']['fire'], alpha=0.5, markersize=10),
            plt.Line2D([0], [0], marker='s', color='w', label='Obstacles',
                      markerfacecolor=self.color_palette['obstacle'], alpha=0.7, markersize=10),
            plt.Line2D([0], [0], color=self.color_palette['path']['shortest'], 
                      lw=2, label='Shortest Path'),
            plt.Line2D([0], [0], color=self.color_palette['path']['safest'], 
                      lw=2, label='Safest Path'),
            plt.Line2D([0], [0], color=self.color_palette['path']['least_crowded'], 
                      lw=2, label='Least Crowded')
        ]
        
        self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.tight_layout()
    
    def plot_convergence(self, convergence_data: List[Dict], figsize=(12, 6)):
        """Plot the convergence of simulation metrics over multiple runs."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Success rate convergence
        x = [d['scenario'] for d in convergence_data]
        success_rates = [d['success_rate'] for d in convergence_data]
        ax1.plot(x, success_rates, 'b-', label='Success Rate', linewidth=2)
        ax1.set_xlabel('Scenario #', fontsize=10)
        ax1.set_ylabel('Success Rate (%)', fontsize=10)
        ax1.set_title('Success Rate Convergence', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 110)
        
        # Add trend line
        if len(x) > 1:
            z = np.polyfit(x, success_rates, 1)
            p = np.poly1d(z)
            ax1.plot(x, p(x), 'r--', alpha=0.5, label='Trend')
        
        # Average evacuation time convergence
        ax2_twin = ax1.twinx()
        evac_times = [d['avg_evacuation_time'] for d in convergence_data]
        ax2_twin.plot(x, evac_times, 'r-', label='Avg Evac Time')
        ax2_twin.set_ylabel('Avg Evacuation Time (s)', color='r')
        ax2_twin.tick_params(axis='y', labelcolor='r')
        
        # Risk distribution convergence
        risk_low = [d['risk_distribution']['low'] for d in convergence_data]
        risk_med = [d['risk_distribution']['medium'] for d in convergence_data]
        risk_high = [d['risk_distribution']['high'] for d in convergence_data]
        
        ax2.stackplot(x, [risk_low, risk_med, risk_high], 
                     labels=['Low Risk', 'Medium Risk', 'High Risk'],
                     colors=[self.color_palette['low_risk'], 
                            self.color_palette['medium_risk'], 
                            self.color_palette['high_risk']],
                     alpha=0.7)
        
        ax2.set_xlabel('Scenario #')
        ax2.set_ylabel('Number of Agents')
        ax2.set_title('Risk Distribution Convergence')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_risk_heatmap(self, env, risk_data, figsize=(10, 8)):
        """Create a heatmap of risk distribution in the environment."""
        # Create grid
        x = np.linspace(0, env.width, 50)
        y = np.linspace(0, env.height, 50)
        X, Y = np.meshgrid(x, y)
        
        # Interpolate risk data to grid
        from scipy.interpolate import griddata
        points = np.array(risk_data['positions'])
        values = np.array(risk_data['values'])
        Z = griddata(points, values, (X, Y), method='cubic', fill_value=0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        heatmap = ax.pcolormesh(X, Y, Z, cmap='YlOrRd', shading='auto', alpha=0.7)
        plt.colorbar(heatmap, label='Risk Level')
        
        # Add environment elements
        self._plot_environment(ax, env)
        
        ax.set_title('Risk Distribution Heatmap')
        ax.set_xlim(0, env.width)
        ax.set_ylim(0, env.height)
        ax.set_aspect('equal')
        
        return fig
    
    def plot_agent_movements(self, agents_history, env, figsize=(12, 10)):
        """Plot the movement paths of all agents."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot environment
        self._plot_environment(ax, env)
        
        # Plot agent paths
        for agent_id, positions in agents_history.items():
            if len(positions) > 1:
                # Get agent type for coloring
                agent_type = positions[-1].get('type', 'adult')
                color = self.color_palette['agent'].get(agent_type, '#000000')
                
                # Plot path
                path = np.array([p['position'] for p in positions])
                ax.plot(path[:, 0], path[:, 1], '-', color=color, alpha=0.5, linewidth=1)
                
                # Plot start and end points
                ax.plot(path[0, 0], path[0, 1], 'o', color=color, markersize=6, alpha=0.7)
                ax.plot(path[-1, 0], path[-1, 1], 's', color=color, markersize=8, alpha=0.9)
        
        ax.set_title('Agent Movement Paths')
        ax.set_xlim(0, env.width)
        ax.set_ylim(0, env.height)
        ax.set_aspect('equal')
        
        # Create legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8, label='Start'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='k', markersize=8, label='End'),
            *[Line2D([0], [0], color=color, lw=2, label=agent_type.capitalize()) 
              for agent_type, color in self.color_palette['agent'].items()]
        ]
        
        ax.legend(handles=legend_elements, loc='upper right')
        
        return fig
    
    def plot_risk_analysis(self, risk_analysis: Dict, figsize=(14, 6)):
        """Plot comprehensive risk analysis."""
        fig = make_subplots(
            rows=1, 
            cols=2,
            subplot_titles=('Success Rate by Agent Type', 'Risk Distribution'),
            specs=[[{'type': 'xy'}, {'type': 'pie'}]]
        )
        
        # Agent type success rates
        agent_types = list(risk_analysis['agent_type_risks'].keys())
        success_rates = list(risk_analysis['agent_type_risks'].values())
        
        fig.add_trace(
            go.Bar(
                x=agent_types,
                y=success_rates,
                marker_color=[self.color_palette['agent'].get(t, '#000000') for t in agent_types],
                text=[f'{r:.1f}%' for r in success_rates],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Risk distribution pie chart
        risk_labels = ['Low', 'Medium', 'High']
        risk_values = [
            risk_analysis['risk_distribution']['low'],
            risk_analysis['risk_distribution']['medium'],
            risk_analysis['risk_distribution']['high']
        ]
        
        fig.add_trace(
            go.Pie(
                labels=risk_labels,
                values=risk_values,
                marker_colors=[
                    self.color_palette['low_risk'],
                    self.color_palette['medium_risk'],
                    self.color_palette['high_risk']
                ],
                hole=0.4,
                textinfo='percent+label'
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text='Risk Analysis',
            showlegend=False,
            height=500
        )
        
        # Update axes
        fig.update_yaxes(title_text='Success Rate (%)', row=1, col=1, range=[0, 100])
        
        return fig
    
    def _plot_environment(self, ax, env):
        """Helper method to plot environment elements."""
        # Plot obstacles
        for obs in env.obstacles:
            ax.add_patch(plt.Rectangle(
                (obs[0] - 0.5, obs[1] - 0.5), 1, 1,
                color='#795548', alpha=0.5, zorder=1
            ))
        
        # Plot exits
        for i, exit_pos in enumerate(env.exits):
            ax.add_patch(plt.Rectangle(
                (exit_pos[0] - 1, exit_pos[1] - 1), 2, 2,
                color='#4CAF50', alpha=0.8, zorder=2
            ))
            ax.text(exit_pos[0], exit_pos[1], f'Exit {i+1}',
                   ha='center', va='center', color='white', fontweight='bold')
        
        # Plot hazards
        for hazard in env.hazards:
            hazard_circle = plt.Circle(
                (hazard['position'][0], hazard['position'][1]),
                hazard['radius'],
                color=self.color_palette['hazard'].get(hazard['type'], '#000000'),
                alpha=0.5,
                zorder=3
            )
            ax.add_patch(hazard_circle)
            ax.text(hazard['position'][0], hazard['position'][1], 
                   hazard['type'].capitalize(),
                   ha='center', va='center', color='white', fontweight='bold')
    
    def create_evacuation_animation(self, agents_history, env, output_file='evacuation_animation.mp4'):
        """Create an animation of the evacuation process."""
        from matplotlib.animation import FuncAnimation
        from matplotlib.collections import PatchCollection
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot static environment
        self._plot_environment(ax, env)
        
        # Initialize agent markers
        agent_markers = []
        agent_texts = []
        
        for agent_id, positions in agents_history.items():
            if positions:
                agent_type = positions[0].get('type', 'adult')
                color = self.color_palette['agent'].get(agent_type, '#000000')
                
                # Create marker for agent
                marker = ax.plot([], [], 'o', color=color, markersize=8, alpha=0.8)[0]
                agent_markers.append(marker)
                
                # Add ID text
                text = ax.text(0, 0, str(agent_id), fontsize=8, ha='center', va='center')
                agent_texts.append(text)
        
        # Animation update function
        def update(frame):
            for i, (agent_id, positions) in enumerate(agents_history.items()):
                if frame < len(positions):
                    pos = positions[frame]['position']
                    agent_markers[i].set_data([pos[0]], [pos[1]])
                    agent_texts[i].set_position((pos[0], pos[1] + 0.5))
            
            return agent_markers + agent_texts
        
        # Determine number of frames
        max_frames = max(len(positions) for positions in agents_history.values())
        
        # Create animation
        anim = FuncAnimation(
            fig, update, frames=max_frames, interval=100, blit=True
        )
        
        # Save animation
        anim.save(output_file, writer='ffmpeg', fps=10, dpi=100)
        plt.close()
        
        return output_file
