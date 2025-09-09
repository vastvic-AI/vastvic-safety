"""Simulation visualization using Matplotlib."""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle, Arrow
import numpy as np
from typing import List, Dict, Any, Optional

from ..core.agent import EvacuationAgent
from ..core.environment import EvacuationEnvironment


class SimulationVisualizer:
    """Class for visualizing the evacuation simulation."""
    
    def __init__(self, env: EvacuationEnvironment, figsize: tuple = (12, 10)):
        """Initialize the visualizer with an environment.
        
        Args:
            env: The evacuation environment to visualize
            figsize: Size of the figure (width, height) in inches
        """
        self.env = env
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.agent_artists = []
        self.hazard_artists = []
        self.obstacle_artists = []
        self.exit_artists = []
        
        # Set up the plot
        self.ax.set_xlim(0, self.env.width)
        self.ax.set_ylim(0, self.env.height)
        self.ax.set_aspect('equal')
        self.ax.set_title('Evacuation Simulation')
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        
        # Draw the initial environment
        self._draw_environment()
    
    def _draw_environment(self):
        """Draw the static elements of the environment."""
        # Clear existing artists
        for artist in self.obstacle_artists + self.exit_artists:
            artist.remove()
        self.obstacle_artists.clear()
        self.exit_artists.clear()
        
        # Draw obstacles
        for x, y in self.env.obstacles:
            rect = Rectangle(
                (x - 0.5, y - 0.5), 1, 1,
                color='gray', alpha=0.7, zorder=1
            )
            self.obstacle_artists.append(rect)
            self.ax.add_patch(rect)
        
        # Draw exits
        for i, (x, y) in enumerate(self.env.exits):
            exit_rect = Rectangle(
                (x - 2, y - 2), 4, 4,
                color='green', alpha=0.7, zorder=2,
                label=f'Exit {i+1}'
            )
            self.exit_artists.append(exit_rect)
            self.ax.add_patch(exit_rect)
            
            # Add exit label
            self.ax.text(x, y, f'Exit {i+1}',
                        ha='center', va='center',
                        color='white', fontweight='bold',
                        zorder=5)
    
    def update_visualization(self):
        """Update the visualization with the current state of the environment."""
        # Clear previous frame's dynamic elements
        for artist in self.agent_artists + self.hazard_artists:
            artist.remove()
        self.agent_artists.clear()
        self.hazard_artists.clear()
        
        # Draw hazards
        self._draw_hazards()
        
        # Draw agents
        self._draw_agents()
        
        # Update the title with current step and metrics
        self.ax.set_title(f'Evacuation Simulation (Step: {self.env.step_count})')
        
        # Redraw the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def _draw_hazards(self):
        """Draw hazard zones on the map."""
        # Draw fire
        for (x, y), intensity in zip(self.env.hazards['fire']['positions'],
                                   self.env.hazards['fire']['intensities']):
            alpha = 0.3 + 0.7 * intensity
            fire = Circle(
                (x, y), 1.5,
                color='red', alpha=alpha,
                zorder=3,
                label='Fire' if not self.hazard_artists else ""
            )
            self.hazard_artists.append(fire)
            self.ax.add_patch(fire)
        
        # Draw smoke
        for (x, y), intensity in zip(self.env.hazards['smoke']['positions'],
                                   self.env.hazards['smoke']['intensities']):
            alpha = 0.1 + 0.4 * intensity
            smoke = Circle(
                (x, y), 3.0,
                color='#888888', alpha=alpha,
                zorder=2,
                label='Smoke' if len(self.hazard_artists) <= 1 else ""
            )
            self.hazard_artists.append(smoke)
            self.ax.add_patch(smoke)
    
    def _draw_agents(self):
        """Draw agents on the map."""
        agent_colors = {
            'adult': '#3498db',    # Blue
            'child': '#e74c3c',    # Red
            'elderly': '#9b59b6',  # Purple
            'mobility_impaired': '#f39c12'  # Orange
        }
        
        for agent in self.env.agents:
            # Agent color based on type
            color = agent_colors.get(agent.agent_type, '#95a5a6')
            
            # Adjust alpha based on health
            alpha = 0.3 + 0.7 * agent.health
            
            # Draw agent
            agent_circle = Circle(
                agent.position, agent.radius,
                color=color, alpha=alpha,
                zorder=4,
                label=agent.agent_type.capitalize() if not self.agent_artists else ""
            )
            self.agent_artists.append(agent_circle)
            self.ax.add_patch(agent_circle)
            
            # Draw velocity vector
            if np.linalg.norm(agent.velocity) > 0.1:
                dx, dy = agent.velocity / np.linalg.norm(agent.velocity) * 2
                arrow = Arrow(
                    agent.position[0], agent.position[1],
                    dx, dy,
                    width=0.3, color=color, alpha=0.8,
                    zorder=5
                )
                self.agent_artists.append(arrow)
                self.ax.add_patch(arrow)
            
            # Draw agent ID (small)
            text = self.ax.text(
                agent.position[0], agent.position[1],
                str(agent.agent_id),
                color='white', fontsize=6,
                ha='center', va='center',
                zorder=6
            )
            self.agent_artists.append(text)
    
    def add_legend(self):
        """Add a legend to the plot."""
        # Get unique labels and handles
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        
        # Add legend if we have any items
        if by_label:
            self.ax.legend(
                by_label.values(), by_label.keys(),
                loc='upper right',
                bbox_to_anchor=(1.3, 1.0),
                frameon=True,
                fancybox=True,
                shadow=True
            )
    
    def animate(self, steps: int = 100, interval: int = 100):
        """Animate the simulation for a number of steps.
        
        Args:
            steps: Number of steps to simulate
            interval: Time between frames in milliseconds
            
        Returns:
            The animation object
        """
        def update(frame):
            # Step the environment
            self.env.step(0.1)  # 0.1 second time step
            
            # Update visualization
            self.update_visualization()
            
            # Update legend
            self.add_legend()
            
            return self.ax.patches
        
        # Create the animation
        anim = FuncAnimation(
            self.fig, update,
            frames=steps,
            interval=interval,
            blit=False,
            repeat=False
        )
        
        return anim
    
    def save_animation(self, filename: str, steps: int = 100, fps: int = 10):
        """Save an animation of the simulation to a file.
        
        Args:
            filename: Output filename (should end with .gif or .mp4)
            steps: Number of steps to simulate
            fps: Frames per second for the output animation
        """
        anim = self.animate(steps=steps, interval=1000//fps)
        anim.save(
            filename,
            writer='ffmpeg' if filename.endswith('.mp4') else 'pillow',
            fps=fps,
            dpi=100,
            savefig_kwargs={'facecolor': 'white'}
        )
        plt.close()


if __name__ == "__main__":
    # Example usage
    env = EvacuationEnvironment(100, 100)
    
    # Add some agents
    for i in range(20):
        x, y = np.random.uniform(20, 80, 2)
        agent = EvacuationAgent(
            agent_id=i,
            position=np.array([x, y], dtype=np.float32),
            agent_type=np.random.choice(['adult', 'child', 'elderly'])
        )
        env.add_agent(agent)
    
    # Add some obstacles
    for x in range(30, 70, 10):
        for y in range(30, 70, 10):
            env.add_obstacle(x, y)
    
    # Add exits
    env.add_exit(50, 5)   # Bottom center
    env.add_exit(5, 50)   # Left center
    env.add_exit(95, 50)  # Right center
    
    # Create and run visualization
    visualizer = SimulationVisualizer(env)
    visualizer.add_legend()
    
    # Show the initial state
    plt.show()
    
    # Run animation
    anim = visualizer.animate(steps=100, interval=100)
    plt.show()
    
    # To save the animation:
    # visualizer.save_animation("evacuation_simulation.gif", steps=100, fps=10)
