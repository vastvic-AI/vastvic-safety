import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import time
from navmesh_ml2d.core.agent import Agent
from navmesh_ml2d.core.navmesh_fixed import GridNavMesh
from stampede_detection import StampedeDetector

# Define TestAgent class locally
class TestAgent:
    def __init__(self, idx, pos, velocity=(0, 0), panic=0.0, exited=False):
        self.idx = idx
        self.pos = np.array(pos, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32)
        self.panic = float(panic)
        self.exited = exited
        self.path = []  # Add path attribute for navigation

def setup_environment():
    # Initialize navigation mesh
    navmesh = GridNavMesh(100, 100)  # 100x100 grid
    
    # Add some obstacles
    obstacles = []
    for x in range(30, 70):
        for y in range(40, 60):
            if (x - 50)**2 + (y - 50)**2 < 100:  # Circular obstacle
                obstacles.append((x, y))
    
    for x, y in obstacles:
        navmesh.grid[y, x] = 1  # Mark as blocked
    
    return navmesh

def update_agents(agents, navmesh, stampede_center=None, stampede_radius=10):
    positions = np.array([agent.pos for agent in agents])
    velocities = np.array([agent.velocity for agent in agents])
    
    # Update agent positions
    for i, agent in enumerate(agents):
        # Simple movement logic - can be replaced with your navigation logic
        if hasattr(agent, 'path') and len(agent.path) > 0:
            target = agent.path[0]
            direction = np.array(target) - np.array(agent.pos)
            if np.linalg.norm(direction) < 1.0:  # Reached waypoint
                agent.path.pop(0)
            else:
                direction = direction / np.linalg.norm(direction)
                agent.pos = agent.pos + direction * 0.5  # Move towards target
        
        # Add some random movement
        agent.pos = agent.pos + np.random.normal(0, 0.1, 2)
        
        # Keep within bounds
        agent.pos = np.clip(agent.pos, 0, 99)
        
        # Increase panic if in stampede zone
        if stampede_center is not None:
            dist = np.linalg.norm(agent.pos - stampede_center)
            if dist < stampede_radius:
                agent.panic = min(1.0, agent.panic + 0.1)

def visualize(agents, navmesh, stampede_center=None, stampede_radius=10):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw navigation mesh
    grid = navmesh.grid
    ax.imshow(grid, cmap='binary', alpha=0.3, origin='lower')
    
    # Draw agents
    positions = np.array([agent.pos for agent in agents])
    panics = np.array([agent.panic for agent in agents])
    
    # Create a colormap from green to red based on panic level
    colors = [(0, 'green'), (0.5, 'yellow'), (1, 'red')]
    cmap = LinearSegmentedColormap.from_list('panic', colors)
    
    scatter = ax.scatter(positions[:, 0], positions[:, 1], c=panics, 
                        cmap=cmap, vmin=0, vmax=1, s=50, edgecolors='black')
    
    # Draw stampede zone if active
    if stampede_center is not None:
        stampede = Circle(stampede_center, stampede_radius, 
                         color='red', alpha=0.2, label='Stampede Zone')
        ax.add_patch(stampede)
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.set_title('Stampede Detection Simulation')
    
    # Add colorbar for panic levels
    cbar = plt.colorbar(scatter, label='Panic Level')
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', 'Medium', 'High'])
    
    return fig

def main():
    st.title("Stampede Detection Visualization")
    
    # Initialize session state
    if 'agents' not in st.session_state:
        st.session_state.agents = [TestAgent(i, np.random.rand(2) * 90 + 5, 
                                          np.random.randn(2) * 0.1, 0.0) 
                                for i in range(20)]
    
    if 'navmesh' not in st.session_state:
        st.session_state.navmesh = setup_environment()
    
    if 'stampede_active' not in st.session_state:
        st.session_state.stampede_active = False
    
    if 'detector' not in st.session_state:
        # Initialize with the same dimensions as our navigation mesh (100x100)
        st.session_state.detector = StampedeDetector(area_width=100, area_height=100)
    
    # Sidebar controls
    st.sidebar.header("Simulation Controls")
    
    if st.sidebar.button("Start Stampede"):
        st.session_state.stampede_active = True
        st.session_state.stampede_center = np.random.rand(2) * 50 + 25
        st.session_state.stampede_radius = 15
    
    if st.sidebar.button("Stop Stampede"):
        st.session_state.stampede_active = False
    
    if st.sidebar.button("Reset Simulation"):
        st.session_state.agents = [TestAgent(i, np.random.rand(2) * 90 + 5, 
                                          np.random.randn(2) * 0.1, 0.0) 
                                for i in range(20)]
        st.session_state.stampede_active = False
    
    # Main visualization
    placeholder = st.empty()
    
    while True:
        # Update agents
        stampede_center = st.session_state.stampede_center if st.session_state.stampede_active else None
        stampede_radius = st.session_state.stampede_radius if st.session_state.stampede_active else 0
        
        update_agents(st.session_state.agents, st.session_state.navmesh, 
                     stampede_center, stampede_radius)
        
        # Detect stampede conditions
        if st.session_state.stampede_active:
            events = st.session_state.detector.detect_from_agent_states(
                agents=st.session_state.agents,
                timestamp=time.time(),
                window_size=2.0
            )
            
            if events:
                st.sidebar.warning(f"⚠️ STAMPEDE DETECTED! {len(events)} events")
        
        # Update visualization
        fig = visualize(st.session_state.agents, st.session_state.navmesh, 
                       stampede_center, stampede_radius)
        
        placeholder.pyplot(fig)
        time.sleep(0.1)

if __name__ == "__main__":
    main()
