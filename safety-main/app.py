import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from navmesh_ml2d.core.agent import Agent
from navmesh_ml2d.core.navmesh import GridNavMesh
from navmesh_ml2d.ui.visualizer import CrowdVisualizer
from navmesh_ml2d.core.crowd import CrowdSim
import time
import cv2
from PIL import Image
import io
import os
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional

def run_sim(params):
    try:
        # Get agents and environment from session state
        agents = st.session_state.get('agents', [])
        grid = st.session_state.get('grid')
        obstacles = st.session_state.get('obstacles', [])
        width = st.session_state.get('width', 100)
        height = st.session_state.get('height', 100)
        
        if not agents or grid is None:
            st.error("Agents not initialized. Please initialize agents first.")
            return
            
        # Create navmesh and add obstacles
        navmesh = GridNavMesh(width, height)
        navmesh.grid = grid
        
        # Initialize hazard map and density map
        navmesh.hazard_map = np.zeros((height, width), dtype=np.float32)
        navmesh.density_map = np.zeros((height, width), dtype=np.float32)
        
        # Filter agents based on selection if any agents are selected
        selected_agents = st.session_state.get('selected_agents', set())
        if selected_agents:
            agents = [agent for agent in agents if agent.agent_id in selected_agents]
            if not agents:
                st.warning("No valid agents selected. Using all agents.")
                agents = st.session_state.agents
        
        # Create crowd simulation with the selected agents
        exits = [(0, 0), (width-1, 0), (0, height-1), (width-1, height-1)]  # Corners as exits
        crowd_sim = CrowdSim(agents=agents, exits=exits, navmesh=navmesh)
        
        # Main simulation loop
        max_steps = 500  # Increased for more complex environments
        start_time = time.time()
        
        # Create a placeholder for the simulation output
        simulation_placeholder = st.empty()
        
        for step in range(max_steps):
            # Check if we should stop
            if st.session_state.get('stop_simulation', False):
                st.warning("Simulation stopped by user.")
                break
                
            # Update simulation
            crowd_sim.step_sim()
            
            # Update visualization every 5 steps
            if step % 5 == 0:
                # Create visualization
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Draw floor plan if available
                if 'uploaded_file' in params and params['uploaded_file'] is not None:
                    ax.imshow(255 - grid * 255, cmap='gray', alpha=0.7)
                else:
                    ax.imshow(grid, cmap='binary', alpha=0.3)
                
                # Track agent paths and metrics
                current_agent_metrics = {}
                
                # Define a colormap for agent groups
                cmap = plt.cm.get_cmap('tab20')
                
                # Draw agents and their paths
                for agent in agents:
                    if not agent.state.get('exited', False):
                        # Get agent position from state
                        pos = agent.state['pos']
                        
                        # Get or assign color based on group ID or agent ID
                        group_id = agent.state.get('group_id', agent.agent_id)
                        color = cmap(group_id % 20)  # Use modulo to cycle through colors
                        
                        # Draw agent
                        circle = plt.Circle(
                            (pos[0], pos[1]),
                            radius=0.5,
                            color=color,
                            alpha=0.7,
                            label=f"Agent {agent.agent_id}"
                        )
                        ax.add_patch(circle)
                        
                        # Draw path if it exists
                        if hasattr(agent, 'path') and len(agent.path) > 1:
                            path_x = [p[0] for p in agent.path]
                            path_y = [p[1] for p in agent.path]
                            ax.plot(path_x, path_y, color=agent.color, alpha=0.3, linewidth=1)
                        
                        # Store metrics
                        current_agent_metrics[agent.agent_id] = agent.get_metrics()
                
                # Set plot limits and labels
                ax.set_xlim(0, width)
                ax.set_ylim(0, height)
                ax.set_aspect('equal')
                ax.invert_yaxis()  # Match image coordinates
                
                # Display the plot
                simulation_placeholder.pyplot(fig)
                plt.close(fig)
                
                # Update metrics
                metrics_text = f"""
                ### Simulation Metrics
                - **Step**: {step}/{max_steps}
                - **Agents Active**: {sum(1 for a in agents if not a.state.get('exited', False))}/{len(agents)}
                - **Agents Exited**: {sum(1 for a in agents if a.state.get('exited', False))}
                - **Simulation Time**: {time.time() - start_time:.2f}s
                """
                
                # Update metrics display
                if 'metrics_placeholder' in st.session_state:
                    st.session_state.metrics_placeholder.markdown(metrics_text)
                
                # Small delay to allow visualization
                time.sleep(0.1)
        
        # Final update
        if not st.session_state.get('stop_simulation', False):
            st.success(f"✅ Simulation completed in {time.time() - start_time:.2f} seconds")
    
    except Exception as e:
        st.error(f"❌ Error in simulation: {str(e)}")
        raise e

def init_environment(uploaded_file=None, threshold=200, width=100, height=100, num_obstacles=20):
    if uploaded_file is not None:
        # Process uploaded floor plan
        grid, obstacles, width, height = process_floor_plan(uploaded_file, threshold)
        return grid, obstacles, width, height
    else:
        # Create a grid with random obstacles
        grid = np.zeros((height, width), dtype=int)
        
        # Add border walls
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1
        
        # Add random obstacles
        obstacles = []
        for _ in range(num_obstacles):
            x = random.randint(1, width-2)
            y = random.randint(1, height-2)
            w = random.randint(1, 5)
            h = random.randint(1, 5)
            
            # Make sure obstacle is within bounds
            if x + w >= width - 1:
                w = width - x - 2
            if y + h >= height - 1:
                h = height - y - 2
                
            if w > 0 and h > 0:
                grid[y:y+h, x:x+w] = 1
                obstacles.append((x, y, w, h))
        
        return grid, obstacles, width, height
def process_floor_plan(uploaded_file, threshold=200):
    """
    Process floor plan image into a navigation grid with white as walkable space.
    Uses advanced image processing with adaptive thresholding and contour detection.

    Args:
        uploaded_file: Uploaded image file (PIL or stream)
        threshold (int): Grayscale threshold (0-255) for initial binarization

    Returns:
        tuple: (grid, obstacles, width, height, original_img, binary_vis)
            - grid: 2D numpy array (0 = walkable, 1 = obstacle)
            - obstacles: List of (x, y) coordinates of obstacles
            - width, height: Dimensions of the grid
            - original_img: RGB image as numpy array (for overlaying path)
            - binary_vis: RGB visualization image (red for obstacles)
    """
    import cv2
    import numpy as np
    from PIL import Image, ImageEnhance
    
    try:
        # Read and preprocess image
        img_pil = Image.open(uploaded_file)
        original_img = np.array(img_pil.convert('RGB'))  # Save original for overlay
        
        # Convert to grayscale if needed
        if img_pil.mode != 'L':
            img = img_pil.convert('L')
        else:
            img = img_pil
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)  # Increase contrast
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Auto-detect if we need to invert (black walls vs white walls)
        if np.mean(img_array) > 128:  # If mostly white
            # Invert so walls are black
            img_array = 255 - img_array
        
        # Apply adaptive thresholding to handle varying lighting
        binary = cv2.adaptiveThreshold(
            img_array, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours of obstacles
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter out small contours (noise)
        min_contour_area = 50
        contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
        
        # Create a new grid with filled obstacles
        height, width = binary.shape
        filled_grid = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(filled_grid, contours, -1, 1, thickness=cv2.FILLED)
        
        # Get obstacle coordinates (convert to 0=walkable, 1=obstacle)
        grid = filled_grid.astype(np.uint8)
        obstacles = np.argwhere(grid == 1)
        
        # Create visualization image (red for obstacles)
        binary_vis = original_img.copy()
        binary_vis[grid == 1] = [255, 0, 0]  # Red for obstacles
        
        return grid, obstacles.tolist(), width, height, original_img, binary_vis
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        empty_grid = np.zeros((100, 100), dtype=np.uint8)
        empty_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        return empty_grid, [], 100, 100, empty_rgb, empty_rgb

# Helper function to process floor plan image
def process_floor_plan_dep(uploaded_file, threshold=128):
    """Process floor plan image into a navigation grid with advanced preprocessing.
    
    Args:
        uploaded_file: Uploaded image file
        threshold: Grayscale threshold (0-255) for initial binarization
        
    Returns:
        tuple: (grid, obstacles, width, height)
            - grid: 2D numpy array (0=walkable, 1=obstacle)
            - obstacles: List of (x,y) coordinates of obstacles
            - width, height: Dimensions of the grid
    """
    import cv2
    import numpy as np
    from PIL import Image, ImageEnhance
    
    try:
        # Read and preprocess image
        img = Image.open(uploaded_file)
        
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)  # Increase contrast
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Auto-detect if we need to invert (black walls vs white walls)
        if np.mean(img_array) > 128:  # If mostly white
            # Invert so walls are black
            img_array = 255 - img_array
        
        # Apply adaptive thresholding to handle varying lighting
        binary = cv2.adaptiveThreshold(
            img_array, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours of obstacles
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter out small contours (noise)
        min_contour_area = 50
        contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
        
        # Create a new grid with filled obstacles
        height, width = binary.shape
        filled_grid = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(filled_grid, contours, -1, 1, thickness=cv2.FILLED)
        
        # Get obstacle coordinates
        obstacles = np.argwhere(filled_grid == 1)
        
        # Create visualization
        vis_img = np.zeros((height, width, 3), dtype=np.uint8)
        vis_img[filled_grid == 1] = (0, 0, 255)  # Red for obstacles
        
        # Show the processed image in the sidebar
        st.sidebar.image(
            vis_img,
            caption='Processed Floor Plan (Red=obstacles)',
            use_column_width=True,
            clamp=True,
            channels='BGR'
        )
        
        st.sidebar.text(f'Grid size: {width}x{height}')
        st.sidebar.text(f'Found {len(obstacles)} obstacle points')
        
        return filled_grid, obstacles.tolist(), width, height
        
    except Exception as e:
        st.error(f"Error processing floor plan: {str(e)}")
        # Return empty grid if there's an error
        empty_grid = np.zeros((100, 100), dtype=np.uint8)
        return empty_grid, [], 100, 100

# Function to create agents with valid starting positions
def create_agents(n, grid, start_region=None, exit_regions=None, speed=2.0, panic=0.2):
    """Create n agents with valid starting positions.
    
    Args:
        n: Number of agents to create
        grid: 2D numpy array representing the environment (0=walkable, 1=obstacle)
        start_region: Either a tuple (x1, y1, x2, y2) defining a rectangular region
                     or a list of (x,y) coordinates where agents can start
        exit_regions: List of tuples, where each tuple is (x1, y1, x2, y2) defining
                     rectangular exit regions, or a list of lists of (x,y) coordinates
        speed: Base movement speed for agents
        panic: Initial panic level for agents (0.0 to 1.0)
    """
    agents = []
    height, width = grid.shape
    
    # Handle start_region
    if start_region is None:
        # If no start region specified, use all walkable cells
        start_region = [(x, y) for y in range(height) for x in range(width) 
                       if grid[y, x] == 0]
    elif isinstance(start_region, tuple) and len(start_region) == 4:
        # Convert (x1, y1, x2, y2) to list of coordinates
        x1, y1, x2, y2 = start_region
        start_region = [(x, y) for x in range(max(0, x1), min(width, x2)) 
                               for y in range(max(0, y1), min(height, y2))
                               if grid[y, x] == 0]
    
    # Handle exit_regions
    if exit_regions is None:
        # If no exit regions, use the edges of the map
        exit_regions = [
            [(x, 0) for x in range(width) if grid[0, x] == 0],  # Top edge
            [(x, height-1) for x in range(width) if grid[height-1, x] == 0],  # Bottom edge
            [(0, y) for y in range(height) if grid[y, 0] == 0],  # Left edge
            [(width-1, y) for y in range(height) if grid[y, width-1] == 0]  # Right edge
        ]
    elif exit_regions and isinstance(exit_regions[0], tuple) and len(exit_regions[0]) == 4:
        # Convert list of (x1,y1,x2,y2) to list of coordinate lists
        new_exit_regions = []
        for region in exit_regions:
            if isinstance(region, tuple) and len(region) == 4:
                x1, y1, x2, y2 = region
                new_region = [(x, y) for x in range(max(0, x1), min(width, x2)) 
                                    for y in range(max(0, y1), min(height, y2))
                                    if grid[y, x] == 0]
                if new_region:  # Only add non-empty regions
                    new_exit_regions.append(new_region)
        exit_regions = new_exit_regions
    
    # If no valid exit regions found, use corners as fallback
    if not exit_regions or all(not region for region in exit_regions):
        exit_regions = [
            [(0, 0)],
            [(width-1, 0)],
            [(0, height-1)],
            [(width-1, height-1)]
        ]
    
    # Filter out any empty exit regions
    exit_regions = [region for region in exit_regions if region]
    if not exit_regions:
        raise ValueError("No valid exit regions found in the environment")
    
    # Ensure we have at least one valid start position
    if not start_region:
        start_region = [(x, y) for y in range(height) for x in range(width) 
                       if grid[y, x] == 0]
        if not start_region:
            raise ValueError("No valid starting positions found in the environment")
    
    # Create a mapping of group IDs to exit points
    group_goals = {}
    group_id = 0
    group_size = np.random.randint(3, 7)  # Start with 3-6 agents in first group
    
    for i in range(min(n, len(start_region))):  # Don't exceed available start positions
        # Start a new group if needed
        if i % group_size == 0:
            group_id += 1
            # Choose a random exit region for this group
            exit_region = exit_regions[np.random.randint(0, len(exit_regions))]
            exit_idx = np.random.randint(0, len(exit_region))
            group_goals[group_id] = exit_region[exit_idx]
            
            # Randomly decide group size for next group (3-6 agents)
            group_size = np.random.randint(3, 7)
        
        # Get random start position from start region
        if not start_region:
            break
            
        start_idx = np.random.randint(0, len(start_region))
        start_x, start_y = start_region.pop(start_idx)
        
        # Use the group's exit point
        goal_x, goal_y = group_goals[group_id]
        
        # Make sure goal is not inside an obstacle
        while 0 <= goal_y < height and 0 <= goal_x < width and grid[goal_y, goal_x] == 1 and len(exit_region) > 1:
            exit_region.pop(exit_idx)
            if exit_region:  # Check if there are still points left
                exit_idx = np.random.randint(0, len(exit_region))
                goal_x, goal_y = exit_region[exit_idx]
            else:
                # If no more points in this exit region, pick a new one
                exit_region = exit_regions[np.random.randint(0, len(exit_regions))]
                exit_idx = np.random.randint(0, len(exit_region))
                goal_x, goal_y = exit_region[exit_idx]
        
        # Create agent with group information
        agents.append(Agent(
            i,
            start=(start_x, start_y),
            goal=(goal_x, goal_y),
            profile={
                'speed': speed * (0.8 + 0.4 * np.random.random()),  # Randomize speed slightly
                'panic': panic,
                'group_id': group_id,  # Assign to group
                'size': 1.0,
                'priority': 1.0,
                'group_goal': (goal_x, goal_y),  # Shared goal for group
                'group_radius': 5.0,  # How far group members can stray
                'cohesion_strength': 0.3,  # How strongly to stay with group
                'alignment_strength': 0.4,  # How much to align with group
                'separation_strength': 0.2  # How much to separate from group
            }
        ))
    
    return agents

# Set page config
st.set_page_config(
    page_title="Crowd Simulation",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("👥 Crowd Simulation with NavMesh")
st.markdown("""
Simulate crowd behavior with pathfinding, obstacle avoidance, and panic dynamics.
Use the sidebar to configure the simulation parameters.
""")

# Sidebar for controls
with st.sidebar:
    st.header("Simulation Controls")
    
    # Simulation parameters
    st.subheader("Parameters")
    num_agents = st.slider("Number of Agents", 10, 200, 30, 10, key="init_num_agents_slider")
    agent_speed = st.slider("Agent Speed", 0.5, 5.0, 2.0, 0.5, key="init_agent_speed_slider")
    panic_level = st.slider("Initial Panic Level", 0.0, 1.0, 0.2, 0.1, key="init_panic_level_slider")
    
    # Environment parameters
    st.subheader("Environment")
    width = st.slider("Width", 50, 200, 100, 10, key="env_width_slider")
    height = st.slider("Height", 50, 200, 100, 10, key="env_height_slider")
    num_obstacles = st.slider("Number of Obstacles", 0, 100, 20, 5, key="num_obstacles_slider")
    
    # Buttons
    st.subheader("Actions")
    run_simulation = st.button("🚀 Run Simulation")
    stop_simulation = st.button("⏹️ Stop")
    
    # Info
    st.markdown("---")
    st.info("Adjust parameters and click 'Run Simulation' to start.")

def initialize_simulation_parameters():
    # Default values
    params = {
        'width': 100,
        'height': 100,
        'num_agents': 30,
        'agent_speed': 2.0,
        'panic_level': 0.2,
        'threshold': 200,
        'uploaded_file': None
    }
    
    # Initialize session state for simulation control
    if 'stop_simulation' not in st.session_state:
        st.session_state.stop_simulation = False
    
    return params

# Initialize parameters
params = initialize_simulation_parameters()

# Initialize session state for agent tracking
if 'agent_paths' not in st.session_state:
    st.session_state.agent_paths = defaultdict(list)
if 'agent_metrics' not in st.session_state:
    st.session_state.agent_metrics = {}
if 'selected_agents' not in st.session_state:
    st.session_state.selected_agents = set()
if 'selected_agent' not in st.session_state:
    st.session_state.selected_agent = 0

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    # File uploader for floor plan
    st.subheader("Upload Floor Plan")
    uploaded_file = st.file_uploader("Upload a floor plan image (black/white recommended)", 
                                   type=['png', 'jpg', 'jpeg'])
    
    # If we have an uploaded file, get its dimensions
    if uploaded_file is not None:
        try:
            # Just peek at the image dimensions without processing the whole thing
            image = Image.open(uploaded_file)
            params['width'], params['height'] = image.size
            # Reset file pointer to the beginning
            uploaded_file.seek(0)
            params['uploaded_file'] = uploaded_file
        except Exception as e:
            st.warning(f"Could not read image dimensions: {e}. Using default size.")
    
    # Threshold slider for image processing
    params['threshold'] = st.slider(
        "Obstacle Threshold", 
        min_value=0, 
        max_value=255, 
        value=200, 
        step=1,
        help="Adjust to fine-tune obstacle detection",
        key="obstacle_threshold_slider"
    )
    
    # Placeholder for the simulation
    simulation_placeholder = st.empty()
    
    # Status
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    # Initialize environment for agent selection
    if 'env_initialized' not in st.session_state or st.session_state.env_initialized is False:
        with status_text.status("Initializing environment..."):
            grid, obstacles, width, height = init_environment(
                uploaded_file=params.get('uploaded_file'),
                threshold=params['threshold'],
                width=params['width'],
                height=params['height']
            )
            st.session_state.env_initialized = True
            st.session_state.grid = grid
            st.session_state.obstacles = obstacles
            st.session_state.width = width
            st.session_state.height = height
    
    # Agent selection before simulation starts
    if 'agents' not in st.session_state and 'env_initialized' in st.session_state and st.session_state.env_initialized:
        st.subheader("Configure Simulation")
        params['num_agents'] = st.slider(
            "Number of Agents", 
            min_value=1, 
            max_value=200, 
            value=30, 
            step=1,
            key="config_num_agents_slider"
        )
        params['agent_speed'] = st.slider(
            "Agent Speed", 
            min_value=0.5, 
            max_value=5.0, 
            value=2.0, 
            step=0.1,
            key="config_agent_speed_slider"
        )
        params['panic_level'] = st.slider(
            "Initial Panic Level", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.2, 
            step=0.05,
            key="config_panic_level_slider"
        )
        
        # Create start and exit regions
        start_region = (10, 10, 20, 20)  # x1, y1, x2, y2
        exit_regions = [(80, 80, 90, 90)]  # List of (x1, y1, x2, y2) regions
        
        if st.button("Initialize Agents"):
            with status_text.status("Creating agents..."):
                agents = create_agents(
                    params['num_agents'],
                    st.session_state.grid,
                    start_region,
                    exit_regions,
                    speed=params['agent_speed'],
                    panic=params['panic_level']
                )
                st.session_state.agents = agents
                st.session_state.selected_agents = set()
                st.write(f"Initialized {len(agents)} agents")
                st.session_state.run_simulation = False  # Reset run state
                st.rerun()
    
    # Show agent selection UI after agents are initialized
    if 'agents' in st.session_state:
        st.subheader("Select Agents to Track")
        agents = st.session_state.agents
        agent_ids = [agent.agent_id for agent in agents]
        
        # Initialize selected_agents if not exists
        if 'selected_agents' not in st.session_state:
            st.session_state.selected_agents = set()
        
        # Convert current selected_agents to a list of valid agent IDs
        valid_selected = [aid for aid in st.session_state.selected_agents if aid in agent_ids]
        
        # Create a container for the agent selection to maintain state
        agent_container = st.container()
        
        # Add a button to clear selection
        if st.button("Clear Selection"):
            st.session_state.selected_agents = set()
            st.rerun()
            
        # Multi-select for agents - using a unique key that persists
        with agent_container:
            selected_agents = st.multiselect(
                "Select one or more agents to track:",
                agent_ids,
                default=valid_selected,
                key=f"agent_selector_{len(agent_ids)}"  # Include count in key to force refresh when agents change
            )
        
        # Update the selected_agents set with the new selection
        st.session_state.selected_agents = set(selected_agents)
        
        # Group selection
        group_ids = sorted(list(set(agent.profile.get('group_id', 0) for agent in agents)))
        selected_group = st.selectbox(
            "Or quickly select a group:",
            ["None"] + [f"Group {gid}" for gid in group_ids],
            index=0,
            key="group_selector"
        )
        
        # If a group is selected, replace current selection with that group
        if selected_group != "None":
            group_id = int(selected_group.split()[-1])
            group_agents = {a.agent_id for a in agents if a.profile.get('group_id') == group_id}
            st.session_state.selected_agents = group_agents
            st.rerun()
        
        # Display selection info
        num_selected = len(st.session_state.selected_agents)
        if num_selected == 0:
            st.warning("No agents selected. All agents will be shown in the simulation.")
        elif num_selected == 1:
            st.success(f"1 agent selected for tracking")
        else:
            st.success(f"{num_selected} agents selected for tracking")

# Handle simulation stop
if 'stop_simulation' in st.session_state and st.session_state.stop_simulation:
    st.session_state.run_simulation = False
    st.session_state.stop_simulation = False
    st.rerun()

# Debug output
st.write("Debug - Simulation State:", {
    'run_simulation': st.session_state.get('run_simulation', False),
    'agents_in_session': 'agents' in st.session_state,
    'agent_count': len(st.session_state.get('agents', [])),
    'grid_exists': 'grid' in st.session_state
})

# Check if run button was clicked and we have agents
if st.session_state.get('run_simulation') and 'agents' in st.session_state:
    # Get agents and environment from session state
    agents = st.session_state.get('agents', [])
    grid = st.session_state.get('grid')
    obstacles = st.session_state.get('obstacles', [])
    width = st.session_state.get('width', 100)
    height = st.session_state.get('height', 100)
    
    if not agents or grid is None:
        st.error("Agents not properly initialized. Please initialize agents first.")
        st.session_state.run_simulation = False
    else:
        # Clear any previous simulation output
        st.session_state.simulation_output = None
        
        # Display status
        status_text = st.empty()
        status_text.info("🚀 Starting simulation...")
        
        # Run the simulation
        try:
            run_sim({
                'num_agents': len(agents),
                'agent_speed': params['agent_speed'],
                'panic_level': params['panic_level'],
                'grid': grid,
                'obstacles': obstacles,
                'width': width,
                'height': height,
                'uploaded_file': params['uploaded_file'],
                'threshold': params['threshold']
            })
            status_text.success("✅ Simulation completed successfully!")
        except Exception as e:
            st.error(f"❌ Error running simulation: {str(e)}")
            status_text.error(f"❌ Error: {str(e)}")
        finally:
            st.session_state.run_simulation = False

with col2:
    # Metrics
    st.subheader("Simulation Metrics")
    metrics_placeholder = st.empty()
    
    # Simulation controls - only show these if we haven't initialized agents yet
    if 'agents' not in st.session_state:
        st.subheader("Simulation Controls")
        params['num_agents'] = st.slider(
            "Number of Agents", 
            min_value=1, 
            max_value=200, 
            value=30, 
            step=1,
            key="num_agents_slider"
        )
        params['agent_speed'] = st.slider(
            "Agent Speed", 
            min_value=0.5, 
            max_value=5.0, 
            value=2.0, 
            step=0.1,
            key="agent_speed_slider"
        )
        params['panic_level'] = st.slider(
            "Initial Panic Level", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.2, 
            step=0.05,
            key="panic_level_slider"
        )
    
    # Buttons
    col1, col2 = st.columns(2)
    with col1:
        run_clicked = st.button("🚀 Run Simulation", key="run_button")
        if run_clicked:
            st.write("Run button clicked!")
            st.session_state.run_simulation = True
            st.session_state.stop_simulation = False
            st.session_state.simulation_started = True
            st.write("Session state updated, rerunning...")
            st.rerun()
    with col2:
        stop_clicked = st.button("⏹️ Stop", key="stop_button")
        if stop_clicked:
            st.write("Stop button clicked!")
            st.session_state.stop_simulation = True
            st.session_state.run_simulation = False
            st.rerun()
    
    # Debug info
    if st.checkbox("Show debug info"):
        st.write("Session state:", st.session_state)



# Run simulation
def run_sim(params):
    try:
        # Get agents and environment from session state
        agents = st.session_state.get('agents', [])
        grid = st.session_state.get('grid')
        obstacles = st.session_state.get('obstacles', [])
        width = st.session_state.get('width', 100)
        height = st.session_state.get('height', 100)
        
        if not agents or grid is None:
            st.error("Agents not initialized. Please initialize agents first.")
            return
            
        # Create navmesh and add obstacles
        navmesh = GridNavMesh(width, height)
        navmesh.grid = grid
        
        # Filter agents based on selection if any agents are selected
        selected_agents = st.session_state.get('selected_agents', set())
        if selected_agents:
            agents = [agent for agent in agents if agent.agent_id in selected_agents]
            if not agents:
                st.warning("No valid agents selected. Using all agents.")
                agents = st.session_state.agents
        
        # Initialize hazard map and density map
        navmesh.hazard_map = np.zeros((height, width), dtype=np.float32)
        navmesh.density_map = np.zeros((height, width), dtype=np.float32)
        
        # Create crowd simulation with the selected agents
        exits = [(0, 0), (width-1, 0), (0, height-1), (width-1, height-1)]  # Corners as exits
        crowd_sim = CrowdSim(agents=agents, exits=exits, navmesh=navmesh)
        
        # Main simulation loop
        max_steps = 500  # Increased for more complex environments
        start_time = time.time()
        
        for step in range(max_steps):
            # Update simulation
            crowd_sim.step_sim()
            
            # Update visualization every 5 steps
            if step % 5 == 0:
                # Create visualization
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Draw floor plan if available
                if uploaded_file is not None:
                    ax.imshow(255 - grid * 255, cmap='gray', alpha=0.7)
                else:
                    ax.imshow(grid, cmap='binary', alpha=0.3)
                
                # Track agent paths and metrics
                current_agent_metrics = {}
                
                # Group agents by their group_id for visualization
                groups = {}
                for agent in agents:
                    if not agent.state.get('exited', False):
                        group_id = agent.profile.get('group_id', 0)
                        if group_id not in groups:
                            groups[group_id] = []
                        groups[group_id].append(agent)
                
                # Draw groups and agents
                for group_id, group_agents in groups.items():
                    # Calculate group center and average velocity
                    if not group_agents:
                        continue
                        
                    group_center = np.mean([a.state['pos'] for a in group_agents], axis=0)
                    
                    # Draw group boundary (transparent circle)
                    if len(group_agents) > 1:  # Only draw for actual groups
                        group_radius = max(5, len(group_agents) * 2)  # Radius based on group size
                        group_circle = plt.Circle(
                            (group_center[0], group_center[1]), 
                            group_radius, 
                            color='blue', 
                            alpha=0.1,
                            fill=True
                        )
                        ax.add_patch(group_circle)
                
                # Draw agents and their paths
                for agent in agents:
                    pos = agent.state['pos']
                    agent_id = agent.agent_id
                    group_id = agent.profile.get('group_id', 0)
                    
                    # Update agent path
                    st.session_state.agent_paths[agent_id].append(pos.copy())
                    
                    # Store current metrics
                    metrics = agent.get_metrics()
                    metrics['step'] = step
                    metrics['timestamp'] = time.time()
                    current_agent_metrics[agent_id] = metrics
                    
                    # Draw agent if not exited
                    if not agent.state.get('exited', False):
                        # Determine color based on selection and group
                        is_selected = agent_id in st.session_state.selected_agents
                        is_in_selected_group = any(
                            a.profile.get('group_id') == group_id 
                            for a in agents 
                            if a.agent_id in st.session_state.selected_agents
                        )
                        
                        # Get color from colormap based on group ID
                        cmap = plt.cm.get_cmap('tab20')
                        group_color = cmap(group_id % 20)
                        
                        if is_selected:
                            color = 'red'
                            size = 40
                            alpha = 1.0
                        elif is_in_selected_group:
                            color = 'orange'
                            size = 35
                            alpha = 0.9
                        else:
                            # Use group color for unselected agents
                            color = group_color
                            size = 30
                            alpha = 0.8
                        
                        # Draw agent
                        ax.scatter(
                            pos[0], 
                            pos[1], 
                            c=[color], 
                            s=size, 
                            alpha=alpha,
                            edgecolors='black' if is_selected or is_in_selected_group else 'none',
                            linewidth=2 if is_selected or is_in_selected_group else 0.5,
                            zorder=10 if is_selected or is_in_selected_group else 1
                        )
                        
                        # Draw path for selected agents and their group
                        if is_selected or is_in_selected_group:
                            path = np.array(st.session_state.agent_paths[agent_id])
                            if len(path) > 1:
                                ax.plot(
                                    path[:, 0], 
                                    path[:, 1], 
                                    '-', 
                                    color=color, 
                                    linewidth=2, 
                                    alpha=0.5,
                                    zorder=5
                                )
                        
                        # Draw line to goal for selected agents
                        if is_selected:
                            goal = agent.goal
                            ax.plot(
                                [pos[0], goal[0]], 
                                [pos[1], goal[1]], 
                                'g--', 
                                alpha=0.3,
                                zorder=1
                            )
                            ax.scatter(
                                goal[0], 
                                goal[1], 
                                c='green', 
                                marker='*', 
                                s=100, 
                                alpha=0.5,
                                zorder=1
                            )
                
                # Update agent metrics
                st.session_state.agent_metrics = current_agent_metrics
                
                ax.set_xlim(0, width)
                ax.set_ylim(height, 0)  # Invert y-axis for image coordinates
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"Step {step}")
                
                # Update the plot in Streamlit
                simulation_placeholder.pyplot(fig)
                plt.close(fig)
                
                # Calculate metrics
                num_exited = sum(1 for agent in agents if agent.state.get('exited', False))
                avg_panic = np.mean([agent.state.get('panic', 0) for agent in agents])
                elapsed = time.time() - start_time
                
                # Agent selection
                agent_ids = [agent.agent_id for agent in agents]
                
                # Initialize selected_agents if not exists
                if 'selected_agents' not in st.session_state:
                    st.session_state.selected_agents = set()
                
                # Only update the multiselect if agent_ids is not empty
                if agent_ids:
                    # Convert current selected_agents to a list of valid agent IDs
                    valid_selected = [aid for aid in st.session_state.selected_agents if aid in agent_ids]
                    
                    # Create a unique key for the multiselect
                    multiselect_key = f"agent_selector_{int(time.time())}"
                    
                    # Multi-select for agents
                    selected_agents = st.sidebar.multiselect(
                        "Select Agents",
                        agent_ids,
                        default=valid_selected,
                        key=multiselect_key
                    )
                    
                    # Update the selected_agents set with the new selection
                    st.session_state.selected_agents = set(selected_agents)
                    
                    # Group selection
                    group_ids = sorted(list(set(agent.profile.get('group_id', 0) for agent in agents)))
                    group_select_key = f"group_selector_{int(time.time())}"
                    selected_group = st.sidebar.selectbox(
                        "Select Group",
                        ["None"] + [f"Group {gid}" for gid in group_ids],
                        index=0,
                        key=group_select_key
                    )
                    
                    # If a group is selected, add all agents from that group to selected_agents
                    if selected_group != "None":
                        group_id = int(selected_group.split()[-1])
                        group_agents = {a.agent_id for a in agents if a.profile.get('group_id') == group_id}
                        st.session_state.selected_agents.update(group_agents)
                
                # Get metrics for all selected agents
                selected_metrics = [st.session_state.agent_metrics.get(aid, {}) for aid in st.session_state.selected_agents]
                agent_metrics = selected_metrics[0] if selected_metrics else {}
                
                # Update metrics
                with metrics_placeholder.container():
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Simulation Metrics")
                        st.metric("Steps", f"{step}/{max_steps}")
                        st.metric("Elapsed Time", f"{elapsed:.1f}s")
                        st.metric("Agents", f"{num_exited}/{len(agents)} exited")
                        st.metric("Completion", f"{(num_exited/len(agents)*100):.1f}%")
                        st.metric("Average Panic", f"{avg_panic:.2f}")
                    
                    with col2:
                        st.markdown("### Selected Agent")
                        if not st.session_state.selected_agents:
                            st.info("No agents selected")
                        elif len(st.session_state.selected_agents) == 1:
                            # Single agent metrics
                            if agent_metrics:
                                st.metric("Agent ID", agent_metrics.get('agent_id', 'N/A'))
                                st.metric("Status", agent_metrics.get('status', 'N/A'))
                                st.metric("Group ID", f"Group {agent_metrics.get('group_id', 'N/A')}")
                                st.metric("Panic Level", f"{agent_metrics.get('panic', 0):.2f}")
                                st.metric("Speed", f"{agent_metrics.get('speed', 0):.2f}")
                                st.metric("Distance to Goal", f"{agent_metrics.get('goal_distance', 0):.1f}")
                                
                                # Show behavior analysis
                                if agent_metrics.get('panic', 0) > 0.7:
                                    st.warning("⚠️ Agent is in panic mode!")
                                if agent_metrics.get('collisions', 0) > 3:
                                    st.warning(f"⚠️ High collision count: {agent_metrics.get('collisions', 0)}")
                        else:
                            # Multiple agents selected - show group metrics
                            st.metric("Agents Selected", len(st.session_state.selected_agents))
                            
                            # Calculate group statistics
                            group_metrics = {
                                'avg_panic': np.mean([m.get('panic', 0) for m in selected_metrics]),
                                'avg_speed': np.mean([m.get('speed', 0) for m in selected_metrics]),
                                'group_id': selected_metrics[0].get('group_id', 'N/A') if selected_metrics else 'N/A',
                                'panicked': sum(1 for m in selected_metrics if m.get('panic', 0) > 0.7)
                            }
                            
                            st.metric("Group ID", f"Group {group_metrics['group_id']}")
                            st.metric("Average Panic", f"{group_metrics['avg_panic']:.2f}")
                            st.metric("Average Speed", f"{group_metrics['avg_speed']:.2f}")
                            
                            if group_metrics['panicked'] > 0:
                                st.warning(f"⚠️ {group_metrics['panicked']} agents in panic mode!")
                            
                            # Show agents in group
                            with st.expander("Group Members"):
                                for i, agent_id in enumerate(st.session_state.selected_agents):
                                    agent = next((a for a in agents if a.agent_id == agent_id), None)
                                    if agent:
                                        status = "🟢" if not agent.state.get('exited', False) else "✅"
                                        panic = "😨" if agent.state.get('panic', 0) > 0.7 else "😊"
                                        st.write(f"{status} {panic} Agent {agent_id} (Panic: {agent.state.get('panic', 0):.2f}, Speed: {agent.get_speed():.1f})")
                
                # Update progress
                progress = (step + 1) / max_steps
                progress_bar.progress(min(progress, 1.0))
                
                # Add a small delay to see the animation
                time.sleep(0.1)
                
                # Check if all agents have exited
                if all(agent.state.get('exited', False) for agent in agents):
                    status_text.success(f"✅ All agents have reached their goals in {step} steps!")
                    break
                
                # Check if we should stop
                if 'stop_simulation' in st.session_state and st.session_state.stop_simulation:
                    status_text.warning("⏹️ Simulation stopped by user.")
                    st.session_state.stop_simulation = False
                    return
        else:
            status_text.warning(f"⏱️ Simulation completed after {max_steps} steps with {sum(1 for a in agents if not a.state.get('exited', False))} agents not reaching their goals.")
    
    except Exception as e:
        status_text.error(f"❌ Error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        if 'grid' in locals():
            st.write(f"Grid shape: {grid.shape}")
        raise e

# Handle simulation state display
if st.session_state.get('run_simulation'):
    st.info("Simulation is running...")
elif st.session_state.get('stop_simulation'):
    st.warning("Simulation stopped.")

# Add some documentation
with st.expander("About this Simulation"):
    st.markdown("""
    ## Crowd Simulation with NavMesh
    
    This simulation demonstrates crowd behavior with the following features:
    
    - **Pathfinding**: Agents navigate around obstacles to reach their goals
    - **Panic Dynamics**: Agents can enter a panic state that affects their movement
    - **Obstacle Avoidance**: Dynamic obstacle detection and avoidance
    - **Metrics**: Real-time tracking of simulation metrics
    
    ### Controls
    - Adjust the number of agents, their speed, and initial panic level
    - Configure the environment size and number of obstacles
    - Click 'Run Simulation' to start
    
    The simulation will run for a fixed number of steps or until all agents reach their goals.
    """)
