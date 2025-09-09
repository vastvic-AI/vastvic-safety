import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
from agent import Agent
from crowd_env import CrowdEnv
from navmesh_ml2d.ui.visualizer import CrowdVisualizer
from stampede_monitor import StampedeMonitor

# --- Large floor: 1000x1000, 6 rooms, lobby, 2 lobby exits ---
width, height = 100, 100
obstacles = []
room_w, room_h = 30, 30
lobby_w, lobby_h = 100, 20
rooms = []
room_doors = []
lobby_doors = []

# Place 6 rooms (3x2 grid), each 300x300, with a door to the lobby
for i in range(3):
    for j in range(2):
        x0, y0 = i*room_w, j*room_h
        rooms.append(((x0, y0), (x0+room_w-1, y0+room_h-1)))
        # Add walls
        for x in range(x0, x0+room_w):
            obstacles.append((x, y0))
            obstacles.append((x, y0+room_h-1))
        for y in range(y0, y0+room_h):
            obstacles.append((x0, y))
            obstacles.append((x0+room_w-1, y))
        # Add door in bottom wall to lobby
        door_x = x0 + room_w//2
        door_y = y0+room_h-1
        room_doors.append((door_x, door_y))
        obstacles.remove((door_x, door_y))

# Lobby is below rooms, at the bottom of the grid
lobby_y0 = height - lobby_h  # e.g., 80 if height=100 and lobby_h=20
# Remove lobby top wall for room doors
for (door_x, door_y) in room_doors:
    for i in range(-2, 3):
        if (door_x+i, lobby_y0) in obstacles:
            obstacles.remove((door_x+i, lobby_y0))
# Add lobby walls
for x in range(0, width):
    obstacles.append((x, lobby_y0))  # Top
    obstacles.append((x, lobby_y0+lobby_h-1))  # Bottom
for y in range(lobby_y0, lobby_y0+lobby_h):
    obstacles.append((0, y))
    obstacles.append((width-1, y))
# Add 2 lobby exits (bottom wall)
lobby_exit1 = (width//4, lobby_y0+lobby_h-1)
lobby_exit2 = (3*width//4, lobby_y0+lobby_h-1)
for dx in range(-5, 6):
    obstacles.remove((lobby_exit1[0]+dx, lobby_exit1[1]))
    obstacles.remove((lobby_exit2[0]+dx, lobby_exit2[1]))
lobby_doors = [lobby_exit1, lobby_exit2]

# Place agents in valid positions with clear paths to exits
import numpy as np
from agent import Agent
agents = []
np.random.seed(42)

# Create a navmesh to check for valid positions
from navmesh_ml2d.core.navmesh import GridNavMesh
navmesh = GridNavMesh(width, height)
# Initialize grid with obstacles
navmesh.grid = np.zeros((height, width), dtype=int)
for x, y in obstacles:
    if 0 <= x < width and 0 <= y < height:
        navmesh.grid[y, x] = 1

# Place agents near room doors with clear paths to lobby
group_id = 0
for room_idx, ((x0, y0), (x1, y1)) in enumerate(rooms):
    door_x, door_y = room_doors[room_idx % len(room_doors)]
    for i in range(5):  # 5 agents per room
        # Place agents near the door but inside the room
        start_x = door_x + np.random.randint(-5, 6)
        start_y = door_y - np.random.randint(1, 6)  # Inside the room
        
        # Ensure position is within room bounds and not on an obstacle
        start_x = max(x0 + 2, min(x1 - 2, start_x))
        start_y = max(y0 + 2, min(y1 - 2, start_y))
        
        # Choose the nearest exit
        exit_dists = [np.hypot(door_x - ex[0], door_y - ex[1]) for ex in lobby_doors]
        goal = lobby_doors[np.argmin(exit_dists)]
        
        agents.append(Agent(
            len(agents), 
            start=(start_x, start_y), 
            goal=goal, 
            profile={
                'speed': 1.5 + np.random.random(),  # Slightly randomized speeds
                'panic': 0.0, 
                'group_id': group_id,
                'size': 1.0, 
                'priority': 1.0
            }
        ))
    group_id += 1

# No scenario events
scenario = []

# --- Create environment and visualizer ---
print("Starting simulation with grid size:", width, height)
env = CrowdEnv(width, height, obstacles=obstacles, agents=agents, scenario=scenario)
visualizer = CrowdVisualizer(env)
stampede_monitor = StampedeMonitor(env)

import os
import matplotlib.pyplot as plt

# Create output directory for frames
os.makedirs('simulation_frames', exist_ok=True)

max_steps = 200
stampede_risk_steps = []
for step in range(max_steps):
    env.step()
    risk, metrics = stampede_monitor.step()
    if risk:
        print(f"[WARNING] Stampede risk detected at step {step}! Metrics: {metrics}")
        stampede_risk_steps.append((step, metrics))
    
    # Save frame every 5 steps
    if step % 5 == 0:
        visualizer.draw()
        plt.savefig(f'simulation_frames/frame_{step:04d}.png')
        plt.close()
    
    # Early exit if all agents have reached their goals
    if env.all_exited():
        print(f"All agents have exited at step {step}")
        break

print(f"Simulation completed. Frames saved to 'simulation_frames' directory.")

# Optionally, save stampede risk steps for reporting
if stampede_risk_steps:
    import csv
    with open('stampede_risk_steps.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['step'] + list(stampede_risk_steps[0][1].keys()))
        writer.writeheader()
        for step, metrics in stampede_risk_steps:
            row = {'step': step}
            row.update(metrics)
            writer.writerow(row)

# --- Print stats ---
print("\nFinal stats:", env.stats())
