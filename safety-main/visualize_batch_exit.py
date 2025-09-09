import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from navmesh import GridNavMesh
import random

# --- Parameters ---
ROOM_WIDTH = 20
ROOM_HEIGHT = 15
N_USERS = 60
N_BATCHES = 2
USERS_PER_BATCH = N_USERS // N_BATCHES
BATCH_SPEEDS = [1] * 30 + [0.5] * 30  # First 30 users at 1 cell/step, next 30 at 0.5 cell/step
N_DOORS = 3

# --- Room setup ---
def create_room(width, height, n_doors):
    """
    Create a grid with at least 6 rooms, separated by internal walls with doors between rooms.
    Ensure every room has at least one door to the outside.
    Returns the grid and a list of door positions (exits to outside only).
    """
    grid = np.zeros((height, width), dtype=np.int32)
    room_rows = 2
    room_cols = 3  # 2x3 = 6 rooms
    room_w = width // room_cols
    room_h = height // room_rows
    internal_doors = []
    # Draw internal vertical walls
    for c in range(1, room_cols):
        x = c * room_w
        for y in range(height):
            grid[y, x] = 1
        door_y = random.randint(1, height-2)
        grid[door_y, x] = 0
        internal_doors.append((x, door_y))
    # Draw internal horizontal walls
    for r in range(1, room_rows):
        y = r * room_h
        for x in range(width):
            grid[y, x] = 1
        door_x = random.randint(1, width-2)
        grid[y, door_x] = 0
        internal_doors.append((door_x, y))
    # Place at least one door for each room on the outside wall
    door_positions = set()
    for r in range(room_rows):
        for c in range(room_cols):
            # For each room, pick a wall segment on the building's edge
            x0 = c * room_w
            y0 = r * room_h
            x1 = (c+1) * room_w - 1 if c < room_cols-1 else width-1
            y1 = (r+1) * room_h - 1 if r < room_rows-1 else height-1
            wall = random.choice(['top', 'bottom', 'left', 'right'])
            if wall == 'top':
                x = random.randint(x0+1, x1-1)
                y = y0
            elif wall == 'bottom':
                x = random.randint(x0+1, x1-1)
                y = y1
            elif wall == 'left':
                x = x0
                y = random.randint(y0+1, y1-1)
            else:
                x = x1
                y = random.randint(y0+1, y1-1)
            door_positions.add((x, y))
    # Optionally add extra doors randomly if n_doors > number of rooms
    extra_doors = max(0, n_doors - room_rows * room_cols)
    for _ in range(extra_doors):
        wall = random.choice(['top', 'bottom', 'left', 'right'])
        if wall == 'top':
            x = random.randint(1, width-2)
            y = 0
        elif wall == 'bottom':
            x = random.randint(1, width-2)
            y = height-1
        elif wall == 'left':
            x = 0
            y = random.randint(1, height-2)
        else:
            x = width-1
            y = random.randint(1, height-2)
        door_positions.add((x, y))
    # Set outside walls (except doors)
    for x in range(width):
        if (x, 0) not in door_positions:
            grid[0, x] = 1
        if (x, height-1) not in door_positions:
            grid[height-1, x] = 1
    for y in range(height):
        if (0, y) not in door_positions:
            grid[y, 0] = 1
        if (width-1, y) not in door_positions:
            grid[y, width-1] = 1
    return grid, list(door_positions)


# --- User setup ---
def random_empty_cell(grid, forbidden):
    h, w = grid.shape
    while True:
        x, y = random.randint(1, w-2), random.randint(1, h-2)
        if grid[y, x] == 0 and (x, y) not in forbidden:
            return (x, y)

def assign_users(grid, n_users, n_batches, users_per_batch):
    users = []
    forbidden = set()
    for i in range(n_users):
        pos = random_empty_cell(grid, forbidden)
        speed = BATCH_SPEEDS[i]
        batch_id = 0 if i < 30 else 1
        users.append({
            'pos': pos,
            'batch': batch_id,
            'speed': speed,
            'path': None,
            'exited': False,
            'collisions': 0,
            'chosen_door': None,
            'exit_time': None,
            'start_pos': pos,
            'door_dist': None,
            'waited': 0  # Track how many times user had to wait
        })
        forbidden.add(pos)
    return users

# --- Pathfinding for each user ---
def compute_paths(users, navmesh, doors):
    for user in users:
        if user['exited']:
            continue
        # Find shortest path to any door
        shortest = None
        for door in doors:
            path = navmesh.astar(user['pos'], door)
            if path and (shortest is None or len(path) < len(shortest)):
                shortest = path
        user['path'] = shortest

# --- Simulation step ---
def step_users(users, navmesh, doors, occupied=None, t=None):
    # occupied: set of positions taken this step (for collision stats)
    if occupied is None:
        occupied = set()
    pos_to_users = {}
    for idx, user in enumerate(users):
        if user['exited'] or not user['path'] or len(user['path']) <= 1:
            continue
        # For speed 0.5, move only on even time steps
        if user['speed'] == 0.5 and (t is not None and t % 2 != 0):
            user['waited'] += 1
            continue
        steps = min(int(user['speed']), len(user['path']) - 1)
        new_pos = user['path'][steps]
        if new_pos in occupied:
            user['waited'] += 1
            continue  # Wait if cell is occupied this step
        user['pos'] = new_pos
        user['path'] = user['path'][steps:]
        occupied.add(new_pos)
        # Check for collision: if multiple users want same cell
        if new_pos in pos_to_users:
            user['collisions'] += 1
            users[pos_to_users[new_pos]]['collisions'] += 1
        else:
            pos_to_users[new_pos] = idx
        if user['pos'] in doors:
            user['exited'] = True
            user['exit_time'] = t
            # Assign chosen door (first time only)
            if user['chosen_door'] is None:
                user['chosen_door'] = user['pos']

# --- Visualization ---
def visualize():
    grid, doors = create_room(ROOM_WIDTH, ROOM_HEIGHT, N_DOORS)
    navmesh = GridNavMesh(ROOM_WIDTH, ROOM_HEIGHT)
    navmesh.grid = grid.copy()
    users = assign_users(grid, N_USERS, N_BATCHES, USERS_PER_BATCH)
    compute_paths(users, navmesh, doors)

    # Data collection
    exit_counts = {door: [] for door in doors}
    exit_times = {door: [] for door in doors}
    door_choices = {door: 0 for door in doors}
    batch_collisions = [0 for _ in range(N_BATCHES)]
    batch_steps = [0 for _ in range(N_BATCHES)]
    batch_speeds = [BATCH_SPEEDS[i] for i in range(N_BATCHES)]
    user_door_dist = [None for _ in range(N_USERS)]
    user_door_choice = [None for _ in range(N_USERS)]
    user_exit_time = [None for _ in range(N_USERS)]

    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.colormaps['tab10']  # tab10 has 10 distinct colors
    
    def update(frame):
        # Update simulation state each frame
        step_users(users, navmesh, doors, t=frame)
        compute_paths(users, navmesh, doors)
        ax.clear()
        # Draw room
        ax.imshow(grid, cmap='Greys', vmin=0, vmax=2, alpha=0.3)
        # Draw doors
        for (x, y) in doors:
            ax.scatter(x, y, marker='s', c='yellow', s=100, edgecolors='black', label='Door' if frame == 0 else None)
        # Draw all user paths on the first frame
        if frame == 0:
            for idx, user in enumerate(users):
                if user['path'] is not None and len(user['path']) > 1:
                    path = np.array(user['path'])
                    color = cmap(user['batch'] % 10)
                    ax.plot(path[:,0], path[:,1], color=color, alpha=0.25, linewidth=2)
        # Draw users
        for idx, user in enumerate(users):
            if not user['exited']:
                color = [cmap(user['batch'] % 10)]
                ax.scatter(user['pos'][0], user['pos'][1], c=color, s=60, edgecolors='k', label=f'Batch {user["batch"]+1}' if frame == 0 else None)
            else:
                # Record exit data
                if user_exit_time[idx] is None:
                    user_exit_time[idx] = user['exit_time']
                    user_door_choice[idx] = user['chosen_door']
                    door_choices[user['chosen_door']] += 1
                    exit_times[user['chosen_door']].append(user['exit_time'])
        ax.set_title(f"Step {frame}: {sum(u['exited'] for u in users)}/{N_USERS} exited")
        ax.set_xlim(-0.5, ROOM_WIDTH-0.5)
        ax.set_ylim(-0.5, ROOM_HEIGHT-0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        # Legend (only once)
        if frame == 0:
            handles = [plt.Line2D([0],[0], marker='o', color='w', label=f'Batch {i+1}', markerfacecolor=cmap(i), markersize=10, markeredgecolor='k') for i in range(N_BATCHES)]
            handles.append(plt.Line2D([0],[0], marker='s', color='w', label='Door', markerfacecolor='yellow', markersize=10, markeredgecolor='k'))
            ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Run until all users have exited or max_steps reached
    max_steps = 1000
    def all_exited():
        return all(u['exited'] for u in users)
    frames = []
    for step in range(max_steps):
        if all_exited():
            break
        frames.append(step)
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=400, repeat=False, blit=False, init_func=lambda: None, repeat_delay=1000)
    plt.tight_layout()
    plt.show()

    # --- After animation: Analysis ---
    # Gather collision and step stats
    for idx, user in enumerate(users):
        batch_collisions[user['batch']] += user['collisions']
        batch_steps[user['batch']] += user['exit_time'] if user['exit_time'] is not None else 100
        # For pattern: record shortest path to each door at start
        navmesh = GridNavMesh(ROOM_WIDTH, ROOM_HEIGHT)
        navmesh.grid = grid.copy()
        dists = [len(navmesh.astar(user['start_pos'], door)) if navmesh.astar(user['start_pos'], door) else np.inf for door in doors]
        user_door_dist[idx] = dists
    # --- Print Matrix/Table ---
    import pandas as pd
    print("\nBatch Speed & Collision Matrix:")
    df = pd.DataFrame({
        'Batch': list(range(1, N_BATCHES+1)),
        'Speed': batch_speeds,
        'Collisions': batch_collisions,
        'Collision Ratio': [c/max(s,1) for c,s in zip(batch_collisions, batch_steps)],
        'Users': USERS_PER_BATCH
    })
    print(df)
    print("\nDoor Choice Matrix:")
    door_labels = [f"Door {i+1} {doors[i]}" for i in range(len(doors))]
    door_choice_counts = [door_choices[door] for door in doors]
    df2 = pd.DataFrame({'Door': door_labels, 'Users Exited': door_choice_counts})
    print(df2)
    # --- Plot exit time per door ---
    import matplotlib.pyplot as plt2
    plt2.figure(figsize=(8,5))
    for i, door in enumerate(doors):
        times = sorted(exit_times[door])
        if times:
            plt2.plot(times, range(1, len(times)+1), label=f"Door {i+1} {door}")
    plt2.xlabel("Time Step")
    plt2.ylabel("Cumulative Exits")
    plt2.title("Cumulative Exits per Door Over Time")
    plt2.legend()
    plt2.tight_layout()
    plt2.show()

if __name__ == '__main__':
    visualize()
