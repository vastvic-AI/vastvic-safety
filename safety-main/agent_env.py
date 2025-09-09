import gym
from gym import spaces
import numpy as np
from navmesh import GridNavMesh

class NavMeshEnv(gym.Env):
    """
    Advanced 2D Navigation Environment with continuous movement, dynamic re-routing, agent profiles, hazard/density avoidance, and advanced crowd behaviors.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, width=10, height=10, obstacles=None, polygons=None, start=(0,0), goal=(9,9), agent_profile=None, scenario=None):
        super(NavMeshEnv, self).__init__()
        self.width = width
        self.height = height
        self.obstacles = obstacles or []
        self.polygons = polygons
        self.start = start
        self.goal = goal
        self.n_actions = 8  # 8 directions for continuous
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(low=0, high=max(width, height)-1, shape=(2,), dtype=np.float32)
        self.nm = GridNavMesh(width, height, self.obstacles, self.polygons)
        self.agent_profile = agent_profile or {'speed': 1.0, 'panic': 0.0, 'group_id': None, 'size': 1.0, 'priority': 1.0}
        self.scenario = scenario or []  # List of scripted events
        self.agent_state = {'pos': list(self.start), 'panic': self.agent_profile['panic'], 'group_id': self.agent_profile['group_id'], 'status': 'normal'}
        self.step_count = 0
        self.reset()

    def reset(self):
        self.agent_state['pos'] = list(self.start)
        self.agent_state['status'] = 'normal'
        self.step_count = 0
        return np.array(self.agent_state['pos'], dtype=np.float32)

    def step(self, action):
        x, y = self.agent_state['pos']
        speed = self.agent_profile['speed']
        # 8-way movement
        directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (1,1), (-1,1), (1,-1)]
        dx, dy = directions[action]
        nx, ny = x + dx * speed, y + dy * speed
        # Social force model: avoid crowded or hazardous areas
        if self.nm.hazard_map[int(ny), int(nx)] > 0:
            self.agent_state['panic'] += 0.1  # Increase panic
            self.agent_state['status'] = 'avoiding hazard'
        if self.nm.density_map[int(ny), int(nx)] > 3:
            self.agent_state['status'] = 'crowded'
            # Slow down or re-route
            nx, ny = x, y  # Stay in place if too crowded
        if self.nm.is_walkable(nx, ny):
            self.agent_state['pos'] = [nx, ny]
        else:
            # Dynamic re-routing
            new_path = self.nm.replan_if_blocked(self.agent_state['pos'], self.goal)
            if new_path and len(new_path) > 1:
                self.agent_state['pos'] = list(new_path[1])
        # Scenario scripting (e.g., block cell, trigger alarm)
        for event in self.scenario:
            if event['step'] == self.step_count:
                event['action'](self)
        done = tuple(map(int, self.agent_state['pos'])) == self.goal
        reward = 1.0 if done else -0.1
        self.step_count += 1
        return np.array(self.agent_state['pos'], dtype=np.float32), reward, done, {}

    def render(self, mode='human'):
        grid = np.copy(self.nm.grid)
        x, y = map(int, self.agent_state['pos'])
        grid[y, x] = 2  # agent
        gx, gy = self.goal
        grid[gy, gx] = 3  # goal
        print("\n".join([" ".join(["A" if cell==2 else "G" if cell==3 else "#" if cell==1 else "." for cell in row]) for row in grid]))

    @staticmethod
    def demo_astar_follow(width=10, height=10, obstacles=None, polygons=None, start=(0,0), goal=(9,9), agent_profile=None, scenario=None, max_steps=100):
        """Demonstrate following the A* path step by step, supporting advanced features."""
        env = NavMeshEnv(width, height, obstacles, polygons, start, goal, agent_profile, scenario)
        path = env.nm.astar(start, goal)
        if not path:
            print("No path found by A* from start to goal.")
            return
        print("\nA* path traversal:")
        for i, (x, y) in enumerate(path[:max_steps]):
            env.agent_state['pos'] = [x, y]
            env.render()
            print(f"Step {i+1}: Position {(x, y)}")
            if (int(x), int(y)) == goal:
                print("Goal reached by A* path!")
                break
        else:
            print("Did not reach goal within 100 steps.")

    """
    2D Grid World Navigation Environment
    The agent must navigate from start to goal, avoiding obstacles.
    Observation: agent (x, y) position
    Action: 0=up, 1=down, 2=left, 3=right
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, width=10, height=10, obstacles=None, start=(0,0), goal=(9,9)):
        super(NavMeshEnv, self).__init__()
        self.width = width
        self.height = height
        self.obstacles = obstacles or []
        self.start = start
        self.goal = goal
        self.n_actions = 4
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(low=0, high=max(width, height)-1, shape=(2,), dtype=np.int32)
        self.nm = GridNavMesh(width, height, self.obstacles)
        self.reset()

    def reset(self):
        self.agent_pos = list(self.start)
        return np.array(self.agent_pos, dtype=np.int32)

    def step(self, action):
        x, y = self.agent_pos
        if action == 0:   # up
            nx, ny = x, y-1
        elif action == 1: # down
            nx, ny = x, y+1
        elif action == 2: # left
            nx, ny = x-1, y
        elif action == 3: # right
            nx, ny = x+1, y
        else:
            nx, ny = x, y
        if self.nm.is_walkable(nx, ny):
            self.agent_pos = [nx, ny]
        done = tuple(self.agent_pos) == self.goal
        reward = 1.0 if done else -0.1
        return np.array(self.agent_pos, dtype=np.int32), reward, done, {}

    def render(self, mode='human'):
        grid = np.copy(self.nm.grid)
        x, y = self.agent_pos
        grid[y, x] = 2  # agent
        gx, gy = self.goal
        grid[gy, gx] = 3  # goal
        print("\n".join([" ".join(["A" if cell==2 else "G" if cell==3 else "#" if cell==1 else "." for cell in row]) for row in grid]))

    def close(self):
        pass

    def print_astar_path(self):
        """Compute and print the A* path from start to goal using GridNavMesh."""
        path = self.nm.astar(self.start, self.goal)
        if not path:
            print("No path found by A* from start to goal.")
            return
        grid = np.copy(self.nm.grid)
        for (x, y) in path:
            if (x, y) != self.start and (x, y) != self.goal:
                grid[y, x] = 4  # Mark A* path
        sx, sy = self.start
        gx, gy = self.goal
        grid[sy, sx] = 2  # Start
        grid[gy, gx] = 3  # Goal
        print("A* Path from start to goal:")
        print("\n".join([" ".join([
            "S" if cell==2 else "G" if cell==3 else "*" if cell==4 else "#" if cell==1 else "." for cell in row
        ]) for row in grid]))

    @staticmethod
    def demo_astar_follow(width=10, height=10, obstacles=None, start=(0,0), goal=(9,9)):
        """Demonstrate following the A* path step by step."""
        env = NavMeshEnv(width, height, obstacles, start, goal)
        path = env.nm.astar(start, goal)
        if not path:
            print("No path found by A* from start to goal.")
            return
        print("\nA* path traversal:")
        for i, (x, y) in enumerate(path):
            env.agent_pos = [x, y]
            env.render()
            print(f"Step {i+1}: Position {x, y}")
        print("Goal reached by A* path!")
