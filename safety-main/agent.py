import numpy as np

class Agent:
    """
    Represents an agent with advanced profile and state for crowd simulation.
    """
    def __init__(self, agent_id, start, goal, profile=None):
        self.agent_id = agent_id
        self.start = np.array(start, dtype=np.float32)
        self.goal = np.array(goal, dtype=np.float32)
        self.profile = profile or {'speed': 1.0, 'panic': 0.0, 'group_id': None, 'size': 1.0, 'priority': 1.0}
        self.state = {
            'pos': np.array(start, dtype=np.float32),
            'panic': self.profile['panic'],
            'group_id': self.profile['group_id'],
            'status': 'normal',
            'path': [],
            'waited': 0,
            'collisions': 0,
            'exited': False,
            'exit_time': None
        }

    def smooth_path(self, path, interp_points=5):
        # Linear interpolation between waypoints for smoothness
        if not path or len(path) < 2:
            return path
        smooth = []
        for i in range(len(path) - 1):
            p0 = np.array(path[i], dtype=np.float32)
            p1 = np.array(path[i+1], dtype=np.float32)
            for alpha in np.linspace(0, 1, interp_points, endpoint=False):
                smooth.append(tuple(p0 + (p1 - p0) * alpha))
        smooth.append(tuple(path[-1]))
        return smooth

    def update(self, navmesh, density_map, hazard_map, t, agents=None, lookahead=3, jitter=0.1):
        if self.state['exited']:
            return
        # If at goal, mark as exited
        if np.allclose(self.state['pos'], self.goal, atol=0.5):
            self.state['exited'] = True
            self.state['exit_time'] = t
            return
        # Social force: avoid hazards/density
        x, y = self.state['pos']
        if hazard_map[int(y), int(x)] > 0:
            self.state['panic'] += 0.1
            self.state['status'] = 'avoiding hazard'
        elif density_map[int(y), int(x)] > 3:
            self.state['status'] = 'crowded'
        else:
            self.state['status'] = 'normal'
        # If path empty or blocked, replan
        if not self.state['path'] or not navmesh.is_walkable(*self.state['pos']):
            raw_path = navmesh.astar(tuple(map(int, self.state['pos'])), tuple(map(int, self.goal)))
            self.state['path'] = self.smooth_path(raw_path)
            if not self.state['path'] or len(self.state['path']) < 2:
                print(f"[Agent {self.agent_id}] No path found from {tuple(map(int, self.state['pos']))} to {tuple(map(int, self.goal))} at t={t}")
        # Move along smoothed path with lookahead and social force
        if self.state['path'] and len(self.state['path']) > 1:
            # Lookahead: target further along path if possible
            target_idx = min(lookahead, len(self.state['path']) - 1)
            target_pos = np.array(self.state['path'][target_idx], dtype=np.float32)
            direction = target_pos - self.state['pos']
            # Social force: repulsion from nearby agents
            if agents is not None:
                repulse = np.zeros(2)
                for other in agents:
                    if other is self or getattr(other, 'state', None) is None:
                        continue
                    op = other.state['pos'] if isinstance(other.state['pos'], np.ndarray) else np.array(other.state['pos'])
                    dist = np.linalg.norm(self.state['pos'] - op)
                    if 0 < dist < 2.0:
                        repulse += (self.state['pos'] - op) / (dist**2 + 1e-4)
                direction = direction + 1.5 * repulse
            # Add jitter
            direction += np.random.uniform(-jitter, jitter, size=2)
            norm = np.linalg.norm(direction)
            if norm > 0:
                step = min(self.profile['speed'], norm)
                self.state['pos'] += (direction / norm) * step
                # Remove path steps if passed
                while len(self.state['path']) > 1 and np.linalg.norm(self.state['pos'] - np.array(self.state['path'][1])) < 0.3:
                    self.state['path'] = self.state['path'][1:]
        else:
            self.state['waited'] += 1

