import numpy as np
import heapq

class GridNavMesh:
    """
    Advanced NavMesh supporting both grid and polygonal navigation, continuous (x, y) movement, dynamic obstacles, density maps, and hazard fields.
    """
    def __init__(self, width, height, obstacles=None, polygons=None):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.int32)
        self.polygons = polygons  # List of polygonal areas, or None for grid fallback
        if obstacles:
            for (x, y) in obstacles:
                self.grid[y, x] = 1  # 1 means obstacle
        self.dynamic_obstacles = set()
        self.density_map = np.zeros((height, width), dtype=np.float32)
        self.hazard_map = np.zeros((height, width), dtype=np.float32)

    def is_walkable(self, x, y):
        # For continuous: check polygons; for grid: check grid
        if self.polygons:
            # TODO: Implement point-in-polygon check
            return any(self.point_in_polygon((x, y), poly) for poly in self.polygons)
        ix, iy = int(x), int(y)
        return 0 <= x < self.width and 0 <= y < self.height and self.grid[iy, ix] == 0 and (ix, iy) not in self.dynamic_obstacles

    def add_dynamic_obstacle(self, x, y):
        self.dynamic_obstacles.add((int(x), int(y)))

    def remove_dynamic_obstacle(self, x, y):
        self.dynamic_obstacles.discard((int(x), int(y)))

    def update_density(self, agent_positions):
        self.density_map.fill(0)
        for x, y in agent_positions:
            self.density_map[int(y), int(x)] += 1

    def update_hazard(self, hazard_positions):
        self.hazard_map.fill(0)
        for x, y in hazard_positions:
            self.hazard_map[int(y), int(x)] = 1

    def neighbors(self, x, y):
        # 8-way for continuous, 4-way for grid
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (1,1), (-1,1), (1,-1)]:
            nx, ny = x + dx, y + dy
            if self.is_walkable(nx, ny):
                yield (nx, ny)

    def point_in_polygon(self, point, polygon):
        # Ray casting algorithm for point-in-polygon
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n+1):
            p2x, p2y = polygon[i % n]
            if min(p1y, p2y) < y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def astar(self, start, goal, avoid_hazards=True, avoid_density=True, avoid_dynamic=True, flow_field=None, cache=None, max_steps=500):
        """
        Pathfinder-level A* pathfinding from start to goal with:
        - Weighted cost for hazards, congestion, dynamic obstacles
        - Predictive planning: avoids dynamic obstacles (can pass a set of (x, y, t) for time-dependent obstacles)
        - Partial path following and local detours
        - Path caching (reuse if valid)
        - Penalize sharp turns for natural paths
        - Follows flow field if provided (herding effect)
        """
        import heapq
        if cache is not None and (start, goal) in cache:
            cached_path = cache[(start, goal)]
            if all(self.is_walkable(x, y) for x, y in cached_path):
                return cached_path
        open_set = [(0 + self.heuristic(start, goal), 0, start, [start], None)] # (est_total, cost, pos, path, prev_dir)
        visited = set()
        steps = 0
        while open_set and steps < max_steps:
            est_total, cost, current, path, prev_dir = heapq.heappop(open_set)
            if current == goal:
                if cache is not None:
                    cache[(start, goal)] = path
                return path
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.neighbors(*current):
                if neighbor in visited:
                    continue
                penalty = 0
                x, y = neighbor
                # Hazard penalty
                if avoid_hazards:
                    penalty += self.hazard_map[int(y), int(x)] * 100
                # Density penalty
                if avoid_density:
                    penalty += self.density_map[int(y), int(x)] * 2
                # Dynamic obstacle penalty
                if avoid_dynamic and (int(x), int(y)) in self.dynamic_obstacles:
                    penalty += 1000
                # Flow field (herding): prefer major flows
                if flow_field is not None:
                    penalty -= flow_field[int(y), int(x)] * 2
                # Sharp turn penalty
                if prev_dir is not None:
                    dir_vec = (x - current[0], y - current[1])
                    if dir_vec != prev_dir:
                        penalty += 0.5
                # Early exit if neighbor is a dynamic obstacle
                if avoid_dynamic and (int(x), int(y)) in self.dynamic_obstacles:
                    continue
                heapq.heappush(open_set, (
                    cost + 1 + penalty + self.heuristic(neighbor, goal),
                    cost + 1 + penalty,
                    neighbor,
                    path + [neighbor],
                    (x - current[0], y - current[1])
                ))
            steps += 1
        return []  # No path found

    def replan_if_blocked(self, agent_pos, goal, prev_path=None, **kwargs):
        # Try local detour first, then full replan
        if not self.is_walkable(*agent_pos):
            # Try local detour: try all neighbors
            for neighbor in self.neighbors(*agent_pos):
                if self.is_walkable(*neighbor):
                    return [agent_pos, neighbor] + (prev_path[2:] if prev_path else [])
            # Otherwise, full replan
            return self.astar(agent_pos, goal, **kwargs)
        # If path is blocked ahead in prev_path, replan from first blocked
        if prev_path:
            for idx, pos in enumerate(prev_path):
                if not self.is_walkable(*pos):
                    return self.astar(agent_pos, goal, **kwargs)
        return None


    def heuristic(self, a, b):
        # Euclidean for continuous, Manhattan for grid
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5

    def replan_if_blocked(self, agent_pos, goal):
        # If agent is blocked, replan path
        if not self.is_walkable(*agent_pos):
            return self.astar(agent_pos, goal)
        return None

    def __init__(self, width, height, obstacles=None):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.int32)
        if obstacles:
            for (x, y) in obstacles:
                self.grid[y, x] = 1  # 1 means obstacle

    def is_walkable(self, x, y):
        ix, iy = int(x), int(y)
        return 0 <= x < self.width and 0 <= y < self.height and self.grid[iy, ix] == 0

    def neighbors(self, x, y):
        # 4-way connectivity (N, S, E, W)
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if self.is_walkable(nx, ny):
                yield (nx, ny)

    def heuristic(self, a, b):
        # Manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def astar(self, start, goal):
        """A* pathfinding from start to goal. Returns list of (x, y) or [] if no path."""
        open_set = [(0 + self.heuristic(start, goal), 0, start, [start])]
        visited = set()
        while open_set:
            est_total, cost, current, path = heapq.heappop(open_set)
            if current == goal:
                return path
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.neighbors(*current):
                if neighbor not in visited:
                    heapq.heappush(open_set, (
                        cost + 1 + self.heuristic(neighbor, goal),
                        cost + 1,
                        neighbor,
                        path + [neighbor]
                    ))
        return []  # No path found
