import numpy as np
from navmesh import GridNavMesh
from agent import Agent

class CrowdEnv:
    """
    Manages multiple agents, environment state, scenario, and step logic.
    """
    def __init__(self, width, height, obstacles=None, polygons=None, agents=None, scenario=None):
        self.width = width
        self.height = height
        try:
            self.navmesh = GridNavMesh(width, height, obstacles=obstacles, polygons=polygons)
        except TypeError:
            self.navmesh = GridNavMesh(width, height, obstacles=obstacles)
        self.agents = agents or []
        self.scenario = scenario or []
        self.t = 0
        self.density_map = np.zeros((height, width), dtype=np.float32)
        self.hazard_map = np.zeros((height, width), dtype=np.float32)

    def step(self):
        # Update density/hazard maps
        self.density_map.fill(0)
        for agent in self.agents:
            x, y = map(int, agent.state['pos'])
            self.density_map[y, x] += 1
        # Apply scenario events
        for event in self.scenario:
            if event['step'] == self.t:
                event['action'](self)
        # Agents act
        for agent in self.agents:
            agent.update(self.navmesh, self.density_map, self.hazard_map, self.t)
        self.t += 1

    def all_exited(self):
        return all(agent.state['exited'] for agent in self.agents)

    def stats(self):
        return {
            'exited': sum(agent.state['exited'] for agent in self.agents),
            'avg_waited': np.mean([agent.state['waited'] for agent in self.agents]),
            'avg_collisions': np.mean([agent.state['collisions'] for agent in self.agents]),
            'max_panic': np.max([agent.state['panic'] for agent in self.agents]),
        }
