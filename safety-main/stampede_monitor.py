from stampede_detection import StampedeDetector
import numpy as np

class StampedeMonitor:
    """
    Monitors the crowd for stampede risk using behavioral and density indicators.
    """
    def __init__(self, env, detector=None):
        self.env = env
        # Use default thresholds suitable for Indian crowd if not specified
        self.detector = detector or StampedeDetector(
            area_width=env.width, area_height=env.height,
            density_threshold=0.15, local_density_threshold=0.25,
            speed_threshold=2.5, alignment_threshold=0.93, panic_ratio_threshold=0.33)
        self.last_metrics = None
        self.last_risk = False

    def get_agent_states(self):
        agent_states = []
        for a in self.env.agents:
            state = a.state.copy()
            # Estimate velocity if previous position is available
            if hasattr(a, 'prev_pos'):
                state['vel'] = np.array(state['pos']) - np.array(a.prev_pos)
            else:
                state['vel'] = np.zeros(2)
            agent_states.append(state)
        return agent_states

    def step(self):
        # Call this after environment step
        for a in self.env.agents:
            if not hasattr(a, 'prev_pos'):
                a.prev_pos = np.array(a.state['pos'])
            else:
                a.prev_pos = np.array(a.state['pos'])
        agent_states = self.get_agent_states()
        risk, metrics = self.detector.detect_from_agent_states(agent_states)
        self.last_metrics = metrics
        self.last_risk = risk
        return risk, metrics
