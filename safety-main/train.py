import gym
import numpy as np
from stable_baselines3 import PPO
from agent_env import NavMeshEnv

# --- Acme Imports ---
import acme
from acme import specs
from acme.wrappers import GymWrapper
from acme.agents.jax import dqn
import jax
import jax.numpy as jnp
import dm_env

# Helper to convert Gym env to dm_env if needed
class GymToDmEnv(dm_env.Environment):
    def __init__(self, gym_env):
        self._env = gym_env
        self._reset = True

    def reset(self):
        obs = self._env.reset()
        self._reset = False
        return dm_env.restart(obs)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        if done:
            self._reset = True
            return dm_env.termination(reward, obs)
        else:
            return dm_env.transition(reward, obs)

    def observation_spec(self):
        return specs.Array(self._env.observation_space.shape, self._env.observation_space.dtype, 'observation')

    def action_spec(self):
        return specs.DiscreteArray(self._env.action_space.n, dtype=self._env.action_space.dtype, name='action')

# --- Acme RL Training Example ---
def run_acme_dqn():
    print("\n--- Acme DQN Training Demo ---")
    # Create and wrap environment
    gym_env = NavMeshEnv(width=10, height=10, obstacles=[(3,3), (3,4), (3,5), (4,5), (5,5), (6,5), (7,5)], start=(0,0), goal=(9,9))
    env = GymWrapper(gym_env)
    env = GymToDmEnv(env)
    environment_spec = specs.make_environment_spec(env)

    # Create DQN agent
    agent = dqn.DQNBuilder().make_agent(environment_spec, seed=0)

    # Training loop (minimal example)
    timestep = env.reset()
    for step in range(1000):
        action = agent.select_action(timestep.observation)
        timestep = env.step(action)
        agent.observe(action, next_timestep=timestep)
        agent.update()
        if timestep.last():
            timestep = env.reset()
    print("Acme DQN training finished.")

# Uncomment to run Acme demo
# run_acme_dqn()

# ---- Stable Baselines3 PPO Training (existing code) ----
obstacles = [(3,3), (3,4), (3,5), (4,5), (5,5), (6,5), (7,5)]
env = NavMeshEnv(width=10, height=10, obstacles=obstacles, start=(0,0), goal=(9,9))

print("\n--- A* Pathfinding Demo ---")
env.print_astar_path()
NavMeshEnv.demo_astar_follow(width=10, height=10, obstacles=obstacles, start=(0,0), goal=(9,9))

def make_env():
    return NavMeshEnv(width=10, height=10, obstacles=obstacles, start=(0,0), goal=(9,9))

from stable_baselines3.common.env_util import DummyVecEnv
vec_env = DummyVecEnv([make_env])

model = PPO('MlpPolicy', vec_env, verbose=1)
model.learn(total_timesteps=20000)

obs = env.reset()
done = False
steps = 0
print("\nAgent navigation:")
while not done and steps < 100:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    env.render()
    print(f"Step {steps+1}: Position {obs}, Reward {reward}")
    steps += 1
if done:
    print("Goal reached!")
else:
    print("Agent did not reach the goal in 100 steps.")
