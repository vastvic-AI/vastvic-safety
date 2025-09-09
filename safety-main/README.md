# 2D NavMesh + ML-Agent Equivalent in Python

This project demonstrates a simple 2D navigation mesh (NavMesh) and reinforcement learning agent (ML-Agent) similar to Unity's system, using Python.

## Features
- 2D grid world with obstacles (NavMesh equivalent)
- A* pathfinding (in `navmesh.py`)
- Custom OpenAI Gym environment (`agent_env.py`)
- RL agent (PPO, via stable-baselines3) that learns to navigate from start to goal
- Simple text-based rendering (can be extended to matplotlib visualization)

## Files
- `navmesh.py` — GridNavMesh class with A* pathfinding
- `agent_env.py` — Custom Gym environment for agent navigation
- `train.py` — RL training and agent test script
- `requirements.txt` — Dependencies

## Usage
1. **Recommended: create and activate a virtual environment**
   
   PowerShell (Windows):
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
   
   Command Prompt (Windows):
   ```bat
   python -m venv venv
   venv\Scripts\activate.bat
   ```
   
   macOS/Linux:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Or, to install dependencies from both this module and the evacuation_simulation module in a single command, run from the repository root:
   ```bash
   pip install -r pathfinder-main/requirements.txt -r pathfinder-main/evacuation_simulation/requirements.txt
   ```
3. **Train and test the agent:**
   ```bash
   python train.py
   ```

## Customization
- Edit `obstacles`, `start`, and `goal` in `train.py` or `agent_env.py` to change the environment.
- The `GridNavMesh` class in `navmesh.py` can be used standalone for pathfinding.

## Notes
- This is a minimal, educational example. For more advanced features (continuous space, 3D, complex agents), consider using PyBullet, PettingZoo, or Unity ML-Agents directly.
