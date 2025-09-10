# vastvic-path

Evacuation simulation and pathfinding app built with Streamlit. This README explains how to set up, install dependencies, and run the app quickly.

## Prerequisites
- Python 3.9+ installed and on your PATH
- pip (comes with Python)

## 1. Create a virtual environment (recommended)

Windows (PowerShell):
```powershell
python -m venv venv .\venv\Scripts\Activate.ps1
```

Windows (Command Prompt):
```bat
python -m venv venv venv\Scripts\activate.bat
```

macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

## 2. Install dependencies
Install both requirement files in a single command (run from the repo root):
```bash
pip install -r pathfinder-main/requirements.txt -r pathfinder-main/evacuation_simulation/requirements.txt
```
If install fails for numpy, run
```bash
pip install numpy
```

## 4. Run the app
```bash
streamlit run app.py
```

The terminal will show a local URL (e.g., http://localhost:8501). Open it in your browser.

## Using the simulator
1. Upload a floorplan image
   - Sidebar → "Floor Plan" → Upload PNG/JPG. White=free, Black=walls, Green/Red=exits, Yellow/Orange=hazards.
   - Adjust "Grid resolution (max dimension, cells)" for more/less detail.
2. Configure parameters (sidebar)
   - Agents count, group settings, panic level, average speed, max time/runtime.
   - Hazard cost weight, exit capacity, pre-evacuation delays, and local avoidance.
3. Start/Stop/Reset
   - Use the sidebar buttons. Changing the uploaded image or grid resolution safely re-initializes the sim.
4. Visuals and metrics
   - Main canvas shows agents and environment. Metrics panel shows agents remaining, evacuated %, trapped, stampede risk, and exit usage for all exits.
   - Tabs: Evacuation Curve, Heatmap (visits), Trajectories, Agent Details.
5. Exports
   - Click "Export Outputs" to save charts and CSV/JSON into `evac_outputs/`.
   - Time-series (overall and per-exit throughput) can be downloaded from the "Throughput and Time-Series" section.

## Tips
- Output files are saved to `evac_outputs/` when you export from the app.
- If you see a Streamlit warning about `use_container_width`, use `width='stretch'` instead in future updates.
- For MP4 export, the app uses `imageio` and `imageio-ffmpeg` (included in requirements). If your system has no FFmpeg, `imageio-ffmpeg` provides a bundled binary.
- If PowerShell blocks activation, run: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then re-activate the venv.

## 5. Deactivate the environment (when done)
```bash
deactivate
```

## Project structure (key files)
- `app.py` — Streamlit evacuation simulator UI + logic
- `pathfinder-main/requirements.txt` — main module dependencies
- `pathfinder-main/evacuation_simulation/requirements.txt` — evacuation module dependencies
- `evac_outputs/` — exported charts/tables after runs (auto-created)

## Features (high-level)
- Load a floorplan image, auto-detect exits, hazards, and walls
- Global pathfinding with optional path smoothing
- Local avoidance to reduce collisions and congestion
- Real-time visualization with optional smooth interpolation
- Metrics, exit usage, throughput/time-series exports, and insurance analytics report

## Files
- `app.py` — Streamlit evacuation simulator UI and main loop
- `path_smoothing.py` — Line-of-sight and Chaikin smoothing utilities
- `local_avoidance.py` — Crowd local avoidance move ranking
- `pathfinder-main/requirements.txt` — Main module dependencies
- `pathfinder-main/evacuation_simulation/requirements.txt` — Evacuation module dependencies
- `evacuation_simulation/` — Core simulation logic, utils, and visualization support
- `insurance/` — Insurance analytics report generation
- `evac_outputs/` — Exported charts, CSV/JSON, and reports (generated)
