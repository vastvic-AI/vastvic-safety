# Evacuation Simulator with Insurance Risk Assessment

A comprehensive simulation of crowd evacuation with insurance risk assessment features.

## Features

- Real-time visualization of evacuation scenarios
- Multiple agent types with different behaviors (adults, children, elderly)
- Dynamic hazard simulation (fire, smoke)
- Insurance risk assessment and claim calculation
- Multiple exit points with intelligent routing
- Obstacle avoidance and crowd dynamics

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd evacuation_simulation
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Simulation

To start the simulation, run:

```bash
streamlit run app.py
```

This will start a local web server and open the simulation in your default web browser.

## Controls

- **Start Simulation**: Begin the evacuation simulation
- **Pause**: Pause the simulation
- **Reset**: Reset the simulation to its initial state
- **Add Fire**: Introduce a fire hazard at a random location

## Project Structure

- `core/`: Core simulation logic
  - `agent.py`: Agent behavior and movement
  - `environment.py`: Simulation environment and hazards
- `app.py`: Streamlit web interface
- `requirements.txt`: Python dependencies

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
