import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' for better compatibility
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arrow
import time
import traceback

# Import our modules
from core.environment import EvacuationEnvironment
from core.agent import EvacuationAgent

# Page config
st.set_page_config(
    page_title="Evacuation Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container { padding: 1rem; }
    .stButton>button { width: 100%; margin: 5px 0; }
    .metric-card { background: #f8f9fa; border-radius: 10px; padding: 15px; margin: 5px 0; }
</style>
""", unsafe_allow_html=True)

class EvacuationViz:
    def __init__(self):
        self.fig, self.ax = None, None
        self.env = EvacuationEnvironment(100, 100)
        self.sim_step = 0
        self.sim_running = False
        self.initialize_plot()
        self.setup_environment()
    
    def initialize_plot(self):
        """Initialize or reinitialize the plot."""
        if self.fig:
            plt.close(self.fig)
        self.fig, self.ax = plt.subplots(figsize=(10, 8), dpi=100)
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_facecolor('#f0f0f0')
        self.fig.tight_layout()

    def setup_environment(self):
        # Add walls
        for x in range(101):
            self.env.add_obstacle(x, 0)
            self.env.add_obstacle(x, 100)
        for y in range(101):
            self.env.add_obstacle(0, y)
            self.env.add_obstacle(100, y)
            
        # Add exits
        self.env.add_exit(50, 0)
        self.env.add_exit(0, 50)
        self.env.add_exit(100, 50)
        self.env.add_exit(50, 100)
        
        # Add agents
        for i in range(30):
            x, y = np.random.uniform(20, 80, 2)
            agent = EvacuationAgent(i, np.array([x, y]))
            self.env.add_agent(agent)

    def update_viz(self):
        try:
            if not self.fig or not self.ax:
                self.initialize_plot()
            
            self.ax.clear()
            self.ax.set_xlim(0, 100)
            self.ax.set_ylim(0, 100)
            self.ax.set_aspect('equal')
            self.ax.grid(True, alpha=0.3)
            self.ax.set_title(f'Evacuation Simulation - Step {self.sim_step}')
            
            # Draw environment
            self.draw_environment()
            
            # Draw agents and their paths
            self.draw_agents()
            
            # Add legend
            self.add_legend()
            
            # Tight layout and draw
            self.fig.tight_layout()
            self.fig.canvas.draw()
            
            return self.fig
            
        except Exception as e:
            st.error(f"Error in update_viz: {str(e)}")
            st.text(traceback.format_exc())
            self.initialize_plot()
            return self.fig
        
        # Draw obstacles
        for x, y in self.env.obstacles:
            self.ax.add_patch(Rectangle((x-0.5, y-0.5), 1, 1, color='gray'))
            
        # Draw exits
        for i, (x, y) in enumerate(self.env.exits):
            self.ax.add_patch(Rectangle((x-2, y-2), 4, 4, color='green'))
            self.ax.text(x, y, f'Exit {i}', color='white', 
                        ha='center', va='center', fontsize=8)
            
    def draw_agents(self):
        """Draw all agents with their current state and insurance info."""
        for agent in self.env.agents:
            # Set color based on agent type
            color_map = {
                'adult': 'blue',
                'child': 'red',
                'elderly': 'purple',
                'mobility_impaired': 'orange'
            }
            color = color_map.get(agent.agent_type, 'gray')
            
            # Draw agent with border indicating insurance status
            has_insurance = hasattr(agent, 'insurance_coverage') and agent.insurance_coverage.get('coverage_active', False)
            edgecolor = 'gold' if has_insurance else 'black'
            linewidth = 2 if has_insurance else 1
            
            circle = Circle(agent.position, agent.radius, 
                          color=color, 
                          edgecolor=edgecolor,
                          linewidth=linewidth,
                          alpha=0.7)
            self.ax.add_patch(circle)
            
            # Draw path if it exists
            if hasattr(agent, 'path') and agent.path:
                path_x = [p[0] for p in agent.path]
                path_y = [p[1] for p in agent.path]
                self.ax.plot(path_x, path_y, ':', color=color, alpha=0.3, linewidth=1)
            
            # Add agent info with insurance status
            info = [
                f"ID:{agent.agent_id} {agent.agent_type[0].upper()}",
                f"P:{agent.panic_level:.1f} H:{agent.health:.1f}"
            ]
            
            # Add insurance info if available
            if has_insurance:
                coverage = agent.insurance_coverage
                info.append(f"Ins: ${coverage.get('personal_accident', 0)//1000}K")
            
            # Add risk factors
            if hasattr(agent, 'risk_factors'):
                total_risk = sum(agent.risk_factors.values()) / max(len(agent.risk_factors), 1)
                risk_color = 'red' if total_risk > 0.5 else 'orange' if total_risk > 0.2 else 'green'
                info.append(f"Risk: {total_risk:.1f}")
            
            # Position text above agent
            text_y_offset = agent.radius + 2
            for i, line in enumerate(info):
                self.ax.text(agent.position[0], 
                           agent.position[1] + text_y_offset + (i * 2.5), 
                           line,
                           color='black', 
                           ha='center', 
                           va='bottom',
                           fontsize=5,
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
            
    def add_legend(self):
        """Add a comprehensive legend to the plot."""
        legend_elements = [
            # Agent types
            plt.Rectangle((0,0), 1, 1, color='blue', alpha=0.7, label='Adult'),
            plt.Rectangle((0,0), 1, 1, color='red', alpha=0.7, label='Child'),
            plt.Rectangle((0,0), 1, 1, color='purple', alpha=0.7, label='Elderly'),
            plt.Rectangle((0,0), 1, 1, color='orange', alpha=0.7, label='Mobility Impaired'),
            # Environment
            plt.Rectangle((0,0), 1, 1, color='green', alpha=0.7, label='Exit'),
            plt.Rectangle((0,0), 1, 1, color='gray', alpha=0.7, label='Obstacle'),
            # Insurance indicators
            plt.Circle((0,0), 0.5, facecolor='white', edgecolor='gold', 
                      linewidth=2, label='Insured Agent'),
            plt.Circle((0,0), 0.5, facecolor='white', edgecolor='black', 
                      linewidth=1, label='Uninsured Agent'),
            # Risk indicators
            plt.Rectangle((0,0), 1, 1, color='red', alpha=0.3, label='High Risk'),
            plt.Rectangle((0,0), 1, 1, color='orange', alpha=0.3, label='Medium Risk'),
            plt.Rectangle((0,0), 1, 1, color='green', alpha=0.3, label='Low Risk')
        ]
        
        # Create two columns for the legend
        legend1 = self.ax.legend(handles=legend_elements[:7], 
                               loc='upper left', 
                               fontsize=7,
                               title="Agents & Environment")
        
        # Add the second legend manually
        legend2 = self.ax.legend(handles=legend_elements[7:], 
                               loc='lower left',
                               fontsize=7,
                               title="Risk Indicators")
        
        # Add the first legend back (second one was removed when adding the second)
        self.ax.add_artist(legend1)
        
        # Ensure the figure is drawn
        self.fig.tight_layout()
        self.fig.canvas.draw()
        

def calculate_insurance_metrics(agents):
    """Calculate insurance-related metrics."""
    insured = 0
    total_coverage = 0
    total_risk = 0
    
    for agent in agents:
        if hasattr(agent, 'insurance_coverage') and agent.insurance_coverage.get('coverage_active', False):
            insured += 1
            total_coverage += agent.insurance_coverage.get('personal_accident', 0)
        
        if hasattr(agent, 'risk_factors'):
            total_risk += sum(agent.risk_factors.values()) / max(len(agent.risk_factors), 1)
    
    avg_risk = total_risk / len(agents) if agents else 0
    insurance_rate = (insured / len(agents)) * 100 if agents else 0
    
    return {
        'insured_count': insured,
        'insurance_rate': insurance_rate,
        'avg_risk': avg_risk,
        'total_coverage': total_coverage
    }

def main():
    # Initialize session state
    if 'sim' not in st.session_state:
        st.session_state.sim = EvacuationViz()
    
    # Sidebar controls
    with st.sidebar:
        st.title('Controls')
        
        # Simulation controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button('▶️ Start Simulation'):
                st.session_state.sim.sim_running = True
        with col2:
            if st.button('⏹️ Stop Simulation'):
                st.session_state.sim.sim_running = False
        
        if st.button('🔄 Reset Simulation'):
            st.session_state.sim = EvacuationViz()
            st.rerun()
        
        st.markdown("---")
        
        # Hazard controls
        st.subheader('Hazards')
        col1, col2 = st.columns(2)
        with col1:
            if st.button('🔥 Add Fire'):
                x, y = np.random.uniform(20, 80, 2)
                st.session_state.sim.env.add_hazard(x, y, 'fire', 1.0, 15.0)
        with col2:
            if st.button('💨 Add Smoke'):
                x, y = np.random.uniform(20, 80, 2)
                st.session_state.sim.env.add_hazard(x, y, 'smoke', 0.7, 10.0)
        
        st.markdown("---")
        
        # Insurance controls
        st.subheader('Insurance')
        if st.button('📝 Toggle Random Insurance'):
            for agent in st.session_state.sim.env.agents:
                if np.random.random() > 0.7:  # 30% chance of having insurance
                    if not hasattr(agent, 'insurance_coverage'):
                        agent.initialize_insurance()
                    agent.insurance_coverage['coverage_active'] = not agent.insurance_coverage.get('coverage_active', False)
    
    # Main content
    st.title('Evacuation Simulation with Insurance')
    
    # Calculate metrics
    agents = st.session_state.sim.env.agents
    insurance_metrics = calculate_insurance_metrics(agents)
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('👥 Agents', len(agents))
    with col2:
        st.metric('🏃 Evacuated', len([a for a in agents if hasattr(a, 'evacuated') and a.evacuated]))
    with col3:
        st.metric('🛡️ Insured', f"{insurance_metrics['insurance_rate']:.1f}%")
    with col4:
        st.metric('⚠️ Avg Risk', f"{insurance_metrics['avg_risk']*100:.1f}%")
    
    # Insurance details
    with st.expander("📊 Insurance Details"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric('Total Coverage', f"${insurance_metrics['total_coverage']/1000:.1f}K")
        with col2:
            st.metric('Potential Claims', f"${insurance_metrics['total_coverage'] * insurance_metrics['avg_risk']/1000:.1f}K")
        
        # Risk distribution
        risk_levels = {'Low': 0, 'Medium': 0, 'High': 0}
        for agent in agents:
            if hasattr(agent, 'risk_factors'):
                risk = sum(agent.risk_factors.values()) / max(len(agent.risk_factors), 1)
                if risk < 0.3:
                    risk_levels['Low'] += 1
                elif risk < 0.7:
                    risk_levels['Medium'] += 1
                else:
                    risk_levels['High'] += 1
        
        st.bar_chart(risk_levels)
    
    # Create placeholder for the plot
    viz_placeholder = st.empty()
    
    # Run simulation loop
    if st.session_state.sim.sim_running:
        try:
            st.session_state.sim.sim_step += 1
            st.session_state.sim.env.step(0.1)
            
            # Update visualization
            with viz_placeholder.container():
                st.pyplot(st.session_state.sim.update_viz())
            
            # Rerun to continue the simulation
            time.sleep(0.1)
            st.rerun()
            
        except Exception as e:
            st.error(f"Error in simulation loop: {str(e)}")
            st.text(traceback.format_exc())
            st.session_state.sim.sim_running = False
    else:
        # Just show the current state
        with viz_placeholder.container():
            st.pyplot(st.session_state.sim.update_viz())

if __name__ == '__main__':
    main()
    
    if st.button('▶️ Start Simulation'):
        st.session_state.sim.sim_running = True
        
    if st.button('⏸️ Pause'):
        st.session_state.sim.sim_running = False
        
    if st.button('🔄 Reset'):
        st.session_state.sim = EvacuationViz()
        st.rerun()

# Main content
st.title('Evacuation Simulation')

# Simulation display
col1, col2 = st.columns([3, 1])

with col1:
    viz_placeholder = st.empty()
    
    # Run simulation loop
    if st.session_state.sim.sim_running:
        st.session_state.sim.sim_step += 1
        st.session_state.sim.env.step(0.1)
        viz_placeholder.pyplot(st.session_state.sim.update_viz())
        time.sleep(0.1)
        st.rerun()
    else:
        viz_placeholder.pyplot(st.session_state.sim.update_viz())

with col2:
    st.metric('Agents', len(st.session_state.sim.env.agents))
    st.metric('Time Step', st.session_state.sim.sim_step)
    
    # Add hazard button
    if st.button('🔥 Add Fire'):
        x, y = np.random.uniform(20, 80, 2)
        st.session_state.sim.env.add_hazard('fire', (x, y), 0.8)
        st.rerun()
