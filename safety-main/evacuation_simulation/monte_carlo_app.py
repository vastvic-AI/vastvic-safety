import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import time
import io
from PIL import Image
from pathlib import Path
import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Use absolute imports
from evacuation_simulation.core import (
    EvacuationEnvironment,
    EvacuationAgent,
    MonteCarloEvacuation,
    EvacuationScenario
)
from evacuation_simulation.visualization.monte_carlo_viz import MonteCarloVisualizer

# Page config
st.set_page_config(
    page_title="Evacuation Monte Carlo Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container { padding: 1rem; }
    .stButton>button { width: 100%; margin: 5px 0; }
    .metric-card { background: #f8f9fa; border-radius: 10px; padding: 15px; margin: 5px 0; }
    .stProgress > div > div > div > div { background-color: #4CAF50; }
</style>
""", unsafe_allow_html=True)

class MonteCarloApp:
    def __init__(self):
        self.simulator = None
        self.viz = MonteCarloVisualizer()
        self.env = self._create_environment()
        self.results = None
        self.animation_path = None
        self.fig = None
        self.animation_running = False
        self.current_step = 0
        self.max_steps = 1000
        self.animation_speed = 100  # ms between frames
        
        # Load floor plan image
        self.floor_plan_img = None
        try:
            floor_plan_path = Path(__file__).parent / "assets" / "floor_plan.png"
            if floor_plan_path.exists():
                self.floor_plan_img = plt.imread(str(floor_plan_path))
        except Exception as e:
            st.warning(f"Could not load floor plan: {e}")
    
    def _create_environment(self):
        """Create a sample environment with floor plan and exits."""
        # Create environment with dimensions matching floor plan
        env_width = 100
        env_height = 100
        env = EvacuationEnvironment(env_width, env_height)
        
        # Add exits at the edges of the floor plan
        exit_width = 5  # Width of exit areas
        env.exits = [
            (0, env_height//2),  # Left middle
            (env_width, env_height//2),  # Right middle
            (env_width//2, 0),  # Bottom middle
            (env_width//2, env_height)  # Top middle
        ]
        
        # Add some obstacles (walls, furniture, etc.)
        self._add_obstacles(env)
        
        return env
    
    def _add_obstacles(self, env):
        """Add obstacles to the environment based on floor plan."""
        # Add perimeter walls
        for x in range(env.width):
            env.add_obstacle(x, 0)
            env.add_obstacle(x, env.height-1)
        for y in range(env.height):
            env.add_obstacle(0, y)
            env.add_obstacle(env.width-1, y)
        
        # Add some internal walls/obstacles
        for x in range(30, 70):
            env.add_obstacle(x, 40)
            env.add_obstacle(x, 60)
        for y in range(30, 70):
            env.add_obstacle(40, y)
            env.add_obstacle(60, y)
        
        # Add walls
        for x in range(101):
            env.add_obstacle(x, 0)
            env.add_obstacle(x, 100)
        for y in range(101):
            env.add_obstacle(0, y)
            env.add_obstacle(100, y)
        
        # Add some internal walls
        for x in range(30, 70):
            env.add_obstacle(x, 40)
            env.add_obstacle(x, 60)
        
        # Add exits
        env.add_exit(0, 50)
        env.add_exit(100, 50)
        env.add_exit(50, 0)
        env.add_exit(50, 100)
        
        # Add agents
        agent_types = ['adult', 'child', 'elderly', 'mobility_impaired']
        for i in range(50):
            x, y = np.random.uniform(20, 80, 2)
            agent = EvacuationAgent(
                agent_id=i,
                position=np.array([x, y]),
                agent_type=np.random.choice(agent_types, p=[0.5, 0.2, 0.2, 0.1]),
                panic_level=np.random.uniform(0, 0.5),
                health=np.random.uniform(0.7, 1.0)
            )
            env.add_agent(agent)
        
        # Add hazards
        env.add_hazard('fire', (30, 30), 1.0)
        env.add_hazard('smoke', (70, 70), 0.7)
        
        return env
    
    def run_simulation(self, num_simulations=100, real_time=False):
        """Run the Monte Carlo simulation with optional real-time visualization."""
        self.simulator = MonteCarloEvacuation(num_simulations=num_simulations)
        
        if real_time:
            # Setup real-time visualization
            self.fig, self.ax = plt.subplots(figsize=(12, 10))
            self.ax.set_aspect('equal')
            self.ax.set_xlim(0, 100)
            self.ax.set_ylim(0, 100)
            self.ax.grid(True, alpha=0.3)
            
            # Create placeholders for visualization
            self.agent_plots = []
            self.hazard_plots = []
            self.path_plots = []
            self.legend_handles = []
            
            # Run simulation with real-time updates
            self.animation_running = True
            self.current_step = 0
            
            # Create a placeholder for the animation
            self.animation_placeholder = st.empty()
            
            # Run the simulation with callbacks for visualization
            self.results = self.simulator.run_simulation(
                self.env,
                step_callback=self._update_visualization,
                max_steps=self.max_steps
            )
            
            # Final update
            self._update_visualization(self.env, self.current_step)
            self.animation_running = False
        else:
            # Run simulation without real-time visualization
            self.results = self.simulator.run_simulation(self.env, max_steps=self.max_steps)
    
    def _get_insurance_recommendations(self, risk_level):
        """Get insurance recommendations based on risk level."""
        if risk_level < 0.3:
            return {
                'level': 'Low Risk',
                'recommendations': [
                    'Basic evacuation insurance coverage',
                    'Standard personal accident coverage',
                    'Minimal business interruption coverage'
                ],
                'coverage': 'Basic',
                'color': 'green'
            }
        elif risk_level < 0.7:
            return {
                'level': 'Medium Risk',
                'recommendations': [
                    'Enhanced evacuation coverage',
                    'Comprehensive personal accident insurance',
                    'Business interruption coverage',
                    'Temporary accommodation coverage'
                ],
                'coverage': 'Enhanced',
                'color': 'orange'
            }
        else:
            return {
                'level': 'High Risk',
                'recommendations': [
                    'Premium evacuation coverage',
                    'Full personal accident coverage',
                    'Extended business interruption coverage',
                    'Temporary relocation coverage',
                    'Hazard-specific coverage',
                    'Emergency response services'
                ],
                'coverage': 'Premium',
                'color': 'red'
            }

    def _update_visualization(self, env, step):
        """Update the visualization for the current simulation step."""
        if not self.animation_running:
            return False  # Stop the animation
            
        self.current_step = step
        
        # Clear previous plots
        for plot in self.agent_plots + self.hazard_plots + self.path_plots:
            if plot in self.ax.patches or plot in self.ax.lines:
                plot.remove()
        
        self.agent_plots = []
        self.hazard_plots = []
        self.path_plots = []
        
        # Update environment elements
        for hazard in env.hazards:
            h = plt.Circle(
                (hazard['position'][0], hazard['position'][1]),
                hazard.get('radius', 5),
                color='red',
                alpha=0.3 * hazard.get('intensity', 1.0),
                zorder=5
            )
            self.ax.add_patch(h)
            self.hazard_plots.append(h)
        
        # Update agents
        for agent in env.agents:
            if hasattr(agent, 'position'):
                # Agent position
                a = plt.Circle(
                    (agent.position[0], agent.position[1]),
                    0.5,
                    color='blue',
                    alpha=0.8,
                    zorder=10
                )
                self.ax.add_patch(a)
                self.agent_plots.append(a)
                
                # Agent path
                if hasattr(agent, 'path') and len(agent.path) > 1:
                    path_x = [p[0] for p in agent.path]
                    path_y = [p[1] for p in agent.path]
                    line, = self.ax.plot(path_x, path_y, 'b--', alpha=0.5, linewidth=1)
                    self.path_plots.append(line)
        
        # Update title and step counter
        self.ax.set_title(f'Evacuation Simulation - Step {step}/{self.max_steps}')
        
        # Convert plot to image and display in Streamlit
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        self.animation_placeholder.image(img)
        
        # Small delay to control animation speed
        time.sleep(self.animation_speed / 1000.0)
        
        return True
    
    def display_results(self):
        """Display simulation results with interactive visualizations."""
        if not self.results:
            st.warning("No simulation results to display. Run a simulation first.")
            return
            
        # Create two columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display floor plan with simulation results
            st.subheader("Evacuation Simulation Results")
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Add floor plan background if available
            if self.floor_plan_img is not None:
                ax.imshow(self.floor_plan_img, extent=[0, 100, 0, 100], alpha=0.7, zorder=0)
            
            # Draw hazards
            for hazard_type, hazard_data in self.env.hazards.items():
                for pos, intensity in zip(hazard_data['positions'], hazard_data['intensities']):
                    circle = plt.Circle(
                        (pos[0], pos[1]),
                        hazard_data.get('radius', 5.0),
                        color='red',
                        alpha=0.3 * intensity,
                        zorder=5,
                        label=f"{hazard_type.title()} Hazard"
                    )
                    ax.add_patch(circle)
            
            # Draw agents and their paths
            for agent in self.env.agents:
                if hasattr(agent, 'position'):
                    # Draw agent path if available
                    if hasattr(agent, 'path') and len(agent.path) > 1:
                        path = np.array(agent.path)
                        ax.plot(path[:, 0], path[:, 1], 'b--', alpha=0.3, linewidth=1, zorder=5)
                    
                    # Draw agent
                    agent_color = 'green' if hasattr(agent, 'evacuated') and agent.evacuated else 'blue'
                    circle = plt.Circle(
                        (agent.position[0], agent.position[1]),
                        0.8,  # Slightly larger for better visibility
                        color=agent_color,
                        alpha=0.8,
                        zorder=10,
                        label='Agent (Evacuated)' if hasattr(agent, 'evacuated') and agent.evacuated else 'Agent'
                    )
                    ax.add_patch(circle)
            
            # Draw exits
            for exit_pos in self.env.exits:
                rect = plt.Rectangle(
                    (exit_pos[0] - 1, exit_pos[1] - 1), 2, 2,
                    color='green', alpha=0.7, zorder=10,
                    label='Exit'
                )
                ax.add_patch(rect)
            
            # Add legend with unique items
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))  # Remove duplicates
            if by_label:
                ax.legend(by_label.values(), by_label.keys(), 
                         loc='upper right', bbox_to_anchor=(1.3, 1))
            
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_aspect('equal')
            ax.axis('on')  # Keep axis for better orientation
            ax.grid(True, alpha=0.3)
            st.pyplot(fig, bbox_inches='tight')
        
        with col2:
            # Display insurance recommendations and metrics
            st.subheader("Risk Assessment")
            self._display_risk_metrics()
            
            st.subheader("Insurance Recommendations")
            self._display_insurance_recommendations()
    
    def _display_risk_metrics(self):
        """Display key risk metrics from the simulation."""
        if not hasattr(self.env, 'agents') or not self.env.agents:
            st.warning("No agent data available for risk assessment.")
            return
        
        # Calculate metrics
        total_agents = len(self.env.agents)
        evacuated = sum(1 for a in self.env.agents if hasattr(a, 'evacuated') and a.evacuated)
        avg_health = np.mean([a.health for a in self.env.agents if hasattr(a, 'health')]) * 100
        
        # Calculate risk level (simple heuristic)
        risk_level = (1 - (evacuated / total_agents)) * 100 if total_agents > 0 else 0
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Evacuation Rate", f"{evacuated}/{total_agents} ({evacuated/total_agents*100:.1f}%)")
            st.metric("Average Health", f"{avg_health:.1f}%")
        
        with col2:
            st.metric("Risk Level", f"{risk_level:.1f}%")
            
            # Color-coded risk indicator
            if risk_level < 30:
                risk_color = "green"
            elif risk_level < 70:
                risk_color = "orange"
            else:
                risk_color = "red"
                
            st.markdown(f"""
            <div style='background-color:#f8f9fa; padding:10px; border-radius:5px; margin-top:10px;'>
                <div style='background-color:{risk_color}; width:{risk_level}%; height:10px; border-radius:3px;'></div>
                <div style='display:flex; justify-content:space-between; font-size:0.8em; margin-top:5px;'>
                    <span>Low</span>
                    <span>Medium</span>
                    <span>High</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def _display_insurance_recommendations(self):
        """Display detailed insurance recommendations based on simulation results."""
        # Calculate risk metrics
        if not hasattr(self.env, 'agents') or not self.env.agents:
            st.warning("No agent data available for risk assessment.")
            return
        
        # Calculate risk factors
        total_agents = len(self.env.agents)
        evacuated = sum(1 for a in self.env.agents if hasattr(a, 'evacuated') and a.evacuated)
        evacuation_rate = (evacuated / total_agents) if total_agents > 0 else 0
        
        # Calculate average health and risk factors
        avg_health = 0
        risk_factors = {}
        
        for agent in self.env.agents:
            if hasattr(agent, 'health'):
                avg_health += agent.health
            
            if hasattr(agent, 'risk_factors'):
                for factor, value in agent.risk_factors.items():
                    if factor not in risk_factors:
                        risk_factors[factor] = 0
                    risk_factors[factor] += value
        
        avg_health = (avg_health / total_agents * 100) if total_agents > 0 else 100
        
        # Normalize risk factors
        if total_agents > 0:
            risk_factors = {k: v / total_agents for k, v in risk_factors.items()}
        
        # Calculate overall risk level (0-100)
        risk_level = (1 - evacuation_rate) * 70 + (100 - avg_health) * 0.3
        risk_level = max(0, min(100, risk_level))  # Clamp between 0-100
        
        # Get recommendations based on risk level
        if risk_level < 30:
            coverage = "Basic"
            color = "green"
            recommendations = [
                "Basic evacuation coverage",
                "Standard personal accident insurance",
                "Minimal business interruption coverage"
            ]
        elif risk_level < 70:
            coverage = "Enhanced"
            color = "orange"
            recommendations = [
                "Enhanced evacuation coverage",
                "Comprehensive personal accident insurance",
                "Business interruption coverage",
                "Temporary accommodation coverage"
            ]
        else:
            coverage = "Premium"
            color = "red"
            recommendations = [
                "Premium evacuation coverage",
                "Full personal accident coverage",
                "Extended business interruption coverage",
                "Temporary relocation coverage",
                "Hazard-specific coverage",
                "Emergency response services"
            ]
        
        # Display risk summary
        st.markdown("### Risk Assessment Summary")
        
        # Risk level gauge
        st.markdown(f"#### Overall Risk Level: :{color}[{coverage}]")
        st.progress(risk_level / 100)
        
        # Key metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Evacuation Rate", f"{evacuated}/{total_agents} ({evacuation_rate*100:.1f}%)")
            st.metric("Average Health", f"{avg_health:.1f}%")
        
        # Top risk factors
        st.markdown("#### Top Risk Factors")
        if risk_factors:
            top_risks = sorted(risk_factors.items(), key=lambda x: x[1], reverse=True)[:3]
            for factor, value in top_risks:
                st.markdown(f"- **{factor.replace('_', ' ').title()}**: {value*100:.1f}%")
        else:
            st.info("No specific risk factors identified.")
        
        # Recommendations
        st.markdown("### Recommended Coverage")
        st.markdown(f"Based on the simulation results, we recommend **{coverage}** level coverage:")
        
        for rec in recommendations:
            st.markdown(f"- ✅ {rec}")
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📝 Get Custom Quote", use_container_width=True):
                st.session_state.show_quote_form = True
        with col2:
            if st.button("📊 View Detailed Analysis", use_container_width=True):
                st.session_state.show_detailed_analysis = True
        
        # Quote form
        if st.session_state.get('show_quote_form', False):
            with st.form("quote_form"):
                st.markdown("### Get a Custom Quote")
                name = st.text_input("Full Name")
                email = st.text_input("Email Address")
                phone = st.text_input("Phone Number")
                
                if st.form_submit_button("Submit Request"):
                    # In a real app, this would send the data to your backend
                    st.success("Thank you! A representative will contact you shortly with a personalized quote.")
                    st.session_state.show_quote_form = False
        
        # Detailed analysis
        if st.session_state.get('show_detailed_analysis', False):
            st.markdown("### Detailed Risk Analysis")
            
            # Create a DataFrame for visualization
            if risk_factors:
                df = pd.DataFrame({
                    'Risk Factor': [f.replace('_', ' ').title() for f in risk_factors.keys()],
                    'Risk Level': [v * 100 for v in risk_factors.values()]
                })
                
                # Sort by risk level
                df = df.sort_values('Risk Level', ascending=False)
                
                # Create a bar chart
                fig = px.bar(
                    df, 
                    x='Risk Level', 
                    y='Risk Factor',
                    orientation='h',
                    title='Risk Factor Analysis',
                    labels={'Risk Level': 'Risk Level (%)', 'Risk Factor': ''}
                )
                
                # Customize the chart
                fig.update_layout(
                    height=300 + len(risk_factors) * 20,  # Dynamic height
                    margin=dict(l=0, r=0, t=40, b=0),
                    xaxis=dict(range=[0, 100]),
                    yaxis=dict(autorange="reversed"),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Add mitigation strategies
            st.markdown("### Risk Mitigation Strategies")
            st.markdown("Based on the identified risks, consider these mitigation strategies:")
            
            if 'fire' in [f.lower() for f in risk_factors.keys()]:
                st.markdown("#### Fire Safety")
                st.markdown("""
                - Install additional fire extinguishers in high-risk areas
                - Conduct regular fire drills
                - Install smoke detectors and fire alarms
                - Consider a sprinkler system
                """)
            
            if 'crowd' in [f.lower() for f in risk_factors.keys()]:
                st.markdown("#### Crowd Management")
                st.markdown("""
                - Implement crowd control measures during peak times
                - Add more emergency exits
                - Install crowd monitoring systems
                - Train staff in crowd management
                """)
            
            if 'structural' in [f.lower() for f in risk_factors.keys()]:
                st.markdown("#### Structural Safety")
                st.markdown("""
                - Conduct structural integrity assessment
                - Reinforce weak structural elements
                - Install emergency lighting
                - Ensure clear signage for evacuation routes
                """)
            
            # Close button for detailed analysis
            if st.button("Close Detailed Analysis"):
                st.session_state.show_detailed_analysis = False
                st.experimental_rerun()
            
        # Display floor plan
        st.subheader("Floor Plan")
        try:
            # Try to load a floor plan image if available
            floor_plan_path = Path(__file__).parent / "assets" / "floor_plan.png"
            if floor_plan_path.exists():
                st.image(str(floor_plan_path), use_column_width=True)
            else:
                # Create a simple floor plan visualization
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.set_xlim(0, 100)
                ax.set_ylim(0, 100)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                
                # Draw walls
                for x, y in self.env.obstacles:
                    ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, color='gray', alpha=0.5))
                
                # Draw exits
                for i, (x, y) in enumerate(self.env.exits):
                    ax.add_patch(plt.Rectangle((x-1, y-1), 2, 2, color='green', alpha=0.7))
                    ax.text(x, y, f"Exit {i+1}", ha='center', va='center', color='white')
                
                # Draw hazards
                for hazard_type, data in self.env.hazards.items():
                    for (x, y), intensity in zip(data['positions'], data['intensities']):
                        color = 'red' if hazard_type == 'fire' else 'orange'
                        ax.add_patch(plt.Circle((x, y), 5, color=color, alpha=0.3*intensity))
                        ax.text(x, y, hazard_type[0].upper(), ha='center', va='center', color='white')
                
                st.pyplot(fig)
                
        except Exception as e:
            st.warning(f"Could not load floor plan: {e}")
            
        # Display insurance recommendations based on risk level
        risk_level = sum(sum(data['intensities']) for data in self.env.hazards.values()) / 10.0
        risk_level = min(max(risk_level, 0), 1.0)  # Clamp between 0 and 1
        self._display_insurance_recommendations(risk_level)
        
        # Show summary metrics
        st.header("Simulation Results")
        
        # Get the last scenario for visualization
        last_scenario = self.simulator.scenarios[-1]
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Simulations", len(self.simulator.scenarios))
        with col2:
            st.metric("Success Rate", f"{last_scenario.success_rate:.1f}%")
        with col3:
            st.metric("Avg Evacuation Time", f"{last_scenario.avg_evacuation_time:.1f}s")
        with col4:
            risk = sum(last_scenario.risk_distribution.values()) / len(last_scenario.risk_distribution)
            st.metric("Avg Risk Level", f"{risk*100:.1f}%")
        
        # Tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "Convergence Analysis", 
            "Risk Distribution", 
            "Agent Paths",
            "Risk Analysis"
        ])
        
        with tab1:
            st.subheader("Convergence Analysis")
            fig = self.viz.plot_convergence(self.simulator.convergence_data)
            st.pyplot(fig)
            
        with tab2:
            st.subheader("Risk Distribution")
            # Generate sample risk data (in a real app, this would come from the simulation)
            risk_data = {
                'positions': np.random.rand(100, 2) * 100,  # Random positions
                'values': np.random.rand(100)  # Random risk values
            }
            fig = self.viz.plot_risk_heatmap(self.env, risk_data)
            st.pyplot(fig)
        
        with tab3:
            st.subheader("Agent Paths")
            # Generate sample agent history (in a real app, this would come from the simulation)
            agents_history = {}
            for i, agent in enumerate(last_scenario.agents):
                if hasattr(agent, 'path') and agent.path:
                    agents_history[agent.agent_id] = [
                        {'position': pos, 'type': agent.agent_type} 
                        for pos in agent.path
                    ]
            
            if agents_history:
                fig = self.viz.plot_agent_movements(agents_history, self.env)
                st.pyplot(fig)
                
                # Add button to generate animation
                if st.button("Generate Evacuation Animation"):
                    with st.spinner('Generating animation...'):
                        self.animation_path = self.viz.create_evacuation_animation(
                            agents_history, 
                            self.env,
                            output_file='evacuation_animation.mp4'
                        )
                    st.success('Animation generated!')
                    st.video(self.animation_path)
            else:
                st.warning("No agent path data available for visualization.")
        
        with tab4:
            st.subheader("Risk Analysis")
            risk_analysis = self.simulator.get_risk_analysis()
            fig = self.viz.plot_risk_analysis(risk_analysis)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed risk factors
            st.subheader("Detailed Risk Factors")
            risk_data = []
            for agent in last_scenario.agents:
                if hasattr(agent, 'risk_factors'):
                    risk_data.append({
                        'Agent ID': agent.agent_id,
                        'Type': agent.agent_type,
                        'Total Risk': sum(agent.risk_factors.values()) / len(agent.risk_factors),
                        **agent.risk_factors
                    })
            
            if risk_data:
                df = pd.DataFrame(risk_data)
                st.dataframe(df.style.background_gradient(
                    cmap='YlOrRd',
                    subset=['Total Risk']
                ))

def main():
    st.title("Evacuation Monte Carlo Simulator")
    st.markdown("---")
    
    # Initialize session state
    if 'app' not in st.session_state:
        st.session_state.app = MonteCarloApp()
    
    app = st.session_state.app
    
    # Simulation controls
    with st.sidebar:
        st.header("Simulation Controls")
        num_simulations = st.slider("Number of Simulations", 1, 1000, 100, 10)
        real_time = st.checkbox("Enable Real-time Visualization", value=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Run Simulation"):
                with st.spinner("Running simulation..."):
                    try:
                        if real_time:
                            app.run_simulation(num_simulations, real_time=True)
                        else:
                            progress_bar = st.progress(0)
                            app.run_simulation(num_simulations, real_time=False)
                            progress_bar.progress(100)
                        st.success("Simulation completed successfully!")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
        
        with col2:
            if st.button("Stop Simulation"):
                app.animation_running = False
                st.warning("Simulation stopped by user")
        
        # Add hazard controls
        hazard_type = st.selectbox(
            "Hazard Type",
            ["fire", "smoke", "water", "debris"]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            hazard_x = st.slider("X Position", 0, 100, 50)
        with col2:
            hazard_y = st.slider("Y Position", 0, 100, 50)
        
        hazard_intensity = st.slider("Intensity", 0.1, 2.0, 1.0, 0.1)
        hazard_radius = st.slider("Radius", 5.0, 50.0, 15.0, 1.0)
        
        if st.button("Add Hazard"):
            app.env.add_hazard(
                hazard_x, hazard_y,
                hazard_type,
                hazard_intensity,
                hazard_radius
            )
            st.success(f"Added {hazard_type} hazard at ({hazard_x}, {hazard_y})")
    
    # Main content
    app.display_results()

if __name__ == "__main__":
    main()
