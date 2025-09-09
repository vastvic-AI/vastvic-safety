# Entrypoint for launching the Streamlit UI
import os
import sys
import subprocess

# Ensure project root is in PYTHONPATH for package imports
project_root = os.path.dirname(os.path.abspath(__file__))
os.environ['PYTHONPATH'] = project_root + os.pathsep + os.environ.get('PYTHONPATH', '')

subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'navmesh_ml2d/ui/streamlit_app.py'])
