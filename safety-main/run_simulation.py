import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath('.'))

# Now import and run the Streamlit app
import streamlit.cli as stcli

def main():
    # Set the path to the Streamlit app
    app_path = os.path.join('navmesh_ml2d', 'ui', 'streamlit_app.py')
    
    # Run the Streamlit app
    sys.argv = ["streamlit", "run", app_path]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
