import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath('.'))

# Try to import the module
try:
    print("Trying to import...")
    from insurance.insurance_recommendation import recommend_for_agent
    print("Import successful!")
    # Test the function
    result = recommend_for_agent({})
    print(f"Test function result: {result}")
except ImportError as e:
    print(f"Import failed: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
