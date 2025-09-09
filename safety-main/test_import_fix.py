import os
import sys

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

print(f"Project root: {project_root}")
print("Contents of insurance directory:")
try:
    print(os.listdir(os.path.join(project_root, 'insurance')))
    
    print("\nTrying to import...")
    from insurance.insurance_recommendation import recommend_for_agent
    print("Import successful!")
    print("Function signature:", recommend_for_agent.__code__.co_varnames[:recommend_for_agent.__code__.co_argcount])
except Exception as e:
    print(f"Error: {e}")
