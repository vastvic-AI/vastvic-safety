import sys
import os

# Print current working directory
print(f"Current working directory: {os.getcwd()}")

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())
print(f"Python path: {sys.path}")

# List files in the current directory
print("\nFiles in current directory:")
for f in os.listdir('.'):
    if os.path.isfile(f):
        print(f"- {f}")

# List files in the insurance directory
insurance_dir = os.path.join(os.getcwd(), 'insurance')
if os.path.exists(insurance_dir):
    print("\nFiles in insurance directory:")
    for f in os.listdir(insurance_dir):
        print(f"- {f}")

# Try to import
try:
    print("\nTrying to import...")
    from insurance.insurance_recommendation import recommend_for_agent
    print("Import successful!")
    # Test the function
    result = recommend_for_agent({})
    print(f"Test function result: {result}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
