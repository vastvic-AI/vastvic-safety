import sys
print("Python path:")
for path in sys.path:
    print(f"- {path}")

try:
    from insurance.insurance_recommendation import recommend_for_agent
    print("\nImport successful!")
    print("Function signature:", recommend_for_agent.__code__.co_varnames[:recommend_for_agent.__code__.co_argcount])
except ImportError as e:
    print(f"\nImport failed: {e}")
    print("\nCurrent working directory:", __file__)
