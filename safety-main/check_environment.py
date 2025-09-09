import sys
import platform
import importlib.metadata

def check_environment():
    print("=== Python Environment Check ===")
    print(f"Python Version: {platform.python_version()}")
    print(f"Python Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    
    print("\n=== Required Dependencies ===")
    required = [
        'numpy',
        'matplotlib',
        'scipy',
        'pandas',
        'streamlit',
        'Pillow',
        'tqdm',
        'pyyaml',
        'opencv-python',
        'scikit-learn'
    ]
    
    for package in required:
        try:
            version = importlib.metadata.version(package)
            print(f"✓ {package}: {version}")
        except importlib.metadata.PackageNotFoundError:
            print(f"✗ {package}: Not installed")
    
    print("\n=== Environment Path ===")
    for path in sys.path:
        print(f"- {path}")

if __name__ == "__main__":
    check_environment()
