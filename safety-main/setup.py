from setuptools import setup, find_packages

setup(
    name="pathfinder_simulations",
    version="0.1.0",
    packages=find_packages(include=['pathfinder_main*']),
    install_requires=[
        'numpy>=1.20.0',
        'matplotlib>=3.4.0',
        'streamlit>=1.0.0',
        'Pillow>=8.0.0',
        'numba>=0.53.0',
        'torch>=1.9.0',
        'plotly>=5.0.0',
        'pandas>=1.3.0',
    ],
    python_requires='>=3.8',
    package_dir={"": "."},
)
