from setuptools import setup, find_packages

setup(
    name="evacuation_simulator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'matplotlib>=3.5.0',
        'scipy>=1.7.0',
        'streamlit>=1.24.0',
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Evacuation simulation with insurance risk assessment",
    url="https://github.com/yourusername/evacuation-simulator",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
