"""
Tests for the GridNavMesh class.
"""

import numpy as np
import pytest
from navmesh_ml2d.core.navmesh import GridNavMesh

@pytest.fixture
def simple_navmesh():
    """Create a simple 10x10 navmesh with some obstacles."""
    obstacles = [(2, 2), (2, 3), (3, 2), (3, 3),  # 2x2 block
                (5, 5), (5, 6), (6, 5), (6, 6)]  # Another 2x2 block
    return GridNavMesh(10, 10, obstacles)

def test_initialization():
    """Test that the navmesh initializes correctly."""
    navmesh = GridNavMesh(10, 10)
    assert navmesh.width == 10
    assert navmesh.height == 10
    assert np.all(navmesh.grid == 0)

def test_obstacle_placement(simple_navmesh):
    """Test that obstacles are correctly placed."""
    assert simple_navmesh.grid[2, 2] == 1
    assert simple_navmesh.grid[3, 3] == 1
    assert simple_navmesh.grid[5, 5] == 1
    assert simple_navmesh.grid[6, 6] == 1
    assert simple_navmesh.grid[0, 0] == 0

def test_is_walkable(simple_navmesh):
    """Test the is_walkable method."""
    assert simple_navmesh.is_walkable(0, 0) is True
    assert simple_navmesh.is_walkable(2, 2) is False  # Obstacle
    assert simple_navmesh.is_walkable(-1, 0) is False  # Out of bounds
    assert simple_navmesh.is_walkable(0, 10) is False  # Out of bounds

def test_astar_path_finding(simple_navmesh):
    """Test A* path finding."""
    # Test simple path
    path = simple_navmesh.astar((0, 0), (9, 9))
    assert len(path) > 0
    assert path[0] == (0, 0)  # Start
    assert path[-1] == (9, 9)  # Goal
    
    # Test path around obstacle
    path = simple_navmesh.astar((1, 1), (4, 4))
    assert len(path) > 0
    assert path[0] == (1, 1)
    assert path[-1] == (4, 4)
    assert (2, 2) not in path  # Should go around the obstacle

def test_dynamic_obstacles(simple_navmesh):
    """Test dynamic obstacle handling."""
    # Add dynamic obstacle
    simple_navmesh.add_dynamic_obstacle(1, 1)
    assert not simple_navmesh.is_walkable(1, 1)
    
    # Test path avoids dynamic obstacle
    path = simple_navmesh.astar((0, 0), (2, 2))
    assert len(path) > 0
    assert (1, 1) not in path
    
    # Remove dynamic obstacle
    simple_navmesh.remove_dynamic_obstacle(1, 1)
    assert simple_navmesh.is_walkable(1, 1)

def test_hazard_avoidance(simple_navmesh):
    """Test hazard avoidance in path finding."""
    # Add hazard
    simple_navmesh.add_hazard(5, 5, radius=2, intensity=1.0)
    
    # Get path that should avoid the hazard
    path = simple_navmesh.astar((0, 0), (9, 9), avoid_hazards=True)
    
    # Check if path avoids the hazard area
    hazard_area = [(x, y) for x in range(3, 8) for y in range(3, 8)]
    assert not any(point in hazard_area for point in path)

def test_density_avoidance(simple_navmesh):
    """Test crowd density avoidance in path finding."""
    # Simulate high density area
    simple_navmesh.density_map[5, 5] = 10.0
    simple_navmesh.density_map[5, 6] = 10.0
    simple_navmesh.density_map[6, 5] = 10.0
    simple_navmesh.density_map[6, 6] = 10.0
    
    # Get path that should avoid high density
    path = simple_navmesh.astar((0, 0), (9, 9), avoid_density=True)
    
    # Check if path avoids the high density area
    density_area = [(5, 5), (5, 6), (6, 5), (6, 6)]
    assert not any(point in density_area for point in path)

def test_flow_field(simple_navmesh):
    """Test flow field integration."""
    # Update flow field toward goal (9, 9)
    simple_navmesh.update_flow_field([(9, 9)])
    
    # Check flow field direction near start
    flow = simple_navmesh.flow_field.get((1, 1), (0, 0))
    assert flow[0] > 0.5  # Mostly right
    assert flow[1] > 0.5  # Mostly down
    
    # Test path with flow field
    path = simple_navmesh.astar((0, 0), (9, 9), flow_field=True)
    assert len(path) > 0
    assert path[0] == (0, 0)
    assert path[-1] == (9, 9)

def test_performance_large_grid():
    """Test performance on a larger grid."""
    # Create a larger grid
    navmesh = GridNavMesh(100, 100)
    
    # Add some random obstacles
    import random
    for _ in range(500):
        x = random.randint(0, 99)
        y = random.randint(0, 99)
        navmesh.grid[y, x] = 1
    
    # Test path finding
    start = (0, 0)
    goal = (99, 99)
    path = navmesh.astar(start, goal, max_steps=10000)
    
    # Just verify we get a path (may be empty if no path exists)
    if path:
        assert path[0] == start
        assert path[-1] == goal
