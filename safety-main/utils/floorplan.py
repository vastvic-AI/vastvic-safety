"""
Utility functions for floor plan parsing, obstacle and exit handling, and validation.
Ensures exits are never blocked by obstacles and provides visualization helpers.
"""
import numpy as np


def parse_floorplan_image(img_arr, threshold=128, invert=False):
    """
    Parse a grayscale floor plan image array into an obstacle grid.
    Args:
        img_arr (np.ndarray): Grayscale image array (H x W)
        threshold (int): Threshold for obstacle detection
        invert (bool): If True, invert obstacle/walkable logic
    Returns:
        np.ndarray: Obstacle grid (0=walkable, 1=obstacle)
    """
    if invert:
        obstacle_grid = (img_arr < threshold).astype(np.uint8)
    else:
        obstacle_grid = (img_arr > threshold).astype(np.uint8)
    return obstacle_grid


def place_exits_on_grid(grid, exits):
    """
    Ensure all exit positions in the grid are set to walkable (0).
    Args:
        grid (np.ndarray): Grid to modify (in-place)
        exits (list of (x, y)): Exit coordinates
    """
    for (x, y) in exits:
        grid[y, x] = 0


def validate_exits_open(grid, exits):
    """
    Assert that all exit positions are walkable in the grid.
    Raises AssertionError if any exit is blocked.
    """
    for (x, y) in exits:
        assert grid[y, x] == 0, f"Exit at ({x},{y}) is blocked by an obstacle!"


def visualize_grid_with_exits(grid, exits):
    """
    Return a matplotlib-ready RGB image showing obstacles in black and exits in green.
    Args:
        grid (np.ndarray): Obstacle grid
        exits (list): List of (x, y) exit positions
    Returns:
        np.ndarray: RGB image for visualization
    """
    img = np.stack([255*(1-grid)]*3, axis=-1).astype(np.uint8)  # white for walkable, black for obstacles
    for (x, y) in exits:
        img[y, x] = [0, 255, 0]  # green for exits
    return img
