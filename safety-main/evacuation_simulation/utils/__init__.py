"""Utility functions for the evacuation simulation."""

from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np

__all__ = [
    'calculate_distance',
    'normalize_vector',
    'calculate_angle',
    'point_in_polygon',
    'line_intersects_circle',
    'calculate_centroid'
]

def calculate_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points.
    
    Args:
        p1: First point as [x, y]
        p2: Second point as [x, y]
        
    Returns:
        Distance between the points
    """
    return np.linalg.norm(p2 - p1)

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length.
    
    Args:
        v: Input vector
        
    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def calculate_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate the angle between two vectors in radians.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Angle in radians
    """
    v1_u = normalize_vector(v1)
    v2_u = normalize_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def point_in_polygon(point: np.ndarray, polygon: List[np.ndarray]) -> bool:
    """Check if a point is inside a polygon using ray casting algorithm.
    
    Args:
        point: Point to check [x, y]
        polygon: List of polygon vertices as [x, y] points
        
    Returns:
        True if point is inside the polygon, False otherwise
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def line_intersects_circle(
    line_start: np.ndarray, 
    line_end: np.ndarray, 
    circle_center: np.ndarray, 
    radius: float
) -> bool:
    """Check if a line segment intersects a circle.
    
    Args:
        line_start: Start of line segment [x, y]
        line_end: End of line segment [x, y]
        circle_center: Center of circle [x, y]
        radius: Radius of circle
        
    Returns:
        True if line segment intersects the circle, False otherwise
    """
    # Vector from line start to end
    d = line_end - line_start
    # Vector from line start to circle center
    f = line_start - circle_center
    
    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - radius * radius
    
    discriminant = b * b - 4 * a * c
    
    if discriminant < 0:
        return False  # No intersection
    
    discriminant = np.sqrt(discriminant)
    t1 = (-b - discriminant) / (2 * a)
    t2 = (-b + discriminant) / (2 * a)
    
    # Check if either t is in [0, 1]
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)

def calculate_centroid(points: List[np.ndarray]) -> np.ndarray:
    """Calculate the centroid of a set of points.
    
    Args:
        points: List of points as [x, y] arrays
        
    Returns:
        Centroid point as [x, y]
    """
    if not points:
        return np.array([0, 0], dtype=np.float32)
    
    points_array = np.array(points)
    return np.mean(points_array, axis=0, dtype=np.float32)
