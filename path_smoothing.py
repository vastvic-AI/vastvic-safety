from typing import List, Tuple
import numpy as np

Cell = Tuple[int, int]

def _bresenham_line(r0: int, c0: int, r1: int, c1: int):
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = (dr - dc)
    r, c = r0, c0
    while True:
        yield (r, c)
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc

def _los_clear(grid: np.ndarray, a: Cell, b: Cell) -> bool:
    H, W = grid.shape
    for rr, cc in _bresenham_line(a[0], a[1], b[0], b[1]):
        if not (0 <= rr < H and 0 <= cc < W):
            return False
        if grid[rr, cc] != 0 and (rr, cc) != a and (rr, cc) != b:
            return False
    return True


def smooth_path_los(path: List[Cell], grid: np.ndarray) -> List[Cell]:
    if not path:
        return path
    sm = [path[0]]
    j = 0
    while j < len(path) - 1:
        k = len(path) - 1
        found = False
        while k > j + 1:
            if _los_clear(grid, path[j], path[k]):
                sm.append(path[k])
                j = k
                found = True
                break
            k -= 1
        if not found:
            sm.append(path[j + 1])
            j += 1
    return sm


def chaikin(points: List[Tuple[float, float]], iterations: int = 2) -> List[Tuple[float, float]]:
    if len(points) < 3 or iterations <= 0:
        return points
    pts = points[:]
    for _ in range(iterations):
        new_pts = [pts[0]]
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            Q = (0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1)
            R = (0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1)
            new_pts.extend([Q, R])
        new_pts.append(pts[-1])
        pts = new_pts
    return pts


def smooth_for_visual(path: List[Cell], grid: np.ndarray, use_chaikin: bool = True, iterations: int = 1) -> List[Tuple[float, float]]:
    """Return a list of waypoints (float coords) derived from grid path.
    First apply LOS smoothing, then optional Chaikin for curvature.
    """
    if not path:
        return []
    los = smooth_path_los(path, grid)
    pts = [(float(r) + 0.0, float(c) + 0.0) for r, c in los]
    if use_chaikin:
        pts = chaikin(pts, iterations=iterations)
    return pts
