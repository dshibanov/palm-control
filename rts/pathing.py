"""A* on a tile grid (8-neighbour, no corner cutting, multi-goal)."""

import heapq
import math

import numpy as np

_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def astar(blocked: np.ndarray, start: tuple, goals: set):
    """Shortest tile path from start to any tile in goals, or None. blocked[y, x] is True for walls."""
    if not goals:
        return None
    h, w = blocked.shape
    if start in goals:
        return [start]

    def heur(t):
        return min(max(abs(t[0] - g[0]), abs(t[1] - g[1])) for g in goals)

    open_ = [(heur(start), 0.0, start)]
    came: dict = {}
    best = {start: 0.0}
    while open_:
        _, g, cur = heapq.heappop(open_)
        if cur in goals:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if g > best.get(cur, math.inf):
            continue
        for dx, dy in _DIRS:
            nx, ny = cur[0] + dx, cur[1] + dy
            if not (0 <= nx < w and 0 <= ny < h) or blocked[ny, nx]:
                continue
            if dx and dy and (blocked[cur[1], nx] or blocked[ny, cur[0]]):
                continue
            ng = g + (1.414 if dx and dy else 1.0)
            if ng < best.get((nx, ny), math.inf):
                best[(nx, ny)] = ng
                came[(nx, ny)] = cur
                heapq.heappush(open_, (ng + heur((nx, ny)), ng, (nx, ny)))
    return None
