"""Game state and rules. Pure Python (no pygame) so it is easy to test."""

import math
import random
from dataclasses import dataclass, field

import numpy as np

from .pathing import astar

TILE = 32
MAP_W, MAP_H = 48, 32
UNIT_SPEED = 96.0
WORKER_COST, HOUSE_COST = 50, 30
GATHER_TIME, CARRY, BUILD_TIME, TRAIN_TIME = 2.0, 10, 6.0, 5.0
HALL_SUPPLY, HOUSE_SUPPLY = 5, 4
FOOTPRINT = {"hall": (3, 3), "house": (2, 2)}
UNIT_RADIUS = 12


def rect_tiles(r):
    tx, ty, w, h = r
    return [(x, y) for y in range(ty, ty + h) for x in range(tx, tx + w)]


def rect_ring(r):
    tx, ty, w, h = r
    return [(x, y) for y in range(ty - 1, ty + h + 1) for x in range(tx - 1, tx + w + 1)
            if not (tx <= x < tx + w and ty <= y < ty + h)]


def rect_px(r):
    return r[0] * TILE, r[1] * TILE, r[2] * TILE, r[3] * TILE


def px_in_rect(wx, wy, r):
    x, y, w, h = rect_px(r)
    return x <= wx < x + w and y <= wy < y + h


def tile_center(t):
    return (t[0] + 0.5) * TILE, (t[1] + 0.5) * TILE


@dataclass
class Mine:
    id: int
    rect: tuple
    gold: int


@dataclass
class Building:
    id: int
    kind: str
    rect: tuple
    done: bool = True
    progress: float = 1.0
    queue: int = 0
    train_t: float = 0.0


@dataclass
class Unit:
    id: int
    x: float
    y: float
    state: str = "idle"  # idle move to_mine gather to_hall to_site build
    path: list = field(default_factory=list)
    carry: int = 0
    timer: float = 0.0
    mine_id: int | None = None
    site_id: int | None = None


class World:
    def __init__(self, seed=1):
        self.w, self.h = MAP_W, MAP_H
        self.trees = np.zeros((self.h, self.w), bool)
        self.units: list[Unit] = []
        self.buildings: list[Building] = []
        self.mines: list[Mine] = []
        self.gold = 100
        self._id = 0
        self.sel_units: list[int] = []
        self.sel_building: int | None = None

        rng = random.Random(seed)
        self.trees[0, :] = self.trees[-1, :] = True
        self.trees[:, 0] = self.trees[:, -1] = True
        for _ in range(16):
            cx, cy = rng.randrange(3, self.w - 3), rng.randrange(3, self.h - 3)
            for _ in range(rng.randint(6, 16)):
                x, y = cx + rng.randint(-2, 2), cy + rng.randint(-2, 2)
                if 0 < x < self.w - 1 and 0 < y < self.h - 1:
                    self.trees[y, x] = True
        hall = self._add_building("hall", (5, 13, 3, 3))
        self._add_mine((13, 12, 2, 2), 500)
        self._add_mine((20, 23, 2, 2), 500)
        for r in [hall.rect] + [m.rect for m in self.mines]:
            tx, ty, w, h = r
            self.trees[max(ty - 2, 1):ty + h + 2, max(tx - 2, 1):tx + w + 2] = False
        for i in range(3):
            x, y = tile_center((9, 13 + i))
            self.units.append(Unit(self._next(), x, y))

    # ---- helpers ----
    def _next(self):
        self._id += 1
        return self._id

    def _add_building(self, kind, rect, done=True):
        b = Building(self._next(), kind, rect, done, 1.0 if done else 0.0)
        self.buildings.append(b)
        return b

    def _add_mine(self, rect, gold):
        self.mines.append(Mine(self._next(), rect, gold))

    def blocked(self) -> np.ndarray:
        b = self.trees.copy()
        for r in [x.rect for x in self.buildings] + [m.rect for m in self.mines]:
            for tx, ty in rect_tiles(r):
                b[ty, tx] = True
        return b

    @staticmethod
    def unit_tile(u):
        return int(u.x // TILE), int(u.y // TILE)

    def _path(self, u, goals) -> bool:
        blocked = self.blocked()
        free = {g for g in goals if 0 <= g[0] < self.w and 0 <= g[1] < self.h and not blocked[g[1], g[0]]}
        tiles = astar(blocked, self.unit_tile(u), free)
        if tiles is None:
            return False
        u.path = [tile_center(t) for t in tiles[1:]]
        return True

    def _go(self, u, state, goals) -> bool:
        if self._path(u, goals):
            u.state = state
            return True
        u.state, u.path = "idle", []
        return False

    def get_building(self, bid):
        return next((b for b in self.buildings if b.id == bid), None)

    def get_mine(self, mid):
        return next((m for m in self.mines if m.id == mid), None)

    def get_unit(self, uid):
        return next((u for u in self.units if u.id == uid), None)

    # ---- economy ----
    @property
    def supply_cap(self):
        return sum(HALL_SUPPLY if b.kind == "hall" else HOUSE_SUPPLY for b in self.buildings if b.done)

    @property
    def supply_used(self):
        return len(self.units) + sum(b.queue for b in self.buildings)

    def train_worker(self, b: Building) -> bool:
        if b.kind != "hall" or not b.done or self.gold < WORKER_COST or self.supply_used >= self.supply_cap:
            return False
        self.gold -= WORKER_COST
        b.queue += 1
        return True

    def can_place(self, kind, tile) -> bool:
        w, h = FOOTPRINT[kind]
        rect = (tile[0], tile[1], w, h)
        if tile[0] < 1 or tile[1] < 1 or tile[0] + w > self.w - 1 or tile[1] + h > self.h - 1:
            return False
        blocked = self.blocked()
        tiles = rect_tiles(rect)
        if any(blocked[y, x] for x, y in tiles):
            return False
        return not any(self.unit_tile(u) in tiles for u in self.units)

    def place_building(self, kind, tile) -> bool:
        """Place a house with the nearest selected worker as builder."""
        cost = HOUSE_COST
        workers = [u for u in map(self.get_unit, self.sel_units) if u]
        if not workers or self.gold < cost or not self.can_place(kind, tile):
            return False
        w, h = FOOTPRINT[kind]
        b = self._add_building(kind, (tile[0], tile[1], w, h), done=False)
        self.gold -= cost
        cx, cy = tile_center((tile[0], tile[1]))
        builder = min(workers, key=lambda u: math.hypot(u.x - cx, u.y - cy))
        builder.site_id = b.id
        if not self._go(builder, "to_site", rect_ring(b.rect)):
            self.buildings.remove(b)
            self.gold += cost
            return False
        return True

    # ---- selection & commands ----
    def select_point(self, wx, wy):
        best = None
        for u in self.units:
            d = math.hypot(u.x - wx, u.y - wy)
            if d <= UNIT_RADIUS + 4 and (best is None or d < best[0]):
                best = (d, u)
        if best:
            self.sel_units, self.sel_building = [best[1].id], None
            return
        for b in self.buildings:
            if px_in_rect(wx, wy, b.rect):
                self.sel_units, self.sel_building = [], b.id
                return
        self.sel_units, self.sel_building = [], None

    def select_box(self, x0, y0, x1, y1):
        x0, x1, y0, y1 = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
        ids = [u.id for u in self.units if x0 <= u.x <= x1 and y0 <= u.y <= y1]
        self.sel_units, self.sel_building = ids, None

    def command_at(self, wx, wy) -> str | None:
        units = [u for u in map(self.get_unit, self.sel_units) if u]
        if not units:
            return None
        for m in self.mines:
            if px_in_rect(wx, wy, m.rect) and m.gold > 0:
                for u in units:
                    u.mine_id = m.id
                    self._send_to_mine(u, m)
                return "gather"
        for b in self.buildings:
            if not b.done and px_in_rect(wx, wy, b.rect):
                for u in units:
                    u.site_id = b.id
                    self._go(u, "to_site", rect_ring(b.rect))
                return "build"
        target = (int(wx // TILE), int(wy // TILE))
        blocked = self.blocked()
        cands = sorted(
            (t for t in ((target[0] + dx, target[1] + dy) for dx in range(-4, 5) for dy in range(-4, 5))
             if 0 <= t[0] < self.w and 0 <= t[1] < self.h and not blocked[t[1], t[0]]),
            key=lambda t: (t[0] - target[0]) ** 2 + (t[1] - target[1]) ** 2)
        if not cands:
            return None
        for i, u in enumerate(units):
            self._go(u, "move", {cands[i % len(cands)]})
        return "move"

    def _send_to_mine(self, u, m):
        self._go(u, "to_mine", rect_ring(m.rect))

    def _send_to_hall(self, u):
        halls = [b for b in self.buildings if b.kind == "hall" and b.done]
        if not halls:
            u.state, u.path = "idle", []
            return
        hall = min(halls, key=lambda b: math.hypot(tile_center(b.rect[:2])[0] - u.x, tile_center(b.rect[:2])[1] - u.y))
        self._go(u, "to_hall", rect_ring(hall.rect))

    # ---- simulation ----
    def update(self, dt):
        for u in self.units:
            if u.state in ("move", "to_mine", "to_hall", "to_site"):
                self._step(u, dt)
                if not u.path:
                    self._arrive(u)
            elif u.state == "gather":
                m = self.get_mine(u.mine_id)
                u.timer += dt
                if u.timer >= GATHER_TIME:
                    take = min(CARRY, m.gold) if m else 0
                    if m:
                        m.gold -= take
                    u.carry = take
                    self._send_to_hall(u)
            elif u.state == "build":
                b = self.get_building(u.site_id)
                if b is None or b.done:
                    u.state = "idle"
                    continue
                b.progress = min(1.0, b.progress + dt / BUILD_TIME)
                if b.progress >= 1.0:
                    b.done = True
                    for o in self.units:
                        if o.site_id == b.id:
                            o.state, o.site_id = "idle", None
        for b in self.buildings:
            if b.kind == "hall" and b.done and b.queue > 0:
                b.train_t += dt
                if b.train_t >= TRAIN_TIME:
                    b.train_t = 0.0
                    b.queue -= 1
                    self._spawn_worker(b)

    def _spawn_worker(self, hall):
        blocked = self.blocked()
        for t in rect_ring(hall.rect):
            if 0 <= t[0] < self.w and 0 <= t[1] < self.h and not blocked[t[1], t[0]]:
                x, y = tile_center(t)
                self.units.append(Unit(self._next(), x, y))
                return

    @staticmethod
    def _step(u, dt):
        if not u.path:
            return
        tx, ty = u.path[0]
        dx, dy = tx - u.x, ty - u.y
        d = math.hypot(dx, dy)
        move = UNIT_SPEED * dt
        if d <= move:
            u.x, u.y = tx, ty
            u.path.pop(0)
        else:
            u.x += dx / d * move
            u.y += dy / d * move

    def _arrive(self, u):
        if u.state == "move":
            u.state = "idle"
        elif u.state == "to_mine":
            m = self.get_mine(u.mine_id)
            if m and m.gold > 0:
                u.state, u.timer = "gather", 0.0
            else:
                u.state = "idle"
        elif u.state == "to_hall":
            self.gold += u.carry
            u.carry = 0
            m = self.get_mine(u.mine_id)
            if m and m.gold > 0:
                self._send_to_mine(u, m)
            else:
                u.state = "idle"
        elif u.state == "to_site":
            b = self.get_building(u.site_id)
            u.state = "build" if b and not b.done else "idle"
