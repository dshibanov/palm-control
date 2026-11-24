"""Palm RTS: a tiny WC3-style game controlled by hand gestures.

Hand mode (default)                      Mouse mode (--mouse, for testing the game)
  right hand index tip = cursor            mouse = cursor
  index tap           = select / click     left click / left drag = select / box select
  middle tap          = command            right click = command
  pinch + move + open = box select
  left open palm      = pan camera         WASD / arrows / screen edges
  left victory (V)    = build mode         B
  left fist           = cancel             Esc
Keys (both): M switch hand <-> mouse control, T train worker, B build house, Esc cancel, Q quit.
"""

import argparse
import math
import time

import numpy as np
import pygame

from rts.world import (FOOTPRINT, HOUSE_COST, TILE, WORKER_COST, World, UNIT_RADIUS, rect_px, tile_center)

W, H = 1100, 720
HUD_TOP, HUD_BOT = 34, 96
GRASS, GRASS2 = (62, 110, 60), (66, 116, 63)


class Game:
    def __init__(self, args):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("Palm RTS")
        self.font = pygame.font.SysFont("dejavusans", 15)
        self.big = pygame.font.SysFont("dejavusans", 20, bold=True)
        self.clock = pygame.time.Clock()
        self.world = World()
        self.cam = np.array([0.0, 200.0])
        self.build_mode = False
        self.box_start = None
        self.cursor = (W // 2, H // 2)
        self.msg, self.msg_until = "", 0.0
        self.args = args
        self.mouse_mode = args.mouse
        self.hi = None
        self.cursor_hand = args.cursor_hand
        self.pan_hand = "Left" if args.cursor_hand == "Right" else "Right"
        self.last_palm = None
        self.hands = None
        self.set_mode(self.mouse_mode)
        pygame.mouse.set_visible(False)  # we draw our own cursor in both modes

    # ---- helpers ----
    def set_mode(self, mouse: bool):
        """Switch between mouse and hand control. Hand mode owns the camera; mouse mode releases it."""
        self.mouse_mode = mouse
        self.box_start, self.last_palm, self.hands = None, None, None
        if mouse:
            if self.hi:
                self.hi.stop()
                self.hi = None
            self.cursor = pygame.mouse.get_pos()
        elif self.hi is None:
            from hand_input import HandInput
            self.hi = HandInput(camera=self.args.camera, click_drop=self.args.click_drop, hand=self.cursor_hand).start()

    def say(self, text, secs=1.5):
        self.msg, self.msg_until = text, time.monotonic() + secs

    def to_world(self, p):
        return p[0] + self.cam[0], p[1] + self.cam[1]

    def buttons(self):
        w, out, x = self.world, [], 12
        if w.sel_building is not None:
            b = w.get_building(w.sel_building)
            if b and b.kind == "hall" and b.done:
                ok = w.gold >= WORKER_COST and w.supply_used < w.supply_cap
                out.append((f"Train Worker [T]  {WORKER_COST}g", pygame.Rect(x, H - 64, 220, 48), self.act_train, ok))
                x += 232
        if w.sel_units:
            ok = w.gold >= HOUSE_COST
            label = "Cancel build [Esc]" if self.build_mode else f"Build House [B]  {HOUSE_COST}g"
            out.append((label, pygame.Rect(x, H - 64, 220, 48), self.act_build, ok or self.build_mode))
        return out

    def act_train(self):
        b = self.world.get_building(self.world.sel_building) if self.world.sel_building else None
        if not b:
            return
        if not self.world.train_worker(b):
            self.say("Cannot train: need gold or supply (build a house)")

    def act_build(self):
        if self.build_mode:
            self.build_mode = False
        elif self.world.sel_units:
            self.build_mode = True
        else:
            self.say("Select a worker first")

    def ghost_tile(self):
        wx, wy = self.to_world(self.cursor)
        return int(round(wx / TILE)) - 1, int(round(wy / TILE)) - 1

    # ---- actions from input ----
    def do_select(self, pos):
        if pos[1] > H - HUD_BOT:
            for _, rect, fn, ok in self.buttons():
                if rect.collidepoint(pos) and ok:
                    fn()
                    return
            return
        if self.build_mode:
            if self.world.place_building("house", self.ghost_tile()):
                self.build_mode = False
            else:
                self.say("Cannot build there")
            return
        self.world.select_point(*self.to_world(pos))

    def do_command(self, pos):
        if self.build_mode:
            self.build_mode = False
            return
        self.world.command_at(*self.to_world(pos))

    def do_box(self, a, b):
        if math.hypot(b[0] - a[0], b[1] - a[1]) < 10 or max(a[1], b[1]) > H - HUD_BOT:
            return
        (x0, y0), (x1, y1) = self.to_world(a), self.to_world(b)
        self.world.select_box(x0, y0, x1, y1)

    # ---- input ----
    def handle_keys(self, ev):
        if ev.key == pygame.K_q:
            return False
        if ev.key == pygame.K_m:
            self.set_mode(not self.mouse_mode)
            self.say("Mouse control" if self.mouse_mode else "Hand control")
        elif ev.key == pygame.K_ESCAPE:
            self.build_mode = False
        elif ev.key == pygame.K_b:
            self.act_build()
        elif ev.key == pygame.K_t:
            self.act_train()
        return True

    def poll_mouse(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            self.box_start = ev.pos
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1 and self.box_start:
            if math.hypot(ev.pos[0] - self.box_start[0], ev.pos[1] - self.box_start[1]) < 10:
                self.do_select(ev.pos)
            else:
                self.do_box(self.box_start, ev.pos)
            self.box_start = None
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3:
            self.do_command(ev.pos)

    def poll_hands(self):
        snap = self.hi.snapshot()
        self.hands = snap
        hand = snap.hands.get(self.cursor_hand)
        if hand:
            self.cursor = (int(hand.cursor[0] * (W - 1)), int(hand.cursor[1] * (H - 1)))
        for e in self.hi.drain_events():
            if e.kind == "click" and e.hand == self.cursor_hand:
                self.do_select(self.cursor)
            elif e.kind == "rclick" and e.hand == self.cursor_hand:
                self.do_command(self.cursor)
            elif e.kind == "pinch_down" and e.hand == self.cursor_hand:
                self.box_start = self.cursor
            elif e.kind == "pinch_up" and e.hand == self.cursor_hand and self.box_start:
                self.do_box(self.box_start, self.cursor)
                self.box_start = None
            elif e.kind == "gesture" and e.hand == self.pan_hand:
                if e.name == "Victory":
                    self.act_build()
                elif e.name == "Closed_Fist":
                    self.build_mode = False
        # pan with the other hand's open palm
        ph = snap.hands.get(self.pan_hand)
        if ph and ph.builtin == "Open_Palm":
            if self.last_palm is not None:
                self.cam[0] -= (ph.palm[0] - self.last_palm[0]) * W * 2.0
                self.cam[1] -= (ph.palm[1] - self.last_palm[1]) * H * 2.0
            self.last_palm = ph.palm
        else:
            self.last_palm = None

    def pan_keys(self, dt):
        keys = pygame.key.get_pressed()
        v = 500 * dt
        dx = (keys[pygame.K_d] or keys[pygame.K_RIGHT]) - (keys[pygame.K_a] or keys[pygame.K_LEFT])
        dy = (keys[pygame.K_s] or keys[pygame.K_DOWN]) - (keys[pygame.K_w] or keys[pygame.K_UP])
        mx, my = self.cursor
        edge = 18
        dx += (mx >= W - edge) - (mx <= edge)
        dy += (my >= H - edge) - (my <= edge + HUD_TOP * 0)
        self.cam += (dx * v, dy * v)

    # ---- drawing ----
    def draw_world(self):
        w, s = self.world, self.screen
        cx, cy = int(self.cam[0]), int(self.cam[1])
        s.fill((20, 24, 20))
        for ty in range(max(cy // TILE, 0), min((cy + H) // TILE + 1, w.h)):
            for tx in range(max(cx // TILE, 0), min((cx + W) // TILE + 1, w.w)):
                r = (tx * TILE - cx, ty * TILE - cy, TILE, TILE)
                pygame.draw.rect(s, GRASS if (tx + ty) % 2 else GRASS2, r)
                if w.trees[ty, tx]:
                    pygame.draw.circle(s, (30, 80, 40), (r[0] + 16, r[1] + 16), 14)
                    pygame.draw.circle(s, (24, 62, 32), (r[0] + 16, r[1] + 16), 14, 2)
        for m in w.mines:
            x, y, ww, hh = rect_px(m.rect)
            pygame.draw.rect(s, (200, 170, 40) if m.gold > 0 else (90, 90, 80), (x - cx, y - cy, ww, hh), border_radius=8)
            s.blit(self.font.render(str(m.gold), True, (20, 20, 10)), (x - cx + 8, y - cy + 22))
        for b in w.buildings:
            x, y, ww, hh = rect_px(b.rect)
            col = (150, 110, 80) if b.kind == "hall" else (170, 140, 110)
            r = pygame.Rect(x - cx, y - cy, ww, hh)
            pygame.draw.rect(s, col if b.done else (110, 100, 90), r, border_radius=6)
            pygame.draw.rect(s, (255, 255, 255) if b.id == w.sel_building else (40, 30, 20), r, 3, border_radius=6)
            if not b.done:
                pygame.draw.rect(s, (0, 0, 0), (r.x, r.bottom + 3, r.w, 6))
                pygame.draw.rect(s, (80, 220, 80), (r.x, r.bottom + 3, int(r.w * b.progress), 6))
            elif b.queue:
                pygame.draw.rect(s, (0, 0, 0), (r.x, r.bottom + 3, r.w, 6))
                pygame.draw.rect(s, (80, 160, 255), (r.x, r.bottom + 3, int(r.w * b.train_t / 5.0), 6))
                s.blit(self.font.render(f"queue {b.queue}", True, (255, 255, 255)), (r.x, r.bottom + 10))
        for u in w.units:
            p = (int(u.x - cx), int(u.y - cy))
            if u.id in w.sel_units:
                pygame.draw.circle(s, (120, 255, 120), p, UNIT_RADIUS + 5, 2)
            pygame.draw.circle(s, (220, 190, 150), p, UNIT_RADIUS)
            pygame.draw.circle(s, (40, 40, 40), p, UNIT_RADIUS, 2)
            if u.carry:
                pygame.draw.circle(s, (230, 200, 40), (p[0] + 8, p[1] - 8), 5)
            if u.state == "gather":
                pygame.draw.arc(s, (255, 255, 120), (p[0] - 16, p[1] - 16, 32, 32), 0, 6.28 * min(u.timer / 2.0, 1), 3)
            elif u.state == "build":
                pygame.draw.circle(s, (255, 255, 255), (p[0], p[1] - 20), 4)
        if self.build_mode:
            tx, ty = self.ghost_tile()
            ww, hh = FOOTPRINT["house"]
            ok = w.can_place("house", (tx, ty))
            g = pygame.Surface((ww * TILE, hh * TILE), pygame.SRCALPHA)
            g.fill((80, 255, 80, 130) if ok else (255, 70, 70, 130))
            s.blit(g, (tx * TILE - cx, ty * TILE - cy))

    def draw_hud(self):
        w, s = self.world, self.screen
        pygame.draw.rect(s, (28, 28, 34), (0, 0, W, HUD_TOP))
        s.blit(self.big.render(f"Gold {w.gold}", True, (240, 205, 60)), (12, 6))
        s.blit(self.big.render(f"Supply {w.supply_used}/{w.supply_cap}", True, (200, 220, 255)), (150, 6))
        mode = self.big.render("MOUSE" if self.mouse_mode else "HAND", True, (255, 200, 120) if self.mouse_mode else (120, 255, 160))
        s.blit(mode, (W - mode.get_width() - 12, 6))
        hint = self.font.render("[M] switch", True, (140, 140, 150))
        s.blit(hint, (W - mode.get_width() - hint.get_width() - 24, 9))
        pygame.draw.rect(s, (28, 28, 34), (0, H - HUD_BOT, W, HUD_BOT))
        if w.sel_units:
            s.blit(self.font.render(f"{len(w.sel_units)} worker(s) selected", True, (200, 255, 200)), (12, H - HUD_BOT + 4))
        elif w.sel_building is not None:
            s.blit(self.font.render("Town Hall" if w.get_building(w.sel_building).kind == "hall" else "House", True, (255, 230, 200)), (12, H - HUD_BOT + 4))
        else:
            s.blit(self.font.render("tap to select - middle tap = command - pinch & drag = box select", True, (150, 150, 160)), (12, H - HUD_BOT + 4))
        for label, rect, _, ok in self.buttons():
            hot = rect.collidepoint(self.cursor)
            pygame.draw.rect(s, (70, 110, 70) if hot and ok else (55, 55, 70) if ok else (45, 40, 40), rect, border_radius=6)
            pygame.draw.rect(s, (160, 160, 180), rect, 2, border_radius=6)
            s.blit(self.font.render(label, True, (240, 240, 240) if ok else (120, 120, 120)), (rect.x + 10, rect.y + 15))
        if time.monotonic() < self.msg_until:
            t = self.big.render(self.msg, True, (255, 220, 120))
            s.blit(t, (W // 2 - t.get_width() // 2, HUD_TOP + 10))

    def draw_overlay(self):
        s = self.screen
        if self.box_start:
            a, b = self.box_start, self.cursor
            r = pygame.Rect(min(a[0], b[0]), min(a[1], b[1]), abs(a[0] - b[0]), abs(a[1] - b[1]))
            pygame.draw.rect(s, (120, 255, 120), r, 2)
        if not self.mouse_mode and self.hands:
            snap = self.hands
            if snap.frame is not None:
                rgb = snap.frame[:, :, ::-1].copy()
                surf = pygame.image.frombuffer(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB")
                s.blit(surf, (W - 200, H - HUD_BOT - 152))
            for i, (label, h) in enumerate(snap.hands.items()):
                t = f"{label}: {h.builtin or '-'}{' PINCH' if h.pinch else ''}"
                s.blit(self.font.render(t, True, (255, 255, 255)), (W - 200, H - HUD_BOT - 172 - i * 18))
            if self.cursor_hand not in snap.hands:
                s.blit(self.big.render(f"show {self.cursor_hand} hand", True, (255, 120, 120)), (W // 2 - 90, HUD_TOP + 40))
            if self.hi.error:
                s.blit(self.big.render(self.hi.error, True, (255, 80, 80)), (12, HUD_TOP + 10))
        pinch = bool(self.hands and self.cursor_hand in self.hands.hands and self.hands.hands[self.cursor_hand].pinch)
        col = (255, 120, 120) if pinch else (255, 255, 255)
        pygame.draw.circle(s, (0, 0, 0), self.cursor, 11, 4)
        pygame.draw.circle(s, col, self.cursor, 9, 2)
        pygame.draw.line(s, col, (self.cursor[0] - 14, self.cursor[1]), (self.cursor[0] + 14, self.cursor[1]), 1)
        pygame.draw.line(s, col, (self.cursor[0], self.cursor[1] - 14), (self.cursor[0], self.cursor[1] + 14), 1)

    def run(self):
        running = True
        try:
            while running:
                dt = min(self.clock.tick(60) / 1000.0, 0.1)
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        running = False
                    elif ev.type == pygame.KEYDOWN:
                        running = self.handle_keys(ev) and running
                    elif self.mouse_mode:
                        self.poll_mouse(ev)
                if self.mouse_mode:
                    self.cursor = pygame.mouse.get_pos()
                else:
                    self.poll_hands()
                self.pan_keys(dt)
                self.cam[0] = min(max(self.cam[0], 0), self.world.w * TILE - W)
                self.cam[1] = min(max(self.cam[1], -HUD_TOP), self.world.h * TILE - H + HUD_BOT)
                self.world.update(dt)
                self.draw_world()
                self.draw_hud()
                self.draw_overlay()
                pygame.display.flip()
        finally:
            if self.hi:
                self.hi.stop()
            pygame.quit()


def main():
    ap = argparse.ArgumentParser(description="Palm RTS")
    ap.add_argument("--mouse", action="store_true", help="play with mouse/keyboard instead of hands")
    ap.add_argument("--camera", type=int, default=2)
    ap.add_argument("--hand", "--cursor-hand", dest="cursor_hand", default="right", type=str.lower,
                    choices=["left", "right"], help="hand that moves the cursor; the other pans/builds (default right)")
    ap.add_argument("--click-drop", type=float, default=None, help="override the saved tap sensitivity")
    args = ap.parse_args()
    args.cursor_hand = args.cursor_hand.capitalize()
    Game(args).run()


if __name__ == "__main__":
    main()
