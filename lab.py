"""Control lab: develop, tune and calibrate one hand control at a time.

  uv run lab.py -m cursor   calibrate / tune hand-cursor movement
  uv run lab.py -m click    tune tap detection (index = select, middle = command)

Switch modes live with Tab or in the Params panel (top-right button, or P).
Cursor mode: Params has a "tracking point" tab and a "smoothing filter" tab (filter presets: add, remove,
turn on/off; the ones that are on are drawn on the plot at the bottom right; closing Params saves them).
Common keys: S save (per hand, calibration.json)   R reset the mode's defaults   Q / Esc quit.
White ring = filtered cursor (what the game uses). Red dot = raw, unfiltered.
Preview (bottom-left): camera with the active region (yellow) and the tracked point.
Modes live in lab_modes.py.
"""

import argparse
import time
from collections import deque

import pygame

import calibration
from hand_input import HandInput
from lab_modes import GREY, H, MODES, W

BTN = pygame.Rect(W - 112, 8, 100, 30)
PANEL = pygame.Rect(W - 472, 46, 460, 364)  # stays above the bottom plot/preview row
TAB_H = 28


class Lab:
    def __init__(self, args):
        self.args = args
        self.font = pygame.font.SysFont("dejavusans", 15)
        self.big = pygame.font.SysFont("dejavusans", 22, bold=True)
        self.hi = HandInput(camera=args.camera, hand=args.hand).start()
        self.modes = {name: cls(self) for name, cls in MODES.items()}
        self.mode = self.modes[args.mode]
        self._panel_open = False
        self.hist: deque = deque(maxlen=45)  # recent cursor px, for jitter
        self.now = time.monotonic()
        self.snap = self.hand = self.cur = self.raw = None
        self.events: list = []
        self.msg, self.msg_until = "", 0.0

    @property
    def panel_open(self):
        return self._panel_open

    @panel_open.setter
    def panel_open(self, v):
        if self._panel_open and not v:  # closing Params saves the filter presets
            self.hi_autosave()
        self._panel_open = v

    def hi_autosave(self):
        calibration.update(self.args.hand, filter_presets=self.hi.presets, filter_driver=self.hi.driver)
        self.say("filter presets saved", 1.5)

    def say(self, text, secs=2.0):
        self.msg, self.msg_until = text, time.monotonic() + secs

    def switch(self, name):
        if name != self.mode.__class__.title and not self.mode.locked:
            self.mode = self.modes[name]
            self.say(f"mode: {name}")

    def save(self):
        hi = self.hi
        calibration.save({"region": list(hi.region), "anchor": hi.anchor, "click_drop": hi.click_drop,
                          "filter_presets": hi.presets, "filter_driver": hi.driver}, self.args.hand)
        self.say(f"saved {self.args.hand}-hand settings")

    # ---- input
    def tab_rects(self):
        w = (PANEL.w - 16) // len(self.modes)
        return {n: pygame.Rect(PANEL.x + 8 + i * w, PANEL.y + 8, w - 4, TAB_H) for i, n in enumerate(self.modes)}

    def handle(self, ev) -> bool:
        if ev.type == pygame.QUIT:
            return False
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and not self.mode.locked:
            if BTN.collidepoint(ev.pos):
                self.panel_open = not self.panel_open
            elif self.panel_open:
                for n, r in self.tab_rects().items():
                    if r.collidepoint(ev.pos):
                        self.switch(n)
                        return True
                self.mode.on_panel_click(ev.pos, PANEL)
        elif ev.type == pygame.KEYDOWN:
            k = ev.key
            if k in (pygame.K_q, pygame.K_ESCAPE):
                return False
            if k == pygame.K_p and not self.mode.locked:
                self.panel_open = not self.panel_open
            elif k == pygame.K_TAB and not self.mode.locked:
                names = list(self.modes)
                self.switch(names[(names.index(self.mode.title) + 1) % len(names)])
            elif k == pygame.K_s:
                self.save()
            elif k == pygame.K_r:
                self.mode.reset()
                self.say(f"{self.mode.title}: reset to defaults (not saved)")
            else:
                self.mode.on_key(k)
        return True

    # ---- frame
    def update(self):
        self.now = time.monotonic()
        self.events = self.hi.drain_events()
        self.snap = self.hi.snapshot()
        self.hand = self.snap.hands.get(self.args.hand)
        self.cur = self.raw = None
        if self.hand:
            self.cur = (self.hand.cursor[0] * (W - 1), self.hand.cursor[1] * (H - 1))
            rx, ry = self.hi.map_cursor(*self.hand.point)
            self.raw = (rx * (W - 1), ry * (H - 1))
            self.hist.append(self.cur)
        else:
            self.hist.clear()
        self.mode.update()

    def draw(self, screen):
        font, snap, hi = self.font, self.snap, self.hi
        screen.fill((24, 26, 30))
        for gx in range(0, W, 100):
            pygame.draw.line(screen, (36, 38, 44), (gx, 0), (gx, H))
        for gy in range(0, H, 100):
            pygame.draw.line(screen, (36, 38, 44), (0, gy), (W, gy))
        self.mode.draw_under(screen)
        if self.raw:
            pygame.draw.circle(screen, (255, 70, 70), (int(self.raw[0]), int(self.raw[1])), 4)
        if self.cur:
            pygame.draw.circle(screen, (255, 255, 255), (int(self.cur[0]), int(self.cur[1])), 14, 2)
        self._preview(screen)
        lines = [f"MODE: {self.mode.title.upper()}   camera fps {snap.fps:4.1f}   hand: {self.args.hand} {'OK' if self.hand else 'NOT FOUND'}"]
        lines += self.mode.stats() + [self.mode.help]
        for i, t in enumerate(lines):
            screen.blit(font.render(t, True, (255, 220, 120) if i == 0 else (230, 230, 230)), (12, 10 + i * 20))
        if time.monotonic() < self.msg_until:
            screen.blit(self.big.render(self.msg, True, (120, 255, 160)), (W // 2 - 200, 120))
        if hi.error:
            screen.blit(self.big.render(hi.error, True, (255, 80, 80)), (12, 120))
        if not self.mode.locked:
            self._draw_params(screen)
        self.mode.draw_over(screen)  # topmost (e.g. the calibration explanation)

    def _preview(self, screen):
        import numpy as np
        snap, hi = self.snap, self.hi
        if snap.frame is None:
            return
        fh, fw = snap.frame.shape[:2]
        sc = 2
        rgb = np.ascontiguousarray(snap.frame[:, :, ::-1])
        surf = pygame.transform.scale(pygame.image.frombuffer(rgb.tobytes(), (fw, fh), "RGB"), (fw * sc, fh * sc))
        ox, oy = 12, H - fh * sc - 12
        screen.blit(surf, (ox, oy))
        x0, y0, x1, y1 = hi.region
        pygame.draw.rect(screen, (255, 220, 60), (ox + x0 * fw * sc, oy + y0 * fh * sc, (x1 - x0) * fw * sc, (y1 - y0) * fh * sc), 2)
        if self.hand:
            pygame.draw.circle(screen, (255, 70, 70), (int(ox + self.hand.point[0] * fw * sc), int(oy + self.hand.point[1] * fh * sc)), 5)

    def _draw_params(self, screen):
        mouse, font = pygame.mouse.get_pos(), self.font
        hot = BTN.collidepoint(mouse)
        pygame.draw.rect(screen, (70, 90, 150) if self.panel_open else (60, 110, 80) if hot else (48, 52, 62), BTN, border_radius=6)
        pygame.draw.rect(screen, (170, 175, 195), BTN, 1, border_radius=6)
        t = font.render("Params  [P]", True, (240, 240, 245))
        screen.blit(t, (BTN.centerx - t.get_width() // 2, BTN.centery - t.get_height() // 2))
        if not self.panel_open:
            return
        pygame.draw.rect(screen, (16, 18, 24), PANEL, border_radius=8)
        pygame.draw.rect(screen, (255, 200, 60), PANEL, 1, border_radius=8)
        for n, r in self.tab_rects().items():
            on = n == self.mode.title
            pygame.draw.rect(screen, (50, 90, 60) if on else (44, 48, 60) if r.collidepoint(mouse) else (32, 35, 44), r, border_radius=5)
            t = font.render(n, True, (240, 240, 245) if on else GREY)
            screen.blit(t, (r.centerx - t.get_width() // 2, r.centery - t.get_height() // 2))
        self.mode.draw_panel(screen, PANEL, mouse)


def main():
    ap = argparse.ArgumentParser(description="Control lab: tune one hand control at a time.")
    ap.add_argument("-m", "--mode", default="cursor", choices=list(MODES), help="which control to work on (default cursor)")
    ap.add_argument("--camera", type=int, default=2)
    ap.add_argument("--hand", default="right", type=str.lower, choices=["left", "right"], help="cursor hand (default right)")
    args = ap.parse_args()
    args.hand = args.hand.capitalize()

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Palm Control lab")
    clock = pygame.time.Clock()
    lab = Lab(args)
    running = True
    while running:
        clock.tick(60)
        for ev in pygame.event.get():
            running = lab.handle(ev) and running
        lab.update()
        lab.draw(screen)
        pygame.display.flip()
    lab.panel_open = False  # saves presets if Params was open
    lab.hi.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
