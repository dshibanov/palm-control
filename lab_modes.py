"""Lab modes. Each mode tunes one control; lab.py is the shared shell (camera, Params panel, save).

A mode implements:
  title, help            name and one-line key help
  locked                 True while it needs the whole screen (e.g. calibrating): Params/mode switch are blocked
  on_key(k), update(), draw_under(screen), draw_over(screen), stats() -> list[str]
  panel_title, draw_panel(screen, rect, mouse), on_panel_click(pos, rect)   (content of the Params panel)
  reset()                restore this mode's defaults
Add a control = add a class here and register it in MODES.
"""

import random
import time
from collections import deque

import numpy as np
import pygame

import calibration
from calibration import CalibrationFlow
from motion import click_signal

W, H = 1100, 720
GOLD, GREEN, GREY = (255, 200, 60), (120, 255, 120), (160, 165, 180)

# friendly names for the 21 MediaPipe hand landmarks
NAMES = ["Wrist", "Thumb base (CMC)", "Thumb knuckle (MCP)", "Thumb joint (IP)", "Thumb tip",
         "Index knuckle (MCP)", "Index middle joint (PIP)", "Index top joint (DIP)", "Index fingertip",
         "Middle knuckle (MCP)", "Middle middle joint", "Middle top joint", "Middle fingertip",
         "Ring knuckle (MCP)", "Ring middle joint", "Ring top joint", "Ring fingertip",
         "Pinky knuckle (MCP)", "Pinky middle joint", "Pinky top joint", "Pinky fingertip"]

INTRO = [
    "Calibration",
    "",
    "This maps the part of the camera view you can comfortably reach onto the whole screen.",
    "Sit in your normal position and keep your hand in front of the camera.",
    "",
    "1.  A target appears in the TOP-LEFT corner.",
    "     Move the tracked point (see Params, default: index KNUCKLE) to where the target is on screen.",
    "     You get 2 seconds to get there and let the cursor settle, then hold still while it captures.",
    "2.  The same for the BOTTOM-RIGHT corner.",
    "",
    "If you move during a capture, or the hand leaves the view, that step simply repeats.",
    "",
    "Press SPACE to begin   (Esc to quit)",
]


class Mode:
    title = ""
    help = ""
    panel_title = ""
    locked = False

    def __init__(self, lab):
        self.lab = lab

    def on_key(self, k): ...
    def update(self): ...
    def draw_under(self, screen): ...
    def draw_over(self, screen): ...
    def stats(self) -> list[str]: return []
    def draw_panel(self, screen, rect, mouse): ...
    def on_panel_click(self, pos, rect): ...
    def reset(self): ...


# --------------------------------------------------------------------------- cursor
class CursorMode(Mode):
    title = "cursor"
    help = "C calibrate   T test   [ ] smoothing   - = lag/jitter   S save   R reset   Tab mode   Q quit"
    panel_title = "Tracking point (what the cursor follows)"
    ROW0, ROW_H = 46, 24
    MARGIN_PX = (66, 58)  # target inset matches region_from_corners default margin (0.06, 0.08)
    DWELL_S, TARGET_R = 0.4, 42

    def __init__(self, lab):
        super().__init__(lab)
        self.phase = "free"  # free | calib | test
        self.flow = CalibrationFlow()
        self.target, self.target_t0, self.dwell0 = None, 0.0, None
        self.times: list[float] = []

    @property
    def locked(self):
        return self.phase == "calib"

    def _new_target(self):
        self.target = (random.randint(80, W - 80), random.randint(80, H - 80))
        self.target_t0, self.dwell0 = time.monotonic(), None

    def on_key(self, k):
        hi, lab = self.lab.hi, self.lab
        if k == pygame.K_c:
            self.phase, self.flow = "calib", CalibrationFlow()
            lab.panel_open = False
        elif k == pygame.K_SPACE and self.phase == "calib" and self.flow.phase == "intro":
            self.flow.start(lab.now)
        elif k == pygame.K_t:
            self.phase = "free" if self.phase == "test" else "test"
            self.times.clear()
            self._new_target()
        elif k == pygame.K_LEFTBRACKET:
            hi.min_cutoff = max(0.1, hi.min_cutoff / 1.25)
        elif k == pygame.K_RIGHTBRACKET:
            hi.min_cutoff = min(10.0, hi.min_cutoff * 1.25)
        elif k == pygame.K_MINUS:
            hi.beta = max(0.0, hi.beta / 1.5)
        elif k == pygame.K_EQUALS:
            hi.beta = min(200.0, max(hi.beta, 0.5) * 1.5)

    def reset(self):
        d, hi = calibration.DEFAULT, self.lab.hi
        hi.region, hi.min_cutoff, hi.beta, hi.anchor = tuple(d["region"]), d["min_cutoff"], d["beta"], d["anchor"]

    def update(self):
        lab, hand = self.lab, self.lab.hand
        if self.phase == "calib":
            region = self.flow.update(lab.now, hand.point if hand else None)
            if region:
                lab.hi.region = region
                self.phase = "free"
                lab.say("calibrated! press S to save, T to test accuracy", 4)
        if self.phase == "test" and self.target and lab.cur:
            if np.hypot(lab.cur[0] - self.target[0], lab.cur[1] - self.target[1]) < self.TARGET_R:
                self.dwell0 = self.dwell0 or lab.now
                if lab.now - self.dwell0 >= self.DWELL_S:
                    self.times.append(lab.now - self.target_t0 - self.DWELL_S)
                    self._new_target()
            else:
                self.dwell0 = None

    def draw_under(self, screen):
        if self.phase == "calib" and self.flow.phase != "intro":
            self._draw_calibration(screen)
        if self.phase == "test" and self.target:
            r = self.TARGET_R
            pygame.draw.circle(screen, (90, 160, 255), self.target, r, 3)
            if self.dwell0:
                pygame.draw.arc(screen, GREEN, (self.target[0] - r, self.target[1] - r, r * 2, r * 2),
                                0, 6.283 * min((self.lab.now - self.dwell0) / self.DWELL_S, 1), 6)

    def draw_over(self, screen):
        if self.phase == "calib" and self.flow.phase == "intro":
            self._draw_calibration(screen)

    def stats(self):
        hi, lab = self.lab.hi, self.lab
        hist = lab.hist
        jit = float(np.std(np.array(hist), axis=0).mean()) if len(hist) > 15 else float("nan")
        out = [f"tracking: {NAMES[hi.anchor]}   state: {self.phase}",
               f"min_cutoff {hi.min_cutoff:5.2f} ([ ])   beta {hi.beta:6.1f} (- =)   jitter(1s std) {jit:5.1f}px",
               "region " + " ".join(f"{v:.2f}" for v in hi.region)]
        if self.times:
            out.append(f"test: {len(self.times)} hits, mean {np.mean(self.times):.2f}s, last {self.times[-1]:.2f}s")
        return out

    # ---- Params panel: pick the tracked landmark
    def _row(self, i, rect):
        return pygame.Rect(rect.x + 8, rect.y + self.ROW0 + i * self.ROW_H, rect.w - 16, 22)

    def draw_panel(self, screen, rect, mouse):
        font, anchor = self.lab.font, self.lab.hi.anchor
        for i, name in enumerate(NAMES):
            r = self._row(i, rect)
            if i == anchor:
                pygame.draw.rect(screen, (50, 90, 60), r, border_radius=4)
            elif r.collidepoint(mouse):
                pygame.draw.rect(screen, (44, 48, 60), r, border_radius=4)
            mark = "* " if i == anchor else "   "
            screen.blit(font.render(f"{mark}{i:2d}  {name}", True, (235, 235, 240) if i == anchor else (190, 192, 200)),
                        (r.x + 6, r.y + 2))
        note = "Tips move on taps (cursor freezes then)." if anchor in (4, 8, 12, 16, 20) else "Changing the point needs a new calibration (C)."
        screen.blit(font.render(note, True, GREY), (rect.x + 10, rect.bottom - 40))
        screen.blit(font.render("click a row to choose  -  S saves it", True, GREY), (rect.x + 10, rect.bottom - 22))

    def on_panel_click(self, pos, rect):
        for i in range(21):
            if self._row(i, rect).collidepoint(pos):
                self.lab.hi.anchor = i
                self.lab.hist.clear()
                self.lab.say(f"tracking: {NAMES[i]}  (recalibrate with C, save with S)", 3)

    # ---- calibration screens
    def _draw_calibration(self, screen):
        flow, font, big, hand_ok = self.flow, self.lab.font, self.lab.big, self.lab.hand is not None
        if flow.phase == "intro":
            dim = pygame.Surface((W, H), pygame.SRCALPHA)
            dim.fill((0, 0, 0, 170))
            screen.blit(dim, (0, 0))
            box = (W // 2 - 410, H // 2 - 190, 820, 380)
            pygame.draw.rect(screen, (10, 12, 16), box, border_radius=10)
            pygame.draw.rect(screen, GOLD, box, 2, border_radius=10)
            for i, line in enumerate(INTRO):
                f = big if i in (0, len(INTRO) - 1) else font
                col = GOLD if i in (0, len(INTRO) - 1) else (225, 225, 230)
                screen.blit(f.render(line, True, col), (W // 2 - 390, H // 2 - 170 + i * 26))
            return
        pt = self.MARGIN_PX if flow.step == 0 else (W - self.MARGIN_PX[0], H - self.MARGIN_PX[1])
        name = "TOP-LEFT" if flow.step == 0 else "BOTTOM-RIGHT"
        col = GREEN if flow.phase == "capture" else GOLD
        pygame.draw.circle(screen, col, pt, 26, 3)
        pygame.draw.line(screen, col, (pt[0] - 34, pt[1]), (pt[0] + 34, pt[1]), 1)
        pygame.draw.line(screen, col, (pt[0], pt[1] - 34), (pt[0], pt[1] + 34), 1)
        pygame.draw.arc(screen, col, (pt[0] - 34, pt[1] - 34, 68, 68), 0, 6.283 * flow.progress, 5)
        if not hand_ok:
            lines = [(f"Show your hand to the camera  ({name} target)", (255, 120, 120))]
        elif flow.phase == "settle":
            left = max(flow.SETTLE_S - flow.elapsed, 0)
            lines = [(f"Move your tracked point to the {name} target", GOLD),
                     (f"capturing in {left:.1f} s  (let the cursor settle)", (225, 225, 230))]
        else:
            ok = flow.last_std < flow.STILL_STD
            lines = [(f"Hold still...  {name}", GREEN),
                     (f"movement {flow.last_std * 100:.1f}  (must stay under {flow.STILL_STD * 100:.1f})", GREEN if ok else (255, 150, 80))]
            if flow.shaky:
                lines.append(("Too much movement, hold still and I'll try again", (255, 150, 80)))
        for i, (t, c) in enumerate(lines):
            surf = (big if i == 0 else font).render(t, True, c)
            screen.blit(surf, (W // 2 - surf.get_width() // 2, H // 2 - 24 + i * 30))
        if flow.step == 1:
            screen.blit(font.render("top-left captured", True, GREEN), (W // 2 - 60, H // 2 + 50))


# --------------------------------------------------------------------------- click
class ClickMode(Mode):
    title = "click"
    help = "- = sensitivity   X clear counters   S save   R reset   Tab mode   Q quit"
    panel_title = "Tap sensitivity"
    GRAPH = pygame.Rect(12, 150, 680, 230)
    SPAN_S = 4.0
    STEP = 0.02

    def __init__(self, lab):
        super().__init__(lab)
        self.sig: deque = deque()  # (t, index signal, middle signal), one per camera frame
        self.last_lms = None
        self.count = {"click": 0, "rclick": 0}
        self.flash = ("", -1e9)

    def on_key(self, k):
        if k == pygame.K_MINUS:
            self._bump(-self.STEP)
        elif k == pygame.K_EQUALS:
            self._bump(self.STEP)
        elif k == pygame.K_x:
            self.count = {"click": 0, "rclick": 0}

    def _bump(self, d):
        hi = self.lab.hi
        hi.click_drop = round(min(0.6, max(0.05, hi.click_drop + d)), 2)

    def reset(self):
        self.lab.hi.click_drop = calibration.DEFAULT["click_drop"]

    def update(self):
        lab, hand = self.lab, self.lab.hand
        for ev in lab.events:
            if ev.hand == lab.args.hand and ev.kind in self.count:
                self.count[ev.kind] += 1
                self.flash = (ev.kind, lab.now)
        if hand and hand.lms is not self.last_lms:
            self.last_lms = hand.lms
            self.sig.append((lab.now, click_signal(hand.lms, "index"), click_signal(hand.lms, "middle")))
        while self.sig and lab.now - self.sig[0][0] > self.SPAN_S:
            self.sig.popleft()

    def draw_under(self, screen):
        lab, g = self.lab, self.GRAPH
        font = lab.font
        pygame.draw.rect(screen, (16, 18, 24), g)
        pygame.draw.rect(screen, (70, 74, 86), g, 1)
        screen.blit(font.render("finger extension vs. the other fingers (index = blue, middle = orange)", True, GREY), (g.x + 8, g.y + 6))
        if len(self.sig) > 2:
            vals = [v for _, a, b in self.sig for v in (a, b)]
            lo, hi_ = min(vals), max(vals)
            pad = max(0.3 - (hi_ - lo), 0) / 2 + 0.05
            lo, hi_ = lo - pad, hi_ + pad

            def pt(t, v):
                return (g.x + (1 - (lab.now - t) / self.SPAN_S) * g.w, g.bottom - 6 - (v - lo) / (hi_ - lo) * (g.h - 36))

            for col, idx in (((90, 160, 255), 1), ((255, 160, 70), 2)):
                pygame.draw.lines(screen, col, False, [pt(s[0], s[idx]) for s in self.sig], 2)
            # threshold: how far below the recent peak the index signal must dip
            recent = [(t, a) for t, a, _ in self.sig if lab.now - t < 0.8]
            peak = max(a for _, a in recent)
            y = pt(0, peak - lab.hi.click_drop)[1]
            pygame.draw.line(screen, (255, 90, 90), (g.right - 0.8 / self.SPAN_S * g.w, y), (g.right, y), 2)
            screen.blit(font.render("must dip below", True, (255, 120, 120)), (g.right - 130, y - 18))
        kind, t0 = self.flash
        if lab.now - t0 < 0.5:
            label = "SELECT  (index tap)" if kind == "click" else "COMMAND  (middle tap)"
            surf = lab.big.render(label, True, (90, 160, 255) if kind == "click" else (255, 160, 70))
            screen.blit(surf, (g.right - surf.get_width() - 12, g.y + 34))

    def stats(self):
        hi, hand = self.lab.hi, self.lab.hand
        return [f"tap drop {hi.click_drop:.2f} (- =)   index taps {self.count['click']}   middle taps {self.count['rclick']}",
                f"pinch: {'YES' if hand and hand.pinch else 'no'}   (taps are ignored while pinching)"]

    def draw_panel(self, screen, rect, mouse):
        font, big, hi = self.lab.font, self.lab.big, self.lab.hi
        y = rect.y + 56
        screen.blit(big.render(f"drop  {hi.click_drop:.2f}", True, (235, 235, 240)), (rect.x + 16, y))
        for name, dx in (("-", 190), ("+", 240)):
            r = pygame.Rect(rect.x + dx, y - 2, 40, 32)
            pygame.draw.rect(screen, (70, 90, 150) if r.collidepoint(mouse) else (48, 52, 62), r, border_radius=6)
            t = big.render(name, True, (240, 240, 245))
            screen.blit(t, (r.centerx - t.get_width() // 2, r.centery - t.get_height() // 2))
        notes = ["How far a finger must dip below its recent peak,",
                 "then come back, to count as a tap.", "",
                 "Lower  = more sensitive, but more false taps.",
                 "Higher = needs a bigger, deliberate tap.", "",
                 "Try: tap index 10 times, then move and",
                 "pinch around. Taps should be counted, and",
                 "nothing else should.", "",
                 "Saved per hand together with the cursor."]
        for i, t in enumerate(notes):
            screen.blit(font.render(t, True, GREY), (rect.x + 16, y + 56 + i * 22))

    def on_panel_click(self, pos, rect):
        y = rect.y + 56
        if pygame.Rect(rect.x + 190, y - 2, 40, 32).collidepoint(pos):
            self._bump(-self.STEP)
        elif pygame.Rect(rect.x + 240, y - 2, 40, 32).collidepoint(pos):
            self._bump(self.STEP)


MODES = {"cursor": CursorMode, "click": ClickMode}
