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
import textwrap
import time
from collections import deque

import numpy as np
import pygame

import calibration
import tracks
from calibration import CalibrationFlow
from motion import FILTER_INFO, FILTERS, click_signal, default_params, make_filter

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
    help = "C calibrate  T circle task (records raw track)  S save  R reset  Tab mode"
    panel_title = "Cursor"
    ROW0, ROW_H, COL_W = 76, 22, 226  # landmark list: 2 columns
    MARGIN_PX = (66, 58)  # target inset matches region_from_corners default margin (0.06, 0.08)

    def __init__(self, lab):
        super().__init__(lab)
        self.phase = "free"  # free | calib | test
        self.flow = CalibrationFlow()
        self.task, self.task_lms, self.saved = None, None, None
        self.filt = FilterPanel(lab)
        self.sub = "point"  # Params panel tab: point | filter

    @property
    def locked(self):
        return self.phase == "calib"

    def _start_task(self):
        self.task, self.task_lms = tracks.TrackTask(size=(W, H)), None
        self.lab.panel_open = False

    def _finish_task(self):
        lab, t = self.lab, self.task
        path = tracks.save(t.to_dict(lab.args.hand, lab.hi.anchor, lab.hi.region, lab.hi.driver))
        self.saved = path
        tracks.plot_and_open_async(path)
        lab.say(f"done! saved {path.name}, opening plot", 5)

    def on_key(self, k):
        lab = self.lab
        if k == pygame.K_c:
            self.phase, self.flow = "calib", CalibrationFlow()
            lab.panel_open = False
        elif k == pygame.K_SPACE and self.phase == "calib" and self.flow.phase == "intro":
            self.flow.start(lab.now)
        elif k == pygame.K_t:
            if self.phase == "test":
                self.phase, self.task = "free", None
                lab.say("circle task cancelled (nothing saved)")
            elif self.phase == "free":
                self.phase = "test"
                self._start_task()

    def reset(self):
        d, hi = calibration.DEFAULT, self.lab.hi
        hi.region, hi.anchor = tuple(d["region"]), d["anchor"]
        self.filt.reset()

    def update(self):
        lab, hand = self.lab, self.lab.hand
        self.filt.update()
        if self.phase == "calib":
            region = self.flow.update(lab.now, hand.point if hand else None)
            if region:
                lab.hi.region = region
                self.phase = "free"
                lab.say("calibrated! press S to save, T for the circle task", 4)
        if self.phase == "test" and self.task:
            if not hand:
                self.task.lost_hand()
            elif hand.lms is not self.task_lms:  # one record per new camera frame
                self.task_lms = hand.lms
                if self.task.step(lab.now, lab.cur, hand.point, lab.hi.map_cursor(*hand.point), lab.raw):
                    self.phase = "free"
                    self._finish_task()

    def draw_under(self, screen):
        self.filt.draw_plot(screen)
        self.filt.draw_markers(screen)
        if self.phase == "calib" and self.flow.phase != "intro":
            self._draw_calibration(screen)
        if self.phase == "test" and self.task and self.task.target:
            t = self.task
            r, (tx, ty) = t.radius, t.target
            pygame.draw.circle(screen, (90, 160, 255), (tx, ty), r, 3)
            num = self.lab.big.render(str(t.idx + 1), True, (90, 160, 255))
            screen.blit(num, (tx - num.get_width() // 2, ty - num.get_height() // 2))
            p = t.progress(self.lab.now)
            if p > 0:
                pygame.draw.arc(screen, GREEN, (tx - r, ty - r, r * 2, r * 2), 0, 6.283 * p, 6)

    def draw_over(self, screen):
        if self.phase == "calib" and self.flow.phase == "intro":
            self._draw_calibration(screen)

    def stats(self):
        hi, lab = self.lab.hi, self.lab
        hist = lab.hist
        jit = float(np.std(np.array(hist), axis=0).mean()) if len(hist) > 15 else float("nan")
        out = [f"tracking: {NAMES[hi.anchor]}   state: {self.phase}",
               f"{self.filt.summary()}   jitter(1s std) {jit:5.1f}px",
               "region " + " ".join(f"{v:.2f}" for v in hi.region)]
        if self.phase == "test" and self.task:
            out.append(f"circle task: {self.task.idx}/{self.task.n} done, {len(self.task.rows)} raw samples recorded (T cancels)")
        elif self.saved:
            out.append(f"last track: {self.saved.name}")
        return out

    # ---- Params panel: pick the tracked landmark
    def _row(self, i, rect):
        col, row = divmod(i, 11)
        return pygame.Rect(rect.x + 8 + col * self.COL_W, rect.y + self.ROW0 + row * self.ROW_H, self.COL_W - 4, 20)

    def _sub_rects(self, rect):
        w = (rect.w - 16) // 2
        return {n: pygame.Rect(rect.x + 8 + i * w, rect.y + 42, w - 4, 26) for i, n in enumerate(("point", "filter"))}

    def draw_panel(self, screen, rect, mouse):
        font, anchor = self.lab.font, self.lab.hi.anchor
        for n, r in self._sub_rects(rect).items():
            on = n == self.sub
            pygame.draw.rect(screen, (50, 70, 110) if on else (44, 48, 60) if r.collidepoint(mouse) else (32, 35, 44), r, border_radius=5)
            t = font.render("tracking point" if n == "point" else "filter presets", True, (240, 240, 245) if on else GREY)
            screen.blit(t, (r.centerx - t.get_width() // 2, r.centery - t.get_height() // 2))
        if self.sub == "filter":
            self.filt.draw_panel(screen, pygame.Rect(rect.x, rect.y + 76, rect.w, rect.h - 76), mouse)
            return
        for i, name in enumerate(NAMES):
            r = self._row(i, rect)
            if i == anchor:
                pygame.draw.rect(screen, (50, 90, 60), r, border_radius=4)
            elif r.collidepoint(mouse):
                pygame.draw.rect(screen, (44, 48, 60), r, border_radius=4)
            screen.blit(font.render(f"{'*' if i == anchor else ' '}{i:2d} {name}", True, (235, 235, 240) if i == anchor else (190, 192, 200)),
                        (r.x + 4, r.y + 1))
        note = "Tips move on taps (cursor freezes then)." if anchor in (4, 8, 12, 16, 20) else "A new point needs a new calibration (C)."
        screen.blit(font.render(note, True, GREY), (rect.x + 10, rect.bottom - 40))
        screen.blit(font.render("click a row to choose  -  S saves it", True, GREY), (rect.x + 10, rect.bottom - 22))

    def on_panel_click(self, pos, rect):
        for n, r in self._sub_rects(rect).items():
            if r.collidepoint(pos):
                self.sub = n
                return
        if self.sub == "filter":
            self.filt.on_panel_click(pos, pygame.Rect(rect.x, rect.y + 76, rect.w, rect.h - 76))
            return
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
    help = "- = sensitivity  X clear  S save  R reset  Tab mode"
    panel_title = "Tap sensitivity"
    GRAPH = pygame.Rect(12, 150, 600, 230)
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
        notes = ["How far a finger must dip below its peak,", "then come back, to count as a tap.", "",
                 "Lower = more sensitive, but more false taps.", "Higher = needs a deliberate tap.", "",
                 "Test: tap 10 times, then pinch and move around.", "Only real taps should be counted."]
        for i, t in enumerate(notes):
            screen.blit(font.render(t, True, GREY), (rect.x + 16, y + 52 + i * 21))

    def on_panel_click(self, pos, rect):
        y = rect.y + 56
        if pygame.Rect(rect.x + 190, y - 2, 40, 32).collidepoint(pos):
            self._bump(-self.STEP)
        elif pygame.Rect(rect.x + 240, y - 2, 40, 32).collidepoint(pos):
            self._bump(self.STEP)


# --------------------------------------------------------------------------- cursor filter presets (part of cursor mode)
RAW_COL = (255, 70, 70)
PALETTE = [(90, 160, 255), (255, 160, 70), (120, 220, 120), (190, 130, 255), (80, 220, 220), (240, 220, 90)]


def preset_label(p) -> str:
    return p["filter"] + (" " + "/".join(f"{v:g}" for v in p["params"].values()) if p["params"] else "")


class FilterPanel:
    """Filter presets (filter type + params). Presets that are on are drawn on the cursor-x plot;
    the driver preset moves the real cursor. Edited in the Params panel ('smoothing filter' tab)."""
    PLOT = pygame.Rect(408, H - 300, W - 420, 288)  # bottom-right, next to the camera preview
    SPAN_S = 4.0
    ROW_H, COL2, PARAM_H = 24, 240, 60

    def __init__(self, lab):
        self.lab = lab
        self.sel = 0  # preset being edited
        self.sig: deque = deque()  # (t, {key: cursor x px}) per camera frame; key "raw" or id(preset)
        self.inst: dict = {}  # id(preset) -> (signature, filter instance) for plotted non-driver presets
        self.last_lms = None
        self.pos: dict = {}  # id(preset) -> (x, y) px of its latest cursor, for the markers on the main screen

    # ---- data
    @property
    def presets(self):
        return self.lab.hi.presets

    def _changed(self):
        self.lab.hi.presets_changed()
        self.lab.hist.clear()

    def _selected(self):
        self.sel = min(self.sel, len(self.presets) - 1)
        return self.presets[self.sel]

    def reset(self):
        hi = self.lab.hi
        hi.presets[:] = calibration.default_presets()
        hi.driver, self.sel = 0, 0
        self._changed()

    def add(self):
        if len(self.presets) >= calibration.MAX_PRESETS:
            self.lab.say(f"max {calibration.MAX_PRESETS} presets")
            return
        src = self._selected()
        self.presets.append({"filter": src["filter"], "params": dict(src["params"]), "on": True})
        self.sel = len(self.presets) - 1

    def remove(self, i):
        hi = self.lab.hi
        if len(self.presets) <= 1:
            self.lab.say("keep at least one preset")
            return
        del self.presets[i]
        if hi.driver == i:
            hi.driver = 0
            self.presets[0]["on"] = True
        elif hi.driver > i:
            hi.driver -= 1
        self.sel = min(self.sel, len(self.presets) - 1)
        self._changed()

    def set_type(self, name):
        p = self._selected()
        if p["filter"] != name:
            p["filter"], p["params"] = name, default_params(name)
            self._changed()

    def bump(self, idx, d):
        p = self._selected()
        spec = FILTERS[p["filter"]].SPEC[idx]
        p["params"][spec.key] = spec.bump(p["params"][spec.key], d)
        self._changed()

    def summary(self):
        hi = self.lab.hi
        n_on = sum(p["on"] for p in self.presets)
        return f"cursor filter: {preset_label(self.presets[hi.driver])}   ({n_on} of {len(self.presets)} presets plotted)"

    # ---- plot
    def update(self):
        lab, hand, hi = self.lab, self.lab.hand, self.lab.hi
        if not hand:
            self.pos = {}
        if not (hand and hand.lms is not self.last_lms):
            return
        self.last_lms = hand.lms
        rx, ry = hi.map_cursor(*hand.point)
        pt, vals, keep, pos = np.array([rx, ry]), {"raw": rx * (W - 1)}, set(), {}
        for i, p in enumerate(self.presets):
            key = id(p)
            if i == hi.driver:
                if lab.cur:
                    vals[key] = lab.cur[0]  # the real cursor, including the tap freeze
                    pos[key] = lab.cur
            elif p["on"]:
                keep.add(key)
                sg = (p["filter"], tuple(p["params"].items()))
                ent = self.inst.get(key)
                if ent is None or ent[0] != sg:
                    ent = self.inst[key] = (sg, make_filter(p["filter"], p["params"]))
                out = ent[1](pt, lab.now)
                vals[key] = float(out[0]) * (W - 1)
                pos[key] = (float(out[0]) * (W - 1), float(out[1]) * (H - 1))
        for key in set(self.inst) - keep:  # preset turned off or removed: restart cleanly when turned on again
            del self.inst[key]
        self.pos = pos
        self.sig.append((lab.now, vals))
        while self.sig and lab.now - self.sig[0][0] > self.SPAN_S:
            self.sig.popleft()

    def _lines(self):
        """(key, colour, width, legend text) for everything drawn on the plot."""
        hi = self.lab.hi
        out = [("raw", RAW_COL, 1, "raw")]
        for i, p in enumerate(self.presets):
            if p["on"] or i == hi.driver:
                txt = preset_label(p) + ("  [cursor]" if i == hi.driver else "")
                out.append((id(p), PALETTE[i % len(PALETTE)], 3 if i == hi.driver else 2, txt))
        return out

    def draw_markers(self, screen):
        """A coloured ring on the main screen for every plotted preset (the driver's is the larger one)."""
        hi = self.lab.hi
        for i, p in enumerate(self.presets):
            xy = self.pos.get(id(p))
            if xy:
                drv = i == hi.driver
                pygame.draw.circle(screen, PALETTE[i % len(PALETTE)], (int(xy[0]), int(xy[1])), 22 if drv else 10, 3)

    def draw_plot(self, screen):
        lab, g, font = self.lab, self.PLOT, self.lab.font
        pygame.draw.rect(screen, (16, 18, 24), g)
        pygame.draw.rect(screen, (70, 74, 86), g, 1)
        screen.blit(font.render("cursor x, last 4 s", True, GREY), (g.x + 8, g.y + 6))
        lines = self._lines()
        x, y = g.x + 8, g.y + 28  # legend, wraps to new rows
        for _, col, w, txt in lines:
            t = font.render(txt, True, (225, 225, 230))
            if x + 30 + t.get_width() > g.right - 8:
                x, y = g.x + 8, y + 20
            pygame.draw.line(screen, col, (x, y + 8), (x + 22, y + 8), w)
            screen.blit(t, (x + 28, y))
            x += 28 + t.get_width() + 16
        top = y + 28
        vals = [v for _, d in self.sig for k, v in d.items() if any(k == ln[0] for ln in lines)]
        if len(self.sig) > 2 and vals:
            lo, hi_ = min(vals), max(vals)
            hi_ = max(hi_, lo + 40)
            for key, col, w, _ in lines:
                pts = [(g.x + (1 - (lab.now - t) / self.SPAN_S) * g.w, g.bottom - 10 - (d[key] - lo) / (hi_ - lo) * (g.bottom - 20 - top))
                       for t, d in self.sig if key in d]
                if len(pts) > 1:
                    pygame.draw.lines(screen, col, False, pts, w)

    # ---- panel content, drawn inside `rect`
    def _row(self, i, rect):
        return pygame.Rect(rect.x + 8, rect.y + i * self.ROW_H, self.COL2 - 20, self.ROW_H - 2)

    def _zones(self, i, rect):
        r = self._row(i, rect)
        return {"on": pygame.Rect(r.x + 4, r.y + 4, 16, 16), "driver": pygame.Rect(r.right - 50, r.y + 3, 20, 20),
                "del": pygame.Rect(r.right - 24, r.y + 3, 20, 20)}

    def _add_btn(self, rect):
        return pygame.Rect(rect.x + 8, rect.y + calibration.MAX_PRESETS * self.ROW_H + 6, self.COL2 - 20, 28)

    def _type_btn(self, k, rect):
        return pygame.Rect(rect.x + self.COL2 + (k % 2) * 108, rect.y + (k // 2) * 30, 104, 26)

    def _params_y(self, rect):
        return rect.y + 68

    def _minus_plus(self, idx, rect):
        y = self._params_y(rect) + idx * self.PARAM_H - 2
        return pygame.Rect(rect.right - 78, y, 30, 26), pygame.Rect(rect.right - 42, y, 30, 26)

    def draw_panel(self, screen, rect, mouse):
        font, big, hi = self.lab.font, self.lab.big, self.lab.hi
        self._selected()
        for i, p in enumerate(self.presets):
            r, z, col = self._row(i, rect), self._zones(i, rect), PALETTE[i % len(PALETTE)]
            if i == self.sel:
                pygame.draw.rect(screen, (44, 56, 80), r, border_radius=4)
            elif r.collidepoint(mouse):
                pygame.draw.rect(screen, (34, 38, 48), r, border_radius=4)
            pygame.draw.rect(screen, col if p["on"] else (90, 94, 104), z["on"], 0 if p["on"] else 2, border_radius=3)
            screen.blit(font.render(preset_label(p), True, col if p["on"] else GREY), (r.x + 28, r.y + 4))
            c = z["driver"].center
            pygame.draw.circle(screen, (235, 235, 240), c, 8, 2)
            if i == hi.driver:
                pygame.draw.circle(screen, (235, 235, 240), c, 4)
            d = z["del"]
            for a, b in (((d.x + 4, d.y + 4), (d.right - 4, d.bottom - 4)), ((d.x + 4, d.bottom - 4), (d.right - 4, d.y + 4))):
                pygame.draw.line(screen, (230, 110, 110) if d.collidepoint(mouse) else (130, 90, 90), a, b, 2)
        ab = self._add_btn(rect)
        if len(self.presets) < calibration.MAX_PRESETS:
            pygame.draw.rect(screen, (70, 90, 150) if ab.collidepoint(mouse) else (48, 52, 62), ab, border_radius=6)
            t = font.render("+ add (copy of selected)", True, (240, 240, 245))
            screen.blit(t, (ab.centerx - t.get_width() // 2, ab.centery - t.get_height() // 2))
        for j, line in enumerate(["square = show on plot", "circle = drives the cursor", "x = remove, click a row to edit"]):
            screen.blit(font.render(line, True, GREY), (rect.x + 12, ab.bottom + 8 + j * 18))
        # ---- editor for the selected preset
        p = self._selected()
        for k, n in enumerate(FILTERS):
            r, on = self._type_btn(k, rect), n == p["filter"]
            pygame.draw.rect(screen, (50, 90, 60) if on else (44, 48, 60) if r.collidepoint(mouse) else (32, 35, 44), r, border_radius=5)
            t = font.render(n, True, (240, 240, 245) if on else GREY)
            screen.blit(t, (r.centerx - t.get_width() // 2, r.centery - t.get_height() // 2))
        y0 = self._params_y(rect)
        spec = FILTERS[p["filter"]].SPEC
        if not spec:
            screen.blit(font.render("no parameters", True, GREY), (rect.x + self.COL2 + 4, y0 + 4))
        for i, ps in enumerate(spec):
            y = y0 + i * self.PARAM_H
            screen.blit(font.render(ps.label, True, (255, 220, 120)), (rect.x + self.COL2 + 4, y))
            screen.blit(font.render(f"{p['params'][ps.key]:g}", True, (255, 255, 255)), (rect.x + self.COL2 + 96, y))
            for r, sym in zip(self._minus_plus(i, rect), "-+"):
                pygame.draw.rect(screen, (70, 90, 150) if r.collidepoint(mouse) else (48, 52, 62), r, border_radius=6)
                t = big.render(sym, True, (240, 240, 245))
                screen.blit(t, (r.centerx - t.get_width() // 2, r.centery - t.get_height() // 2))
            for j, line in enumerate(textwrap.wrap(ps.hint, 30)[:2]):
                screen.blit(font.render(line, True, GREY), (rect.x + self.COL2 + 4, y + 22 + j * 16))
        for j, line in enumerate(textwrap.wrap(FILTER_INFO[p["filter"]], 56)[:2]):
            screen.blit(font.render(line, True, (140, 145, 160)), (rect.x + 12, rect.bottom - 40 + j * 18))

    def on_panel_click(self, pos, rect):
        hi = self.lab.hi
        for i in range(len(self.presets)):
            z = self._zones(i, rect)
            if z["del"].collidepoint(pos):
                return self.remove(i)
            if z["on"].collidepoint(pos):
                if i == hi.driver:
                    self.lab.say("the cursor preset is always plotted")
                else:
                    self.presets[i]["on"] = not self.presets[i]["on"]
                return
            if z["driver"].collidepoint(pos):
                hi.driver, self.presets[i]["on"] = i, True
                return self._changed()
            if self._row(i, rect).collidepoint(pos):
                self.sel = i
                return
        if self._add_btn(rect).collidepoint(pos):
            return self.add()
        for k, n in enumerate(FILTERS):
            if self._type_btn(k, rect).collidepoint(pos):
                return self.set_type(n)
        for i in range(len(FILTERS[self._selected()["filter"]].SPEC)):
            m, p = self._minus_plus(i, rect)
            if m.collidepoint(pos):
                return self.bump(i, -1)
            if p.collidepoint(pos):
                return self.bump(i, 1)


MODES = {"cursor": CursorMode, "click": ClickMode}
