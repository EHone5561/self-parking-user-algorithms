"""Headless check of a planner against the parking simulator's rules.

Re-implements (does not import) the parts of self-parking-sim/demo_self_parking_sim.py that
decide a round: map preparation, start pose, vehicle model, collisions, success, score.

    python tools/check_parking.py [planner_module] [--variant dense_center|single_free_slot --seeds 20]
"""

import argparse
import importlib
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ALGOS = Path(__file__).resolve().parent.parent          # self-parking-user-algorithms
SIM = ALGOS.parent / "self-parking-sim"                  # sibling simulator checkout
sys.path.insert(0, str(ALGOS))

DT, L, MAX_STEER, MAX_ACCEL, MAX_BRAKE = 1 / 60, 2.6, math.radians(35), 3.0, 7.0
STEER_RATE, COAST, TIMEOUT = math.radians(180), 0.8, 180.0
LF, LR, W = 1.6, 1.4, 1.6
CAR_LENGTH, CAR_WIDTH, SLOT_LEN_MARGIN, SLOT_W_MARGIN = 3.0, 1.6, 1.2, 0.6
SUCCESS_IOU = 0.30
ORIENT_THR = math.cos(math.radians(48))
FLIP_DEADZONE = math.radians(2)
STAGES = {  # STAGE_RULES: time, distance factor, speed, steer flips, weights
    "": (65, 0.90, 2.0, 4, dict(time=8, distance=6, speed=5, steer_flip=4, parking_iou=9, parking_orientation=5, parking_stop=3)),
    "dense_center": (80, 0.95, 2.8, 5, dict(time=9, distance=6, speed=6, steer_flip=5, parking_iou=9, parking_orientation=6, parking_stop=3)),
    "single_free_slot": (95, 1.05, 3.2, 6, dict(time=11, distance=7, speed=6, steer_flip=5, parking_iou=10, parking_orientation=7, parking_stop=3)),
}


def score(variant, extent, elapsed, dist, avg_speed, flips, io, orient, expected, final_speed):
    t_target, d_factor, s_target, f_target, w = STAGES[variant]
    diag = math.hypot(extent[1] - extent[0], extent[3] - extent[2])
    c = lambda v: max(0.0, min(1.0, v))
    parts = {
        "time": c(t_target / elapsed) * w["time"],
        "distance": c(diag * d_factor / max(dist, 1e-6)) * w["distance"],
        "speed": c(avg_speed / s_target) * w["speed"],
        "steer_flip": c(1 - flips / (f_target * 2)) * w["steer_flip"],
        "parking_iou": c((io - SUCCESS_IOU) / (1 - SUCCESS_IOU) + 0.5) * w["parking_iou"],
        "parking_orientation": (1.0 if orient == expected else 0.0) * w["parking_orientation"],
        "parking_stop": c(1 - final_speed / 0.3) * w["parking_stop"],
    }
    return min(100.0, 50 + sum(parts.values()))


def load_map(variant, seed):
    m = loadmat(SIM / "parking_assets_layers_75x50.mat")
    extent = tuple(np.array(m["extent"]).squeeze().tolist())
    slots = np.array(m["slots"], float)
    occ = np.array(m["occupied_idx"]).astype(bool).ravel()
    lines = np.array(m["lines"], float)
    if variant:
        rng = random.Random(seed)
        n = len(occ)
        k = min(int(occ.sum()), n - 1)
        order = list(range(n))
        rng.shuffle(order)
        occ = np.zeros(n, bool)
        occ[order[:k]] = True
        if variant == "dense_center":
            free = np.where(~occ)[0]
            rng.shuffle(free)
            occ[free[:max(4, len(free) // 2)]] = True
        elif variant == "single_free_slot":
            occ = np.ones(n, bool)
            occ[rng.randrange(n)] = False
            # scale_map_geometry(M, 0.8): shrink about the map center
            mx, my = (extent[0] + extent[1]) / 2, (extent[2] + extent[3]) / 2
            sx = lambda v: mx + (v - mx) * 0.8
            sy = lambda v: my + (v - my) * 0.8
            extent = (sx(extent[0]), sx(extent[1]), sy(extent[2]), sy(extent[3]))
            slots = np.stack([sx(slots[:, 0]), sx(slots[:, 1]), sy(slots[:, 2]), sy(slots[:, 3])], 1)
            lines = np.stack([sx(lines[:, 0]), sy(lines[:, 1]), sx(lines[:, 2]), sy(lines[:, 3])], 1)
    # resize_slots_to_vehicle
    cx, cy = (slots[:, 0] + slots[:, 1]) / 2, (slots[:, 2] + slots[:, 3]) / 2
    hw, hl = (CAR_WIDTH + SLOT_W_MARGIN) / 2, (CAR_LENGTH + SLOT_LEN_MARGIN) / 2
    slots = np.stack([np.clip(cx - hw, extent[0], extent[1]), np.clip(cx + hw, extent[0], extent[1]),
                      np.clip(cy - hl, extent[2], extent[3]), np.clip(cy + hl, extent[2], extent[3])], 1)
    # open_top_parking_lane
    top = extent[3]
    horiz = [l[1] for l in lines if abs(l[1] - l[3]) <= 0.25]
    top_slot_y = max(horiz)
    lines = np.array([l for l in lines
                      if not (abs(l[1] - top) <= 0.25 and abs(l[3] - top) <= 0.25)
                      and not (abs(l[1] - l[3]) <= 0.25 and abs(l[1] - top_slot_y) <= 0.6)])
    rects = []
    for x1, y1, x2, y2 in lines:
        if x1 == x2:
            r = [x1 - .25, x1 + .25, min(y1, y2), max(y1, y2)]
        elif y1 == y2:
            r = [min(x1, x2), max(x1, x2), y1 - .25, y1 + .25]
        else:
            r = [min(x1, x2) - .25, max(x1, x2) + .25, min(y1, y2) - .25, max(y1, y2) + .25]
        r = [max(r[0], extent[0]), min(r[1], extent[1]), max(r[2], extent[2]), min(r[3], extent[3])]
        if r[1] > r[0] and r[3] > r[2]:
            rects.append(r)
    payload = {"extent": list(extent), "cellSize": 0.5, "slots": slots.tolist(),
               "occupied_idx": [int(v) for v in occ], "lines": lines.tolist(),
               "walls_rects": np.array(m["walls_rects"], float).tolist(),
               "expected_orientation": "rear_in" if variant == "single_free_slot" else "front_in"}
    return extent, slots, occ, rects, payload


def car_poly(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + px * c - py * s, y + px * s + py * c) for px, py in ((LF, W / 2), (LF, -W / 2), (-LR, -W / 2), (-LR, W / 2))]


def sat(a, b):
    for poly in (a, b):
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            ax, ay = -(y2 - y1), x2 - x1
            n = math.hypot(ax, ay)
            if n == 0:
                continue
            ax, ay = ax / n, ay / n
            pa = [p[0] * ax + p[1] * ay for p in a]
            pb = [p[0] * ax + p[1] * ay for p in b]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return False
    return True


def rect_poly(r):
    return [(r[0], r[2]), (r[1], r[2]), (r[1], r[3]), (r[0], r[3])]


def clip(poly, r):
    def cut(pts, inside, inter):
        out = []
        for i in range(len(pts)):
            cur, prev = pts[i], pts[i - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(inter(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(inter(prev, cur))
        return out
    xmin, xmax, ymin, ymax = r
    def ix(xv):
        return lambda p, q: (xv, p[1] + (q[1] - p[1]) * (xv - p[0]) / (q[0] - p[0]))
    def iy(yv):
        return lambda p, q: (p[0] + (q[0] - p[0]) * (yv - p[1]) / (q[1] - p[1]), yv)
    pts = poly
    for inside, inter in ((lambda p: p[0] >= xmin, ix(xmin)), (lambda p: p[0] <= xmax, ix(xmax)),
                          (lambda p: p[1] >= ymin, iy(ymin)), (lambda p: p[1] <= ymax, iy(ymax))):
        pts = cut(pts, inside, inter) if pts else []
    return pts


def area(p):
    return abs(sum(p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1] for i in range(len(p)))) / 2 if len(p) > 2 else 0.0


def iou(poly, r):
    inter = area(clip(poly, r))
    if inter <= 0:
        return 0.0
    return inter / (area(poly) + (r[1] - r[0]) * (r[3] - r[2]) - inter)


def orientation(yaw, r):
    vertical = (r[3] - r[2]) >= (r[1] - r[0])
    axis = math.sin(yaw) if vertical else math.cos(yaw)
    if abs(axis) < ORIENT_THR:
        return "unknown"
    return "front_in" if axis >= 0 else "rear_in"


def run(mod, extent, slots, occ, rects, payload, target_idx, variant=""):
    mod.handle_map_payload(payload)
    xmin, xmax, ymin, ymax = extent
    x, y, yaw, v, delta, t = xmin + 4, ymin + 6, math.radians(90), 0.0, 0.0, 0.0
    target = tuple(slots[target_idx])
    gears, prev_gear, dist = 0, "D", 0.0
    speed_sum, samples, flips, prev_sign, worst = 0.0, 0, 0, 0, 0.0
    wall0 = time.perf_counter()
    while t < TIMEOUT:
        # Pace to real time for the first seconds so background planning costs sim time like in the simulator
        if t < 10 and time.perf_counter() - wall0 < t:
            time.sleep(t - (time.perf_counter() - wall0))
        obs = {"t": t, "state": {"x": x, "y": y, "yaw": yaw, "v": v}, "target_slot": list(target),
               "limits": {"dt": DT, "L": L, "maxSteer": MAX_STEER, "maxAccel": MAX_ACCEL,
                          "maxBrake": MAX_BRAKE, "steerRate": STEER_RATE}}
        t0 = time.perf_counter()
        cmd = mod.planner_step(obs)
        worst = max(worst, time.perf_counter() - t0)
        tgt = max(-MAX_STEER, min(MAX_STEER, float(cmd.get("steer", 0))))
        acc = max(0, min(1, float(cmd.get("accel", 0))))
        brk = max(0, min(1, float(cmd.get("brake", 0))))
        gear = "R" if str(cmd.get("gear", "D")).upper().startswith("R") else "D"
        if gear != prev_gear:
            gears += 1
            prev_gear = gear
        delta += max(-STEER_RATE * DT, min(STEER_RATE * DT, tgt - delta))
        sign = 0 if abs(delta) < FLIP_DEADZONE else (1 if delta > 0 else -1)
        if sign:
            if prev_sign and sign != prev_sign:
                flips += 1
            prev_sign = sign
        sgn = 1 if gear == "D" else -1
        vs = math.copysign(1, v) if abs(v) > 1e-6 else 0
        a = sgn * MAX_ACCEL * acc - vs * MAX_BRAKE * brk
        if acc < 1e-3 and brk < 1e-3 and abs(v) > 1e-3:
            a -= vs * COAST
        v += a * DT
        nx, ny = x + v * math.cos(yaw) * DT, y + v * math.sin(yaw) * DT
        dist += math.hypot(nx - x, ny - y)
        x, y = nx, ny
        speed_sum += abs(v)
        samples += 1
        yaw = (yaw + v / L * math.tan(delta) * DT + math.pi) % (2 * math.pi) - math.pi
        t += DT
        poly = car_poly(x, y, yaw)
        inside = all(target[0] <= px <= target[1] and target[2] <= py <= target[3] for px, py in poly)
        hit = None
        if any(not (xmin <= px <= xmax and ymin <= py <= ymax) for px, py in poly):
            hit = "boundary"
        if not hit:
            for i, r in enumerate(slots):
                if occ[i] and sat(poly, rect_poly(r)):
                    hit = "occupied_slot"
                    break
        if not hit:
            for r in rects:
                if sat(poly, rect_poly(r)):
                    hit = f"line {[round(c, 1) for c in r]}"
                    break
        if hit and not inside:
            return "collision", t, (x, y), f"{hit} worst_step={worst * 1000:.0f}ms"
        io = iou(poly, target)
        if io >= SUCCESS_IOU and abs(v) <= 0.2 and orientation(yaw, target) != "unknown":
            o = orientation(yaw, target)
            sc = score(variant, extent, t, dist, speed_sum / samples, flips, io, o,
                       payload["expected_orientation"], abs(v))
            return "success", t, (x, y), (f"score={sc:.1f} iou={io:.2f} {o} gears={gears} flips={flips} "
                                          f"dist={dist:.0f}m worst_step={worst * 1000:.0f}ms")
    return "timeout", t, (x, y), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("planner", nargs="?", default="astar_planner")
    ap.add_argument("--variant", default="")
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()
    mod = importlib.import_module(args.planner)
    results = {}
    for seed in range(args.seeds):
        extent, slots, occ, rects, payload = load_map(args.variant, seed)
        for i in range(len(slots)):
            cx, cy = (slots[i][0] + slots[i][1]) / 2, (slots[i][2] + slots[i][3]) / 2
            far = [j for j in range(len(slots)) if not occ[j] and (slots[j][0] + slots[j][1]) / 2 > extent[0] + 6
                   and math.hypot((slots[j][0] + slots[j][1]) / 2 - extent[0] - 4,
                                  (slots[j][2] + slots[j][3]) / 2 - extent[2] - 6) >= 8]
            if occ[i] or (far and i not in far):
                continue
            res = run(mod, extent, slots, occ, rects, payload, i, args.variant)
            results[res[0]] = results.get(res[0], 0) + 1
            print(f"seed {seed} slot {i:2d} ({cx:4.1f},{cy:4.1f}): {res[0]:9s} t={res[1]:5.1f}s "
                  f"at ({res[2][0]:.1f},{res[2][1]:.1f}) {res[3]}")
    print(results)


if __name__ == "__main__":
    main()
