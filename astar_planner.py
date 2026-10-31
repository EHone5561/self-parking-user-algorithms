"""A* 경로 계획 + Pure Pursuit 추종 플래너.

`student_planner.py`와 같은 인터페이스(handle_map_payload, planner_step)를 제공합니다.
`my_agent_astar.py`로 실행하면 이 플래너가 쓰입니다.

1. 맵을 0.25m 격자로 만들고, 주차선·점유 슬롯·외벽을 차량 반폭만큼 부풀려 장애물로 표시
2. 목표 슬롯의 열린 쪽 앞에 '진입 준비점'을 두고, 시작점→준비점을 A*로 탐색
3. 준비점→슬롯 중앙은 직선으로 이어 붙여, 슬롯에 똑바로 들어가게 함
4. 경로를 Pure Pursuit로 따라가고, 남은 거리에 맞춰 감속해 슬롯 중앙에 정지
"""

from typing import Any, Dict, List, Optional, Tuple
import heapq
import math
import threading

import numpy as np

CELL = 0.25           # A* 격자 크기 (m)
INFLATE = 1.15        # 장애물 팽창 반경: 차량 반폭 0.8 + 여유
NEAR_COST = 4.0       # 장애물 근처를 지날 때의 추가 비용 (통로 가운데로 다니도록)
NEAR_DIST = 5.0       # 이 거리 안이면 추가 비용
APPROACH = 9.0        # 슬롯 입구에서 준비점까지 거리 (m)
LOOKAHEAD = 2.6       # Pure Pursuit 전방 주시 거리 (m)
CRUISE = 3.2          # 주행 속도 (m/s)
APPROACH_SPEED = 1.6  # 슬롯 진입 속도 (m/s)
REAR_TO_CENTER = 0.1  # 후축 기준 좌표 → 차체 중심 (Lf 1.6, Lr 1.4)


class AStarPlanner:
    def __init__(self) -> None:
        self.map: Optional[Dict[str, Any]] = None
        self.path: List[Tuple[float, float]] = []
        self.blocked: List[List[bool]] = []
        self.cost: List[List[float]] = []
        self.idx = 0
        self.done = False
        self.planning: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ 맵
    def set_map(self, payload: Dict[str, Any]) -> None:
        self.map = payload
        self.path = []
        self.planning = None
        self.idx = 0
        self.done = False
        xmin, xmax, ymin, ymax = payload["extent"]
        self.origin = (xmin, ymin)
        self.cols = int(round((xmax - xmin) / CELL))
        self.rows = int(round((ymax - ymin) / CELL))

        obstacles = [tuple(r) for r in self._line_rects(payload)]
        obstacles += [tuple(s) for s, occ in zip(payload["slots"], payload["occupied_idx"]) if occ]
        # 외벽: 맵 경계 바깥으로 나가면 boundary 충돌
        obstacles += [(xmin - 1, xmin, ymin - 1, ymax + 1), (xmax, xmax + 1, ymin - 1, ymax + 1),
                      (xmin - 1, xmax + 1, ymin - 1, ymin), (xmin - 1, xmax + 1, ymax, ymax + 1)]
        self.obstacles = obstacles

        # 각 칸의 가장 가까운 장애물까지 거리로 막힘/비용 계산 (시뮬레이터는 0.15초 안에 응답을
        # 기다리므로 numpy로 한 번에 계산)
        xs = xmin + (np.arange(self.cols) + 0.5) * CELL
        ys = ymin + (np.arange(self.rows) + 0.5) * CELL
        gx, gy = np.meshgrid(xs, ys)
        d = np.full(gx.shape, np.inf)
        for x0, x1, y0, y1 in obstacles:
            dx = np.maximum(np.maximum(x0 - gx, 0.0), gx - x1)
            dy = np.maximum(np.maximum(y0 - gy, 0.0), gy - y1)
            d = np.minimum(d, np.hypot(dx, dy))
        self.blocked = (d < INFLATE).tolist()
        self.cost = np.where(d < NEAR_DIST, NEAR_COST * (NEAR_DIST - d) / NEAR_DIST, 0.0).tolist()

    @staticmethod
    def _line_rects(payload: Dict[str, Any], half: float = 0.25) -> List[List[float]]:
        """시뮬레이터와 같은 방식으로 주차선을 얇은 사각형으로 만듭니다."""
        rects = []
        for x1, y1, x2, y2 in payload.get("lines", []):
            hx = 0.0 if (y1 == y2 and x1 != x2) else half   # 가로선은 x 방향으로 늘리지 않음
            hy = 0.0 if (x1 == x2 and y1 != y2) else half   # 세로선은 y 방향으로 늘리지 않음
            rects.append([min(x1, x2) - hx, max(x1, x2) + hx, min(y1, y2) - hy, max(y1, y2) + hy])
        return rects

    def _cell_center(self, r: int, c: int) -> Tuple[float, float]:
        return self.origin[0] + (c + 0.5) * CELL, self.origin[1] + (r + 0.5) * CELL

    def _to_cell(self, x: float, y: float) -> Tuple[int, int]:
        c = min(self.cols - 1, max(0, int((x - self.origin[0]) / CELL)))
        r = min(self.rows - 1, max(0, int((y - self.origin[1]) / CELL)))
        return r, c

    # ------------------------------------------------------------------ 경로
    def _entry_point(self, slot: List[float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """슬롯 중앙과, 주차선이 없는 쪽(입구) 앞의 준비점을 돌려줍니다."""
        cx, cy = (slot[0] + slot[1]) / 2, (slot[2] + slot[3]) / 2
        vertical = (slot[3] - slot[2]) >= (slot[1] - slot[0])
        half = (slot[3] - slot[2]) / 2 if vertical else (slot[1] - slot[0]) / 2
        best = None
        for sign in (1, -1):
            dx, dy = (0, sign) if vertical else (sign, 0)
            # 입구 쪽: 슬롯 중앙에서 그 방향으로 나가는 선분이 주차선과 겹치지 않음
            seg_clear = all(not _segment_hits(cx, cy, cx + dx * (half + 1.0), cy + dy * (half + 1.0), rect)
                            for rect in self._line_rects(self.map))
            px, py = cx + dx * (half + APPROACH), cy + dy * (half + APPROACH)
            r, c = self._to_cell(px, py)
            if seg_clear and not self.blocked[r][c]:
                return (cx, cy), (px, py)
            if best is None and not self.blocked[r][c]:
                best = (px, py)
        return (cx, cy), best or (cx, cy + half + APPROACH)

    def _astar(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        s, g = self._to_cell(*start), self._to_cell(*goal)
        moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
        h = lambda n: math.hypot(n[0] - g[0], n[1] - g[1])
        open_heap = [(h(s), 0.0, s)]
        came = {s: None}
        best = {s: 0.0}
        while open_heap:
            _, cost, node = heapq.heappop(open_heap)
            if node == g:
                break
            if cost > best[node]:
                continue
            for dr, dc, step in moves:
                nr, nc = node[0] + dr, node[1] + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                # 시작 칸 주변은 장애물 팽창 안에 있을 수 있으니 허용
                if self.blocked[nr][nc] and math.hypot(nr - s[0], nc - s[1]) > 4:
                    continue
                new_cost = cost + step * (1.0 + self.cost[nr][nc])
                if new_cost < best.get((nr, nc), float("inf")):
                    best[(nr, nc)] = new_cost
                    came[(nr, nc)] = node
                    heapq.heappush(open_heap, (new_cost + h((nr, nc)), new_cost, (nr, nc)))
        if g not in came:
            return []
        cells = []
        node = g
        while node is not None:
            cells.append(node)
            node = came[node]
        cells.reverse()
        return [self._cell_center(r, c) for r, c in cells]

    def _plan(self, obs: Dict[str, Any]) -> None:
        st = obs["state"]
        center, entry = self._entry_point(obs["target_slot"])
        start = (st["x"] + REAR_TO_CENTER * math.cos(st["yaw"]),
                 st["y"] + REAR_TO_CENTER * math.sin(st["yaw"]))
        path = self._astar(start, entry)
        # 준비점 → 슬롯 중앙은 직선 (0.25m 간격)
        n = max(1, int(math.dist(entry, center) / CELL))
        path += [(entry[0] + (center[0] - entry[0]) * i / n, entry[1] + (center[1] - entry[1]) * i / n)
                 for i in range(1, n + 1)]
        self.path = _smooth(path)
        self.entry = entry
        self.idx = 0
        print(f"[astar] {len(path)} waypoints, start {start[0]:.1f},{start[1]:.1f} -> "
              f"entry {entry[0]:.1f},{entry[1]:.1f} -> slot {center[0]:.1f},{center[1]:.1f}")

    # ------------------------------------------------------------------ 제어
    def control(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        st = obs["state"]
        x, y, yaw, v = st["x"], st["y"], st["yaw"], st["v"]
        if not self.path and self.planning is None:
            # A*는 0.15초 응답 제한보다 오래 걸릴 수 있어 백그라운드에서 돌리고, 끝날 때까지 정지
            self.planning = threading.Thread(target=self._plan, args=(obs,), daemon=True)
            self.planning.start()
        if self.planning is not None and self.planning.is_alive():
            return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": "D"}
        cmd = {"steer": 0.0, "accel": 0.0, "brake": 0.0, "gear": "D"}
        if self.done or not self.path:
            cmd["brake"] = 1.0
            return cmd

        L = obs["limits"]["L"]
        max_steer = obs["limits"]["maxSteer"]

        # 가장 가까운 경로점 (뒤로는 가지 않음)
        best_i, best_d = self.idx, float("inf")
        for i in range(self.idx, min(len(self.path), self.idx + 40)):
            d = math.hypot(self.path[i][0] - x, self.path[i][1] - y)
            if d < best_d:
                best_i, best_d = i, d
        self.idx = best_i

        goal = self.path[-1]
        remaining = sum(math.dist(self.path[i], self.path[i + 1]) for i in range(self.idx, len(self.path) - 1))
        remaining += best_d if self.idx == len(self.path) - 1 else 0.0

        # Pure Pursuit: 전방 주시점을 향하는 조향각
        look = max(1.2, min(LOOKAHEAD, remaining)) if remaining < LOOKAHEAD else LOOKAHEAD
        target = goal
        for i in range(self.idx, len(self.path)):
            if math.hypot(self.path[i][0] - x, self.path[i][1] - y) >= look:
                target = self.path[i]
                break
        alpha = _wrap(math.atan2(target[1] - y, target[0] - x) - yaw)
        dist = max(0.5, math.hypot(target[0] - x, target[1] - y))
        steer = math.atan2(2.0 * L * math.sin(alpha), dist)
        cmd["steer"] = max(-max_steer, min(max_steer, steer))

        # 속도: 회전이 클수록, 남은 거리가 짧을수록 느리게
        to_entry = math.hypot(self.entry[0] - x, self.entry[1] - y)
        desired = CRUISE * max(0.45, 1.0 - abs(cmd["steer"]) / max_steer)
        if to_entry < 6.0 or self.idx >= len(self.path) - int(APPROACH / CELL) - 2:
            desired = min(desired, APPROACH_SPEED)
        desired = min(desired, math.sqrt(2 * 2.5 * max(0.0, remaining - 0.15)))  # 2.5 m/s² 로 정지

        if remaining < 0.15 or (remaining < 0.6 and v < 0.3):
            self.done = True
            cmd["brake"] = 1.0
        elif v < desired - 0.1:
            cmd["accel"] = min(1.0, (desired - v) * 0.8 + 0.15)
        elif v > desired + 0.1:
            cmd["brake"] = min(1.0, (v - desired) * 0.5)
        return cmd


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _rect_dist(x: float, y: float, rect) -> float:
    dx = max(rect[0] - x, 0.0, x - rect[1])
    dy = max(rect[2] - y, 0.0, y - rect[3])
    return math.hypot(dx, dy)


def _segment_hits(x1, y1, x2, y2, rect, steps: int = 20) -> bool:
    return any(_rect_dist(x1 + (x2 - x1) * i / steps, y1 + (y2 - y1) * i / steps, rect) == 0.0
               for i in range(steps + 1))


def _smooth(path: List[Tuple[float, float]], passes: int = 30) -> List[Tuple[float, float]]:
    """격자 계단을 펴기 위한 이동 평균. 끝점과 마지막 직선 구간은 유지합니다."""
    if len(path) < 5:
        return path
    keep_tail = int(APPROACH / CELL) + 2
    pts = [list(p) for p in path]
    for _ in range(passes):
        for i in range(1, len(pts) - keep_tail):
            pts[i][0] = (pts[i - 1][0] + pts[i][0] + pts[i + 1][0]) / 3
            pts[i][1] = (pts[i - 1][1] + pts[i][1] + pts[i + 1][1]) / 3
    return [tuple(p) for p in pts]


planner = AStarPlanner()


def handle_map_payload(map_payload: Dict[str, Any]) -> None:
    planner.set_map(map_payload)


def planner_step(obs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return planner.control(obs)
    except Exception as exc:
        print(f"[astar] error: {exc}")
        return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": "D"}
