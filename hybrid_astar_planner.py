"""Hybrid A* 경로 계획 + Pure Pursuit 추종 플래너 (전진·후진).

`student_planner.py`와 같은 인터페이스(handle_map_payload, planner_step)를 제공합니다.
`my_agent_hybrid.py`로 실행하면 이 플래너가 쓰입니다.

계획
1. 장애물(주차선·점유 슬롯·맵 경계)까지의 거리장을 0.1m 격자로 계산하고,
   차체를 원 4개로 근사해 충돌을 검사합니다.
2. 상태는 (x, y, 방향) 연속값이고, 0.5m × 0.5m × 10° 칸으로 중복을 거릅니다.
3. 한 번 확장할 때 조향 5가지 × 전진/후진으로 1.6m씩 실제 차량 모델대로 굴려 봅니다.
   후진·기어 변경·큰 조향에는 비용을 더 붙입니다.
4. 휴리스틱은 장애물을 고려한 2D 최단거리(목표에서 다익스트라)입니다.
5. 확장한 노드마다 '원호 하나 + 직선'으로 목표 자세(슬롯 중앙, 요구 방향)에 바로 닿는지 시험하고,
   충돌이 없으면 거기서 끝냅니다 (Reeds-Shepp 대신 쓰는 간단한 해석적 확장).

추종
- 경로를 기어 방향이 바뀌는 지점에서 구간으로 나누고, 구간마다 Pure Pursuit로 따라갑니다.
- 구간 끝에서 정지 → 기어 변경 → 다음 구간. 마지막 구간 끝에서 정지하면 주차 완료입니다.
"""

from typing import Any, Dict, List, Optional, Tuple
import heapq
import math
import threading

import numpy as np

# 차량 (시뮬레이터 Params와 같은 값; L과 maxSteer는 관측의 limits로 덮어씀)
WHEELBASE = 2.6
MAX_STEER = math.radians(35)
LF, LR, HALF_W = 1.6, 1.4, 0.8          # 후축 기준 앞·뒤 길이, 반폭
REAR_TO_CENTER = (LF - LR) / 2          # 후축 → 차체 중심

# 충돌
FIELD_CELL = 0.1                        # 거리장 격자 (m)
MARGIN = 0.3                            # 충돌 여유 (m)
CIRCLES = [-LR + (LF + LR) * (k + 0.5) / 4 for k in range(4)]   # 차체를 덮는 원 중심 (후축 기준)
CIRCLE_R = math.hypot(HALF_W, (LF + LR) / 8)
# 차체 둘레 점 (0.2m 간격; 주차선 폭 0.5m보다 촘촘해서 선을 건너뛰지 않음)
OUTLINE = ([(-LR + (LF + LR) * k / 15, s * HALF_W) for k in range(16) for s in (-1, 1)]
           + [(e, -HALF_W + 2 * HALF_W * k / 8) for k in range(1, 8) for e in (-LR, LF)])

# 탐색
XY_RES, YAW_BINS = 0.5, 36
STEP = 1.6                              # 한 번 확장할 때 이동 거리 (m)
STEERS = [-1.0, -0.5, 0.0, 0.5, 1.0]    # 최대 조향 대비 비율
REVERSE_COST = 1.5                      # 후진 거리 가중치
GEAR_COST = 6.0                         # 기어 변경 비용
STEER_COST = 0.3                        # 조향 크기 비용
STEER_CHANGE_COST = 0.6                 # 조향 변화 비용
H_WEIGHT = 1.6                          # 휴리스틱 가중치 (클수록 빠르고 덜 최적)
MAX_EXPAND = 60000
R_MIN = WHEELBASE / math.tan(MAX_STEER) * 1.05   # 해석적 확장에서 쓸 최소 회전 반경

# 추종
PATH_RES = 0.1
CRUISE_FWD, CRUISE_REV = 3.0, 1.5       # m/s
DECEL = 2.0                             # 정지 목표 감속도 (m/s²)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class HybridAStarPlanner:
    # 탐색 비용 (하위 클래스에서 바꿀 수 있게 속성으로 둠)
    reverse_cost = REVERSE_COST
    gear_cost = GEAR_COST
    steer_cost = STEER_COST
    steer_change_cost = STEER_CHANGE_COST
    h_weight = H_WEIGHT

    def __init__(self) -> None:
        self.map: Optional[Dict[str, Any]] = None
        self.worker: Optional[threading.Thread] = None
        self.segments: List[List[Tuple[float, float, float]]] = []
        self.dirs: List[int] = []
        self.seg = 0
        self.idx = 0
        self.done = False
        self.failed = False

    # ================================================================== 맵
    def set_map(self, payload: Dict[str, Any]) -> None:
        self.map = payload
        self.worker = None
        self.segments, self.dirs = [], []
        self.seg = self.idx = 0
        self.done = self.failed = False

    def _build_field(self) -> None:
        p = self.map
        xmin, xmax, ymin, ymax = p["extent"]
        self.extent = (xmin, xmax, ymin, ymax)
        rects = self._line_rects(p)
        rects += [list(s) for s, occ in zip(p["slots"], p["occupied_idx"]) if occ]
        rects += [[xmin - 5, xmin, ymin - 5, ymax + 5], [xmax, xmax + 5, ymin - 5, ymax + 5],
                  [xmin - 5, xmax + 5, ymin - 5, ymin], [xmin - 5, xmax + 5, ymax, ymax + 5]]
        cols = int(math.ceil((xmax - xmin) / FIELD_CELL))
        rows = int(math.ceil((ymax - ymin) / FIELD_CELL))
        gx, gy = np.meshgrid(xmin + (np.arange(cols) + 0.5) * FIELD_CELL,
                             ymin + (np.arange(rows) + 0.5) * FIELD_CELL)
        d = np.full(gx.shape, np.inf)
        for x0, x1, y0, y1 in rects:
            dx = np.maximum(np.maximum(x0 - gx, 0.0), gx - x1)
            dy = np.maximum(np.maximum(y0 - gy, 0.0), gy - y1)
            d = np.minimum(d, np.hypot(dx, dy))
        self.field = d
        self.field_shape = (rows, cols)

    @staticmethod
    def _line_rects(payload: Dict[str, Any], half: float = 0.25) -> List[List[float]]:
        """시뮬레이터와 같은 방식으로 주차선을 얇은 사각형으로 만듭니다."""
        rects = []
        for x1, y1, x2, y2 in payload.get("lines", []):
            hx = 0.0 if (y1 == y2 and x1 != x2) else half
            hy = 0.0 if (x1 == x2 and y1 != y2) else half
            rects.append([min(x1, x2) - hx, max(x1, x2) + hx, min(y1, y2) - hy, max(y1, y2) + hy])
        return rects

    def _clearance(self, x: float, y: float) -> float:
        c = int((x - self.extent[0]) / FIELD_CELL)
        r = int((y - self.extent[2]) / FIELD_CELL)
        if not (0 <= r < self.field_shape[0] and 0 <= c < self.field_shape[1]):
            return -1.0
        return float(self.field[r, c])

    def _free(self, x: float, y: float, yaw: float) -> bool:
        cy, sy = math.cos(yaw), math.sin(yaw)
        # 빠른 통과: 차체를 덮는 원 4개가 모두 비어 있으면 충돌 없음
        if all(self._clearance(x + o * cy, y + o * sy) >= CIRCLE_R + MARGIN for o in CIRCLES):
            return True
        # 원은 앞뒤로 차체보다 크므로, 애매하면 차체 둘레 점들로 정확히 검사
        return all(self._clearance(x + px * cy - py * sy, y + px * sy + py * cy) >= MARGIN
                   for px, py in OUTLINE)

    # ================================================================== 휴리스틱
    def _holonomic(self, goal_xy: Tuple[float, float]) -> None:
        """목표에서 시작하는 2D 다익스트라 거리 (차체 반폭만큼 장애물 팽창)."""
        xmin, xmax, ymin, ymax = self.extent
        cols = int(math.ceil((xmax - xmin) / XY_RES))
        rows = int(math.ceil((ymax - ymin) / XY_RES))
        step = int(round(XY_RES / FIELD_CELL))
        coarse = self.field[step // 2::step, step // 2::step][:rows, :cols]
        free = coarse >= HALF_W
        dist = np.full((rows, cols), np.inf)
        gc = min(cols - 1, int((goal_xy[0] - xmin) / XY_RES))
        gr = min(rows - 1, int((goal_xy[1] - ymin) / XY_RES))
        dist[gr, gc] = 0.0
        heap = [(0.0, gr, gc)]
        moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
        while heap:
            d, r, c = heapq.heappop(heap)
            if d > dist[r, c]:
                continue
            for dr, dc, w in moves:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and free[nr, nc]:
                    nd = d + w * XY_RES
                    if nd < dist[nr, nc]:
                        dist[nr, nc] = nd
                        heapq.heappush(heap, (nd, nr, nc))
        self.h2d = dist
        self.h_shape = (rows, cols)

    def _h(self, x: float, y: float, goal: Tuple[float, float, float]) -> float:
        c = int((x - self.extent[0]) / XY_RES)
        r = int((y - self.extent[2]) / XY_RES)
        euclid = math.hypot(goal[0] - x, goal[1] - y)
        if 0 <= r < self.h_shape[0] and 0 <= c < self.h_shape[1] and math.isfinite(self.h2d[r, c]):
            return max(euclid, float(self.h2d[r, c]))
        return euclid + 10.0

    # ================================================================== 목표
    def _goal(self, slot: List[float]) -> Tuple[float, float, float]:
        """슬롯 중앙에 차체 중심이 오고, 요구 방향(front_in/rear_in)을 보는 후축 자세."""
        cx, cy = (slot[0] + slot[1]) / 2, (slot[2] + slot[3]) / 2
        vertical = (slot[3] - slot[2]) >= (slot[1] - slot[0])
        front_in = self.map.get("expected_orientation", "front_in") != "rear_in"
        if vertical:
            yaw = math.pi / 2 if front_in else -math.pi / 2
        else:
            yaw = 0.0 if front_in else math.pi
        return (cx - REAR_TO_CENTER * math.cos(yaw), cy - REAR_TO_CENTER * math.sin(yaw), yaw)

    # ================================================================== 탐색
    def _key(self, x: float, y: float, yaw: float) -> Tuple[int, int, int]:
        return (int(math.floor(x / XY_RES)), int(math.floor(y / XY_RES)),
                int(round(_wrap(yaw) / (2 * math.pi / YAW_BINS))) % YAW_BINS)

    def _roll(self, x, y, yaw, steer, direction, length) -> Optional[List[Tuple[float, float, float]]]:
        """차량 모델대로 length만큼 굴린 경로 (충돌하면 None)."""
        pts = []
        n = max(1, int(math.ceil(length / PATH_RES)))
        ds = direction * length / n
        for _ in range(n):
            x += ds * math.cos(yaw)
            y += ds * math.sin(yaw)
            yaw = _wrap(yaw + ds / self.L * math.tan(steer))
            if not self._free(x, y, yaw):
                return None
            pts.append((x, y, yaw))
        return pts

    def _analytic(self, x, y, yaw, goal) -> Optional[Tuple[List[Tuple[float, float, float]], List[int]]]:
        """원호 하나 + 직선으로 목표 자세에 닿는 경로. 원호와 직선은 기어가 달라도 됨."""
        gx, gy, gyaw = goal
        nx, ny = -math.sin(gyaw), math.cos(gyaw)          # 목표 축의 법선
        lateral = nx * (x - gx) + ny * (y - gy)
        delta = _wrap(gyaw - yaw)
        arc: List[Tuple[float, float, float]] = []
        arc_dir = 0
        if abs(delta) < math.radians(2):
            if abs(lateral) > 0.1:
                return None
            ex, ey = x, y
        else:
            wx = math.sin(gyaw) - math.sin(yaw)
            wy = math.cos(yaw) - math.cos(gyaw)
            denom = nx * wx + ny * wy
            if abs(denom) < 1e-6:
                return None
            radius = -lateral / denom                     # 부호 있는 1/곡률
            if abs(radius) < R_MIN:
                return None
            # 같은 방향 변화를 전진/후진 중 어느 쪽으로 도는지: delta = direction * s / radius
            arc_dir = 1 if delta / radius > 0 else -1
            length = abs(delta * radius)
            steer = math.atan(self.L / radius)            # 곡률 1/radius는 기어 방향과 무관
            arc = self._roll(x, y, yaw, steer, arc_dir, length)
            if arc is None:
                return None
            ex, ey = arc[-1][0], arc[-1][1]
        along = math.cos(gyaw) * (gx - ex) + math.sin(gyaw) * (gy - ey)
        line_dir = 1 if along >= 0 else -1
        line = self._roll(ex, ey, gyaw, 0.0, line_dir, abs(along)) if abs(along) > 1e-3 else []
        if line is None:
            return None
        pts = arc + line
        dirs = [arc_dir] * len(arc) + [line_dir] * len(line)
        return pts, dirs

    def _search(self, start: Tuple[float, float, float], goal: Tuple[float, float, float]):
        max_steer = self.max_steer
        # 노드: (x, y, yaw, 방향, 조향, 부모, 부모→여기 경로점들)
        nodes = [(start[0], start[1], start[2], 1, 0.0, -1, [])]
        g_cost = {self._key(*start): 0.0}
        heap = [(self.h_weight * self._h(start[0], start[1], goal), 0.0, 0)]
        closed = set()
        expanded = 0
        while heap and expanded < MAX_EXPAND:
            _, g, i = heapq.heappop(heap)
            x, y, yaw, d, st, _, _ = nodes[i]
            key = self._key(x, y, yaw)
            if key in closed:
                continue
            closed.add(key)
            expanded += 1

            if self._h(x, y, goal) < 25.0 or expanded % 20 == 0:
                shot = self._analytic(x, y, yaw, goal)
                if shot is not None:
                    return self._collect(nodes, i, shot), expanded

            for direction in (1, -1):
                for ratio in STEERS:
                    steer = ratio * max_steer
                    pts = self._roll(x, y, yaw, steer, direction, STEP)
                    if pts is None:
                        continue
                    nx, ny, nyaw = pts[-1]
                    nkey = self._key(nx, ny, nyaw)
                    if nkey in closed:
                        continue
                    cost = STEP * (self.reverse_cost if direction < 0 else 1.0)
                    cost += self.steer_cost * abs(ratio) + self.steer_change_cost * abs(steer - st) / max_steer
                    if direction != d and i != 0:
                        cost += self.gear_cost
                    ng = g + cost
                    if ng < g_cost.get(nkey, float("inf")):
                        g_cost[nkey] = ng
                        nodes.append((nx, ny, nyaw, direction, steer, i, pts))
                        heapq.heappush(heap, (ng + self.h_weight * self._h(nx, ny, goal), ng, len(nodes) - 1))
        return None, expanded

    @staticmethod
    def _collect(nodes, i, shot):
        pts, dirs = [], []
        while i >= 0:
            x, y, yaw, d, _, parent, seg = nodes[i]
            pts = seg + pts
            dirs = [d] * len(seg) + dirs
            i = parent
        return pts + shot[0], dirs + shot[1]

    def _plan(self, obs: Dict[str, Any]) -> None:
        try:
            self.L = float(obs["limits"].get("L", WHEELBASE))
            self.max_steer = float(obs["limits"].get("maxSteer", MAX_STEER))
            self._build_field()
            goal = self._goal(obs["target_slot"])
            gcx = goal[0] + REAR_TO_CENTER * math.cos(goal[2])
            gcy = goal[1] + REAR_TO_CENTER * math.sin(goal[2])
            self._holonomic((gcx, gcy))
            st = obs["state"]
            result, expanded = self._search((st["x"], st["y"], st["yaw"]), goal)
            if result is None:
                print(f"[hybrid] no path found ({expanded} nodes)")
                self.failed = True
                return
            pts, dirs = result
            # 기어 방향이 바뀌는 곳에서 구간 나누기
            segments, seg_dirs = [[pts[0]]], [dirs[0]]
            for p, d in zip(pts[1:], dirs[1:]):
                if d != seg_dirs[-1]:
                    segments.append([segments[-1][-1]])
                    seg_dirs.append(d)
                segments[-1].append(p)
            self.segments, self.dirs = segments, seg_dirs
            length = len(pts) * PATH_RES
            print(f"[hybrid] path {length:.1f}m, {len(segments)} segments "
                  f"({''.join('D' if d > 0 else 'R' for d in seg_dirs)}), {expanded} nodes")
        except Exception as exc:  # 스레드 안의 예외도 보이게
            print(f"[hybrid] planning error: {exc}")
            self.failed = True

    # ================================================================== 추종
    def control(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        stop = {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": "D"}
        if self.worker is None:
            # 탐색은 0.15초 응답 제한보다 오래 걸릴 수 있어 백그라운드에서 돌리고, 끝날 때까지 정지
            self.worker = threading.Thread(target=self._plan, args=(obs,), daemon=True)
            self.worker.start()
        if self.worker.is_alive() or self.failed or not self.segments:
            return stop

        st = obs["state"]
        x, y, yaw, v = st["x"], st["y"], st["yaw"], st["v"]
        direction = self.dirs[self.seg]
        gear = "D" if direction > 0 else "R"
        if self.done:
            return {**stop, "gear": gear}
        path = self.segments[self.seg]

        # 구간 안에서 가장 가까운 점 (앞쪽으로만)
        best_i, best_d = self.idx, float("inf")
        for i in range(self.idx, min(len(path), self.idx + 30)):
            dd = math.hypot(path[i][0] - x, path[i][1] - y)
            if dd < best_d:
                best_i, best_d = i, dd
        self.idx = best_i
        ex, ey, eyaw = path[-1]
        # 남은 거리: 구간 끝까지를 진행 방향 기준으로 (지나치면 음수)
        remaining = (len(path) - 1 - self.idx) * PATH_RES
        if self.idx >= len(path) - 3:
            remaining = direction * (math.cos(eyaw) * (ex - x) + math.sin(eyaw) * (ey - y))

        speed = v * direction                              # 진행 방향 기준 속도
        if remaining < 0.08:
            if abs(v) < 0.05:
                if self.seg + 1 < len(self.segments):
                    self.seg += 1
                    self.idx = 0
                else:
                    self.done = True
            return {**stop, "gear": gear}

        # Pure Pursuit (후진은 차 방향을 뒤집어 계산하고 조향 부호를 바꿈)
        look = max(1.2, min(2.5, 1.0 + 0.6 * abs(v)))
        tx, ty = None, None
        for i in range(self.idx, len(path)):
            if math.hypot(path[i][0] - x, path[i][1] - y) >= look:
                tx, ty = path[i][0], path[i][1]
                break
        if tx is None:  # 구간 끝 너머로 연장한 점을 겨냥해 마지막 자세를 맞춤
            extra = max(0.0, look - math.hypot(ex - x, ey - y))
            tx = ex + direction * extra * math.cos(eyaw)
            ty = ey + direction * extra * math.sin(eyaw)
        heading = yaw if direction > 0 else _wrap(yaw + math.pi)
        alpha = _wrap(math.atan2(ty - y, tx - x) - heading)
        ld = max(0.5, math.hypot(tx - x, ty - y))
        steer = math.atan2(2.0 * self.L * math.sin(alpha), ld) * direction
        steer = max(-self.max_steer, min(self.max_steer, steer))

        cruise = CRUISE_FWD if direction > 0 else CRUISE_REV
        desired = cruise * max(0.4, 1.0 - 0.6 * abs(steer) / self.max_steer)
        desired = min(desired, math.sqrt(2 * DECEL * max(0.0, remaining - 0.05)))

        cmd = {"steer": steer, "accel": 0.0, "brake": 0.0, "gear": gear}
        if speed < -0.05:                                  # 반대 방향으로 굴러가는 중: 먼저 세움
            cmd["brake"] = 1.0
        elif speed < desired - 0.05:
            cmd["accel"] = min(1.0, 0.25 + 0.6 * (desired - speed))
        elif speed > desired + 0.05:
            # 남은 거리 안에 멈추는 데 필요한 감속도 v²/2d 를 최대 제동 7 m/s² 대비 비율로
            need = speed * speed / (2 * max(remaining, 0.05))
            cmd["brake"] = min(1.0, max(0.6 * (speed - desired) + 0.1, 1.3 * need / 7.0))
        return cmd


planner = HybridAStarPlanner()


def handle_map_payload(map_payload: Dict[str, Any]) -> None:
    planner.set_map(map_payload)


def planner_step(obs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return planner.control(obs)
    except Exception as exc:
        print(f"[hybrid] error: {exc}")
        return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": "D"}
