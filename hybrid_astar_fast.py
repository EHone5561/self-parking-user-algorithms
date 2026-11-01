"""더 빠르게 주행하는 Hybrid A* 플래너.

경로 탐색은 `hybrid_astar_planner.py`와 같고, 추종 속도만 다르게 정합니다.
- 고정 상한(전진 3, 후진 1.5 m/s) 대신 경로의 곡률로 구간별 속도 프로필을 만듭니다:
  곡선에서는 횡가속도 A_LAT 이하, 직선에서는 V_MAX까지, 구간 끝(정지·기어 변경)
  전에는 DECEL로 줄어들도록 뒤에서부터 계산합니다.
- 속도가 높을수록 Pure Pursuit 전방 주시 거리를 늘립니다.
`my_agent_fast.py`로 실행하면 이 플래너가 쓰입니다.
"""

from typing import Any, Dict, List
import math

from hybrid_astar_planner import PATH_RES, HybridAStarPlanner, _wrap

V_MAX_FWD, V_MAX_REV = 8.0, 4.0   # m/s
A_LAT = 4.5                       # 곡선 횡가속도 한계 (m/s²): v² × 곡률
DECEL = 6.0                       # 정지·감속에 쓰는 감속도 (m/s²); 최대 제동은 7
CURVE_WINDOW = 8                  # 곡률을 앞뒤 0.8m 범위 최댓값으로 (조향 지연 대비)


class FastHybridAStarPlanner(HybridAStarPlanner):
    h_weight = 1.2                # 휴리스틱 가중치를 낮춰 더 짧은 경로 (기본 1.6)

    def _plan(self, obs: Dict[str, Any]) -> None:
        super()._plan(obs)
        self.profiles = [self._profile(seg, d) for seg, d in zip(self.segments, self.dirs)]

    @staticmethod
    def _profile(seg: List[tuple], direction: int) -> List[float]:
        n = len(seg)
        vmax = V_MAX_FWD if direction > 0 else V_MAX_REV
        curv = [abs(_wrap(seg[min(i + 1, n - 1)][2] - seg[i][2])) / PATH_RES for i in range(n)]
        limit = []
        for i in range(n):
            k = max(curv[max(0, i - CURVE_WINDOW):i + CURVE_WINDOW + 1])
            limit.append(min(vmax, math.sqrt(A_LAT / k)) if k > 1e-6 else vmax)
        # 뒤에서부터: 구간 끝에서 0, 그 앞은 DECEL로 멈출 수 있는 속도 이하
        limit[-1] = 0.0
        for i in range(n - 2, -1, -1):
            limit[i] = min(limit[i], math.sqrt(limit[i + 1] ** 2 + 2 * DECEL * PATH_RES))
        return limit

    def control(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        stop = {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": "D"}
        if self.worker is None:
            return super().control(obs)                    # 탐색 시작 (백그라운드)
        if self.worker.is_alive() or self.failed or not self.segments:
            return stop

        st = obs["state"]
        x, y, yaw, v = st["x"], st["y"], st["yaw"], st["v"]
        direction = self.dirs[self.seg]
        gear = "D" if direction > 0 else "R"
        if self.done:
            return {**stop, "gear": gear}
        path = self.segments[self.seg]
        profile = self.profiles[self.seg]

        best_i, best_d = self.idx, float("inf")
        for i in range(self.idx, min(len(path), self.idx + 40)):
            dd = math.hypot(path[i][0] - x, path[i][1] - y)
            if dd < best_d:
                best_i, best_d = i, dd
        self.idx = best_i
        ex, ey, eyaw = path[-1]
        remaining = (len(path) - 1 - self.idx) * PATH_RES
        if self.idx >= len(path) - 3:
            remaining = direction * (math.cos(eyaw) * (ex - x) + math.sin(eyaw) * (ey - y))

        speed = v * direction
        if remaining < 0.08:
            if abs(v) < 0.05:
                if self.seg + 1 < len(self.segments):
                    self.seg += 1
                    self.idx = 0
                else:
                    self.done = True
            return {**stop, "gear": gear}

        look = max(1.2, min(3.0, 1.0 + 0.35 * abs(v)))
        tx = ty = None
        for i in range(self.idx, len(path)):
            if math.hypot(path[i][0] - x, path[i][1] - y) >= look:
                tx, ty = path[i][0], path[i][1]
                break
        if tx is None:
            extra = max(0.0, look - math.hypot(ex - x, ey - y))
            tx = ex + direction * extra * math.cos(eyaw)
            ty = ey + direction * extra * math.sin(eyaw)
        heading = yaw if direction > 0 else _wrap(yaw + math.pi)
        alpha = _wrap(math.atan2(ty - y, tx - x) - heading)
        ld = max(0.5, math.hypot(tx - x, ty - y))
        steer = math.atan2(2.0 * self.L * math.sin(alpha), ld) * direction
        steer = max(-self.max_steer, min(self.max_steer, steer))

        # 속도 프로필 (조금 앞 지점까지 보고 가장 낮은 값) + 남은 거리 안에 멈출 수 있는 속도
        desired = min(profile[self.idx:self.idx + 6])
        desired = min(desired, math.sqrt(2 * DECEL * max(0.0, remaining - 0.05)))
        desired = max(desired, 0.3)                        # 끝 직전에 멈춰 서지 않도록

        cmd = {"steer": steer, "accel": 0.0, "brake": 0.0, "gear": gear}
        if speed < -0.05:
            cmd["brake"] = 1.0
        elif speed < desired - 0.05:
            cmd["accel"] = min(1.0, 0.3 + 0.8 * (desired - speed))
        elif speed > desired + 0.05:
            need = speed * speed / (2 * max(remaining, 0.05))
            cmd["brake"] = min(1.0, max(0.5 * (speed - desired) + 0.1, 1.3 * need / 7.0))
        return cmd


planner = FastHybridAStarPlanner()


def handle_map_payload(map_payload: Dict[str, Any]) -> None:
    planner.set_map(map_payload)


def planner_step(obs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return planner.control(obs)
    except Exception as exc:
        print(f"[fast] error: {exc}")
        return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": "D"}
