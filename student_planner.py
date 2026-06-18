"""학생 자율주차 알고리즘 스켈레톤 모듈.

이 파일만 수정하면 되고, 네트워킹/IPC 관련 코드는 `ipc_client.py`에서
자동으로 처리합니다. 학생은 아래 `PlannerSkeleton` 클래스나 `planner_step`
함수를 원하는 로직으로 교체/확장하면 됩니다.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import heapq
import math
import time

def pretty_print_map_summary(map_payload: Dict[str, Any]) -> None:
    extent = map_payload.get("extent") or [None, None, None, None]
    slots = map_payload.get("slots") or []
    occupied = map_payload.get("occupied_idx") or []
    free_slots = len(slots) - sum(1 for v in occupied if v)
    print("[algo] map extent :", extent)
    print("[algo] total slots:", len(slots), "/ free:", free_slots)
    stationary = map_payload.get("grid", {}).get("stationary")
    if stationary:
        rows = len(stationary)
        cols = len(stationary[0]) if stationary else 0
        print("[algo] grid size  :", rows, "x", cols)


@dataclass
class PlannerSkeleton:
    """경로 계획/제어 로직을 담는 기본 스켈레톤 클래스입니다."""

    map_data: Optional[Dict[str, Any]] = None
    map_extent: Optional[Tuple[float, float, float, float]] = None
    cell_size: float = 0.5
    stationary_grid: Optional[List[List[float]]] = None
    waypoints: List[Tuple[float]] = None
    graph: Dict[Tuple[int, int], List[Tuple[int, int]]] = None
    graph_nodes: List[Tuple[float, float]] = None  # 노드의 월드 좌표

    def __post_init__(self) -> None:
        if self.waypoints is None:
            self.waypoints = []
        if self.graph is None:
            self.graph = {}
        if self.graph_nodes is None:
            self.graph_nodes = []
        self.mygear = "D"

    def _build_graph_from_map(self, map_payload: Dict[str, Any]) -> None:
        """맵 데이터를 기반으로 도로망 그래프 생성"""
        extent = map_payload.get("extent", [0.0, 75.0, 0.0, 50.0])
        cell_size = float(map_payload.get("cellSize", 0.5))
        walls = map_payload.get("walls_rects", [])
        occ_idx = map_payload.get("occupied_idx", [])
        slots = map_payload.get("slots", [])
        ox, oy = extent[0], extent[2]

        # 장애물 그리드
        grid = map_payload.get("grid", {})
        stat = grid.get("stationary", [])
        parked = grid.get("parked", [])
        occ_grid = [[1.0 if stat[y][x] > 0.5 or parked[y][x] > 0.5 else 0.0
                      for x in range(len(stat[0]))] for y in range(len(stat))] if stat else []

    def set_map(self, map_payload: Dict[str, Any]) -> None:
        """시뮬레이터에서 전송한 정적 맵 데이터를 보관합니다."""
        self.map_data = map_payload
        self.map_extent = tuple(
            map(float, map_payload.get("extent", (0.0, 0.0, 0.0, 0.0)))
        )
        self.cell_size = float(map_payload.get("cellSize", 0.5))
        self.stationary_grid = map_payload.get("grid", {}).get("stationary")
        pretty_print_map_summary(map_payload)
        self.waypoints.clear()
        self.mygear = "D"
        self.last_t = -1.0
        self._build_graph_from_map(map_payload)

    def compute_path(self, obs: Dict[str, Any]) -> None:
        """관측과 맵을 이용해 경로(웨이포인트)를 준비합니다."""

        #######################          파싱 단계          ##############################
        slot = obs["target_slot"] # 우리의 목표는 target
        x = obs["state"]["x"] ###### 현재 x 좌표
        y = obs["state"]["y"] ###### 현재 y 좌표
        yaw = obs["state"]["yaw"] ####### 차량이 향하고 있는 방향각 (math.pi / 2 + 0.1) 3.14
        t = float(obs.get("t", 0.0)) ####### 현재 시간 
        v = float(obs.get("state", {}).get("v", 0.0)) ####### 현재 속도 

        target_x = (slot[0] + slot[1]) / 2  
        target_y = (slot[2] + slot[3]) / 2 
        #################################################################################

        if 3.9 < x < 4.1 : # 시작 지점 x가 4.0 인 경우 (전진 주차 문제)
            # self.waypoints.append((4.0, 6.0))
            if target_y > 27.5: # 세번째 윗 줄일 경우
                if target_x < 15: ############ 아랫쪽 줄 첫번 째 칸과 두번 째 칸의 보정
                    conv_val = 1 # 이경우 속도 부족인데
                else:
                    conv_val = 0
                self.waypoints.append((4,30)) # 첫번째 turn point 
                self.waypoints.append((0, target_x + 1.5 + conv_val)) # 두번째 turn point 
            else: # 첫번째, 두번째 줄일 경우

                if (target_x, target_y) < (15, 20): ############ 아랫쪽 줄 첫번 째 칸과 두번 째 칸의 보정
                    conv_val = 2.3 # 이경우 속도 부족인데
                elif (target_x, target_y) < (25, 20):
                    conv_val = 1.3
                else:
                    conv_val = 0

                #우리가 회전할 경로를 waypoints 변수에 저장하기
                self.waypoints.append((4,target_y)) #첫번째 turn point
                self.waypoints.append((0, target_x + 1.5 - conv_val)) # 두번째 turn point
        else: # 시작 지점 x가 11.5 인 경우 (후진 주차 문제)
            # self.waypoints.append((11.5, 11.0))
            if target_y > 27.5: # 세번째 윗 줄일 경우
                self.waypoints.append((11.5,31)) # 첫번째 turn point 
                self.waypoints.append((0, target_x + 1.5)) # 두번째 turn point 
            else: # 첫번째, 두번째 줄일 경우
                
                if (target_x  < 15) and (target_y < 15): ############ 아랫쪽 줄 첫번 째 칸과 두번 째 칸의 보정
                    conv_val = 2 # 이 경우 속도 부족인데 (1, 2 좌표 보정)
                elif (target_x  < 25) and (target_y < 15):
                    conv_val = 1
                else:
                    conv_val = 0
                
                #우리가 회전할 경로를 waypoints 변수에 저장하기
                self.waypoints.append((11.5,target_y)) #첫번째 turn point
                self.waypoints.append((0, target_x + 1.5 + conv_val)) # 두번째 turn point

        #     pass

    def compute_control(self, obs: Dict[str, Any]) -> Dict[str, float]:
        """경로를 따라가기 위한 조향/가감속 명령을 산출합니다."""
        if (self.waypoints == []):
            self.compute_path(obs)
        else:
            pass
        #######################          파싱 단계          ##############################
        slot = obs["target_slot"] # 우리의 목표는 target
        x = obs["state"]["x"] ###### 현재 x 좌표
        y = obs["state"]["y"] ###### 현재 y 좌표
        yaw = obs["state"]["yaw"] ####### 차량이 향하고 있는 방향각 (math.pi / 2 + 0.1) 3.14
        t = float(obs.get("t", 0.0)) ####### 현재 시간 
        v = float(obs.get("state", {}).get("v", 0.0)) ####### 현재 속도 
        
        target_x = (slot[0] + slot[1]) / 2  
        target_y = (slot[2] + slot[3]) / 2 
        #################################################################################
        """경로를 따라가기 위한 조향/가감속 명령을 산출합니다."""
        # 예시: 기본 데모 제어. 학생은 원하는 알고리즘으로 대체하면 됩니다.
        t = float(obs.get("t", 0.0))
        v = float(obs.get("state", {}).get("v", 0.0))
        cmd = {"steer": 0.0, "accel": 0.0, "brake": 0.0, "gear": "D"}
        if self.mygear == "R":
            cmd["gear"] = "R"

        if self.waypoints[0][0] == 4:
            if self.waypoints[0][1] == 30: # 일단 세번 째 줄이라면 
                if v >= 6.0: ### 안정성을 위해 최대 속도 제한 
                    cmd["brake"] = 0.2
                else:
                    if  t < 2 and v < 3.5: #### 기본적으로 속도를 상승시키자
                        cmd["accel"] = 1
                    
                    elif y < 30: #### 타켓 y까지는 속도를 좀 줄여서 나아가자
                        rad_down = -(math.pi / 2 + 0.1)
                        if (abs(yaw - rad_down) > 0.1) and (v < 6):
                            cmd["accel"] = 0.3
                        else:
                            pass
                    
                    elif y >= target_y:
                        cmd["accel"] = 0
                        cmd["brake"] = 1 # 착지로직

                    else: #target_y에 가까워 졌을 경우 
                        if  v < 6:
                            cmd["accel"] = 0.3
                        rad_right = 0  # 오른쪽 방향
                        rad_up = (math.pi / 2 + 0.1) # 위쪽 방향

                        if (x < target_x -5):

                            if abs(yaw - rad_right) > 0.05: ###### 오른쪽으로 한 번 꺾자
                                cmd["accel"] = 0.2
                                # 오차가 크면 크게 꺾고, 작으면 작게 꺾기
                                error = yaw - rad_right
                                cmd["steer"] = max(-0.61, min(0.61, -error * 1.5))
                            else:
                                cmd["steer"] = 0 # 우회전 한 번

                        else:
                            if (x > self.waypoints[1][1] -v ): ###### target_x 에 거리 5를 남겼다면
                                if (target_x, target_y) < (15, 20): ############ 아랫쪽 줄 첫번 째 칸과 두번 째 칸의 보정
                                    cmd["brake"] = 0.1
                                else:
                                    cmd["brake"] = 0.2
                                if abs(yaw - rad_up) > 0.05:
                                    # 오차가 크면 크게 꺾고, 작으면 작게 꺾기
                                    error = yaw - rad_up
                                    cmd["steer"] = max(-0.61, min(0.61, -error * 1.5)) # 좌회전 한번 
                                else:
                                    cmd["steer"] = 0
                                    cmd["brake"] = 1.0
            else: ############################# 여기가 아랫쪽 

                if v >= 6.0: ### 안정성을 위해 최대 속도 제한 
                    cmd["brake"] = 0.2
                else:
                    if  t < 1 and v < 2: #### 기본적으로 속도를 상승시키자
                        cmd["accel"] = 1
                    
                    elif y < target_y: #### 타켓 y까지는 속도를 좀 줄여서 나아가자
                        rad_down = -(math.pi / 2 + 0.1)
                        if (abs(yaw - rad_down) > 0.2) and (v < 6):
                            cmd["accel"] = 0.3
                        else:
                            cmd["accel"] = 0
                            cmd["brake"] = 1 # 착지하는 경우 

                    else: #target_y에 가까워 졌을 경우 
                        cmd["accel"] = 0.2
                        rad_right = 0  # 오른쪽 방향
                        rad_down = - (math.pi / 2 + 0.1)

                        if (x  < target_x -5):

                            if abs(yaw - rad_right) > 0.05: ###### 오른쪽으로 한 번 꺾자
                                if (12.4 <= target_x <= 12.5) and (24.8 <= target_y <= 25.2): ###### 앞 부분 보정 (1 2 가능)
                                    cmd["accel"] = 0.1
                                    cmd["brake"] = 0.3
                                else:
                                    cmd["accel"] = 0.2
                                # 오차가 크면 크게 꺾고, 작으면 작게 꺾기
                                error = yaw - rad_right
                                cmd["steer"] = max(-0.61, min(0.61, -error * 1.5))
                            else:
                                cmd["steer"] = 0 # 우회전 한 번

                        else:
                            if (x > self.waypoints[1][1] -v): ###### target_x 에 거리 5를 남겼다면
                                if (target_x < 18): # 1열 과 2열에 속한 경우 # 트러블 메이커들은 세부 조정
                                    if (target_y < 17): # 1행에 속한 경우
                                        cmd["accel"] = 0.2
                                        cmd["brake"] = 0.2
                                    else: # 2행에 속한 경우 
                                        if (target_x < 15): # 2행 1열에 속한 경우 
                                            cmd["accel"] = 0.1
                                            cmd["brake"] = 0.1
                                        else: # 2행 2열은 트러블 메이커가 아님. 똑같게 
                                            cmd["accel"] = 0
                                            cmd["brake"] = 0.25
                                else:
                                    cmd["accel"] = 0
                                    cmd["brake"] = 0.25
                                if abs(yaw - rad_down) > 0.05:
                                    # 오차가 크면 크게 꺾고, 작으면 작게 꺾기
                                    error = yaw - rad_down
                                    cmd["steer"] = max(-0.61, min(0.61, -error * 1.5))
                                else:
                                    cmd["steer"] = 0
                                    cmd["brake"] = 1.0

        #########################################################################################################3
        #                            
        else: # 여기는 후진 주차 문제
            if self.waypoints[0][1] == 31:
                if v >= 5.5: ### 안정성을 위해 최대 속도 제한 
                    cmd["brake"] = 0.2
                else:
                    if t < 1 and v < 2: #### 기본적으로 속도를 상승시키자
                        cmd["accel"] = 1
                    
                    elif y < self.waypoints[0][1]: #### 타켓 y까지는 속도를 좀 줄여서 나아가자
                        rad_down = -(math.pi / 2 + 0.1)
                        if cmd["gear"] == "D":
                            if (abs(yaw - rad_down) > 0.2) and (v < 6):
                                cmd["accel"] = 0.3

                    else: # target_y에 가까워 졌을 경우 
                        
                        # ────────────────── [변경 구간: 여기서부터 3단계 체인] ──────────────────
                        
                        # 1단계: 기어 전환 조건 충족 시 (기어만 바꾸고 아래 모든 로직 스킵!)

                        if (-0.2 < v < 0.2) and (cmd["gear"] == "D"): 
                            cmd["brake"] = 0
                            cmd["accel"] = 0
                            self.mygear = "R"

                        # 2단계: 기어가 D일 때 실행할 메인 주차 로직 (기어 전환 타이밍이 아닐 때만 진입)
                        elif cmd["gear"] == "D":
                            cmd["accel"] = 0.2
                            rad_right = 0  # 오른쪽 방향

                            if (x < self.waypoints[1][1] ): ############ 멈추자
                                if abs(yaw - rad_right) > 0.05: ###### 오른쪽으로 한 번 꺾자
                                    if (12.4 <= target_x <= 12.5) and (24.8 <= target_y <= 25.2): ###### 앞 부분 보정 (1 2 가능)
                                        cmd["accel"] = 0.1
                                        cmd["brake"] = 0.3
                                    else:
                                        cmd["accel"] = 0.2
                                    error = yaw - rad_right
                                    cmd["steer"] = max(-0.61, min(0.61, -error * 1.5))
                                else:
                                    cmd["steer"] = 0 # 우회전 한 번

                            else:
                                    
                                if (x > self.waypoints[1][1] - v): ###### target_x 에 가까워 졌다면 (속도 보정)
                                    if (target_x, target_y) < (15, 17): ###### 앞 부분 보정 (1 2 가능)
                                        cmd["accel"] = 0
                                        cmd["brake"] = 1.0
                                    else:
                                        cmd["accel"] = 0
                                        cmd["brake"] = 1.0

                        # 3단계: 기어가 R일 때 실행할 후진 로직 (기어 전환 타이밍이 아닐 때만 진입)
                        else: 
                            # 1. 목표: y축 음의 방향 (아래쪽, -1.57)


                            # 1. 목표: y축 음의 방향 (아래쪽, -1.57)
                            rad_down = - (math.pi / 2)  
                            steer_gain = -0.065 * target_x + 4.74
                            steer_gain = max(1.4, min(3.5, steer_gain))

                            # 남은 주차 거리 측정
                            remaining_dist = abs(y - target_y)

                            # [안전장치] 목표 위치 도달 시 최우선 정지
                            if remaining_dist < 0.2:
                                cmd["steer"] = 0
                                cmd["accel"] = 0.0
                                cmd["brake"] = 1.0

                            else:
                                virtual_yaw = (yaw + math.pi) if yaw <= 0 else (yaw - math.pi)
                                rad_up = math.pi / 2  # 성공했던 기준 각도 활용
                                
                                raw_error = virtual_yaw - rad_up
                                error = (raw_error + math.pi) % (2 * math.pi) - math.pi

                                if abs(error) > 0.1:
                                    # 아까 성공했던 조향 부호 (error * steer_gain) 그대로 유지!
                                    steer_val = max(-0.61, min(0.61, error * steer_gain))
                                    cmd["steer"] = steer_val
                                    
                                    # 가속도 및 감속 로직 (그대로 유지)
                                    base_accel = 0.10 + (abs(steer_val) * 0.15)
                                    
                                    if remaining_dist < 1.0:
                                        scale = max(0.2, min(1.0, remaining_dist))
                                        cmd["accel"] = max(0.05, min(1.0, base_accel * scale))
                                        cmd["brake"] = max(0.0, min(0.3, 0.3 * (1.0 - remaining_dist)))
                                    else:
                                        cmd["accel"] = max(0.0, min(1.0, base_accel))
                                        cmd["brake"] = 0.0
                                        
                                    
                                else:
                                    # 정렬 완료 시
                                    cmd["steer"] = 0 
                                    
                                    if remaining_dist < 1.0:
                                        scale = max(0.1, min(1.0, remaining_dist))
                                        cmd["accel"] = max(0.05, 0.15 * scale)
                                        cmd["brake"] = max(0.0, min(0.5, 0.5 * (1.0 - remaining_dist)))
                                    else:
                                        cmd["accel"] = 0.15
                                        cmd["brake"] = 0.0

            else:
                if v >= 5.5: ### 안정성을 위해 최대 속도 제한 
                    cmd["brake"] = 0.2
                else:
                    if t < 1 and v < 2: #### 기본적으로 속도를 상승시키자
                        cmd["accel"] = 1
                    
                    elif y < self.waypoints[0][1]: #### 타켓 y까지는 속도를 좀 줄여서 나아가자
                        rad_down = -(math.pi / 2 + 0.1)
                        if cmd["gear"] == "D":
                            if (abs(yaw - rad_down) > 0.2) and (v < 6):
                                cmd["accel"] = 0.3

                    else: # target_y에 가까워 졌을 경우 
                        
                        # ────────────────── [변경 구간: 여기서부터 3단계 체인] ──────────────────
                        
                        # 1단계: 기어 전환 조건 충족 시 (기어만 바꾸고 아래 모든 로직 스킵!)

                        if (-0.2 < v < 0.2) and (cmd["gear"] == "D"): 
                            cmd["brake"] = 0
                            cmd["accel"] = 0
                            self.mygear = "R"

                        # 2단계: 기어가 D일 때 실행할 메인 주차 로직 (기어 전환 타이밍이 아닐 때만 진입)
                        elif cmd["gear"] == "D":
                            cmd["accel"] = 0.2
                            rad_right = 0  # 오른쪽 방향

                            if (x < self.waypoints[1][1] ): ############ 멈추자
                                if abs(yaw - rad_right) > 0.05: ###### 오른쪽으로 한 번 꺾자
                                    if (12.4 <= target_x <= 12.5) and (24.8 <= target_y <= 25.2): ###### 앞 부분 보정 (1 2 가능)
                                        cmd["accel"] = 0.1
                                        cmd["brake"] = 0.3
                                    else:
                                        cmd["accel"] = 0.2
                                    error = yaw - rad_right
                                    cmd["steer"] = max(-0.61, min(0.61, -error * 1.5))
                                else:
                                    cmd["steer"] = 0 # 우회전 한 번

                            else:
                                if (target_x, target_y) < (15, 20):
                                    target_x -= 2 # 이 경우 속도 부족인데 (1, 2 좌표 보정)
                                elif (target_x, target_y) < (25, 20):
                                    target_x -= 1
                                else:
                                    pass
                                    
                                if (x > self.waypoints[1][1] - v): ###### target_x 에 거리 5를 남겼다면
                                    if (target_x, target_y) < (15, 17): ###### 앞 부분 보정 (1 2 가능)
                                        cmd["accel"] = 0
                                        cmd["brake"] = 1.0
                                    else:
                                        cmd["accel"] = 0
                                        cmd["brake"] = 1.0

                        # 3단계: 기어가 R일 때 실행할 후진 로직 (기어 전환 타이밍이 아닐 때만 진입)
                        else: 

                            rad_up =  (math.pi / 2)  # 목표: y축 양의 방향 (+1.57)
                            steer_gain = -0.065 * target_x + 4.74
                            steer_gain = max(1.4, min(3.5, steer_gain))

                            # 후진 주차 위상 보정
                            raw_error = yaw - rad_up
                            error = (raw_error + math.pi) % (2 * math.pi) - math.pi

                            if abs(error) > 0.1:
                                # 1. 후진 조향각 계산 (후진이므로 error에 부호 매칭)
                                steer_val = max(-0.61, min(0.61, error * steer_gain))
                                cmd["steer"] = steer_val
                                
                                # 핸들이 꺾였을 때 바퀴 저항을 이겨내도록 기본 0.20에 조향 크기별 가중치 추가
                                cmd["accel"] = max(0.0, min(1.0, 0.20 + (abs(steer_val) * 0.15)))
                                cmd["brake"] = 0.0
                                
                            else:
                                # 주향각 정렬 완료 시 조향 풀고 가속 중지
                                cmd["steer"] = 0 
                                cmd["brake"] = 0.4
                                cmd["accel"] = 0.0

                                # 목표 y 위치까지 도달했는지 확인 후 완전 정지
                                if abs(y - target_y) < 0.2:
                                    cmd["brake"] = 1.0

        return cmd
        ################################################################################
        # 예시: 기본 데모 제어. 학생은 원하는 알고리즘으로 대체하면 됩니다.



# 전역 planner 인스턴스 (통신 모듈이 이 객체를 사용합니다.)
planner = PlannerSkeleton()


def handle_map_payload(map_payload: Dict[str, Any]) -> None:
    """통신 모듈에서 맵 패킷을 받을 때 호출됩니다."""

    planner.set_map(map_payload)


def planner_step(obs: Dict[str, Any]) -> Dict[str, Any]:
    """통신 모듈에서 매 스텝 호출하여 명령을 생성합니다."""

    try:
        return planner.compute_control(obs)
    except Exception as exc:
        print(f"[algo] planner_step error: {exc}")
        return {"steer": 0.0, "accel": 0.0, "brake": 0.5, "gear": "D"}


