"""Hybrid A* 플래너(`hybrid_astar_planner.py`)로 시뮬레이터에 접속하는 엔트리포인트.

`ipc_client.py`는 `student_planner`를 import하므로, 그 이름에 Hybrid A* 플래너를 끼워 넣고 실행합니다.
"""

import sys

import hybrid_astar_planner

sys.modules["student_planner"] = hybrid_astar_planner

from ipc_client import main  # noqa: E402

if __name__ == "__main__":
    main()
