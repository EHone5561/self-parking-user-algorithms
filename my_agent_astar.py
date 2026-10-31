"""A* 플래너(`astar_planner.py`)로 시뮬레이터에 접속하는 엔트리포인트.

`ipc_client.py`는 `student_planner`를 import하므로, 그 이름에 A* 플래너를 끼워 넣고 실행합니다.
시뮬레이터를 먼저 켠 뒤 `python my_agent_astar.py`로 접속하세요.
"""

import sys

import astar_planner

sys.modules["student_planner"] = astar_planner

from ipc_client import main  # noqa: E402

if __name__ == "__main__":
    main()
