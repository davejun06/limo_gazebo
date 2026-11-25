# LIMO Gazebo 주행 (Q1 장애물 없음, Q2 장애물 있음)

빌드, 맵핑, 전역 경로 생성, Q1·Q2 실행 방법은 [코드설명.md](코드설명.md)에 있습니다.

- 워크스페이스: `ws_limo_humble/` (빌드: `colcon build --symlink-install --packages-select limo_car pure_pursuit`)
- 모든 터미널에서 먼저: `cd ws_limo_humble && source env.sh`
- Docker 이미지: `yspark98/limo:gazebo_humble` (`docker.sh`)
