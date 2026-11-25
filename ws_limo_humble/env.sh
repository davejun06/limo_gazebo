# 모든 터미널에서 먼저 실행: source env.sh   (워크스페이스 루트에서)
_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-77}
export ROS_LOCALHOST_ONLY=1          # 모든 터미널이 같아야 노드끼리 보임
export GAZEBO_IP=127.0.0.1
export GAZEBO_MASTER_URI=${GAZEBO_MASTER_URI:-http://127.0.0.1:11345}
source /opt/ros/humble/setup.bash
[ -f "$_WS/install/setup.bash" ] && source "$_WS/install/setup.bash"
cd "$_WS"
