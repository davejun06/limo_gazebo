#!/bin/bash

xhost +SI:localuser:root

docker run -it \
  --network host \
  --ipc host \
  --device /dev/dri:/dev/dri \
  --name limo_gazebo_humble \
  --workdir "$PWD" \
  -e DISPLAY="$DISPLAY" \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /etc/localtime:/etc/localtime:ro \
  -v "$PWD":"$PWD":rw \
  yspark98/limo:gazebo_humble
