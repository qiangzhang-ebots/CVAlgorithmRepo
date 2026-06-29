#!/bin/bash

# 实际使用的时候，可以按之前的run_docker.sh(run_docker.sh不知道怎么启动)

# 简易纯净启动版本
IMAGE="ebotsinc/ebots_ros2_perception:dev-latest"
CONTAINER_NAME="perception_container"

export DOCKER_HOME=/root

# 定义容器中工作空间
EBOTS_PERCEPTION_WORKSPACE="/root/perception/workspace"
ARTIFACT_ID=artifacts_cell_tianma

# 允许显示图像界面
xhost +local: > /dev/null

echo "正在启动 perception 容器..."

docker run -it --rm \
  --name $CONTAINER_NAME \
  --network host \
  --pid host \
  --ipc host \
  --privileged \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix/:/tmp/.X11-unix \
  -v $PWD/logs:$DOCKER_HOME/logs \
  -v $PWD/src:$EBOTS_PERCEPTION_WORKSPACE/cvalgorithm/src \
  --tmpfs /root/perception:exec \
  -w $DOCKER_HOME \
  $IMAGE \
  /bin/bash -ci "source $DOCKER_HOME/.entry_point.sh && /bin/bash"
  
  #--tmpfs /root/perception \