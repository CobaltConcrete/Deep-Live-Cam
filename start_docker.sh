#!/bin/bash

docker run -it \
  --gpus all \
  --device=/dev/video0:/dev/video0 \
  -p 8080:8080 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $HOME/projects/Deep-Live-Cam:/workspaces/Deep-Live-Cam \
  -v $HOME/projects/LHM:/workspaces/LHM \
  -w /workspaces \
  cobaltconcrete/lhm-deeplivecam:cu12.1-torch2.3.0-onnxruntimegpu1.21.0-cudnn-8.9.7 \
  /bin/bash
