#!/bin/bash

set -e

install_if_missing () {
  pkg=$1
  ver=$2
  import_name=${3:-$pkg}

  if python -c "import $import_name" 2>/dev/null; then
    echo "[SKIP] $pkg already installed"
  else
    if [ -z "$ver" ]; then
      echo "[INSTALL] $pkg"
      pip install "$pkg"
    else
      echo "[INSTALL] $pkg$ver"
      pip install "$pkg$ver"
    fi
  fi
}

# =========================
# Core dependencies
# =========================

# NOTE: install_if_missing's 3rd arg is the *import* name, which for several
# of these differs from the pip package name. Passing the pip name to
# `python -c "import ..."` silently always fails (SyntaxError on the hyphen,
# or plain ModuleNotFoundError), so every previous run of this script forced
# a reinstall of opencv/pillow/protobuf/onnxruntime-gpu to the pins below on
# *every* invocation, no matter what was already there. In this
# TensorRT-8.6.1 container that's actively dangerous for onnxruntime-gpu:
# the container is pinned to cuDNN 8 (TensorRT 8.6.1 requires cuDNN 8), but
# onnxruntime-gpu>=1.19 pre-built wheels require cuDNN 9 — installing
# ==1.23.2 here would silently break CUDA inference for FasterLivePortrait,
# LHM++, and Deep-Live-Cam alike (see build-onnxruntime-gpu1_21_0-cudnn8_9_7.md).

install_if_missing numpy ">=1.23.5,<2"
install_if_missing typing-extensions ">=4.8.0" typing_extensions
install_if_missing opencv-python "==4.10.0.84" cv2
install_if_missing cv2_enumerate_cameras "==1.1.15"
install_if_missing onnx "==1.18.0"
install_if_missing insightface "==0.7.3"
install_if_missing psutil "==5.9.8"
install_if_missing PySide6 ">=6.7,<7"
install_if_missing pillow "==12.1.1" PIL
install_if_missing tqdm ">=4.65.0"

install_if_missing opennsfw2 "==0.10.2"
install_if_missing protobuf "==4.25.1" google.protobuf

# =========================
# Platform-specific deps (WSL = Linux)
# =========================

if [[ "$(uname -s)" != "Darwin" ]]; then
  install_if_missing onnxruntime-gpu "==1.23.2" onnxruntime
  install_if_missing tensorflow ">=2.15.0"
else
  install_if_missing tensorflow ">=2.15.0"
fi

# =========================
# System libs PySide6/Qt needs to even import (QtGui pulls in EGL/GL/X11
# client libs) — a headless container has none of these by default.
# =========================

apt-get install -y -qq \
  libegl1 libgl1 libxkbcommon0 libdbus-1-3 libnss3 \
  libxcomposite1 libxrandr2 libxi6 libxtst6 libfontconfig1 \
  > /dev/null

# =========================
# Windows-only helper
# =========================

if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
  install_if_missing pygrabber ""
fi