# Building onnxruntime-gpu 1.21.0 from Source (CUDA 12.x + cuDNN 8)

## Why This Is Necessary

Pre-built `onnxruntime-gpu` wheels on PyPI have a cuDNN version gap:

| onnxruntime-gpu version | cuDNN required | cuBLAS required |
|-------------------------|----------------|-----------------|
| 1.18.x                  | 8              | **11** (CUDA 11 era) |
| 1.19.x – 1.21.x+        | **9**          | 12              |

If your container has **CUDA 12.x + cuDNN 8 + cuBLAS 12** (e.g. because another model requires cuDNN 8), no pre-built wheel works. You must build from source.

---

## System Requirements

- CUDA 12.x (tested on 12.1)
- cuDNN 8.9.x
- Python 3.10
- GPU: RTX 4090 Laptop (Ada Lovelace, arch `sm_89`) — adjust `CMAKE_CUDA_ARCHITECTURES` for other GPUs
- 16GB+ RAM recommended (CUDA kernel compilation is very memory-hungry)

---

## Step 1: Install System Dependencies

```bash
apt-get update && apt-get install -y cmake ninja-build patchelf wget git

# Install cuDNN 8 dev headers (libs may already be present)
apt-get install -y libcudnn8-dev

# Verify cuDNN 8 is installed
dpkg -l | grep cudnn
# Should show: libcudnn8 and libcudnn8-dev at version 8.9.x

# Verify headers exist
find /usr/include -name "cudnn*.h" | head -5
```

> **Note:** cuDNN 8.9.7 built for CUDA 12.2 is fully compatible with CUDA 12.1 — CUDA minor versions are backwards compatible.

---

## Step 2: Install a Modern CMake

The system CMake (3.22) is too old. onnxruntime 1.21.0 requires CMake 3.28+.

```bash
pip install cmake --upgrade

# Make it take precedence over system cmake
export PATH=$(python3 -c "import cmake, os; print(os.path.join(os.path.dirname(cmake.__file__), 'data', 'bin'))"):$PATH

# Verify
cmake --version
# Should show 3.28+ (tested with 4.3.2)
```

Add the export to your shell profile to persist it:
```bash
echo 'export PATH=$(python3 -c "import cmake, os; print(os.path.join(os.path.dirname(cmake.__file__), '"'"'data'"'"', '"'"'bin'"'"'))"):$PATH' >> ~/.bashrc
```

---

## Step 3: Downgrade NumPy (if on NumPy 2.x)

onnxruntime 1.21.0 is compiled against NumPy 1.x. If you have NumPy 2.x it will fail to import after installation.

```bash
pip install "numpy<2.0" --force-reinstall
```

---

## Step 4: Clone onnxruntime

```bash
git clone --recursive --branch v1.21.0 --depth 1 \
    https://github.com/microsoft/onnxruntime.git /tmp/onnxruntime
```

---

## Step 5: Fix the Eigen Dependency

The Eigen archive URL in `deps.txt` points to a specific GitLab commit hash that GitLab now serves with a different SHA1 than expected (GitLab changed how it packages archives). We download the stable Eigen 3.4 zip directly and patch `deps.txt` to use it.

```bash
# Download stable Eigen 3.4
wget -O /tmp/eigen.zip https://gitlab.com/libeigen/eigen/-/archive/3.4/eigen-3.4.zip

# Get its actual SHA1
ACTUAL_SHA1=$(sha1sum /tmp/eigen.zip | cut -d' ' -f1)
echo "Eigen SHA1: $ACTUAL_SHA1"

# Patch deps.txt to use the local file
sed -i "s|eigen;https://gitlab.com/libeigen/eigen/-/archive/1d8b82b0740839c0de7f1242a3585e3390ff5f33/eigen-1d8b82b0740839c0de7f1242a3585e3390ff5f33.zip;5ea4d05e62d7f954a46b3213f9b2535bdd866803|eigen;file:///tmp/eigen.zip;${ACTUAL_SHA1}|" \
    /tmp/onnxruntime/cmake/deps.txt

# Verify patch
grep eigen /tmp/onnxruntime/cmake/deps.txt
# Should show: eigen;file:///tmp/eigen.zip;<sha1>
```

---

## Step 6: Add Swap Space (Recommended)

CUDA kernel compilation uses 2–4 GB RAM per parallel job. Add swap to prevent OOM crashes:

```bash
fallocate -l 8G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
free -h  # Verify swap is active
```

---

## Step 7: Run CMake Configure

```bash
cd /tmp/onnxruntime

./build.sh \
  --allow_running_as_root \
  --config Release \
  --build_wheel \
  --use_cuda \
  --cuda_home /usr/local/cuda \
  --cudnn_home /usr \
  --cmake_extra_defines \
    CMAKE_CUDA_ARCHITECTURES=89 \
    CMAKE_POLICY_VERSION_MINIMUM=3.5 \
  --skip_tests \
  --parallel 4
```

### Key flags explained

| Flag | Purpose |
|------|---------|
| `--allow_running_as_root` | Required if running as root (common in containers) |
| `--cudnn_home /usr` | Points to system cuDNN: headers in `/usr/include`, libs in `/usr/lib/x86_64-linux-gnu` |
| `--cuda_home /usr/local/cuda` | Standard CUDA install path |
| `CMAKE_CUDA_ARCHITECTURES=89` | RTX 4090 (Ada Lovelace). Change for other GPUs — see table below |
| `CMAKE_POLICY_VERSION_MINIMUM=3.5` | Suppresses CMake 4.x errors in older sub-project CMakeLists files (dlpack etc.) |
| `--parallel 4` | **Keep this low.** Each nvcc job uses 2–4 GB RAM. 24 jobs = SIGSEGV crash |

### GPU Architecture Reference

| GPU | Architecture | `CMAKE_CUDA_ARCHITECTURES` value |
|-----|-------------|----------------------------------|
| RTX 4090 / 4080 / 4070 | Ada Lovelace | `89` |
| RTX 3090 / 3080 / 3070 | Ampere | `86` |
| RTX 2080 / 2070 | Turing | `75` |
| Tesla V100 | Volta | `70` |
| Tesla T4 | Turing | `75` |
| A100 | Ampere | `80` |

---

## Step 8: Build (this takes 30–60 minutes)

If `build.sh` crashes (SIGSEGV from OOM), invoke make directly from the build directory — it resumes from where it left off:

```bash
cd /tmp/onnxruntime/build/Linux/Release
make -j4 2>&1 | tail -50
```

Monitor progress in a second terminal:
```bash
watch -n10 'find /tmp/onnxruntime/build/Linux/Release -name "*.o" -newer /tmp/onnxruntime/build/Linux/Release/CMakeCache.txt | wc -l'
```

The count should tick up slowly. Each CUDA `.cu` file can take 5–10 minutes.

---

## Step 9: Package the Wheel

Once `make` completes (`[100%] Built target ...`), build the Python wheel:

```bash
cd /tmp/onnxruntime

python3 tools/ci_build/build.py \
  --allow_running_as_root \
  --build_dir /tmp/onnxruntime/build/Linux \
  --config Release \
  --build_wheel \
  --use_cuda \
  --cuda_home /usr/local/cuda \
  --cudnn_home /usr \
  --cmake_extra_defines \
    CMAKE_CUDA_ARCHITECTURES=89 \
    CMAKE_POLICY_VERSION_MINIMUM=3.5 \
  --skip_tests \
  --skip_submodule_sync \
  --parallel 4 \
  --build
```

Find the wheel:
```bash
find /tmp/onnxruntime/build/Linux/Release/dist -name "*.whl"
# e.g. onnxruntime_gpu-1.21.0-cp310-cp310-linux_x86_64.whl
```

---

## Step 10: Install and Verify

```bash
pip install /tmp/onnxruntime/build/Linux/Release/dist/onnxruntime_gpu-1.21.0-*.whl --force-reinstall

# Test
cd /workspaces  # important: run from outside the onnxruntime source dir
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Expected: [..., 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

> **Important:** Always run the test from outside `/tmp/onnxruntime`. If you run it from inside the source directory, Python imports the local source instead of the installed package and you get `ModuleNotFoundError`.

---

## Complete One-Shot Script

Copy-paste this entire block into a fresh container to reproduce the full build:

```bash
#!/bin/bash
set -e

# === CONFIG — adjust these for your environment ===
CUDA_ARCH=89          # 89=RTX4090, 86=RTX3090, 80=A100, 75=T4/RTX2080
ORT_VERSION=v1.21.0
BUILD_DIR=/tmp/onnxruntime
# ==================================================

# 1. System deps
apt-get update && apt-get install -y cmake ninja-build patchelf wget git libcudnn8-dev

# 2. Modern CMake
pip install cmake --upgrade "numpy<2.0"
export PATH=$(python3 -c "import cmake, os; print(os.path.join(os.path.dirname(cmake.__file__), 'data', 'bin'))"):$PATH

# 3. Swap
fallocate -l 8G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile

# 4. Clone
git clone --recursive --branch $ORT_VERSION --depth 1 \
    https://github.com/microsoft/onnxruntime.git $BUILD_DIR

# 5. Fix Eigen
wget -O /tmp/eigen.zip https://gitlab.com/libeigen/eigen/-/archive/3.4/eigen-3.4.zip
EIGEN_SHA1=$(sha1sum /tmp/eigen.zip | cut -d' ' -f1)
sed -i "s|eigen;https://gitlab.com/libeigen/eigen/-/archive/1d8b82b0740839c0de7f1242a3585e3390ff5f33/eigen-1d8b82b0740839c0de7f1242a3585e3390ff5f33.zip;5ea4d05e62d7f954a46b3213f9b2535bdd866803|eigen;file:///tmp/eigen.zip;${EIGEN_SHA1}|" \
    $BUILD_DIR/cmake/deps.txt

# 6. Configure
cd $BUILD_DIR
./build.sh \
  --allow_running_as_root \
  --config Release \
  --build_wheel \
  --use_cuda \
  --cuda_home /usr/local/cuda \
  --cudnn_home /usr \
  --cmake_extra_defines \
    CMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH \
    CMAKE_POLICY_VERSION_MINIMUM=3.5 \
  --skip_tests \
  --parallel 4 || true  # may exit after configure; build continues below

# 7. Build (resume-safe)
cd $BUILD_DIR/build/Linux/Release
make -j4

# 8. Package wheel
cd $BUILD_DIR
python3 tools/ci_build/build.py \
  --allow_running_as_root \
  --build_dir $BUILD_DIR/build/Linux \
  --config Release \
  --build_wheel \
  --use_cuda \
  --cuda_home /usr/local/cuda \
  --cudnn_home /usr \
  --cmake_extra_defines \
    CMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH \
    CMAKE_POLICY_VERSION_MINIMUM=3.5 \
  --skip_tests \
  --skip_submodule_sync \
  --parallel 4 \
  --build

# 9. Install
pip install $BUILD_DIR/build/Linux/Release/dist/onnxruntime_gpu-*.whl --force-reinstall

# 10. Verify (run from outside source dir)
cd /tmp
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `libcudnn.so.9: No such file` | Pre-built wheel needs cuDNN 9 | Build from source (this guide) |
| `CMake 3.28 or higher is required` | System cmake too old | `pip install cmake --upgrade` |
| `SHA1 hash mismatch` for eigen | GitLab changed archive format | Patch `deps.txt` to use local zip (Step 5) |
| `Compatibility with CMake < 3.5` | Old sub-project CMakeLists | Add `CMAKE_POLICY_VERSION_MINIMUM=3.5` |
| `SIGSEGV` during make | OOM with too many parallel jobs | Use `make -j4` instead of `-j24` |
| `ModuleNotFoundError: onnxruntime.capi` | Running python inside source dir | `cd` out of `/tmp/onnxruntime` first |
| `NumPy 1.x compiled module` error | NumPy 2.x incompatible with built wheel | `pip install "numpy<2.0"` |
