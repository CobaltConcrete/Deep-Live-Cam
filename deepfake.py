#!/usr/bin/env python3
"""
deepfake.py — Webcam → Deep-Live-Cam face-swap → MJPEG HTTP stream

Usage:
    python deepfake.py -s ref_image.png --execution-provider cuda

Options:
    -s / --source       Path to source face image (required)
    --camera-index      V4L2 camera index (default: 0)
    --width             Capture width  (default: 640)
    --height            Capture height (default: 480)
    --fps               Target capture FPS (default: 30)
    --port              MJPEG HTTP port (default: 8080)
    --jpeg-quality      MJPEG JPEG quality 1-100 (default: 85)
    --frame-processor   One or more processors (default: face_swapper)
    --many-faces        Process every face in the frame
    --execution-provider  cuda | rocm | coreml | dml | cpu  (default: auto)
    --execution-threads   Number of inference threads (default: auto)
    --max-memory        Max RAM in GB (default: 16)
"""

import os
import sys

# ── bootstrap: make sure the project root (where `modules/` lives) is on the
#    path, regardless of where Python was invoked from.  This mirrors what
#    run.py / modules/run.py do implicitly by living inside the project tree.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Also add project root to PATH so bundled ffmpeg/ffprobe are found
os.environ["PATH"] = _PROJECT_ROOT + os.pathsep + os.environ.get("PATH", "")

# ── Ensure pip-installed cuDNN / CUDA libs are discoverable ────────────────
# onnxruntime-gpu needs libcudnn.so.9 which pip installs under site-packages
# in a path the dynamic linker doesn't know about.  We pre-load the library
# with RTLD_GLOBAL so that onnxruntime_providers_cuda.so can find it.
import ctypes, glob as _glob, site as _site
_cudnn_search_dirs = []
for _base in _site.getsitepackages() + [_site.getusersitepackages()]:
    if isinstance(_base, str):
        _cudnn_search_dirs.append(os.path.join(_base, "nvidia", "cudnn", "lib"))
# Also try the common /usr/local path directly (Docker containers)
_cudnn_search_dirs.append("/usr/local/lib/python{}.{}/dist-packages/nvidia/cudnn/lib".format(
    sys.version_info.major, sys.version_info.minor))
for _cudnn_dir in _cudnn_search_dirs:
    if os.path.isdir(_cudnn_dir):
        _ld = os.environ.get("LD_LIBRARY_PATH", "")
        if _cudnn_dir not in _ld:
            os.environ["LD_LIBRARY_PATH"] = _cudnn_dir + os.pathsep + _ld
        for _so in sorted(_glob.glob(os.path.join(_cudnn_dir, "libcudnn*.so.*"))):
            try:
                ctypes.CDLL(_so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
        break

import time
import signal
import argparse
import threading

# ── single-thread OMP tweak must happen before torch/onnx imports ──────────
if any(a.startswith('--execution-provider') for a in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '6'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
import numpy as np
import cv2

# ── optional torch ──────────────────────────────────────────────────────────
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import onnxruntime

try:
    import tensorflow
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

# ── suppress noisy warnings ─────────────────────────────────────────────────
warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
if HAS_TORCH:
    warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


# ══════════════════════════════════════════════════════════════════════════════
# MJPEG HTTP streamer
# ══════════════════════════════════════════════════════════════════════════════

class MJPEGStreamer:
    """
    Serves rendered frames as an MJPEG stream over HTTP so they can be watched
    in any browser — no display server required.

    Open  http://<host>:<port>/  in your browser to watch.
    """

    def __init__(self, port: int = 8080, jpeg_quality: int = 85):
        from http.server import BaseHTTPRequestHandler, HTTPServer

        self.port    = port
        self.quality = jpeg_quality
        self._frame: bytes = b""
        self._lock   = threading.Lock()
        self._stop   = threading.Event()

        streamer = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                if self.path == "/":
                    body = (
                        b"<html><body style='margin:0;background:#000'>"
                        b"<img src='/stream' style='max-width:100%;height:auto'>"
                        b"</body></html>"
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        "multipart/x-mixed-replace; boundary=frame"
                    )
                    self.end_headers()
                    try:
                        while not streamer._stop.is_set():
                            with streamer._lock:
                                frame = streamer._frame
                            if frame:
                                self.wfile.write(
                                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                                    + frame + b"\r\n"
                                )
                            time.sleep(0.01)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self.send_response(404)
                    self.end_headers()

        self._server = HTTPServer(("0.0.0.0", port), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"[stream] Preview → http://localhost:{port}/  (or your server IP)")

    def push(self, rgb_np: np.ndarray):
        """Encode an RGB uint8 frame and push it to connected clients."""
        ok, buf = cv2.imencode(
            ".jpg", cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, self.quality],
        )
        if ok:
            with self._lock:
                self._frame = buf.tobytes()

    def stop(self):
        self._stop.set()
        self._server.shutdown()


# ══════════════════════════════════════════════════════════════════════════════
# Helpers (mirrors core.py utilities)
# ══════════════════════════════════════════════════════════════════════════════

def encode_execution_providers(providers):
    return [p.replace('ExecutionProvider', '').lower() for p in providers]


def decode_execution_providers(names):
    available     = onnxruntime.get_available_providers()
    encoded_avail = encode_execution_providers(available)
    return [
        prov for prov, enc in zip(available, encoded_avail)
        if any(n in enc for n in names)
    ]


def suggest_default_execution_provider() -> str:
    available = encode_execution_providers(onnxruntime.get_available_providers())
    for pref in ('cuda', 'rocm', 'coreml', 'dml'):
        if pref in available:
            return pref
    return 'cpu'


def suggest_execution_providers():
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads(providers) -> int:
    if 'DmlExecutionProvider'  in providers: return 1
    if 'ROCMExecutionProvider' in providers: return 1
    if 'CUDAExecutionProvider' in providers: return 2
    cpu_count = os.cpu_count() or 4
    return max(4, min(cpu_count - 2, 16))


def limit_resources(max_memory: int) -> None:
    import platform
    if HAS_TENSORFLOW:
        gpus = tensorflow.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tensorflow.config.experimental.set_memory_growth(gpu, True)
    if max_memory:
        memory = max_memory * 1024 ** 3
        if platform.system().lower() == 'windows':
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def release_resources() -> None:
    if 'CUDAExecutionProvider' in onnxruntime.get_available_providers() and HAS_TORCH:
        torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# Webcam capture thread
# ══════════════════════════════════════════════════════════════════════════════

class WebcamCapture:
    """
    Grabs frames from a V4L2 webcam in a background thread so the main loop
    never stalls waiting on the camera driver.
    """

    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        # CAP_V4L2 = cv2.CAP_V4L2 on Linux; fall back to default backend elsewhere
        backend = getattr(cv2, 'CAP_V4L2', cv2.CAP_ANY)
        self._cap = cv2.VideoCapture(index, backend)

        if not self._cap.isOpened():
            # retry with default backend
            self._cap = cv2.VideoCapture(index)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {index}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS,          fps)

        # MJPG gives better USB bandwidth utilisation on most webcams
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self._cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        self._frame: np.ndarray | None = None
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

        print(f"[camera] Opened camera {index}  "
              f"{int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
              f"{int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}  "
              f"{self._cap.get(cv2.CAP_PROP_FPS):.0f} fps")

    def _reader(self):
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = frame   # BGR uint8

    def read(self) -> np.ndarray | None:
        """Return the latest BGR frame, or None if none yet."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self._cap.release()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser(
        description="Webcam → face-swap → MJPEG HTTP stream (headless)"
    )
    ap.add_argument('-s', '--source',      required=True,
                    help='Source face image')
    ap.add_argument('--camera-index',      type=int, default=0,
                    help='V4L2 camera index (default: 0)')
    ap.add_argument('--width',             type=int, default=640)
    ap.add_argument('--height',            type=int, default=480)
    ap.add_argument('--fps',               type=int, default=30,
                    help='Target capture FPS (default: 30)')
    ap.add_argument('--port',              type=int, default=8080,
                    help='MJPEG HTTP port (default: 8080)')
    ap.add_argument('--jpeg-quality',      type=int, default=85,
                    help='MJPEG JPEG quality 1-100 (default: 85)')
    ap.add_argument('--frame-processor',   nargs='+',
                    default=['face_swapper'],
                    choices=['face_swapper', 'face_enhancer',
                             'face_enhancer_gpen256', 'face_enhancer_gpen512'],
                    dest='frame_processor',
                    help='Pipeline of frame processors')
    ap.add_argument('--many-faces',        action='store_true', default=False,
                    help='Process every face in the frame')
    ap.add_argument('--execution-provider', nargs='+',
                    default=[suggest_default_execution_provider()],
                    choices=suggest_execution_providers(),
                    dest='execution_provider')
    ap.add_argument('--execution-threads', type=int, default=None,
                    dest='execution_threads')
    ap.add_argument('--max-memory',        type=int, default=16,
                    dest='max_memory')
    return ap.parse_args()


def main():
    args = parse_args()

    # ── configure globals ───────────────────────────────────────────────────
    import modules.globals as gbl
    from modules.processors.frame.core import get_frame_processors_modules

    gbl.source_path        = args.source
    gbl.target_path        = None          # live webcam — no file target
    gbl.output_path        = None
    gbl.frame_processors   = args.frame_processor
    gbl.headless           = True
    gbl.keep_fps           = True
    gbl.keep_audio         = False
    gbl.keep_frames        = False
    gbl.many_faces         = args.many_faces
    gbl.mouth_mask         = False
    gbl.nsfw_filter        = False
    gbl.map_faces          = False
    gbl.video_encoder      = 'libx264'
    gbl.video_quality      = 18
    gbl.live_mirror        = False
    gbl.live_resizable     = False
    gbl.max_memory         = args.max_memory
    gbl.execution_providers = decode_execution_providers(args.execution_provider)
    gbl.execution_threads  = (
        args.execution_threads
        if args.execution_threads is not None
        else suggest_execution_threads(gbl.execution_providers)
    )

    # fp_ui toggles used by some processors
    # Initialise fp_ui if it doesn't exist yet (some globals.py versions omit it)
    if not hasattr(gbl, 'fp_ui') or gbl.fp_ui is None:
        gbl.fp_ui = {}
    for key in ('face_enhancer', 'face_enhancer_gpen256', 'face_enhancer_gpen512'):
        gbl.fp_ui[key] = key in args.frame_processor

    print(f"[config] providers={gbl.execution_providers}  "
          f"threads={gbl.execution_threads}  "
          f"processors={gbl.frame_processors}")

    limit_resources(args.max_memory)

    # ── pre-check processors ────────────────────────────────────────────────
    # Verify the processor modules are importable and surface any real errors.
    # get_frame_processors_modules() silently swallows ImportError — we don't.
    import importlib
    for _proc_name in gbl.frame_processors:
        _mod_path = f"modules.processors.frame.{_proc_name}"
        try:
            importlib.import_module(_mod_path)
            print(f"[debug] import OK: {_mod_path}")
        except ImportError as _e:
            print(f"[error] Cannot import {_mod_path}: {_e}")
            print(f"        sys.path = {sys.path}")
            return
        except Exception as _e:
            # Other errors (missing model file etc.) are expected at import time
            # for some processor variants — log but don't abort yet.
            print(f"[debug] import warning for {_mod_path}: {type(_e).__name__}: {_e}")

    frame_processors = get_frame_processors_modules(gbl.frame_processors)
    if not frame_processors:
        print("[error] No frame processors loaded — check modules/processors/frame/ exists")
        return
    for fp in frame_processors:
        if not fp.pre_check():
            print(f"[error] pre_check failed for {fp.NAME}")
            return
        if not fp.pre_start():
            print(f"[error] pre_start failed for {fp.NAME}")
            return

    # ── pre-analyse source face ────────────────────────────────────────────
    from modules.face_analyser import get_one_face, get_many_faces
    from modules import imread_unicode

    source_img = imread_unicode(gbl.source_path)
    if source_img is None:
        print(f"[error] Cannot read source image: {gbl.source_path}")
        return
    source_face = get_one_face(source_img)
    if source_face is None:
        print(f"[error] No face detected in source image: {gbl.source_path}")
        return
    print(f"[init] Source face loaded from {gbl.source_path}")

    # ── start camera + streamer ─────────────────────────────────────────────
    cam      = WebcamCapture(args.camera_index, args.width, args.height, args.fps)
    streamer = MJPEGStreamer(port=args.port, jpeg_quality=args.jpeg_quality)

    # ── graceful shutdown ───────────────────────────────────────────────────
    def _shutdown(sig=None, frame=None):
        print("\n[main] Shutting down …")
        cam.stop()
        streamer.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print("[main] Streaming started. Press Ctrl-C to stop.")

    # ── main processing loop ────────────────────────────────────────────────
    frame_count  = 0
    t0           = time.time()
    fps_interval = 5.0          # print FPS every N seconds
    det_count    = 0
    det_interval = max(1, round(args.fps * 0.08))  # re-detect face every N frames
    cached_target_face = None

    while True:
        bgr = cam.read()
        if bgr is None:
            time.sleep(0.005)
            continue

        # Deep-Live-Cam face_swapper.process_frame expects:
        #   process_frame(source_face: Face, temp_frame: ndarray, target_face: Face = None)
        # where source_face is a Face object from get_one_face(), NOT a file path.
        # The frame should be BGR (OpenCV format) — that's what the swapper and
        # insightface detection both operate on.
        temp_frame = bgr

        # Periodically re-detect target face(s) to avoid running detection every frame
        det_count += 1
        if det_count % det_interval == 0:
            if gbl.many_faces:
                cached_target_face = None  # many_faces mode detects internally
            else:
                cached_target_face = get_one_face(temp_frame)

        try:
            for fp in frame_processors:
                if fp.NAME == "DLC.FACE-SWAPPER":
                    if gbl.many_faces:
                        many_faces = get_many_faces(temp_frame)
                        if many_faces:
                            result = temp_frame.copy()
                            swapped_bboxes = []
                            for t_face in many_faces:
                                result = fp.swap_face(source_face, t_face, result)
                                if hasattr(t_face, "bbox") and t_face.bbox is not None:
                                    swapped_bboxes.append(t_face.bbox.astype(int))
                            temp_frame = fp.apply_post_processing(result, swapped_bboxes)
                        # else: no faces detected, pass through
                    elif cached_target_face is not None:
                        temp_frame = fp.process_frame(source_face, temp_frame, cached_target_face)
                    # else: no target face detected yet, pass through
                else:
                    # Other processors (face_enhancer etc.) take (source_face, frame)
                    temp_frame = fp.process_frame(source_face, temp_frame)
        except Exception as exc:
            # Don't crash the loop on a single bad frame (e.g. no face detected)
            print(f"[warn] process_frame error: {exc}")
            continue

        # Convert BGR→RGB for the MJPEG streamer (push() expects RGB)
        rgb = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)

        # Push to MJPEG clients
        streamer.push(rgb)

        # FPS counter
        frame_count += 1
        elapsed = time.time() - t0
        if elapsed >= fps_interval:
            print(f"[perf] {frame_count / elapsed:.1f} fps")
            frame_count = 0
            t0 = time.time()


if __name__ == '__main__':
    main()