#!/usr/bin/env python3
"""
transfer_pose.py — Pose-transfer face swap

Takes:
  -s  / --source     reference face image  (whose identity to use)
  --input-pose       video A  (pose donor  — supplies head orientation)
  --input-target     video B  (swap target — whose face region to replace)

For every pair of frames (A_i, B_i) the result is:
  • Source identity
  • Head pose / expression from video A
  • Composited at the face location in video B

Usage examples
──────────────
  # Two webcams (indices 0 and 1)
  python transfer_pose.py -s ref.png --input-pose 0 --input-target 1

  # One webcam as pose, a pre-recorded clip as target
  python transfer_pose.py -s ref.png --input-pose 0 --input-target actor.mp4

  # Two video files, write output
  python transfer_pose.py -s ref.png --input-pose pose.mp4 --input-target target.mp4 \\
      --output out.mp4

  # Stream to browser
  python transfer_pose.py -s ref.png --input-pose 0 --input-target 1 --stream mjpeg

Extra options (mirrors deepfake.py)
────────────────────────────────────
  --execution-provider  cuda | cpu  (default: auto)
  --execution-threads   int
  --max-memory          GB (default 16)
  --frame-processor     face_swapper [face_enhancer …]
  --many-faces
  --width / --height / --fps        (live capture only)
  --port                            (MJPEG port, default 8080)
  --jpeg-quality                    (MJPEG quality, default 85)
  --det-interval        frames between face re-detections (default: auto)
"""

# ── project root bootstrap ───────────────────────────────────────────────────
import os, sys
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.environ["PATH"] = _PROJECT_ROOT + os.pathsep + os.environ.get("PATH", "")

# ── CUDA / cuDNN pre-load ────────────────────────────────────────────────────
import ctypes, glob as _glob, site as _site
_cudnn_search_dirs = []
for _base in _site.getsitepackages() + [_site.getusersitepackages()]:
    if isinstance(_base, str):
        _cudnn_search_dirs.append(os.path.join(_base, "nvidia", "cudnn", "lib"))
_cudnn_search_dirs.append(
    "/usr/local/lib/python{}.{}/dist-packages/nvidia/cudnn/lib".format(
        sys.version_info.major, sys.version_info.minor
    )
)
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

if any(a.startswith("--execution-provider") for a in sys.argv):
    os.environ["OMP_NUM_THREADS"] = "6"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time, signal, argparse, threading, warnings, copy
import numpy as np
import cv2
import onnxruntime

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import tensorflow
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
if HAS_TORCH:
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


# ══════════════════════════════════════════════════════════════════════════════
# Provider helpers  (identical to deepfake.py)
# ══════════════════════════════════════════════════════════════════════════════

def encode_execution_providers(providers):
    return [p.replace("ExecutionProvider", "").lower() for p in providers]


def decode_execution_providers(names):
    available = onnxruntime.get_available_providers()
    encoded   = encode_execution_providers(available)
    return [p for p, e in zip(available, encoded) if any(n in e for n in names)]


def suggest_default_execution_provider() -> str:
    available = encode_execution_providers(onnxruntime.get_available_providers())
    for pref in ("cuda", "rocm", "coreml", "dml"):
        if pref in available:
            return pref
    return "cpu"


def suggest_execution_providers():
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads(providers) -> int:
    if "DmlExecutionProvider"  in providers: return 1
    if "ROCMExecutionProvider" in providers: return 1
    if "CUDAExecutionProvider" in providers: return 2
    cpu_count = os.cpu_count() or 4
    return max(4, min(cpu_count - 2, 16))


def limit_resources(max_memory: int) -> None:
    if HAS_TENSORFLOW:
        gpus = tensorflow.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tensorflow.config.experimental.set_memory_growth(gpu, True)
    if max_memory:
        memory = max_memory * 1024 ** 3
        import platform
        if platform.system().lower() == "windows":
            k32 = ctypes.windll.kernel32
            k32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


# ══════════════════════════════════════════════════════════════════════════════
# MJPEG streamer  (identical to deepfake.py)
# ══════════════════════════════════════════════════════════════════════════════

class MJPEGStreamer:
    def __init__(self, port: int = 8080, jpeg_quality: int = 85):
        from http.server import BaseHTTPRequestHandler, HTTPServer

        self.port    = port
        self.quality = jpeg_quality
        self._frame: bytes = b""
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        streamer = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *_): pass

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
                        "multipart/x-mixed-replace; boundary=frame",
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
# Video capture  (webcam index or file/RTSP URL)
# ══════════════════════════════════════════════════════════════════════════════

class VideoCapture:
    """
    Non-blocking reader for a webcam index or video file / RTSP stream.

    For live sources (webcam / RTSP) frames are grabbed in a background thread
    so the caller always gets the *latest* frame without stalling.
    For file sources the reader is synchronous (called directly from the main
    loop) to honour the original frame sequence.
    """

    def __init__(self, src, width: int = 640, height: int = 480, fps: int = 30):
        # Determine if src is a live source (int index or rtsp://) or a file.
        self._is_live = isinstance(src, int) or (
            isinstance(src, str) and src.lower().startswith("rtsp://")
        )

        if isinstance(src, int):
            backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
            self._cap = cv2.VideoCapture(src, backend)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(src)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_FPS,          fps)
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self._cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        else:
            self._cap = cv2.VideoCapture(src)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {src!r}")

        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f = self._cap.get(cv2.CAP_PROP_FPS) or fps
        print(f"[capture] Opened {src!r}  {w}×{h}  {f:.0f} fps  live={self._is_live}")

        self._frame: np.ndarray | None = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        self._eof   = False

        if self._is_live:
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()

    # ── background reader (live sources only) ────────────────────────────────

    def _reader(self):
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.002)

    # ── public API ────────────────────────────────────────────────────────────

    def read(self) -> np.ndarray | None:
        """Return next BGR frame.  None if not yet available or EOF."""
        if self._is_live:
            with self._lock:
                return None if self._frame is None else self._frame.copy()
        else:
            ok, frame = self._cap.read()
            if not ok:
                self._eof = True
                return None
            return frame

    @property
    def eof(self) -> bool:
        return self._eof

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    def stop(self):
        self._stop.set()
        if self._is_live:
            self._thread.join(timeout=2)
        self._cap.release()


# ══════════════════════════════════════════════════════════════════════════════
# FFmpeg writer  (for file output, VS Code / browser compatible)
# ══════════════════════════════════════════════════════════════════════════════

class FFmpegWriter:
    def __init__(self, path: str, width: int, height: int, fps: float):
        import subprocess
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}", "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "pipe:0",
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-crf", "18",
            path,
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.width, self.height = width, height
        print(f"[writer] Writing to {path!r}  {width}×{height}  {fps:.1f} fps")

    def write(self, bgr: np.ndarray):
        if bgr.shape[:2] != (self.height, self.width):
            bgr = cv2.resize(bgr, (self.width, self.height))
        self._proc.stdin.write(bgr.tobytes())

    def close(self):
        self._proc.stdin.close()
        self._proc.wait()
        print("[writer] Done.")


# ══════════════════════════════════════════════════════════════════════════════
# Pose transplant
# ══════════════════════════════════════════════════════════════════════════════

def transplant_pose(pose_face, target_face):
    """
    Return a *new* face object whose identity slot is taken from `pose_face`
    (embedding, kps, landmark geometry — everything the swap model uses to
    determine head pose / expression) but whose BOUNDING BOX is replaced with
    `target_face`'s bounding box.

    Why this matters
    ────────────────
    insightface's INSwapper.get() crops the input frame using the face's 5-point
    kps (keypoints) to compute an affine warp M.  The cropped region is fed to
    the swap model; the model therefore "sees" the pose encoded in the kps, not
    the raw pixel content.  The affine M also determines where the result is
    pasted back.

    By using pose_face.kps for the crop/pose but repositioning those kps to sit
    at target_face's location we get:
      • Source identity rendered with A's head pose
      • Composited at the correct face location in B's frame

    The repositioning is a rigid 2-D similarity transform:
      1. Translate pose_face.kps so their centroid sits at target_face's centroid
      2. Scale so the inter-ocular span matches target_face's inter-ocular span
         (preserves relative landmark geometry = pose; only changes absolute size)
    """
    if pose_face is None or target_face is None:
        return target_face  # fallback: no pose info, swap normally

    try:
        synthetic = copy.copy(pose_face)   # shallow copy — shares normed_embedding

        # ── 5-point keypoints from both faces ───────────────────────────────
        # insightface kps layout: [left_eye, right_eye, nose, left_mouth, right_mouth]
        p_kps = np.array(pose_face.kps,   dtype=np.float32)  # (5, 2)
        t_kps = np.array(target_face.kps, dtype=np.float32)  # (5, 2)

        # ── centroid alignment ───────────────────────────────────────────────
        p_center = p_kps.mean(axis=0)
        t_center = t_kps.mean(axis=0)

        # ── scale: match inter-ocular distance (eyes are indices 0 and 1) ───
        p_iod = np.linalg.norm(p_kps[1] - p_kps[0]) + 1e-6
        t_iod = np.linalg.norm(t_kps[1] - t_kps[0]) + 1e-6
        scale = t_iod / p_iod

        # ── build transformed kps ────────────────────────────────────────────
        new_kps = (p_kps - p_center) * scale + t_center

        synthetic.kps  = new_kps
        synthetic.bbox = target_face.bbox.copy()   # paste location from B

        # Transplant 2D landmarks for mouth-mask / Poisson blend helpers
        if (
            hasattr(pose_face, "landmark_2d_106")
            and pose_face.landmark_2d_106 is not None
            and hasattr(target_face, "landmark_2d_106")
            and target_face.landmark_2d_106 is not None
        ):
            p_lm = np.array(pose_face.landmark_2d_106,   dtype=np.float32)
            t_lm = np.array(target_face.landmark_2d_106, dtype=np.float32)
            p_lm_center = p_lm.mean(axis=0)
            t_lm_center = t_lm.mean(axis=0)
            # Reuse same scale (consistent with kps rescaling)
            synthetic.landmark_2d_106 = (p_lm - p_lm_center) * scale + t_lm_center

        return synthetic

    except Exception as exc:
        print(f"[warn] transplant_pose failed ({exc}), falling back to target_face")
        return target_face


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def parse_args():
    ap = argparse.ArgumentParser(
        description="Pose-transfer face swap: source id + pose from A → composited on B"
    )
    ap.add_argument("-s", "--source",      required=True,
                    help="Source face image (identity)")
    ap.add_argument("--input-pose",        required=True,
                    help="Video A: pose donor  (webcam index or file/RTSP URL)")
    ap.add_argument("--input-target",      required=True,
                    help="Video B: swap target (webcam index or file/RTSP URL)")
    ap.add_argument("--output",            default=None,
                    help="Output video file path (optional; auto-named if omitted)")
    ap.add_argument("--stream",            choices=["mjpeg", "window", "none"],
                    default="none",
                    help="Live preview: mjpeg (browser), window (cv2), none")
    # Capture params (live sources)
    ap.add_argument("--width",             type=int, default=640)
    ap.add_argument("--height",            type=int, default=480)
    ap.add_argument("--fps",               type=int, default=30)
    # MJPEG params
    ap.add_argument("--port",              type=int, default=8080)
    ap.add_argument("--jpeg-quality",      type=int, default=85)
    # Frame processors
    ap.add_argument("--frame-processor",   nargs="+",
                    default=["face_swapper"],
                    choices=["face_swapper", "face_enhancer",
                             "face_enhancer_gpen256", "face_enhancer_gpen512"],
                    dest="frame_processor")
    ap.add_argument("--many-faces",        action="store_true", default=False)
    ap.add_argument("--det-interval",      type=int, default=None,
                    help="Re-detect faces every N frames (default: auto from --fps)")
    # Execution
    ap.add_argument("--execution-provider", nargs="+",
                    default=[suggest_default_execution_provider()],
                    choices=suggest_execution_providers(),
                    dest="execution_provider")
    ap.add_argument("--execution-threads", type=int, default=None,
                    dest="execution_threads")
    ap.add_argument("--max-memory",        type=int, default=16,
                    dest="max_memory")
    return ap.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── convert string indices to int ────────────────────────────────────────
    src_pose   = int(args.input_pose)   if _is_int(args.input_pose)   else args.input_pose
    src_target = int(args.input_target) if _is_int(args.input_target) else args.input_target

    # ── configure globals ────────────────────────────────────────────────────
    import modules.globals as gbl
    from modules.processors.frame.core import get_frame_processors_modules

    gbl.source_path        = args.source
    gbl.target_path        = None
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
    gbl.video_encoder      = "libx264"
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
    if not hasattr(gbl, "fp_ui") or gbl.fp_ui is None:
        gbl.fp_ui = {}
    for key in ("face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512"):
        gbl.fp_ui[key] = key in args.frame_processor

    print(
        f"[config] providers={gbl.execution_providers}  "
        f"threads={gbl.execution_threads}  "
        f"processors={gbl.frame_processors}"
    )
    limit_resources(args.max_memory)

    # ── load processors ──────────────────────────────────────────────────────
    import importlib
    for _proc_name in gbl.frame_processors:
        _mod_path = f"modules.processors.frame.{_proc_name}"
        try:
            importlib.import_module(_mod_path)
            print(f"[debug] import OK: {_mod_path}")
        except ImportError as _e:
            print(f"[error] Cannot import {_mod_path}: {_e}")
            return
        except Exception as _e:
            print(f"[debug] import warning for {_mod_path}: {type(_e).__name__}: {_e}")

    frame_processors = get_frame_processors_modules(gbl.frame_processors)
    if not frame_processors:
        print("[error] No frame processors loaded")
        return
    for fp in frame_processors:
        if not fp.pre_check():
            print(f"[error] pre_check failed for {fp.NAME}")
            return
        if not fp.pre_start():
            print(f"[error] pre_start failed for {fp.NAME}")
            return

    # ── pre-load source face ─────────────────────────────────────────────────
    from modules.face_analyser import get_one_face, get_many_faces
    from modules import imread_unicode

    source_img = imread_unicode(gbl.source_path)
    if source_img is None:
        print(f"[error] Cannot read source image: {gbl.source_path}")
        return
    source_face = get_one_face(source_img)
    if source_face is None:
        print(f"[error] No face in source image: {gbl.source_path}")
        return
    print(f"[init] Source face loaded from {gbl.source_path}")

    # ── open captures ────────────────────────────────────────────────────────
    cap_pose   = VideoCapture(src_pose,   args.width, args.height, args.fps)
    cap_target = VideoCapture(src_target, args.width, args.height, args.fps)

    # ── optional output writer ────────────────────────────────────────────────
    writer = None
    if args.output:
        out_path = args.output
    else:
        # auto-name only when both inputs are files and user didn't specify --output
        if not isinstance(src_target, int) and not str(src_target).startswith("rtsp://"):
            base = os.path.splitext(os.path.basename(str(src_target)))[0]
            out_path = f"{base}_pose_transfer.mp4"
        else:
            out_path = None
    # Writer is created lazily on first frame (need to know frame size)

    # ── optional streamer ────────────────────────────────────────────────────
    streamer = None
    if args.stream == "mjpeg":
        streamer = MJPEGStreamer(port=args.port, jpeg_quality=args.jpeg_quality)

    # ── graceful shutdown ─────────────────────────────────────────────────────
    def _shutdown(sig=None, frame=None):
        print("\n[main] Shutting down …")
        cap_pose.stop()
        cap_target.stop()
        if streamer:
            streamer.stop()
        if writer:
            writer.close()
        if args.stream == "window":
            cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print("[main] Processing started. Press Ctrl-C to stop.")

    # ── detection cadence ─────────────────────────────────────────────────────
    det_interval = args.det_interval or max(1, round(args.fps * 0.08))
    det_count    = 0
    cached_pose_face   = None   # face from video A
    cached_target_face = None   # face from video B

    # ── perf counter ─────────────────────────────────────────────────────────
    frame_count  = 0
    t0           = time.time()
    fps_interval = 5.0

    while True:
        bgr_pose   = cap_pose.read()
        bgr_target = cap_target.read()

        # For file sources: stop when either stream is exhausted
        if cap_pose.eof or cap_target.eof:
            print("[main] End of input — exiting.")
            break

        # Skip if either stream hasn't produced a frame yet (live startup)
        if bgr_pose is None or bgr_target is None:
            time.sleep(0.005)
            continue

        temp_frame = bgr_target   # we write the result into a copy of B

        # ── periodic face re-detection ───────────────────────────────────────
        det_count += 1
        if det_count % det_interval == 0:
            if gbl.many_faces:
                cached_pose_face   = None   # many_faces mode detects inside the loop
                cached_target_face = None
            else:
                cached_pose_face   = get_one_face(bgr_pose)
                cached_target_face = get_one_face(bgr_target)

        try:
            for fp in frame_processors:
                if fp.NAME == "DLC.FACE-SWAPPER":
                    if gbl.many_faces:
                        # ── multi-face: pair each detected face in B with the
                        #    first detected face in A (best we can do without
                        #    explicit correspondence tracking)
                        pose_faces   = get_many_faces(bgr_pose)
                        target_faces = get_many_faces(bgr_target)
                        pose_face_ref = pose_faces[0] if pose_faces else None

                        if target_faces and pose_face_ref is not None:
                            result = temp_frame.copy()
                            swapped_bboxes = []
                            for t_face in target_faces:
                                synthetic = transplant_pose(pose_face_ref, t_face)
                                result = fp.swap_face(source_face, synthetic, result)
                                if hasattr(t_face, "bbox") and t_face.bbox is not None:
                                    swapped_bboxes.append(t_face.bbox.astype(int))
                            temp_frame = fp.apply_post_processing(result, swapped_bboxes)

                    elif cached_pose_face is not None and cached_target_face is not None:
                        # ── single-face path ─────────────────────────────────
                        # Build synthetic face: A's pose transplanted to B's location
                        synthetic = transplant_pose(cached_pose_face, cached_target_face)
                        temp_frame = fp.process_frame(source_face, temp_frame, synthetic)

                    # else: no faces detected yet — pass frame through unchanged

                else:
                    # face_enhancer and friends only need (source_face, frame)
                    temp_frame = fp.process_frame(source_face, temp_frame)

        except Exception as exc:
            print(f"[warn] process_frame error: {exc}")
            continue

        # ── output ───────────────────────────────────────────────────────────
        h, w = temp_frame.shape[:2]

        # Lazy-init file writer
        if out_path and writer is None:
            fps_out = cap_target.fps if not isinstance(src_target, int) else args.fps
            writer  = FFmpegWriter(out_path, w, h, fps_out)

        if writer:
            writer.write(temp_frame)

        if streamer:
            rgb = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
            streamer.push(rgb)

        if args.stream == "window":
            cv2.imshow("transfer_pose", temp_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # ── FPS counter ───────────────────────────────────────────────────────
        frame_count += 1
        elapsed = time.time() - t0
        if elapsed >= fps_interval:
            print(f"[perf] {frame_count / elapsed:.1f} fps")
            frame_count = 0
            t0 = time.time()

    _shutdown()


if __name__ == "__main__":
    main()