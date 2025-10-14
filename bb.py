import cv2
import sys
import os
import argparse
import glob
import time
from datetime import datetime

# Configuration
DEVICE = "/dev/v4l/by-id/usb-Actions_Micro_UGREEN-25854_-1575730626-video-index0"  # Default video device (stable by-id symlink)
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_PATH = f"frame_{TIMESTAMP}.png"


def list_devices() -> None:
    """Print available V4L2 devices using convenient names if present."""
    by_id_dir = "/dev/v4l/by-id"
    if os.path.isdir(by_id_dir):
        print("[/dev/v4l/by-id]")
        for name in sorted(os.listdir(by_id_dir)):
            path = os.path.join(by_id_dir, name)
            try:
                target = os.path.realpath(path)
            except OSError:
                target = "<unresolvable>"
            if os.path.islink(path):
                print(f"{name} -> {target}")
        print()
    print("[/dev]")
    for dev in sorted(glob.glob("/dev/video*")):
        print(dev)
    print()


# Helpers
def capture_from_device_path(device_path: str, warmup_frames: int = 15, frames_to_save: int = 1, interval_ms: int = 0) -> str:
    """Attempt to capture a single frame from a specific device path.

    Returns the saved file path on success, or an empty string on failure.
    """
    print(f"Using device: {device_path} (script={__file__})")
    if not os.path.exists(device_path):
        print(f"Error: Video device {device_path} not found.")
        return ""

    # Initialize video capture with path only (avoid index fallbacks to prevent mislabeling)
    cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Error: Could not open {device_path} with V4L2. Trying default backend by path...")
        cap = cv2.VideoCapture(device_path, cv2.CAP_ANY)
    if not cap.isOpened():
        print(f"Error: Could not open video device {device_path} with any backend.")
        return ""

    # Try multiple formats and resolutions
    formats = [
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
        ('YUYV', cv2.VideoWriter_fourcc(*'YUYV'))
    ]
    resolutions = [(1920, 1080), (1280, 720), (640, 480)]

    ret, frame = False, None
    for fmt_name, fmt_code in formats:
        for width, height in resolutions:
            cap.set(cv2.CAP_PROP_FOURCC, fmt_code)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            # Warm up reads to let the real signal replace any device splash screen
            for _ in range(max(0, warmup_frames)):
                cap.read()
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"Success: Captured frame with {fmt_name} {width}x{height}")
                break
        if ret and frame is not None:
            break

    if not ret or frame is None:
        print("Error: Failed to capture frame. Ensure video source is active.")
        cap.release()
        return ""

    # Save one or multiple frames with device hint in the filename
    base = os.path.basename(device_path).replace('/', '_')
    last_output = ""
    for i in range(max(1, frames_to_save)):
        if i > 0 and interval_ms > 0:
            time.sleep(interval_ms / 1000.0)
            # grab next frame
            ret, frame = cap.read()
            if not ret or frame is None:
                break
        suffix = "" if frames_to_save == 1 else f"_f{i+1}"
        output_path = f"frame_{base}_{TIMESTAMP}{suffix}.png"
        cv2.imwrite(output_path, frame)
        print(f"Frame saved as {output_path} from {device_path}")
        last_output = output_path

    cap.release()
    return last_output


# CLI
parser = argparse.ArgumentParser(description="Capture a single frame from a V4L2 device")
parser.add_argument("--device", "-d", default=os.getenv("VIDEO_DEV", DEVICE), help="Device path or index (e.g., /dev/video0 or 0)")
parser.add_argument("--list", "-l", action="store_true", help="List V4L2 devices and exit")
parser.add_argument("--scan", "-s", action="store_true", help="Scan all /dev/video* devices and save a test frame from each")
parser.add_argument("--warmup", type=int, default=15, help="Warmup frames to read before saving (reduce splash frames)")
parser.add_argument("--frames", type=int, default=1, help="Number of frames to save per device")
parser.add_argument("--interval-ms", type=int, default=0, help="Delay between successive saved frames in milliseconds")
args = parser.parse_args()

if args.list:
    list_devices()
    sys.exit(0)

DEVICE = str(args.device)

if args.scan or DEVICE.lower() in {"all", "auto"}:
    # Scan all /dev/video* devices and attempt capture from each
    devices = sorted(glob.glob("/dev/video*"))
    if not devices:
        print("No /dev/video* devices found.")
        sys.exit(1)

    print("Scanning devices:")
    for d in devices:
        print(f" - {d}")
    print()

    saved = []
    for d in devices:
        path = capture_from_device_path(d, warmup_frames=max(5, args.warmup), frames_to_save=1, interval_ms=0)
        if path:
            saved.append((d, path))

    if not saved:
        print("No frames captured from any device.")
        sys.exit(1)

    print("\nSummary of captured frames:")
    for d, p in saved:
        print(f"{d} -> {p}")
    sys.exit(0)

# Single-device mode
if not os.path.exists(DEVICE):
    print(f"Error: Video device {DEVICE} not found.")
    sys.exit(1)

out = capture_from_device_path(DEVICE, warmup_frames=args.warmup, frames_to_save=args.frames, interval_ms=args.interval_ms) 
if not out:
    sys.exit(1)
