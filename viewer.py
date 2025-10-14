import time
import os
import argparse
from datetime import datetime
import cv2

from capture import BackgroundCapture


def main():
    parser = argparse.ArgumentParser(description="Viewer at ~2 FPS, 800x600, with optional headless mode")
    parser.add_argument("--device", default="/dev/v4l/by-id/usb-Actions_Micro_UGREEN-25854_-1575730626-video-index0")
    parser.add_argument("--headless", action="store_true", help="Run without GUI; save frames to disk at ~2 FPS")
    parser.add_argument("--outdir", default="frames_out", help="Directory for saved frames in headless mode")
    parser.add_argument("--frames", type=int, default=10, help="Number of frames to save in headless mode")
    args = parser.parse_args()

    cap = BackgroundCapture(
        device_path=args.device,
        warmup_frames=45,
        target_fps=10.0,  # background read pace; we will display/save at ~2 FPS
    )
    cap.start()

    try:
        if args.headless or not os.environ.get("DISPLAY"):
            os.makedirs(args.outdir, exist_ok=True)
            saved = 0
            # Save ~2 FPS (500ms)
            while saved < max(1, args.frames):
                frame = cap.get_frame(timeout_sec=0.5)
                if frame is None:
                    continue
                # Resize to 800x600 for parity with GUI mode
                disp = cv2.resize(frame, (800, 600), interpolation=cv2.INTER_AREA)
                ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                out_path = os.path.join(args.outdir, f"frame_{ts}.png")
                cv2.imwrite(out_path, disp)
                print(f"Saved {out_path}")
                saved += 1
                time.sleep(0.5)
            return

        window_name = "Viewer 800x600 (2 FPS)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)

        while True:
            frame = cap.get_frame(timeout_sec=0.2)
            if frame is not None:
                # Resize to 800x600 for display only
                disp = cv2.resize(frame, (800, 600), interpolation=cv2.INTER_AREA)
                cv2.imshow(window_name, disp)

            # ~2 frames per second display
            key = cv2.waitKey(500)  # 500 ms
            if key == 27 or key == ord('q'):
                break

    finally:
        cap.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()


