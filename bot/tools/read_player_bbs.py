import os
import sys
import argparse
import json
from typing import Dict, List, Optional, Tuple
import time
import cv2
import numpy as np

# Ensure project root on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAND_ROOT = os.path.dirname(ROOT)
for path in (ROOT, GRAND_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from capture import BackgroundCapture
from bot.core.roi import load_tables_config
from bot.core.matchers import nms_boxes


DEFAULT_SEAT_BB_ROIS: Dict[int, Tuple[int, int, int, int]] = {
    0: (209, 183, 90, 8),
    1: (423, 134, 90, 8),
    2: (400, 168, 90, 8),
    3: (420, 289, 90, 8),
    4: (320, 318, 90, 8),
    5: (200, 289, 90, 8),
}


def _load_gray_with_optional_mask(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None
    mask = None
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)[1]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return gray, mask


def _match_score_and_loc(image_gray, tmpl_gray, mask=None):
    try:
        res = cv2.matchTemplate(image_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED, mask=mask)
    except Exception:
        return -1.0, None
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return float(max_val), (int(max_loc[0]), int(max_loc[1]))


def _load_rois_from_json(path: Optional[str]) -> Optional[Dict[int, Tuple[int, int, int, int]]]:
    if not path or not os.path.isfile(path):
        return None
    with open(path, 'r') as f:
        data = json.load(f)
    rois: Dict[int, Tuple[int, int, int, int]] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                sk = int(k)
            except Exception:
                continue
            if isinstance(v, (list, tuple)) and len(v) == 4:
                rois[sk] = (int(v[0]), int(v[1]), int(v[2]), int(v[3]))
    elif isinstance(data, list):
        for idx, v in enumerate(data):
            if isinstance(v, (list, tuple)) and len(v) == 4:
                rois[idx] = (int(v[0]), int(v[1]), int(v[2]), int(v[3]))
    return rois if rois else None


def _load_existing_templates(outdir: str) -> List:
    existing = []
    if not os.path.isdir(outdir):
        return existing
    for name in os.listdir(outdir):
        path = os.path.join(outdir, name)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        existing.append(img)
    return existing


def _is_duplicate_by_tm(patch_gray, existing_gray_list, threshold: float) -> bool:
    for tmpl in existing_gray_list:
        if tmpl.shape != patch_gray.shape:
            continue
        try:
            res = cv2.matchTemplate(patch_gray, tmpl, cv2.TM_CCOEFF_NORMED)
        except Exception:
            continue
        score = float(res[0, 0])
        if score >= threshold:
            return True
    return False


def _scan_number_simple_debug(crop_gray, dot_tmpl, digit_tmpls: Dict[str, Tuple[any, Optional[any]]], threshold: float, min_var: float, debug_dir: str, step_prefix: str) -> Optional[str]:
    """Debug version that saves every scanning step"""
    try:
        os.makedirs(debug_dir, exist_ok=True)
        step_count = 0
        rw = crop_gray.shape[1]
        x_scan = 0
        end_limit = rw
        chars: List[str] = []
        dot_found = False
        while x_scan < end_limit:
            matched = False
            for width in range(1, end_limit - x_scan + 1):
                slice_img = crop_gray[:, x_scan:x_scan+width]
                
                # Skip slices that are too small (will cause infinite scores)
                if slice_img.size == 0:
                    continue
                
                # digits first - try all templates for each digit
                best_d = None
                best_s = -1.0
                best_lx = 0
                best_w = 0
                for d, tmpl_list in digit_tmpls.items():
                    for dt, dm in tmpl_list:
                        # Skip if slice is smaller than template (will cause infinite scores)
                        if slice_img.shape[0] < dt.shape[0] or slice_img.shape[1] < dt.shape[1]:
                            continue
                        try:
                            res = cv2.matchTemplate(slice_img, dt, cv2.TM_CCOEFF_NORMED, mask=dm) if dm is not None else cv2.matchTemplate(slice_img, dt, cv2.TM_CCOEFF_NORMED)
                        except Exception:
                            res = None
                        if res is None:
                            continue
                        _, sc, _, loc = cv2.minMaxLoc(res)
                        if sc > best_s and loc is not None:
                            best_s = float(sc)
                            best_d = d
                            best_lx = int(loc[0])
                            best_w = dt.shape[1]
                
                # Save this step
                step_count += 1
                fname = f"{step_prefix}_step{step_count:03d}_x{x_scan}_w{width}"
                if best_d is not None:
                    pix_var = 0.0
                    if min_var > 0.0:
                        abs_x = x_scan + best_lx
                        glyph = crop_gray[:, abs_x:abs_x + best_w]
                        if glyph.size > 0:
                            _, stddev = cv2.meanStdDev(glyph)
                            pix_var = float(stddev.mean()**2)
                    
                    # Handle non-finite scores
                    if not np.isfinite(best_s):
                        score_str = "inf" if np.isinf(best_s) else "nan"
                        fname += f"_DIGIT_{best_d}_score{score_str}_var{pix_var:.1f}_INVALID"
                    elif best_s >= threshold and (min_var == 0.0 or pix_var >= min_var):
                        fname += f"_DIGIT_{best_d}_score{best_s:.3f}_var{pix_var:.1f}_ACCEPT"
                        matched = True
                    else:
                        fname += f"_DIGIT_{best_d}_score{best_s:.3f}_var{pix_var:.1f}_REJECT"
                else:
                    fname += "_NO_MATCH"
                
                fname += ".png"
                cv2.imwrite(os.path.join(debug_dir, fname), slice_img)
                
                if best_d is not None and np.isfinite(best_s) and best_s >= threshold:
                    abs_x = x_scan + best_lx
                    if min_var > 0.0:
                        glyph = crop_gray[:, abs_x:abs_x + best_w]
                        if glyph.size == 0:
                            pix_var = 0.0
                        else:
                            _, stddev = cv2.meanStdDev(glyph)
                            pix_var = float(stddev.mean()**2)
                        if pix_var < min_var:
                            x_scan = abs_x + max(1, best_w // 2)
                            continue
                    chars.append(best_d)
                    x_scan = abs_x + best_w + 1
                    matched = True
                    break
                # single dot only after at least one digit
                if not matched and not dot_found and dot_tmpl is not None and len(chars) > 0:
                    try:
                        resd = cv2.matchTemplate(slice_img, dot_tmpl, cv2.TM_CCOEFF_NORMED)
                    except Exception:
                        resd = None
                    if resd is not None:
                        _, ds, _, dl = cv2.minMaxLoc(resd)
                        if np.isfinite(ds) and ds >= threshold and dl is not None:
                            lx = int(dl[0])
                            abs_x = x_scan + lx
                            dw = dot_tmpl.shape[1]
                            if min_var > 0.0:
                                glyph = crop_gray[:, abs_x:abs_x + dw]
                                if glyph.size == 0:
                                    pix_var = 0.0
                                else:
                                    _, stddev = cv2.meanStdDev(glyph)
                                    pix_var = float(stddev.mean()**2)
                                if pix_var < min_var:
                                    x_scan = abs_x + max(1, dw // 2)
                                    continue
                            chars.append('.')
                            x_scan = abs_x + dw + 1
                            matched = True
                            dot_found = True
                            break
            if not matched:
                break
        if chars:
            return ''.join(chars)
        return None
    except Exception:
        return None


def _scan_number_simple(crop_gray, dot_tmpl, digit_tmpls: Dict[str, Tuple[any, Optional[any]]], threshold: float, min_var: float = 0.0) -> Optional[str]:
    try:
        rw = crop_gray.shape[1]
        x_scan = 0
        end_limit = rw
        chars: List[str] = []
        dot_found = False
        while x_scan < end_limit:
            matched = False
            for width in range(1, end_limit - x_scan + 1):
                slice_img = crop_gray[:, x_scan:x_scan+width]
                
                # Skip slices that are too small
                if slice_img.size == 0:
                    continue
                
                # digits first - try all templates for each digit
                best_d = None
                best_s = -1.0
                best_lx = 0
                best_w = 0
                for d, tmpl_list in digit_tmpls.items():
                    for dt, dm in tmpl_list:
                        # Skip if slice is smaller than template (prevents infinite scores)
                        if slice_img.shape[0] < dt.shape[0] or slice_img.shape[1] < dt.shape[1]:
                            continue
                        try:
                            res = cv2.matchTemplate(slice_img, dt, cv2.TM_CCOEFF_NORMED, mask=dm) if dm is not None else cv2.matchTemplate(slice_img, dt, cv2.TM_CCOEFF_NORMED)
                        except Exception:
                            res = None
                        if res is None:
                            continue
                        _, sc, _, loc = cv2.minMaxLoc(res)
                        if sc > best_s and loc is not None:
                            best_s = float(sc)
                            best_d = d
                            best_lx = int(loc[0])
                            best_w = dt.shape[1]
                if best_d is not None and best_s >= threshold:
                    abs_x = x_scan + best_lx
                    if min_var > 0.0:
                        glyph = crop_gray[:, abs_x:abs_x + best_w]
                        if glyph.size == 0 or float(cv2.meanStdDev(glyph)[1].mean()**2) < min_var:
                            x_scan = abs_x + max(1, best_w // 2)
                            continue
                    chars.append(best_d)
                    x_scan = abs_x + best_w + 1
                    matched = True
                    break
                # single dot only after at least one digit
                if not matched and not dot_found and dot_tmpl is not None and len(chars) > 0:
                    try:
                        resd = cv2.matchTemplate(slice_img, dot_tmpl, cv2.TM_CCOEFF_NORMED)
                    except Exception:
                        resd = None
                    if resd is not None:
                        _, ds, _, dl = cv2.minMaxLoc(resd)
                        if ds >= threshold and dl is not None:
                            lx = int(dl[0])
                            abs_x = x_scan + lx
                            dw = dot_tmpl.shape[1]
                            if min_var > 0.0:
                                glyph = crop_gray[:, abs_x:abs_x + dw]
                                if glyph.size == 0 or float(cv2.meanStdDev(glyph)[1].mean()**2) < min_var:
                                    x_scan = abs_x + max(1, dw // 2)
                                    continue
                            chars.append('.')
                            x_scan = abs_x + dw + 1
                            matched = True
                            dot_found = True
                            break
            if not matched:
                break
        if chars:
            return ''.join(chars)
        return None
    except Exception:
        return None


 


def main():
    parser = argparse.ArgumentParser(description='Detect per-seat BB label and save the amount image to its left within provided ROIs')
    parser.add_argument('--device', default='/dev/v4l/by-id/usb-Actions_Micro_UGREEN-25854_-1575730626-video-index0')
    parser.add_argument('--tables', default='bot/config/tables.json')
    parser.add_argument('--table-id', type=int, default=0)
    parser.add_argument('--templates', default='bot/templates/pot', help='Directory with bb.png for BB label (fallbacks enabled)')
    parser.add_argument('--outdir', default='bot/templates/player_bb_values', help='Directory to save per-seat amount crops')
    parser.add_argument('--tm-threshold', type=float, default=0.995, help='Dedup same-size crops by template match')
    parser.add_argument('--bb-threshold', type=float, default=0.99)
    parser.add_argument('--fps', type=float, default=60.0)
    parser.add_argument('--numbers-dir', default='bot/templates/player_bb/numbers', help='Directory with dot.png and 0-9.png for parsing amounts')
    parser.add_argument('--digit-threshold', type=float, default=0.95)
    parser.add_argument('--no-save', action='store_true', help='Do not save cropped images, only print detected numbers')
    parser.add_argument('--left-pad', type=int, default=30, help='Extra pixels to include to the left of seat ROI to avoid truncating numbers')
    parser.add_argument('--debug-bb', action='store_true', help='Save full table frames with BB positions/ROIs drawn')
    parser.add_argument('--debug-bb-dir', default='bot/frames_out/bb_debug', help='Directory to save BB debug frames')
    parser.add_argument('--scan-width', type=int, default=70, help='Pixels to scan to the left of BB for digits (narrow band)')
    parser.add_argument('--debug-bb-on-match', action='store_true', help='Also save debug frame when a BB is matched, even if value not changed')
    parser.add_argument('--min-digit-threshold', type=float, default=0.97, help='Fallback threshold for digits/dot if no value detected')
    parser.add_argument('--min-digit-var', type=float, default=15.0, help='Reject glyph matches below this pixel variance (helps ignore flat dark regions)')
    parser.add_argument('--save-tries-seats', default=None, help='Comma-separated seat numbers to save match attempts for (e.g. "2,4") or "all"')
    parser.add_argument('--save-tries-dir', default='bot/frames_out/seat_tries', help='Directory to save per-seat match attempt images')
    parser.add_argument('--save-all-attempts', action='store_true', help='Save every frame attempt for debugging, even when BB not detected or value is null')
    parser.add_argument('--save-all-dir', default='bot/frames_out/all_attempts', help='Directory to save all attempt frames when --save-all-attempts is enabled')
    parser.add_argument('--input-folder', default=None, help='Process images from folder instead of live capture (images should be cropped ROI images)')
    parser.add_argument('--input-folder-seat', type=int, default=None, help='When using --input-folder, specify which seat these images belong to (required for folder mode)')
    parser.add_argument('--debug-scan-steps', action='store_true', help='Save every scanning step for debugging (creates subfolder per image with all scan attempts)')
    parser.add_argument('--debug-scan-dir', default='bot/frames_out/scan_steps', help='Directory to save scan step debugging images')
    args = parser.parse_args()

    # Load ROIs: prefer config tables.json landmarks.bb_text_rois, fallback to defaults
    profiles = load_tables_config(args.tables)
    profile = next((p for p in profiles if getattr(p, 'table_id', None) == args.table_id), None)
    if profile is None or profile.roi.w <= 0 or profile.roi.h <= 0:
        raise SystemExit('Invalid or missing table ROI for requested table-id')
    seat_rois: Dict[int, Tuple[int, int, int, int]] = {}
    src = profile.landmarks.get('bb_text_rois') if isinstance(profile.landmarks, dict) else None
    if isinstance(src, dict):
        for k, v in src.items():
            try:
                sk = int(k)
            except Exception:
                continue
            if isinstance(v, (list, tuple)) and len(v) == 4:
                seat_rois[sk] = (int(v[0]), int(v[1]), int(v[2]), int(v[3]))
    if not seat_rois:
        seat_rois = dict(DEFAULT_SEAT_BB_ROIS)

    # profile already loaded

    # Load templates (BB label); try provided dir, then common fallbacks
    bb_tmpl, bb_mask = _load_gray_with_optional_mask(os.path.join(args.templates, 'bb.png'))
    if bb_tmpl is None:
        fallback_paths = [
            'bot/templates/pot/bb.png',
            'bot/templates/player_bb/bb.png',
        ]
        for fp in fallback_paths:
            bb_tmpl, bb_mask = _load_gray_with_optional_mask(fp)
            if bb_tmpl is not None:
                break
    if bb_tmpl is None:
        raise SystemExit('Missing bb.png template (checked --templates/bb.png, bot/templates/pot/bb.png, bot/templates/player_bb/bb.png)')

    # Load number templates (dot + digits)
    dot_tmpl, dot_mask = _load_gray_with_optional_mask(os.path.join(args.numbers_dir, 'dot.png'))
    
    # Load digit templates - support multiple templates per digit (e.g., 2.png, 22.png, 222.png)
    digit_tmpls: Dict[str, List[Tuple[any, Optional[any]]]] = {}
    if os.path.isdir(args.numbers_dir):
        for fname in os.listdir(args.numbers_dir):
            if fname.endswith('.png') and fname != 'dot.png':
                # Extract first character as the digit (e.g., "2.png" -> "2", "22.png" -> "2")
                digit = fname[0]
                if digit in '0123456789':
                    dt, dm = _load_gray_with_optional_mask(os.path.join(args.numbers_dir, fname))
                    if dt is not None:
                        if digit not in digit_tmpls:
                            digit_tmpls[digit] = []
                        digit_tmpls[digit].append((dt, dm))
                        print(f"Loaded digit template: {fname} -> digit '{digit}'")
    
    print(f"Loaded {sum(len(tmpls) for tmpls in digit_tmpls.values())} digit templates for {len(digit_tmpls)} digits")

    # Prepare per-seat output dirs and dedupe caches
    os.makedirs(args.outdir, exist_ok=True)
    seat_dirs: Dict[int, str] = {}
    seat_existing_gray: Dict[int, List] = {}
    for seat in seat_rois.keys():
        sd = os.path.join(args.outdir, f'seat{seat}')
        os.makedirs(sd, exist_ok=True)
        seat_dirs[seat] = sd
        seat_existing_gray[seat] = _load_existing_templates(sd)

    # Parse save-tries seats option
    save_tries_targets = None
    if args.save_tries_seats:
        val = str(args.save_tries_seats).strip().lower()
        if val == 'all':
            save_tries_targets = 'all'
        else:
            try:
                save_tries_targets = set(int(x.strip()) for x in val.split(',') if x.strip() != '')
            except Exception:
                save_tries_targets = None

    # Prepare save-all-attempts directory if enabled
    if args.save_all_attempts:
        os.makedirs(args.save_all_dir, exist_ok=True)

    # Handle folder input mode
    if args.input_folder:
        if args.input_folder_seat is None:
            raise SystemExit('--input-folder-seat is required when using --input-folder')
        
        if not os.path.isdir(args.input_folder):
            raise SystemExit(f'Input folder does not exist: {args.input_folder}')
        
        # Get all image files from folder
        image_files = []
        for fname in sorted(os.listdir(args.input_folder)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(args.input_folder, fname))
        
        if not image_files:
            raise SystemExit(f'No image files found in: {args.input_folder}')
        
        print(f"Processing {len(image_files)} images from folder for seat {args.input_folder_seat}")
        
        # Process each image
        for img_path in image_files:
            try:
                # Load image (already cropped ROI)
                roi_bgr = cv2.imread(img_path)
                if roi_bgr is None:
                    print(f"Warning: Could not load {img_path}")
                    continue
                
                roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
                seat = args.input_folder_seat
                
                # Initialize variables
                out_text = None
                bb_score = None
                bb_loc = None
                
                # Find BB label
                bb_score, bb_loc = _match_score_and_loc(roi_gray, bb_tmpl, bb_mask)
                try:
                    res_bb = cv2.matchTemplate(roi_gray, bb_tmpl, cv2.TM_CCOEFF_NORMED, mask=bb_mask) if roi_gray.size else None
                except Exception:
                    res_bb = None
                had_bb_match = False
                if res_bb is not None:
                    ys, xs = np.where(res_bb >= args.bb_threshold)
                    if xs.size > 0:
                        idx = int(np.argmax(xs))
                        bb_loc = (int(xs[idx]), int(ys[idx]))
                        bb_score = float(res_bb[ys[idx], xs[idx]])
                        had_bb_match = True
                
                # Try to parse value if BB detected
                if np.isfinite(bb_score) and bb_score >= args.bb_threshold and bb_loc is not None:
                    bx = int(bb_loc[0])
                    by = int(bb_loc[1])
                    left_w = max(0, min(bx, roi_gray.shape[1]))
                    if left_w > 0:
                        crop_gray = roi_gray[:, 0:left_w]
                        
                        # Use debug scanner if enabled
                        if args.debug_scan_steps:
                            img_name_base = os.path.splitext(os.path.basename(img_path))[0]
                            debug_dir = os.path.join(args.debug_scan_dir, img_name_base)
                            step_prefix = f"seat{seat}"
                            out_text = _scan_number_simple_debug(crop_gray, dot_tmpl, digit_tmpls, args.digit_threshold, args.min_digit_var, debug_dir, step_prefix)
                        else:
                            out_text = _scan_number_simple(crop_gray, dot_tmpl, digit_tmpls, args.digit_threshold, args.min_digit_var)
                
                # Fallback: try scanning entire ROI if no BB detected
                if not out_text and (bb_score is None or not np.isfinite(bb_score) or bb_score < args.bb_threshold):
                    if args.debug_scan_steps:
                        img_name_base = os.path.splitext(os.path.basename(img_path))[0]
                        debug_dir = os.path.join(args.debug_scan_dir, img_name_base, "fallback")
                        step_prefix = f"seat{seat}_fallback"
                        out_text = _scan_number_simple_debug(roi_gray, dot_tmpl, digit_tmpls, args.digit_threshold, args.min_digit_var, debug_dir, step_prefix)
                    else:
                        out_text = _scan_number_simple(roi_gray, dot_tmpl, digit_tmpls, args.digit_threshold, args.min_digit_var)
                
                # Print result
                img_name = os.path.basename(img_path)
                if out_text:
                    print(f"✓ {img_name}: BB={bb_score:.3f}, Value={out_text}")
                else:
                    bb_str = f"{bb_score:.3f}" if bb_score is not None and np.isfinite(bb_score) else "no_match"
                    print(f"✗ {img_name}: BB={bb_str}, Value=NULL")
                
                # Save all attempts if enabled
                if args.save_all_attempts:
                    try:
                        ts_ms = int(time.time() * 1000)
                        
                        # Determine if this was a successful read
                        success = (bb_score is not None and np.isfinite(bb_score) and 
                                  bb_score >= args.bb_threshold and out_text is not None)
                        
                        # Create subfolder structure
                        seat_dir = os.path.join(args.save_all_dir, f'seat{seat}')
                        subfolder = 'read' if success else 'failed'
                        target_dir = os.path.join(seat_dir, subfolder)
                        os.makedirs(target_dir, exist_ok=True)
                        
                        # Create descriptive filename
                        if bb_score is not None and np.isfinite(bb_score) and bb_score >= args.bb_threshold:
                            bb_status = f"bb_ok_{bb_score:.3f}"
                        else:
                            bb_score_str = f"{bb_score:.3f}" if bb_score is not None and np.isfinite(bb_score) else "no_match"
                            bb_status = f"bb_fail_{bb_score_str}"
                        
                        if out_text:
                            value_status = f"val_{out_text}"
                        else:
                            value_status = "val_null"
                        
                        fname = f"{bb_status}_{value_status}_{ts_ms}.png"
                        out_path = os.path.join(target_dir, fname)
                        
                        # Save the ROI
                        cv2.imwrite(out_path, roi_bgr)
                        
                        # Save annotated version if BB was detected
                        if bb_score is not None and np.isfinite(bb_score) and bb_loc is not None:
                            annotated = roi_bgr.copy()
                            th, tw = bb_tmpl.shape[:2]
                            bx, by = int(bb_loc[0]), int(bb_loc[1])
                            cv2.rectangle(annotated, (bx, by), (bx + tw, by + th), (0, 255, 0), 2)
                            if out_text:
                                cv2.putText(annotated, out_text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            cv2.putText(annotated, f"score:{bb_score:.3f}", (5, annotated.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                            ann_path = out_path.replace('.png', '_annotated.png')
                            cv2.imwrite(ann_path, annotated)
                    except Exception:
                        pass
                        
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        print("\nFolder processing complete!")
        return

    # Live capture mode
    cap = BackgroundCapture(args.device, warmup_frames=45, target_fps=max(8.0, args.fps))
    cap.start()
    try:
        prev_vals: Dict[int, Optional[str]] = {s: None for s in seat_rois.keys()}
        interval = 1.0 / max(0.1, args.fps)
        frame_count = 0
        while True:
            frame = cap.get_frame(timeout_sec=0.3)
            if frame is None:
                time.sleep(0.01)
                continue

            table = frame[profile.roi.y:profile.roi.y+profile.roi.h, profile.roi.x:profile.roi.x+profile.roi.w]
            gray_table = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)

            # Prepare visualization frame if debugging
            vis = table.copy() if args.debug_bb else None
            any_value_changed = False

            for seat, (x, y, w, h) in seat_rois.items():
                # Expand ROI to the left by left-pad (bounded by table)
                rx = max(0, int(x) - max(0, int(args.left_pad)))
                rw = max(0, int(x) + int(w) - rx)
                # Use BGR ROI for saving crops
                roi_bgr = table[y:y+h, rx:rx+rw]
                roi_gray = gray_table[y:y+h, rx:rx+rw]
                if roi_gray.size == 0 or roi_bgr.size == 0:
                    continue
                # Determine whether we should save per-seat try images for this seat
                should_save_tries = (save_tries_targets == 'all') or (isinstance(save_tries_targets, set) and seat in save_tries_targets)

                # Initialize out_text for save-all-attempts logic
                out_text = None
                bb_score = None
                bb_loc = None

                # Find BB label; prefer the rightmost high-confidence match to avoid false matches on the left
                bb_score, bb_loc = _match_score_and_loc(roi_gray, bb_tmpl, bb_mask)
                try:
                    res_bb = cv2.matchTemplate(roi_gray, bb_tmpl, cv2.TM_CCOEFF_NORMED, mask=bb_mask) if roi_gray.size else None
                except Exception:
                    res_bb = None
                had_bb_match = False
                if res_bb is not None:
                    ys, xs = np.where(res_bb >= args.bb_threshold)
                    if xs.size > 0:
                        idx = int(np.argmax(xs))  # pick rightmost among above-threshold matches
                        bb_loc = (int(xs[idx]), int(ys[idx]))
                        bb_score = float(res_bb[ys[idx], xs[idx]])
                        had_bb_match = True
                # Require finite score and threshold
                if np.isfinite(bb_score) and bb_score >= args.bb_threshold and bb_loc is not None:
                    bx = int(bb_loc[0])
                    by = int(bb_loc[1])
                    left_w = max(0, min(bx, rw))
                    if left_w > 0:
                        crop_bgr = roi_bgr[:, 0:left_w]
                        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

                        # Optionally parse amount text from the cropped region (simple scanner with variance filter)
                        out_text = _scan_number_simple(crop_gray, dot_tmpl, digit_tmpls, args.digit_threshold, args.min_digit_var)

                        # Print amount if parsed and changed
                        if out_text and out_text != prev_vals.get(seat):
                            print(f"seat={seat} value={out_text}")
                            prev_vals[seat] = out_text
                            any_value_changed = True

                            # If requested, save attempts once per value into a folder named with the value
                            if should_save_tries:
                                try:
                                    value_dir = os.path.join(args.save_tries_dir, f"seat{seat}", str(out_text))
                                    if not os.path.exists(value_dir):
                                        os.makedirs(value_dir, exist_ok=True)
                                        # Save context images
                                        # 1) Full raw BB ROI (expanded ROI including BB label)
                                        try:
                                            cv2.imwrite(os.path.join(value_dir, 'bb_roi.png'), roi_bgr)
                                        except Exception:
                                            pass
                                        # 2) Left-of-BB crop and scan band (for simple scanner, band == crop)
                                        cv2.imwrite(os.path.join(value_dir, 'roi.png'), crop_gray)
                                        pr = crop_gray
                                        parse_x1 = 0
                                        cv2.imwrite(os.path.join(value_dir, 'band.png'), pr)
                                        # No per-attempt crops for simple scanner; keep structure compatible
                                except Exception:
                                    pass

                        # Save crop unless disabled and skip if duplicate
                        if not args.no_save and np.isfinite(bb_score):
                            if not _is_duplicate_by_tm(crop_gray, seat_existing_gray[seat], args.tm_threshold):
                                ts = int(time.time())
                                if out_text:
                                    base_path = os.path.join(seat_dirs[seat], f"{out_text}.png")
                                    if os.path.exists(base_path):
                                        out_path = os.path.join(seat_dirs[seat], f"{out_text}_{ts}.png")
                                    else:
                                        out_path = base_path
                                else:
                                    out_path = os.path.join(seat_dirs[seat], f'bb_{ts}.png')
                                cv2.imwrite(out_path, crop_bgr)
                                seat_existing_gray[seat].append(crop_gray)
                                print(json.dumps({"event": "seat_bb_crop", "seat": seat, "saved": out_path, "value": out_text, "score": round(float(bb_score), 3)}))

                        # Draw debug annotations
                        if vis is not None:
                            # Draw seat ROI (blue)
                            cv2.rectangle(vis, (rx, y), (rx + rw, y + h), (255, 0, 0), 2)
                            # Draw matched BB label (yellow)
                            th, tw = bb_tmpl.shape[:2]
                            cv2.rectangle(vis, (rx + bx, y + by), (rx + bx + tw, y + by + th), (0, 255, 255), 2)
                            # Label seat and value if available
                            label_text = f"S{seat}"
                            if out_text:
                                label_text += f": {out_text}"
                            cv2.putText(vis, label_text, (rx, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                elif vis is not None:
                    # If no match, still draw seat ROI (blue) for context
                    cv2.rectangle(vis, (rx, y), (rx + rw, y + h), (255, 0, 0), 2)
                    cv2.putText(vis, f"S{seat}", (rx, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Save all attempts if enabled (even failures)
                if args.save_all_attempts:
                    try:
                        ts_ms = int(time.time() * 1000)
                        
                        # Determine if this was a successful read
                        success = (bb_score is not None and np.isfinite(bb_score) and 
                                  bb_score >= args.bb_threshold and out_text is not None)
                        
                        # Create subfolder structure: save_all_dir/seat{N}/[failed|read]/
                        seat_dir = os.path.join(args.save_all_dir, f'seat{seat}')
                        subfolder = 'read' if success else 'failed'
                        target_dir = os.path.join(seat_dir, subfolder)
                        os.makedirs(target_dir, exist_ok=True)
                        
                        # Create descriptive filename with status
                        if bb_score is not None and np.isfinite(bb_score) and bb_score >= args.bb_threshold:
                            bb_status = f"bb_ok_{bb_score:.3f}"
                        else:
                            bb_score_str = f"{bb_score:.3f}" if bb_score is not None and np.isfinite(bb_score) else "no_match"
                            bb_status = f"bb_fail_{bb_score_str}"
                        
                        if out_text:
                            value_status = f"val_{out_text}"
                        else:
                            value_status = "val_null"
                        
                        fname = f"{bb_status}_{value_status}_{ts_ms}.png"
                        out_path = os.path.join(target_dir, fname)
                        
                        # Save the full ROI (BGR) for inspection
                        cv2.imwrite(out_path, roi_bgr)
                        
                        # Also save annotated version with BB position if available
                        if bb_score is not None and np.isfinite(bb_score) and bb_loc is not None:
                            annotated = roi_bgr.copy()
                            th, tw = bb_tmpl.shape[:2]
                            bx, by = int(bb_loc[0]), int(bb_loc[1])
                            cv2.rectangle(annotated, (bx, by), (bx + tw, by + th), (0, 255, 0), 2)
                            if out_text:
                                cv2.putText(annotated, out_text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            cv2.putText(annotated, f"score:{bb_score:.3f}", (5, annotated.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                            ann_path = out_path.replace('.png', '_annotated.png')
                            cv2.imwrite(ann_path, annotated)
                    except Exception:
                        pass

            # Render tabulated dashboard every frame
            try:
                # Clear screen
                try:
                    sys.stdout.write("\033[2J\033[H")
                except Exception:
                    pass
                
                # Build table
                header = ["Seat", "Player BB"]
                colw = [8, 15]
                lines = []
                
                # Header
                lines.append(header[0].ljust(colw[0]) + "  " + header[1].ljust(colw[1]))
                lines.append("-" * (colw[0] + colw[1] + 2))
                
                # Rows for each seat
                for seat in sorted(seat_rois.keys()):
                    val = prev_vals.get(seat)
                    if val:
                        bb_text = val
                    else:
                        bb_text = 'none'
                    lines.append(str(seat).ljust(colw[0]) + "  " + bb_text.ljust(colw[1]))
                
                # Footer
                lines.append("")
                lines.append(f"Frame: {frame_count}")
                
                print("\n".join(lines))
            except Exception:
                pass

            # Save debug frame if requested and any seat value changed this cycle
            if vis is not None and (any_value_changed or args.debug_bb_on_match):
                try:
                    os.makedirs(args.debug_bb_dir, exist_ok=True)
                    ts_dbg = int(time.time() * 1000)
                    out_dbg = os.path.join(args.debug_bb_dir, f"bb_debug_table{args.table_id}_{ts_dbg}.png")
                    cv2.imwrite(out_dbg, vis)
                except Exception:
                    pass

            time.sleep(interval)
            frame_count += 1
    finally:
        cap.stop()


if __name__ == '__main__':
    main()


