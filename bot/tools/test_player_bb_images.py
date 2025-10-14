import os
import sys
import argparse
import json
from typing import Dict, Optional, Tuple
import cv2


# Ensure project root on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAND_ROOT = os.path.dirname(ROOT)
for path in (ROOT, GRAND_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)


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


def _scan_number_simple(crop_gray, dot_tmpl, digit_tmpls: Dict[str, Tuple[any, Optional[any]]], threshold: float, save_dir: Optional[str] = None, min_var: float = 0.0) -> Optional[str]:
    try:
        rw = crop_gray.shape[1]
        x_scan = 0
        end_limit = rw
        chars = []
        dot_found = False
        match_idx = 0
        while x_scan < end_limit:
            matched = False
            for width in range(1, end_limit - x_scan + 1):
                slice_img = crop_gray[:, x_scan:x_scan+width]
                # digits first
                best_d = None
                best_s = -1.0
                best_lx = 0
                best_w = 0
                for d, (dt, dm) in digit_tmpls.items():
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
                    chars.append(best_d)
                    abs_x = x_scan + best_lx
                    if save_dir is not None:
                        try:
                            os.makedirs(save_dir, exist_ok=True)
                            glyph = crop_gray[:, abs_x:abs_x + best_w]
                            if glyph.size > 0:
                                tag = ""
                                if min_var > 0.0 and float(cv2.meanStdDev(glyph)[1].mean()**2) < min_var:
                                    tag = "_lowvar"
                                cv2.imwrite(os.path.join(save_dir, f"match_{match_idx}_digit_{best_d}_x{abs_x}_w{best_w}{tag}.png"), glyph)
                                match_idx += 1
                        except Exception:
                            pass
                    # Reject low-variance glyphs if threshold set
                    if min_var > 0.0:
                        glyph = crop_gray[:, abs_x:abs_x + best_w]
                        if glyph.size == 0 or float(cv2.meanStdDev(glyph)[1].mean()**2) < min_var:
                            # undo append and skip
                            chars.pop()
                            # move pointer a bit to avoid infinite loop
                            x_scan = abs_x + max(1, best_w // 2)
                            continue
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
                            chars.append('.')
                            lx = int(dl[0])
                            abs_x = x_scan + lx
                            dw = dot_tmpl.shape[1]
                            if save_dir is not None:
                                try:
                                    os.makedirs(save_dir, exist_ok=True)
                                    glyph = crop_gray[:, abs_x:abs_x + dw]
                                    if glyph.size > 0:
                                        tag = ""
                                        if min_var > 0.0 and float(cv2.meanStdDev(glyph)[1].mean()**2) < min_var:
                                            tag = "_lowvar"
                                        cv2.imwrite(os.path.join(save_dir, f"match_{match_idx}_dot_x{abs_x}_w{dw}{tag}.png"), glyph)
                                        match_idx += 1
                                except Exception:
                                    pass
                            if min_var > 0.0:
                                glyph = crop_gray[:, abs_x:abs_x + dw]
                                if glyph.size == 0 or float(cv2.meanStdDev(glyph)[1].mean()**2) < min_var:
                                    chars.pop()
                                    x_scan = abs_x + max(1, dw // 2)
                                    continue
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


def _scan_number_simple_rtl(crop_gray, dot_tmpl, digit_tmpls: Dict[str, Tuple[any, Optional[any]]], threshold: float, save_dir: Optional[str] = None, min_var: float = 0.0) -> Optional[str]:
    try:
        rw = crop_gray.shape[1]
        x_scan = rw
        match_idx = 0
        chars_rev = []
        dot_used = False
        while x_scan > 0:
            matched = False
            for width in range(1, x_scan + 1):
                x_l = max(0, x_scan - width)
                slice_img = crop_gray[:, x_l:x_scan]
                # digits first
                best_d = None
                best_s = -1.0
                best_lx = 0
                best_w = 0
                for d, (dt, dm) in digit_tmpls.items():
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
                    chars_rev.append(best_d)
                    # Move pointer to just left of the matched digit's left edge
                    abs_x = x_l + best_lx
                    if save_dir is not None:
                        try:
                            os.makedirs(save_dir, exist_ok=True)
                            glyph = crop_gray[:, abs_x:abs_x + best_w]
                            if glyph.size > 0:
                                tag = ""
                                if min_var > 0.0 and float(cv2.meanStdDev(glyph)[1].mean()**2) < min_var:
                                    tag = "_lowvar"
                                cv2.imwrite(os.path.join(save_dir, f"match_{match_idx}_digit_{best_d}_x{abs_x}_w{best_w}{tag}.png"), glyph)
                                match_idx += 1
                        except Exception:
                            pass
                    if min_var > 0.0:
                        glyph = crop_gray[:, abs_x:abs_x + best_w]
                        if glyph.size == 0 or float(cv2.meanStdDev(glyph)[1].mean()**2) < min_var:
                            chars_rev.pop()
                            x_scan = abs_x - max(1, best_w // 2)
                            continue
                    x_scan = abs_x - 1
                    matched = True
                    break
                # allow a single dot, but only after at least one digit captured
                if not matched and not dot_used and dot_tmpl is not None and len(chars_rev) > 0:
                    try:
                        resd = cv2.matchTemplate(slice_img, dot_tmpl, cv2.TM_CCOEFF_NORMED)
                    except Exception:
                        resd = None
                    if resd is not None:
                        _, ds, _, dl = cv2.minMaxLoc(resd)
                        if ds >= threshold and dl is not None:
                            chars_rev.append('.')
                            lx = int(dl[0])
                            abs_x = x_l + lx
                            dw = dot_tmpl.shape[1]
                            if save_dir is not None:
                                try:
                                    os.makedirs(save_dir, exist_ok=True)
                                    glyph = crop_gray[:, abs_x:abs_x + dw]
                                    if glyph.size > 0:
                                        tag = ""
                                        if min_var > 0.0 and float(cv2.meanStdDev(glyph)[1].mean()**2) < min_var:
                                            tag = "_lowvar"
                                        cv2.imwrite(os.path.join(save_dir, f"match_{match_idx}_dot_x{abs_x}_w{dw}{tag}.png"), glyph)
                                        match_idx += 1
                                except Exception:
                                    pass
                            if min_var > 0.0:
                                glyph = crop_gray[:, abs_x:abs_x + dw]
                                if glyph.size == 0 or float(cv2.meanStdDev(glyph)[1].mean()**2) < min_var:
                                    chars_rev.pop()
                                    x_scan = abs_x - max(1, dw // 2)
                                    continue
                            x_scan = abs_x - 1
                            matched = True
                            dot_used = True
                            break
            if not matched:
                break
        if chars_rev:
            return ''.join(reversed(chars_rev))
        return None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description='Test player BB number parsing on static images (numeric region crops)')
    parser.add_argument('--images', nargs='*', help='Paths to one or more images to parse')
    parser.add_argument('--dir', default=None, help='Directory containing images to parse')
    parser.add_argument('--numbers-dir', default='bot/templates/player_bb/numbers', help='Directory with dot.png and 0-9.png')
    parser.add_argument('--digit-threshold', type=float, default=0.99)
    parser.add_argument('--rtl', action='store_true', help='Parse right-to-left (anchor from the right edge)')
    parser.add_argument('--save-matches-dir', default=None, help='If set, save each matched glyph slice into this directory (per-image subfolder)')
    parser.add_argument('--min-var', type=float, default=10.0, help='Reject glyph matches below this pixel variance (helps ignore flat dark regions)')
    args = parser.parse_args()

    # Load templates
    dot_tmpl, _ = _load_gray_with_optional_mask(os.path.join(args.numbers_dir, 'dot.png'))
    digit_tmpls: Dict[str, Tuple[any, Optional[any]]] = {}
    for d in '0123456789':
        dt, dm = _load_gray_with_optional_mask(os.path.join(args.numbers_dir, f'{d}.png'))
        if dt is not None:
            digit_tmpls[d] = (dt, dm)

    # Collect image paths from --images and/or --dir
    paths = []
    if args.images:
        paths.extend(args.images)
    if args.dir:
        try:
            exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
            for name in sorted(os.listdir(args.dir)):
                p = os.path.join(args.dir, name)
                if os.path.isfile(p) and os.path.splitext(name)[1].lower() in exts:
                    paths.append(p)
        except Exception:
            pass

    if not paths:
        print(json.dumps({"error": "no_input", "hint": "Provide --images or --dir"}))
        return

    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(json.dumps({"image": path, "error": "cannot_read"}))
            continue
        if img.ndim == 3 and img.shape[2] == 4:
            bgr = img[:, :, :3]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        save_dir = None
        if args.save_matches_dir:
            stem = os.path.splitext(os.path.basename(path))[0]
            save_dir = os.path.join(args.save_matches_dir, stem)
        if args.rtl:
            text = _scan_number_simple_rtl(gray, dot_tmpl, digit_tmpls, args.digit_threshold, save_dir, args.min_var)
        else:
            text = _scan_number_simple(gray, dot_tmpl, digit_tmpls, args.digit_threshold, save_dir, args.min_var)
        print(json.dumps({"image": path, "text": text}))


if __name__ == '__main__':
    main()


