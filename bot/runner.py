import json
import time
import argparse
import os
import sys
import cv2

# Ensure project root on sys.path to import capture.py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from capture import BackgroundCapture
from bot.core.roi import load_tables_config
from bot.core.templates import TemplateLibrary
from bot.detectors.dealer import DealerDetector
from bot.detectors.turn import TurnDetector
from bot.detectors.cards import CardsDetector
from bot.detectors.numbers import NumbersDetector
from bot.table_detector import TableDetector
from bot.multi import MultiTableManager


def build_detector_for_profile(profile, tlib: TemplateLibrary, match_cfg: dict) -> TableDetector:
    # NOTE: For v1, we rely on landmarks in tables.json for ROIs
    dealer_tpl = tlib.templates.get('dealer', None)
    glow_tpl = tlib.templates.get('glow', None)
    card_tpls = {k: v for k, v in tlib.templates.items() if k.startswith('card_')}
    digit_tpls = {k: v for k, v in tlib.templates.items() if k.startswith('digit_')}

    detectors = {}
    if dealer_tpl and 'dealer_seat_rois' in profile.landmarks:
        detectors['dealer'] = DealerDetector(
            dealer_tpl,
            threshold=match_cfg['groups'].get('dealer', {}).get('threshold', match_cfg['defaults']['threshold']),
            seat_rois=profile.landmarks['dealer_seat_rois']
        )
    if glow_tpl and 'glow_seat_rois' in profile.landmarks:
        detectors['turn'] = TurnDetector(
            glow_tpl,
            threshold=match_cfg['groups'].get('glow', {}).get('threshold', match_cfg['defaults']['threshold']),
            seat_glow_rois=profile.landmarks['glow_seat_rois']
        )
    if card_tpls and 'hole_rois' in profile.landmarks and 'board_rois' in profile.landmarks:
        detectors['cards'] = CardsDetector(
            card_tpls,
            threshold=match_cfg['groups'].get('cards', {}).get('threshold', match_cfg['defaults']['threshold']),
            hole_rois=profile.landmarks['hole_rois'],
            board_rois=profile.landmarks['board_rois']
        )
    if digit_tpls and 'pot_roi' in profile.landmarks:
        detectors['numbers'] = NumbersDetector(
            digit_tpls,
            threshold=match_cfg['groups'].get('digits', {}).get('threshold', match_cfg['defaults']['threshold']),
            roi=profile.landmarks['pot_roi']
        )

    return TableDetector(detectors)


def main():
    parser = argparse.ArgumentParser(description='Poker vision runner (detect-only)')
    parser.add_argument('--device', default='/dev/v4l/by-id/usb-Actions_Micro_UGREEN-25854_-1575730626-video-index0')
    parser.add_argument('--tables', default='bot/config/tables.json')
    parser.add_argument('--templates', default='bot/templates')
    parser.add_argument('--match', default='bot/config/match.json')
    parser.add_argument('--fps', type=float, default=4.0)
    parser.add_argument('--once', action='store_true', help='Process a single frame then exit')
    args = parser.parse_args()

    # Load configs
    profiles = load_tables_config(args.tables)
    with open(args.match, 'r') as f:
        match_cfg = json.load(f)

    tlib = TemplateLibrary(args.templates)
    # Auto-load board card templates named like As.png, Td.png, etc.
    tlib.load_dir('board_cards', name_prefix='card_')

    mtm = MultiTableManager(
        profiles,
        detector_factory=lambda p: build_detector_for_profile(p, tlib, match_cfg)
    )

    cap = BackgroundCapture(args.device, warmup_frames=45, target_fps=max(8.0, args.fps))
    cap.start()

    try:
        frame_interval = 1.0 / max(0.1, args.fps)
        while True:
            frame = cap.get_frame(timeout_sec=0.2)
            if frame is None:
                time.sleep(0.01)
                continue
            results = mtm.process_all(frame)
            print(json.dumps({"ts": int(time.time()*1000), "tables": results}))
            if args.once:
                break
            time.sleep(frame_interval)
    finally:
        cap.stop()


if __name__ == '__main__':
    main()
