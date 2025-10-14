import json
import argparse
import os
import sys

# Ensure project root on sys.path (not strictly needed here but consistent)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    parser = argparse.ArgumentParser(description='Write placeholder table ROIs; replace with interactive calibration later')
    parser.add_argument('--out', default='bot/config/tables.json')
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    args = parser.parse_args()

    # Placeholder: split into 4 equal quadrants
    w2 = args.width // 2
    h2 = args.height // 2
    data = {
        "tables": [
            {"id": 0, "roi": {"x": 0, "y": 0, "w": w2, "h": h2}, "scale": {"width": 1280, "height": 720}, "landmarks": {}},
            {"id": 1, "roi": {"x": w2, "y": 0, "w": w2, "h": h2}, "scale": {"width": 1280, "height": 720}, "landmarks": {}},
            {"id": 2, "roi": {"x": 0, "y": h2, "w": w2, "h": h2}, "scale": {"width": 1280, "height": 720}, "landmarks": {}},
            {"id": 3, "roi": {"x": w2, "y": h2, "w": w2, "h": h2}, "scale": {"width": 1280, "height": 720}, "landmarks": {}}
        ]
    }

    with open(args.out, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
