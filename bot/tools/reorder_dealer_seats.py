import os
import sys
import json
import argparse
import math
from typing import Dict, List, Tuple

# Ensure project root on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAND_ROOT = os.path.dirname(ROOT)
for path in (ROOT, GRAND_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from bot.core.roi import load_tables_config


def _angle(center_x: float, center_y: float, rect: Tuple[int, int, int, int]) -> float:
    x, y, w, h = rect
    cx = x + w / 2.0
    cy = y + h / 2.0
    a = math.atan2(-(cy - center_y), (cx - center_x))  # clockwise, 0 at +X
    if a < 0:
        a += 2 * math.pi
    return a


def _extract_rects(dsr) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    items: List[Tuple[str, Tuple[int, int, int, int]]] = []
    if isinstance(dsr, dict):
        for k, v in dsr.items():
            if isinstance(v, (list, tuple)) and len(v) == 4:
                items.append((str(k), (int(v[0]), int(v[1]), int(v[2]), int(v[3]))))
    elif isinstance(dsr, list):
        for i, v in enumerate(dsr):
            if isinstance(v, (list, tuple)) and len(v) == 4:
                items.append((str(i), (int(v[0]), int(v[1]), int(v[2]), int(v[3]))))
    return items


def main():
    parser = argparse.ArgumentParser(description='Reorder dealer_seat_rois indices across tables to match a reference table\'s angular order')
    parser.add_argument('--tables', default='bot/config/tables.json')
    parser.add_argument('--ref-table-id', type=int, default=0, help='Reference table id whose current seat indices define the desired order')
    parser.add_argument('--table-id', type=int, default=None, help='If set, only reorder this table id (besides the reference)')
    parser.add_argument('--dry-run', action='store_true', help='Only print planned changes without writing')
    parser.add_argument('--backup', action='store_true', help='Write a .bak copy of the tables file before modifying')
    args = parser.parse_args()

    # Load raw JSON
    with open(args.tables, 'r') as f:
        data = json.load(f)

    # Build helper maps: id -> table dict and id -> profile for dimensions
    tables_by_id: Dict[int, dict] = {}
    for t in data.get('tables', []):
        tables_by_id[int(t['id'])] = t

    if args.ref_table_id not in tables_by_id:
        raise SystemExit('Reference table id not found in tables.json')

    # Compute desired label order from reference table's current labels in angular order
    ref_t = tables_by_id[args.ref_table_id]
    roi = ref_t['roi']
    scale_cx = roi['w'] / 2.0
    scale_cy = roi['h'] / 2.0
    ref_items = _extract_rects(ref_t.get('landmarks', {}).get('dealer_seat_rois', {}))
    if not ref_items:
        raise SystemExit('Reference table has no dealer_seat_rois')
    # Sort ref by angle and record the label order
    ref_sorted = sorted(ref_items, key=lambda kv: _angle(scale_cx, scale_cy, kv[1]))
    desired_labels_in_order = [lbl for lbl, _ in ref_sorted]

    # Process target tables
    changes = []
    for tid, t in tables_by_id.items():
        if tid == args.ref_table_id:
            continue
        if args.table_id is not None and tid != args.table_id:
            continue
        lm = t.get('landmarks', {})
        dsr = lm.get('dealer_seat_rois')
        items = _extract_rects(dsr)
        if not items:
            continue
        if len(items) != len(desired_labels_in_order):
            # skip if sizes mismatch
            continue
        # Sort current items by angle
        roi_t = t['roi']
        cx = roi_t['w'] / 2.0
        cy = roi_t['h'] / 2.0
        sorted_items = sorted(items, key=lambda kv: _angle(cx, cy, kv[1]))

        # Remap: desired label at rank r gets rect at rank r
        new_map: Dict[str, List[int]] = {}
        for r, desired_label in enumerate(desired_labels_in_order):
            _, rect = sorted_items[r]
            new_map[str(desired_label)] = [int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])]

        # Collect change if different
        if isinstance(dsr, dict):
            old_map = {str(k): v for k, v in dsr.items() if isinstance(v, (list, tuple)) and len(v) == 4}
        else:
            old_map = {str(i): v for i, v in enumerate(dsr) if isinstance(v, (list, tuple)) and len(v) == 4}
        if old_map != new_map:
            changes.append((tid, old_map, new_map))

    # Print planned changes
    for tid, old_map, new_map in changes:
        print(json.dumps({
            "table_id": tid,
            "change": {
                "from": old_map,
                "to": new_map
            }
        }))

    if args.dry_run or not changes:
        return

    if args.backup:
        with open(args.tables + '.bak', 'w') as f:
            json.dump(data, f, indent=2)

    # Apply changes
    for tid, _, new_map in changes:
        t = tables_by_id[tid]
        lm = t.setdefault('landmarks', {})
        lm['dealer_seat_rois'] = new_map

    with open(args.tables, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Rewrote dealer_seat_rois for {len(changes)} table(s)")


if __name__ == '__main__':
    main()


