import os
import sys
import argparse
import json
import cv2

# Ensure project root on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bot.core.templates import TemplateLibrary
from bot.core.matchers import match_template, find_best_match


def main():
    parser = argparse.ArgumentParser(description='Batch check templates against a screenshot')
    parser.add_argument('--image', required=True)
    parser.add_argument('--templates', default='bot/templates')
    parser.add_argument('--match', default='bot/config/match.json')
    args = parser.parse_args()

    with open(args.match, 'r') as f:
        match_cfg = json.load(f)

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit('Failed to load image')

    tlib = TemplateLibrary(args.templates)
    # Expect user to populate tlib via naming convention or by extending this tool
    print('Template check placeholder: load and test specific templates here.')


if __name__ == '__main__':
    main()
