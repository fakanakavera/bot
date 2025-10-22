#!/usr/bin/env python3
"""
Check if digit templates are valid (have pixel variation)
"""
import os
import sys
import argparse
import cv2
import numpy as np

def check_template(path: str, name: str):
    """Check if a template is valid"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"❌ {name}: Could not load image")
        return False
    
    # Convert to grayscale if needed
    if img.ndim == 3:
        if img.shape[2] == 4:  # BGRA
            gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
            mask = img[:, :, 3]
        else:  # BGR
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = None
    else:
        gray = img
        mask = None
    
    # Check variance
    mean, stddev = cv2.meanStdDev(gray, mask=mask)
    variance = float(stddev.mean()**2)
    
    # Check if all pixels are the same
    if variance < 1.0:
        print(f"⚠️  {name}: Very low variance ({variance:.2f}) - might be flat/invalid")
        print(f"   Shape: {gray.shape}, Mean: {mean[0][0]:.1f}, StdDev: {stddev[0][0]:.2f}")
        return False
    else:
        print(f"✓ {name}: Valid (variance={variance:.2f}, shape={gray.shape})")
        return True

def main():
    parser = argparse.ArgumentParser(description='Validate digit templates')
    parser.add_argument('--numbers-dir', default='bot/templates/player_bb/numbers', 
                       help='Directory with dot.png and 0-9.png')
    args = parser.parse_args()
    
    if not os.path.isdir(args.numbers_dir):
        print(f"❌ Directory does not exist: {args.numbers_dir}")
        return 1
    
    print(f"Checking templates in: {args.numbers_dir}\n")
    
    all_valid = True
    
    # Check dot
    dot_path = os.path.join(args.numbers_dir, 'dot.png')
    if os.path.exists(dot_path):
        if not check_template(dot_path, 'dot.png'):
            all_valid = False
    else:
        print(f"⚠️  dot.png: Not found")
        all_valid = False
    
    print()
    
    # Check digits
    for d in '0123456789':
        digit_path = os.path.join(args.numbers_dir, f'{d}.png')
        if os.path.exists(digit_path):
            if not check_template(digit_path, f'{d}.png'):
                all_valid = False
        else:
            print(f"⚠️  {d}.png: Not found")
            all_valid = False
    
    print()
    if all_valid:
        print("✓ All templates are valid!")
        return 0
    else:
        print("❌ Some templates have issues - this may cause infinite scores")
        return 1

if __name__ == '__main__':
    sys.exit(main())

