#!/usr/bin/env python3
"""
Test degradation impact on LOCAL images
"""

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageDraw
import argparse
import random
import os
import glob

print("=" * 60)
print("WriteVision - Local Image Degradation Test")
print("=" * 60)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"\n✓ Using device: {device}")

# Load model
print("\n✓ Loading TrOCR model...")
model_name = "microsoft/trocr-base-handwritten"
processor = TrOCRProcessor.from_pretrained(model_name, use_fast=False)
model = VisionEncoderDecoderModel.from_pretrained(model_name)
model = model.to(device)
model.eval()
print(f"✓ Successfully loaded: {model_name}")

def add_holes(img, num_holes=3):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for _ in range(num_holes):
        hole_w = random.randint(40, 80)
        hole_h = random.randint(15, 30)
        x = random.randint(0, max(1, w - hole_w))
        y = random.randint(0, max(1, h - hole_h))
        draw.rectangle([x, y, x + hole_w, y + hole_h], fill='white')
    return img

def add_fade(img):
    from PIL import ImageFilter
    return img.filter(ImageFilter.GaussianBlur(radius=2.0))

def add_stain(img):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for _ in range(2):
        size = random.randint(30, 60)
        x = random.randint(0, max(1, w - size))
        y = random.randint(0, max(1, h - size))
        draw.ellipse([x, y, x + size, y + size], 
                     fill=(180, 180, 180), outline=(120, 120, 120))
    return img

def recognize_text(image):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_length=64)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, default='.',
                       help='Directory containing images')
    parser.add_argument('--pattern', type=str, default='*.png',
                       help='Image file pattern (e.g., *.jpg, *.png)')
    args = parser.parse_args()
    
    # Find images
    pattern = os.path.join(args.image_dir, args.pattern)
    image_files = glob.glob(pattern)
    
    if not image_files:
        print(f"\n✗ No images found matching: {pattern}")
        print("\nTry:")
        print("  python writevision_local_test.py --image-dir /path/to/images --pattern '*.jpg'")
        return
    
    print(f"\n✓ Found {len(image_files)} images")
    
    # Show available images
    print("\nAvailable images:")
    for i, img_path in enumerate(image_files):
        print(f"  [{i}] {os.path.basename(img_path)}")
    
    # Test first 3 (or all if fewer)
    test_files = image_files[:min(3, len(image_files))]
    
    print("\n" + "=" * 60)
    print(f"TESTING {len(test_files)} IMAGES")
    print("=" * 60)
    
    for idx, img_path in enumerate(test_files):
        print(f"\n[{os.path.basename(img_path)}]")
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        print(f"  Size: {image.size}")
        
        # Recognize clean
        text_clean = recognize_text(image)
        print(f"  Clean: '{text_clean}'")
        
        # Test degradations
        degradations = {
            'holes': add_holes(image),
            'fade': add_fade(image),
            'stain': add_stain(image)
        }
        
        for deg_type, img_deg in degradations.items():
            text_deg = recognize_text(img_deg)
            print(f"  {deg_type.capitalize():6s}: '{text_deg}'", end='')
            
            if text_clean != text_deg:
                print(" ⚠️  DIFFERENT")
            else:
                print(" ✓ same")
            
            # Save degraded version
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_path = f"degraded_{base_name}_{deg_type}.png"
            img_deg.save(output_path)
    
    print("\n" + "=" * 60)
    print("✓ Degraded images saved with prefix: degraded_*")
    print("=" * 60)

if __name__ == "__main__":
    main()
