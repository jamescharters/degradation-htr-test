#!/usr/bin/env python3
"""
Minimal Degradation-Awareness Test
Runs on M4 Mac in 1-2 hours
"""

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageDraw, ImageFont
import random
from jiwer import cer
import numpy as np

print("=" * 60)
print("DEGRADATION-AWARE HTR: Minimal Viability Test")
print("=" * 60)

# Setup device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"\n✓ Using device: {device}")

# Load model
print("✓ Loading TrOCR-small (optimized for M4)...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
model = model.to(device)
print("✓ Model loaded successfully")

# ============================================
# PART 1: CREATE SYNTHETIC TEST DATA
# ============================================

def create_text_image(text, width=400, height=64):
    """Create simple handwritten-style text image"""
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf", 32)
    except:
        font = ImageFont.load_default()
    
    # Add slight randomness for "handwritten" feel
    x = 10 + random.randint(-3, 3)
    y = 10 + random.randint(-3, 3)
    draw.text((x, y), text, fill='black', font=font)
    
    return img

def add_holes(img, num_holes=3):
    """Add white rectangles (holes) to image"""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    
    for _ in range(num_holes):
        hole_w = random.randint(30, 60)
        hole_h = random.randint(10, 20)
        x = random.randint(0, max(1, w - hole_w))
        y = random.randint(0, max(1, h - hole_h))
        draw.rectangle([x, y, x + hole_w, y + hole_h], fill='white')
    
    return img

def add_fade(img):
    """Add blur to simulate faded ink"""
    from PIL import ImageFilter
    return img.filter(ImageFilter.GaussianBlur(radius=1.5))

def add_stain(img):
    """Add dark blobs to simulate stains"""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    
    for _ in range(2):
        size = random.randint(20, 40)
        x = random.randint(0, max(1, w - size))
        y = random.randint(0, max(1, h - size))
        draw.ellipse([x, y, x + size, y + size], 
                     fill=(200, 200, 200), outline=(150, 150, 150))
    
    return img

# Test samples
test_texts = [
    "The quick brown fox",
    "jumps over the lazy dog",
    "Hello world from here",
    "Machine learning works",
    "Degradation test case"
]

print(f"\n✓ Creating {len(test_texts)} synthetic test images...")

# ============================================
# PART 2: TEST DEGRADATION IMPACT
# ============================================

print("\n" + "=" * 60)
print("EXPERIMENT 1: Does degradation hurt performance?")
print("=" * 60)

results = []

for idx, text_gt in enumerate(test_texts):
    print(f"\n[Sample {idx+1}/{len(test_texts)}] Ground truth: '{text_gt}'")
    
    # Create clean image
    img_clean = create_text_image(text_gt)
    
    # Test clean
    pixel_values = processor(img_clean, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_length=64)
    text_pred_clean = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    cer_clean = cer(text_gt.lower(), text_pred_clean.lower())
    
    # Create and test degraded versions
    degradations = {
        'holes': add_holes(img_clean),
        'fade': add_fade(img_clean),
        'stain': add_stain(img_clean)
    }
    
    sample_result = {'text': text_gt, 'clean_cer': cer_clean}
    
    print(f"  Clean:  CER={cer_clean:.3f}, pred='{text_pred_clean}'")
    
    for deg_type, img_deg in degradations.items():
        pixel_values = processor(img_deg, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_length=64)
        text_pred_deg = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        cer_deg = cer(text_gt.lower(), text_pred_deg.lower())
        
        sample_result[f'{deg_type}_cer'] = cer_deg
        print(f"  {deg_type.capitalize():6s}: CER={cer_deg:.3f}, pred='{text_pred_deg}'")
        
        # Save sample images for inspection
        if idx == 0:  # Save first sample
            img_clean.save(f"sample_clean.png")
            img_deg.save(f"sample_{deg_type}.png")
    
    results.append(sample_result)

# ============================================
# PART 3: ANALYZE RESULTS
# ============================================

print("\n" + "=" * 60)
print("SUMMARY: Average CER by Degradation Type")
print("=" * 60)

avg_clean = np.mean([r['clean_cer'] for r in results])
avg_holes = np.mean([r['holes_cer'] for r in results])
avg_fade = np.mean([r['fade_cer'] for r in results])
avg_stain = np.mean([r['stain_cer'] for r in results])

print(f"\nClean text:     {avg_clean:.3f}")
print(f"With holes:     {avg_holes:.3f}  (Δ = +{(avg_holes - avg_clean):.3f})")
print(f"With fading:    {avg_fade:.3f}  (Δ = +{(avg_fade - avg_clean):.3f})")
print(f"With stains:    {avg_stain:.3f}  (Δ = +{(avg_stain - avg_clean):.3f})")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)

if avg_holes > avg_clean + 0.05 or avg_fade > avg_clean + 0.05:
    print("\n✓ POSITIVE: Degradation significantly hurts performance!")
    print("  → Degradation-aware training is worth pursuing")
    print("  → Your MPhil idea has merit")
    print("\nNext steps:")
    print("  1. Annotate a real manuscript folio")
    print("  2. Test degradation-weighted loss")
    print("  3. If that works, proceed with full MPhil proposal")
else:
    print("\n✗ NEGATIVE: Degradation doesn't hurt much")
    print("  → Model may already be robust")
    print("  → Try with real manuscripts (not synthetic)")
    print("  → Or consider different degradation types")

print("\n✓ Sample images saved: sample_clean.png, sample_holes.png, etc.")
print("=" * 60)
