# 1. Make sure you're in your project directory
cd degradation-htr-test

# 2. Add the missing dependency (requests)
uv add requests

# 3. Create the script
cat > writevision_adapted.py << 'EOF'
#!/usr/bin/env python3
"""
Adapted from rafia9005/WriteVision for degradation testing
Fixed tokenizer issues for M4 Mac
"""

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import base64
import argparse
import random

print("=" * 60)
print("WriteVision - Degradation Testing (Adapted)")
print("=" * 60)

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"\n✓ Using device: {device}")

# Load model with proper tokenizer settings
print("\n✓ Loading TrOCR model...")
try:
    model_name = "microsoft/trocr-base-handwritten"
    
    # Load processor WITHOUT fast tokenizer (this fixes the error)
    processor = TrOCRProcessor.from_pretrained(
        model_name,
        use_fast=False  # CRITICAL: avoids the NoneType error
    )
    
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    print(f"✓ Successfully loaded: {model_name}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit(1)

# Image loading functions
def load_image_from_url(url):
    """Load image from URL"""
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

def load_image_from_path(path):
    """Load image from local path"""
    return Image.open(path).convert("RGB")

def load_image_from_base64(b64_string):
    """Load image from base64 string"""
    image_data = base64.b64decode(b64_string)
    return Image.open(BytesIO(image_data)).convert("RGB")

# Degradation functions
def add_holes(img, num_holes=3):
    """Add white rectangles (holes)"""
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
    """Add blur to simulate faded ink"""
    from PIL import ImageFilter
    return img.filter(ImageFilter.GaussianBlur(radius=2.0))

def add_stain(img):
    """Add dark blobs"""
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

# Recognition function
def recognize_text(image):
    """Recognize text from image"""
    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values.to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_length=64)
    
    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]
    
    return generated_text

# Main function
def main():
    parser = argparse.ArgumentParser(description='WriteVision - HTR with degradation testing')
    parser.add_argument('--input', type=str, required=True,
                       choices=['url', 'path', 'base64'],
                       help='Input type')
    parser.add_argument('--source', type=str, required=True,
                       help='Image source (URL, path, or base64)')
    parser.add_argument('--test-degradation', action='store_true',
                       help='Test with degradation types')
    
    args = parser.parse_args()
    
    # Load image
    print(f"\n✓ Loading image from {args.input}...")
    if args.input == 'url':
        image = load_image_from_url(args.source)
    elif args.input == 'path':
        image = load_image_from_path(args.source)
    else:
        image = load_image_from_base64(args.source)
    
    print(f"✓ Image loaded: {image.size}")
    
    # Recognize clean image
    print("\n" + "=" * 60)
    print("RECOGNITION RESULTS")
    print("=" * 60)
    
    print("\n[Clean image]")
    text_clean = recognize_text(image)
    print(f"  Recognized text: '{text_clean}'")
    
    image.save("result_clean.png")
    print("  Saved: result_clean.png")
    
    # Test with degradation if requested
    if args.test_degradation:
        print("\n[Testing with degradation...]")
        
        degradations = {
            'holes': add_holes(image),
            'fade': add_fade(image),
            'stain': add_stain(image)
        }
        
        for deg_type, img_deg in degradations.items():
            print(f"\n  {deg_type.capitalize()}:")
            text_deg = recognize_text(img_deg)
            print(f"    Recognized: '{text_deg}'")
            
            filename = f"result_{deg_type}.png"
            img_deg.save(filename)
            print(f"    Saved: {filename}")
            
            if text_clean != text_deg:
                print(f"    ⚠️  Different from clean!")
            else:
                print(f"    ✓ Same as clean")
    
    print("\n" + "=" * 60)
    print("✓ Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()

