#!/usr/bin/env python3
"""
Quill Physics Simulator
Generates synthetic medieval-style writing with controllable physical parameters
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw
import json
from pathlib import Path

print("=" * 60)
print("QUILL PHYSICS SIMULATOR")
print("=" * 60)

class QuillSimulator:
    """
    Simulates ink deposition from quill writing using simplified fluid dynamics
    """
    def __init__(self, canvas_size=(256, 256), dt=0.01):
        self.width, self.height = canvas_size
        self.dt = dt  # Time step
        
        # Physical constants
        self.ink_viscosity = 0.01  # Pa·s (typical ink)
        self.evaporation_rate = 0.001  # 1/s
        self.capillary_coefficient = 0.1
        
        # Canvas - accumulated ink height at each pixel
        self.canvas = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Time evolution history
        self.history = []
        self.current_time = 0.0
        
    def reset(self):
        """Clear canvas for new letter"""
        self.canvas = np.zeros((self.height, self.width), dtype=np.float32)
        self.history = []
        self.current_time = 0.0
    
    def deposit_ink(self, x, y, amount, spread):
        """
        Deposit ink at position (x, y) with Gaussian spread
        
        Args:
            x, y: position (float coordinates)
            amount: ink volume
            spread: stroke width (sigma of Gaussian)
        """
        # Create grid for Gaussian
        x_grid = np.arange(self.width)
        y_grid = np.arange(self.height)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Gaussian ink deposition
        gaussian = np.exp(-((X - x)**2 + (Y - y)**2) / (2 * spread**2))
        gaussian = gaussian * amount / (2 * np.pi * spread**2)
        
        # Add to canvas
        self.canvas += gaussian
        
    def apply_fluid_dynamics(self):
        """
        Apply simplified thin film fluid dynamics
        Models ink spreading and evaporation
        """
        # Laplacian for diffusion (ink spreads)
        from scipy.ndimage import laplace
        diffusion = self.capillary_coefficient * laplace(self.canvas)
        
        # Evaporation (thicker ink evaporates more)
        evaporation = -self.evaporation_rate * self.canvas * self.dt
        
        # Update canvas
        self.canvas += (diffusion + evaporation) * self.dt
        
        # Physical constraint: ink height >= 0
        self.canvas = np.maximum(self.canvas, 0)
    
    def draw_stroke(self, start, end, pressure, speed, samples=50):
        """
        Draw a single stroke from start to end
        
        Args:
            start: (x, y) starting point
            end: (x, y) ending point
            pressure: quill pressure (affects stroke width)
            speed: writing speed (affects ink deposition rate)
            samples: number of points along stroke
        """
        x0, y0 = start
        x1, y1 = end
        
        # Interpolate path
        t_values = np.linspace(0, 1, samples)
        
        stroke_metadata = {
            'start': start,
            'end': end,
            'pressure': pressure,
            'speed': speed,
            'start_time': self.current_time,
            'points': []
        }
        
        for t in t_values:
            # Current position
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            
            # Stroke width depends on pressure
            width = 2.0 + pressure * 3.0  # Range: 2-5 pixels
            
            # Ink amount depends on speed (slower = more ink)
            ink_amount = 0.1 / (speed + 0.1)
            
            # Deposit ink
            self.deposit_ink(x, y, ink_amount, spread=width)
            
            # Apply fluid dynamics
            self.apply_fluid_dynamics()
            
            # Save snapshot periodically
            if len(self.history) == 0 or self.current_time - self.history[-1]['time'] > 0.05:
                self.history.append({
                    'time': self.current_time,
                    'canvas': self.canvas.copy()
                })
            
            stroke_metadata['points'].append({
                'x': float(x),
                'y': float(y),
                'time': self.current_time,
                'width': float(width),
                'ink_amount': float(ink_amount)
            })
            
            self.current_time += self.dt / speed  # Time depends on speed
        
        stroke_metadata['end_time'] = self.current_time
        
        return stroke_metadata

class LetterGenerator:
    """
    Generates synthetic letters with controllable stroke parameters
    """
    def __init__(self, canvas_size=(256, 256)):
        self.canvas_size = canvas_size
        
        # Define letter strokes (simplified medieval uncial-style)
        self.letter_strokes = {
            'A': [
                [(64, 200), (128, 50)],   # Left diagonal
                [(128, 50), (192, 200)],  # Right diagonal
                [(90, 140), (166, 140)]   # Cross bar
            ],
            'B': [
                [(80, 50), (80, 200)],    # Vertical stem
                [(80, 50), (160, 90)],    # Upper curve
                [(160, 90), (80, 125)],   # Return to stem
                [(80, 125), (170, 165)],  # Lower curve
                [(170, 165), (80, 200)]   # Return to stem
            ],
            'C': [
                [(180, 80), (120, 50)],   # Top curve
                [(120, 50), (80, 100)],   # Left curve top
                [(80, 100), (80, 150)],   # Left vertical
                [(80, 150), (120, 200)],  # Left curve bottom
                [(120, 200), (180, 170)]  # Bottom curve
            ],
            'E': [
                [(80, 50), (80, 200)],    # Vertical stem
                [(80, 50), (180, 50)],    # Top horizontal
                [(80, 125), (160, 125)],  # Middle horizontal
                [(80, 200), (180, 200)]   # Bottom horizontal
            ],
            'I': [
                [(128, 60), (128, 190)]   # Simple vertical
            ],
            'N': [
                [(70, 200), (70, 50)],    # Left vertical
                [(70, 50), (186, 200)],   # Diagonal
                [(186, 200), (186, 50)]   # Right vertical
            ],
            'O': [
                [(128, 50), (180, 100)],  # Top right
                [(180, 100), (180, 150)], # Right side
                [(180, 150), (128, 200)], # Bottom right
                [(128, 200), (76, 150)],  # Bottom left
                [(76, 150), (76, 100)],   # Left side
                [(76, 100), (128, 50)]    # Top left
            ],
            'T': [
                [(60, 60), (196, 60)],    # Top horizontal
                [(128, 60), (128, 190)]   # Vertical stem
            ]
        }
    
    def generate_letter(self, letter, pressure_variation=0.3, speed_variation=0.3):
        """
        Generate a letter with random variation in physical parameters
        
        Args:
            letter: Letter to generate ('A', 'B', etc.)
            pressure_variation: Random variation in pressure (0-1)
            speed_variation: Random variation in speed (0-1)
        
        Returns:
            final_image: Final rendered letter
            metadata: All stroke parameters and timing
        """
        if letter not in self.letter_strokes:
            raise ValueError(f"Letter '{letter}' not defined")
        
        sim = QuillSimulator(canvas_size=self.canvas_size)
        
        strokes = self.letter_strokes[letter]
        metadata = {
            'letter': letter,
            'strokes': [],
            'canvas_size': self.canvas_size
        }
        
        for i, (start, end) in enumerate(strokes):
            # Randomize physical parameters
            pressure = 0.5 + np.random.randn() * pressure_variation
            pressure = np.clip(pressure, 0.1, 1.0)
            
            speed = 1.0 + np.random.randn() * speed_variation
            speed = np.clip(speed, 0.3, 2.0)
            
            print(f"  Stroke {i+1}/{len(strokes)}: "
                  f"pressure={pressure:.2f}, speed={speed:.2f}")
            
            # Draw stroke
            stroke_meta = sim.draw_stroke(start, end, pressure, speed)
            stroke_meta['stroke_id'] = i
            metadata['strokes'].append(stroke_meta)
        
        # Final canvas
        final_image = sim.canvas.copy()
        
        # Normalize to 0-255 range for image
        final_image_normalized = (final_image / final_image.max() * 255).astype(np.uint8) if final_image.max() > 0 else final_image.astype(np.uint8)
        
        # Invert (black ink on white background)
        final_image_normalized = 255 - final_image_normalized
        
        metadata['history'] = sim.history
        metadata['total_time'] = sim.current_time
        
        return final_image_normalized, metadata

def generate_word(word, output_dir='synthetic_letters', spacing=20):
    """
    Generate a complete word with multiple letters
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    word = word.upper()
    
    print(f"\nGenerating word: '{word}'")
    print("=" * 60)
    
    # Generate each letter
    letter_images = []
    all_metadata = []
    
    for i, letter in enumerate(word):
        if letter == ' ':
            # Add space
            letter_images.append(np.ones((256, 50), dtype=np.uint8) * 255)
            continue
        
        print(f"\nLetter {i+1}/{len(word)}: {letter}")
        
        gen = LetterGenerator()
        img, metadata = gen.generate_letter(
            letter,
            pressure_variation=0.2,
            speed_variation=0.2
        )
        
        letter_images.append(img)
        all_metadata.append(metadata)
        
        # Save individual letter
        Image.fromarray(img).save(output_dir / f"{word}_letter_{i}_{letter}.png")
        
        # Save metadata
        # Remove numpy arrays from history for JSON serialization
        metadata_save = metadata.copy()
        metadata_save['history'] = [
            {'time': h['time'], 'shape': h['canvas'].shape}
            for h in metadata['history']
        ]
        
        with open(output_dir / f"{word}_letter_{i}_{letter}_metadata.json", 'w') as f:
            json.dump(metadata_save, f, indent=2)
    
    # Concatenate letters horizontally
    word_image = np.concatenate(letter_images, axis=1)
    
    # Save word image
    Image.fromarray(word_image).save(output_dir / f"{word}_complete.png")
    
    print(f"\n✓ Generated word '{word}'")
    print(f"✓ Saved to: {output_dir}/")
    
    return word_image, all_metadata

def visualize_temporal_evolution(metadata, output_path='evolution.gif'):
    """
    Create animation showing letter being written over time
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    
    history = metadata['history']
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    def update(frame):
        ax.clear()
        canvas = history[frame]['canvas']
        # Invert for display
        display = 255 - (canvas / canvas.max() * 255 if canvas.max() > 0 else canvas)
        ax.imshow(display, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f"Time: {history[frame]['time']:.2f}s")
        ax.axis('off')
    
    anim = FuncAnimation(fig, update, frames=len(history), interval=50)
    anim.save(output_path, writer=PillowWriter(fps=20))
    plt.close()
    
    print(f"✓ Saved animation: {output_path}")

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    import sys
    
    # Generate sample letters
    print("\n[Test 1] Generating individual letters...")
    gen = LetterGenerator()
    
    for letter in ['A', 'B', 'C']:
        print(f"\nGenerating letter '{letter}'...")
        img, metadata = gen.generate_letter(letter)
        
        # Save
        Image.fromarray(img).save(f"letter_{letter}.png")
        print(f"✓ Saved: letter_{letter}.png")
    
    # Generate word
    print("\n" + "=" * 60)
    print("[Test 2] Generating word...")
    word_img, word_meta = generate_word("CANONE")
    
    # Visualize first letter evolution
    print("\n" + "=" * 60)
    print("[Test 3] Creating temporal animation...")
    if word_meta:
        visualize_temporal_evolution(word_meta[0], 'letter_evolution.gif')
    
    print("\n" + "=" * 60)
    print("✓ All synthetic data generated!")
    print("\nGenerated files:")
    print("  - letter_A.png, letter_B.png, letter_C.png")
    print("  - synthetic_letters/CANONE_*.png")
    print("  - synthetic_letters/CANONE_*_metadata.json")
    print("  - letter_evolution.gif")
    print("=" * 60)
