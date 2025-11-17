#!/usr/bin/env python3
"""
Quill Simulator
Generates synthetic letter-writing data, including a final image,
metadata, and intermediate anchor frames for PINN training.
"""
import numpy as np
from PIL import Image
import json
from pathlib import Path

class QuillSimulator:
    def __init__(self, width=200, height=200, ink_flow_rate=1.5, evaporation_rate=0.05, diffusion_rate=0.1):
        self.width = width
        self.height = height
        self.ink_flow_rate = ink_flow_rate
        self.evaporation_rate = evaporation_rate
        self.diffusion_rate = diffusion_rate
        
        self.canvas = np.zeros((height, width), dtype=np.float32)
        self.history = []
        
        # Diffusion kernel
        self.kernel = np.array([[0.05, 0.2, 0.05],
                                [0.2, -1, 0.2],
                                [0.05, 0.2, 0.05]])

    def _apply_diffusion(self):
        laplacian = np.convolve(self.canvas.flatten(), self.kernel.flatten(), 'same').reshape(self.height, self.width)
        self.canvas += self.diffusion_rate * laplacian

    def _apply_evaporation(self):
        self.canvas -= self.evaporation_rate * self.canvas
        self.canvas = np.clip(self.canvas, 0, 2.0)

    def _apply_ink(self, x, y, pressure):
        # Pen tip size
        tip_radius = 2
        
        y_grid, x_grid = np.ogrid[-y:self.height-y, -x:self.width-x]
        mask = x_grid*x_grid + y_grid*y_grid <= tip_radius*tip_radius
        
        ink_amount = self.ink_flow_rate * pressure
        self.canvas[mask] += ink_amount

    def run_simulation(self, strokes, total_time, steps_per_second=50):
        num_steps = int(total_time * steps_per_second)
        time_points = np.linspace(0, total_time, num_steps)
        dt = total_time / num_steps

        current_stroke_idx = 0
        
        for t in time_points:
            # Find active stroke
            active_stroke = None
            for stroke in strokes:
                if stroke['start_time'] <= t < stroke['end_time']:
                    active_stroke = stroke
                    break

            if active_stroke:
                # Interpolate pen position
                progress = (t - active_stroke['start_time']) / (active_stroke['end_time'] - active_stroke['start_time'])
                
                start_pt = np.array(active_stroke['points'][0])
                end_pt = np.array(active_stroke['points'][-1])

                # Simple linear interpolation for this example
                current_pos = start_pt * (1 - progress) + end_pt * progress
                x, y = int(current_pos[0]), int(current_pos[1])
                
                self._apply_ink(x, y, pressure=1.0)
            
            # Physics updates
            # self._apply_diffusion() # Diffusion makes it too blurry for this PINN
            self._apply_evaporation()
            
            # Store a snapshot
            self.history.append((t, self.canvas.copy()))

        print(f"✓ Simulation complete. Ran {num_steps} steps.")

    def get_frame_at_time(self, time):
        if not self.history:
            return None
        
        # Find the closest snapshot in history
        closest_time, closest_frame = min(self.history, key=lambda item: abs(item[0] - time))
        return closest_frame

    def save_frame_at_time(self, time, filename):
        frame = self.get_frame_at_time(time)
        if frame is not None:
            # Normalize frame for saving
            img_data = np.clip(frame, 0, 1)
            img_data = (1.0 - img_data) * 255 # Invert: ink is black
            img = Image.fromarray(img_data.astype(np.uint8), 'L')
            img.save(filename)
            print(f"  ✓ Saved frame for t={time:.2f} to {filename}")

def generate_letter_C(width, height):
    # Defines the strokes for the letter 'C'
    total_time = 2.0  # seconds
    
    # A single, curved stroke
    center_x, center_y = width / 2, height / 2
    radius = width / 3
    
    start_angle = np.pi / 4
    end_angle = -np.pi / 4 + 2 * np.pi
    
    # Generate points on the arc
    points = []
    num_points = 50
    for i in range(num_points):
        angle = start_angle + (end_angle - start_angle) * (i / (num_points - 1))
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        points.append((x, y))

    strokes = [
        {
            "start_time": 0.2, # Start after a short delay
            "end_time": 1.8,
            "points": points
        }
    ]
    
    metadata = {
        "letter": "C",
        "total_time": total_time,
        "strokes": strokes
    }
    
    return metadata

if __name__ == "__main__":
    output_dir = Path("synthetic_letters")
    output_dir.mkdir(exist_ok=True)
    
    sim = QuillSimulator(width=100, height=100)
    metadata = generate_letter_C(sim.width, sim.height)
    
    sim.run_simulation(metadata['strokes'], metadata['total_time'])
    
    output_prefix = output_dir / f"CANONE_letter_{0}_{metadata['letter']}"

    # Save metadata
    with open(f"{output_prefix}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✓ Saved metadata to {output_prefix}_metadata.json")

    # Save final image and anchor frames
    print("\nSaving final image and anchor frames...")
    
    # <<< NEW: Save anchor frames for V4 training >>>
    sim.save_frame_at_time(0.0, f"{output_prefix}_t0.00.png")  # Optional, but good for checking
    sim.save_frame_at_time(metadata['total_time'] * 0.25, f"{output_prefix}_t0.25.png")
    sim.save_frame_at_time(metadata['total_time'] * 0.50, f"{output_prefix}_t0.50.png")
    sim.save_frame_at_time(metadata['total_time'] * 0.75, f"{output_prefix}_t0.75.png")
    
    # Save the final frame (t=1.0 of the simulation time)
    sim.save_frame_at_time(metadata['total_time'], f"{output_prefix}.png")
