#!/usr/bin/env python3
"""
Quill Simulator (DEFINITIVE & CORRECTED)
This version fixes all data generation bugs, ensuring the stroke path is correctly
interpolated and the output metadata.json has the correct, clean data structure.
"""
import numpy as np
from PIL import Image
import json
from pathlib import Path

class QuillSimulator:
    def __init__(self, width=100, height=100, ink_flow_rate=1.5, evaporation_rate=0.05):
        self.width = width
        self.height = height
        self.ink_flow_rate = ink_flow_rate
        self.evaporation_rate = evaporation_rate
        self.canvas = np.zeros((height, width), dtype=np.float32)
        self.history = []

    def _apply_evaporation(self):
        self.canvas -= self.evaporation_rate * self.canvas
        self.canvas = np.clip(self.canvas, 0, 2.0)

    def _apply_ink(self, x, y, pressure):
        tip_radius = 2
        y_grid, x_grid = np.ogrid[-y:self.height-y, -x:self.width-x]
        mask = x_grid*x_grid + y_grid*y_grid <= tip_radius*tip_radius
        self.canvas[mask] += self.ink_flow_rate * pressure

    def run_simulation(self, strokes, total_time, steps_per_second=50):
        num_steps = int(total_time * steps_per_second)
        time_points = np.linspace(0, total_time, num_steps)
        
        for t in time_points:
            active_stroke = None
            for stroke in strokes:
                if stroke['start_time'] <= t < stroke['end_time']:
                    active_stroke = stroke
                    break

            if active_stroke:
                progress = (t - active_stroke['start_time']) / (active_stroke['end_time'] - active_stroke['start_time'])
                points_arr = np.array(active_stroke['points'])
                idx_float = progress * (len(points_arr) - 1)
                idx0, idx1 = int(np.floor(idx_float)), int(np.ceil(idx_float))
                if idx1 >= len(points_arr): idx1 = idx0 = len(points_arr) - 1
                
                if idx0 == idx1: current_pos = points_arr[idx0]
                else:
                    local_progress = idx_float - idx0
                    current_pos = points_arr[idx0] * (1 - local_progress) + points_arr[idx1] * local_progress
                
                x, y = int(current_pos[0]), int(current_pos[1])
                self._apply_ink(x, y, pressure=1.0)
            
            self._apply_evaporation()
            self.history.append((t, self.canvas.copy()))
        print(f"✓ Simulation complete. Ran {num_steps} steps.")

    def get_frame_at_time(self, time):
        if not self.history: return None
        _, closest_frame = min(self.history, key=lambda item: abs(item[0] - time))
        return closest_frame

    def save_frame_at_time(self, time, filename):
        frame = self.get_frame_at_time(time)
        if frame is not None:
            img_data = np.clip(frame, 0, 1)
            img_data = (1.0 - img_data) * 255
            img = Image.fromarray(img_data.astype(np.uint8), 'L')
            img.save(filename)
            print(f"  ✓ Saved frame for t={time:.2f} to {filename}")

def generate_letter_C(width, height):
    total_time = 2.0
    center_x, center_y = width / 2, height / 2
    radius = width / 3
    start_angle, end_angle = np.pi / 4, -np.pi / 4 + 2 * np.pi
    points = []
    num_points = 50
    for i in range(num_points):
        angle = start_angle + (end_angle - start_angle) * (i / (num_points - 1))
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        # Ensure points are saved as lists, which is JSON-friendly
        points.append([x, y]) 
    strokes = [{"start_time": 0.2, "end_time": 1.8, "points": points}]
    metadata = {"letter": "C", "total_time": total_time, "strokes": strokes}
    return metadata

if __name__ == "__main__":
    output_dir = Path("synthetic_letters")
    output_dir.mkdir(exist_ok=True)
    sim = QuillSimulator(width=100, height=100)
    metadata = generate_letter_C(sim.width, sim.height)
    sim.run_simulation(metadata['strokes'], metadata['total_time'])
    output_prefix = output_dir / f"CANONE_letter_{0}_{metadata['letter']}"
    with open(f"{output_prefix}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✓ Saved metadata to {output_prefix}_metadata.json")
    print("\nSaving final image...")
    sim.save_frame_at_time(metadata['total_time'], f"{output_prefix}.png")
