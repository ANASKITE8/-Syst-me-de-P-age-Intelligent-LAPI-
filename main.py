"""
Professional Vehicle Detection System with UI
==============================================
Main script with visual dashboard for demonstration purposes
"""

import cv2
from ultralytics import YOLO
from collections import defaultdict
import cvzone
import os
import time
from datetime import datetime, timedelta


class VehicleDetectionSystem:
    def __init__(self, video_path, model_path='models/yolo11s.pt'):
        # Initialize paths
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.video_path = os.path.join(self.script_dir, video_path)
        
        # Load YOLO model
        print("🚀 Initializing AI Vehicle Detection System...")
        self.model = YOLO(model_path)
        self.class_list = self.model.names
        
        # Video capture
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise Exception("❌ Could not open video file")
        
        # Video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Detection settings
        self.line_y = 430
        self.line_start_x = 690
        self.line_end_x = 1130
        
        # Tracking data
        self.class_counts = defaultdict(int)
        self.crossed_ids = set()
        self.vehicle_colors = {
            'car': (50, 205, 50),
            'truck': (255, 140, 0),
            'bus': (65, 105, 225),
            'motorcycle': (255, 20, 147),
            'bicycle': (0, 255, 255),
            'train': (138, 43, 226)
        }
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.paused = False
        
        # Output video
        output_path = os.path.join(self.script_dir, 'outputs', 'output_professional.mp4')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.output_video = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps,
            (self.frame_width, self.frame_height)
        )
        
        print(f"✅ System Ready | Resolution: {self.frame_width}x{self.frame_height} | FPS: {self.fps:.1f}")
        print(f"📹 Total Frames: {self.total_frames}")
        print("\n" + "="*70)
        print("🎮 CONTROLS:")
        print("   SPACE    - Pause/Resume")
        print("   ESC      - Exit")
        print("   R        - Reset counts")
        print("="*70 + "\n")
    
    def draw_dashboard(self, frame):
        """Draw professional dashboard overlay"""
        overlay = frame.copy()
        
        # Dashboard background (top-left)
        cv2.rectangle(overlay, (10, 10), (420, 280), (30, 30, 30), -1)
        cv2.rectangle(overlay, (10, 10), (420, 280), (100, 100, 100), 2)
        
        # Title bar
        cv2.rectangle(overlay, (10, 10), (420, 50), (50, 50, 150), -1)
        cv2.putText(overlay, "TRAFFIC MONITORING SYSTEM", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Statistics section
        y_pos = 80
        total_vehicles = sum(self.class_counts.values())
        
        # Total count
        cv2.putText(overlay, "TOTAL VEHICLES:", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay, str(total_vehicles), (250, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y_pos += 30
        cv2.line(overlay, (20, y_pos), (400, y_pos), (100, 100, 100), 1)
        y_pos += 25
        
        # Individual counts
        for class_name, count in sorted(self.class_counts.items()):
            color = self.vehicle_colors.get(class_name, (255, 255, 255))
            
            cv2.putText(overlay, f"{class_name.upper()}:", (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(overlay, str(count), (250, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Progress bar
            bar_width = int((count / max(total_vehicles, 1)) * 120)
            cv2.rectangle(overlay, (280, y_pos - 12), (280 + bar_width, y_pos - 2), color, -1)
            
            y_pos += 30
        
        # Performance stats (top-right)
        cv2.rectangle(overlay, (self.frame_width - 310, 10), 
                     (self.frame_width - 10, 160), (30, 30, 30), -1)
        cv2.rectangle(overlay, (self.frame_width - 310, 10), 
                     (self.frame_width - 10, 160), (100, 100, 100), 2)
        
        cv2.rectangle(overlay, (self.frame_width - 310, 10), 
                     (self.frame_width - 10, 50), (50, 150, 50), -1)
        cv2.putText(overlay, "PERFORMANCE", (self.frame_width - 290, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # FPS
        elapsed_time = time.time() - self.start_time
        current_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        cv2.putText(overlay, f"FPS: {current_fps:.1f}", 
                   (self.frame_width - 290, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Progress
        progress = (self.frame_count / self.total_frames) * 100 if self.total_frames > 0 else 0
        cv2.putText(overlay, f"Progress: {progress:.1f}%", 
                   (self.frame_width - 290, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Time
        time_str = str(timedelta(seconds=int(elapsed_time)))
        cv2.putText(overlay, f"Time: {time_str}", 
                   (self.frame_width - 290, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 255), 1)
        
        # Status indicator
        status_text = "PAUSED" if self.paused else "PROCESSING"
        status_color = (0, 165, 255) if self.paused else (0, 255, 0)
        cv2.rectangle(overlay, (10, self.frame_height - 50), 
                     (200, self.frame_height - 10), (30, 30, 30), -1)
        cv2.putText(overlay, status_text, (20, self.frame_height - 23),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Blend overlay
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        return frame
    
    def draw_detection_line(self, frame):
        """Draw the detection line"""
        cv2.line(frame, (self.line_start_x, self.line_y), 
                (self.line_end_x, self.line_y), (0, 0, 255), 4)
        cv2.circle(frame, (self.line_start_x, self.line_y), 8, (0, 0, 255), -1)
        cv2.circle(frame, (self.line_end_x, self.line_y), 8, (0, 0, 255), -1)
        
        cv2.rectangle(frame, (self.line_start_x - 5, self.line_y - 35),
                     (self.line_start_x + 130, self.line_y - 10), (0, 0, 255), -1)
        cv2.putText(frame, "COUNTING LINE", (self.line_start_x, self.line_y - 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def process_detections(self, frame, results):
        """Process detections and draw annotations"""
        if results[0].boxes is None or len(results[0].boxes) == 0:
            return frame
        
        boxes = results[0].boxes.xyxy.cpu()
        class_indices = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu()
        
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
        else:
            track_ids = list(range(len(boxes)))
        
        for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            class_name = self.class_list[class_idx]
            color = self.vehicle_colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.circle(frame, (cx, cy), 6, color, -1)
            cv2.circle(frame, (cx, cy), 8, color, 2)
            cv2.line(frame, (cx, cy), (cx, self.line_y), color, 1)
            
            # Label
            label = f"ID:{track_id} {class_name.upper()} {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Check crossing
            if cy > self.line_y and track_id not in self.crossed_ids:
                self.crossed_ids.add(track_id)
                self.class_counts[class_name] += 1
                print(f"✓ {class_name.upper()} crossed | ID: {track_id}")
        
        return frame
    
    def run(self):
        """Main processing loop"""
        try:
            while self.cap.isOpened():
                if not self.paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("\n✅ Video completed!")
                        break
                    
                    self.frame_count += 1
                    
                    results = self.model.track(frame, persist=True, 
                                              classes=[1, 2, 3, 5, 6, 7], 
                                              verbose=False)
                    
                    self.draw_detection_line(frame)
                    frame = self.process_detections(frame, results)
                    frame = self.draw_dashboard(frame)
                    self.output_video.write(frame)
                else:
                    frame = self.draw_dashboard(frame)
                
                cv2.imshow("AI Traffic Monitoring System", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == 32:  # SPACE
                    self.paused = not self.paused
                elif key == ord('r') or key == ord('R'):
                    self.class_counts.clear()
                    self.crossed_ids.clear()
                    print("\n🔄 Counts reset")
            
            self.cleanup()
            
        except KeyboardInterrupt:
            print("\n⏹ Interrupted")
            self.cleanup()
    
    def cleanup(self):
        """Cleanup and print summary"""
        elapsed_time = time.time() - self.start_time
        
        print("\n" + "="*70)
        print("📊 FINAL STATISTICS")
        print("="*70)
        print(f"⏱ Time: {str(timedelta(seconds=int(elapsed_time)))}")
        print(f"🎬 Frames: {self.frame_count}/{self.total_frames}")
        print(f"⚡ Avg FPS: {self.frame_count/elapsed_time:.2f}")
        
        total = sum(self.class_counts.values())
        if total > 0:
            print(f"\n🚗 VEHICLE COUNTS:")
            for name, count in sorted(self.class_counts.items(), key=lambda x: x[1], reverse=True):
                pct = (count / total) * 100
                print(f"   {name.upper():15s}: {count:3d} ({pct:5.1f}%)")
            print(f"   {'TOTAL':15s}: {total:3d}")
        
        print("="*70)
        
        self.cap.release()
        self.output_video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        system = VehicleDetectionSystem('test_videos/4.mp4')
        system.run()
    except FileNotFoundError:
        print("❌ Video file not found!")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        input("\n✨ Press Enter to exit...")
    