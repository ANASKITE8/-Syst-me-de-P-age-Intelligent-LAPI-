"""
Vehicle Detection Module for License Plate Recognition Pipeline
================================================================
This module handles Step 1: Vehicle Detection and Tracking

Outputs:
- Visual video with annotations (output_professional.mp4)
- JSON file with detection data (detections.json)
- Cropped vehicle images in outputs/vehicles/ folder
"""

import cv2
from ultralytics import YOLO
from collections import defaultdict
import json
import os
from datetime import datetime
from pathlib import Path


class VehicleDetector:
    """
    Vehicle Detection and Tracking System
    Detects vehicles, tracks them, and prepares data for license plate detection
    """
    
    def __init__(self, model_path='models/yolo11s.pt', output_dir='outputs'):
        """
        Initialize the Vehicle Detector
        
        Args:
            model_path: Path to YOLO model file
            output_dir: Directory to save outputs
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'vehicles').mkdir(exist_ok=True)
        (self.output_dir / 'frames').mkdir(exist_ok=True)
        
        # Load YOLO model
        print("🚀 Loading YOLO model...")
        self.model = YOLO(model_path)
        self.class_list = self.model.names
        
        # Vehicle classes we're interested in (COCO dataset)
        self.vehicle_classes = {
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            6: 'train',
            7: 'truck'
        }
        
        # Detection data storage
        self.all_detections = []
        self.frame_count = 0
        
        print("✅ Vehicle Detector initialized")
    
    def detect_vehicles(self, frame, frame_number, save_crops=True):
        """
        Detect vehicles in a single frame
        
        Args:
            frame: Input frame (numpy array)
            frame_number: Current frame number
            save_crops: Whether to save cropped vehicle images
            
        Returns:
            detections: List of detection dictionaries
            annotated_frame: Frame with bounding boxes drawn
        """
        # Run YOLO detection with tracking
        results = self.model.track(
            frame, 
            persist=True, 
            classes=list(self.vehicle_classes.keys()),
            verbose=False
        )
        
        detections = []
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            class_indices = results[0].boxes.cls.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            # Get tracking IDs
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
            else:
                track_ids = list(range(len(boxes)))
            
            # Process each detection
            for idx, (box, track_id, class_idx, conf) in enumerate(
                zip(boxes, track_ids, class_indices, confidences)
            ):
                x1, y1, x2, y2 = map(int, box)
                class_name = self.vehicle_classes.get(class_idx, 'unknown')
                
                # Create detection record
                detection = {
                    'frame_number': frame_number,
                    'track_id': track_id,
                    'vehicle_type': class_name,
                    'confidence': float(conf),
                    'bbox': {
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'width': x2 - x1,
                        'height': y2 - y1
                    },
                    'center': {
                        'x': int((x1 + x2) / 2),
                        'y': int((y1 + y2) / 2)
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
                detections.append(detection)
                self.all_detections.append(detection)
                
                # Save cropped vehicle image
                if save_crops and conf > 0.5:  # Only save high confidence detections
                    vehicle_crop = frame[y1:y2, x1:x2]
                    crop_filename = f"vehicle_{frame_number:06d}_{track_id:04d}.jpg"
                    crop_path = self.output_dir / 'vehicles' / crop_filename
                    cv2.imwrite(str(crop_path), vehicle_crop)
                    detection['crop_path'] = str(crop_path)
        
        return detections, results
    
    def annotate_frame(self, frame, detections):
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input frame
            detections: List of detections for this frame
            
        Returns:
            annotated_frame: Frame with annotations
        """
        annotated = frame.copy()
        
        # Vehicle type colors
        colors = {
            'car': (50, 205, 50),
            'truck': (255, 140, 0),
            'bus': (65, 105, 225),
            'motorcycle': (255, 20, 147),
            'bicycle': (0, 255, 255),
            'train': (138, 43, 226)
        }
        
        for det in detections:
            x1 = det['bbox']['x1']
            y1 = det['bbox']['y1']
            x2 = det['bbox']['x2']
            y2 = det['bbox']['y2']
            
            color = colors.get(det['vehicle_type'], (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            cx, cy = det['center']['x'], det['center']['y']
            cv2.circle(annotated, (cx, cy), 5, color, -1)
            
            # Draw label
            label = f"ID:{det['track_id']} {det['vehicle_type']} {det['confidence']:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(annotated, (x1, y1 - label_h - 10), 
                         (x1 + label_w + 10, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated
    
    def process_video(self, video_path, output_video_path=None, 
                     save_frames=False, frame_skip=1):
        """
        Process entire video and detect vehicles
        
        Args:
            video_path: Path to input video
            output_video_path: Path to save annotated video
            save_frames: Whether to save individual frames
            frame_skip: Process every Nth frame (1 = all frames)
            
        Returns:
            statistics: Detection statistics dictionary
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")
        
        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
        
        # Video writer
        if output_video_path is None:
            output_video_path = self.output_dir / 'output_professional.mp4'
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        frame_number = 0
        vehicle_counts = defaultdict(int)
        
        print("🎬 Processing video...")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_number += 1
                
                # Skip frames if needed
                if frame_number % frame_skip != 0:
                    continue
                
                # Progress indicator
                if frame_number % 30 == 0:
                    progress = (frame_number / total_frames) * 100
                    print(f"⏳ Frame {frame_number}/{total_frames} ({progress:.1f}%)")
                
                # Detect vehicles
                detections, _ = self.detect_vehicles(frame, frame_number, save_crops=True)
                
                # Count vehicles
                for det in detections:
                    vehicle_counts[det['vehicle_type']] += 1
                
                # Annotate frame
                annotated = self.annotate_frame(frame, detections)
                
                # Save frame if needed
                if save_frames:
                    frame_path = self.output_dir / 'frames' / f"frame_{frame_number:06d}.jpg"
                    cv2.imwrite(str(frame_path), annotated)
                
                # Write to output video
                out.write(annotated)
                
                # Display (optional)
                cv2.imshow("Vehicle Detection", annotated)
                if cv2.waitKey(1) == 27:  # ESC to exit
                    print("\n⏹ Processing stopped by user")
                    break
        
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
        
        # Save detection data to JSON
        self.save_detections_json()
        
        # Calculate statistics
        statistics = {
            'total_frames_processed': frame_number,
            'total_detections': len(self.all_detections),
            'unique_vehicles': len(set(d['track_id'] for d in self.all_detections)),
            'vehicle_counts': dict(vehicle_counts),
            'output_video': str(output_video_path),
            'detections_json': str(self.output_dir / 'detections.json')
        }
        
        return statistics
    
    def save_detections_json(self, filename='detections.json'):
        """
        Save all detections to JSON file for next pipeline step
        
        Args:
            filename: Output JSON filename
        """
        output_path = self.output_dir / filename
        
        data = {
            'metadata': {
                'total_detections': len(self.all_detections),
                'unique_vehicles': len(set(d['track_id'] for d in self.all_detections)),
                'timestamp': datetime.now().isoformat(),
                'model': self.model_path
            },
            'detections': self.all_detections
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Saved detections to {output_path}")
    
    def get_statistics(self):
        """Get detection statistics"""
        vehicle_counts = defaultdict(int)
        for det in self.all_detections:
            vehicle_counts[det['vehicle_type']] += 1
        
        return {
            'total_detections': len(self.all_detections),
            'unique_vehicles': len(set(d['track_id'] for d in self.all_detections)),
            'vehicle_counts': dict(vehicle_counts)
        }


def main():
    """Example usage of VehicleDetector"""
    
    # Initialize detector
    detector = VehicleDetector(
        model_path='models/yolo11s.pt',
        output_dir='outputs'
    )
    
    # Process video
    video_path = 'test_videos/4.mp4'
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    print("="*70)
    print("🚗 VEHICLE DETECTION PIPELINE - STEP 1")
    print("="*70)
    
    # Process
    stats = detector.process_video(
        video_path=video_path,
        save_frames=False,  # Set to True if you need individual frames
        frame_skip=1  # Process every frame (change to 2 for every other frame)
    )
    
    # Print results
    print("\n" + "="*70)
    print("📊 DETECTION COMPLETE")
    print("="*70)
    print(f"✅ Frames processed: {stats['total_frames_processed']}")
    print(f"✅ Total detections: {stats['total_detections']}")
    print(f"✅ Unique vehicles: {stats['unique_vehicles']}")
    print(f"\n🚗 Vehicle counts:")
    for vehicle_type, count in stats['vehicle_counts'].items():
        print(f"   {vehicle_type}: {count}")
    print(f"\n📁 Outputs:")
    print(f"   Video: {stats['output_video']}")
    print(f"   JSON: {stats['detections_json']}")
    print(f"   Crops: outputs/vehicles/")
    print("="*70)
    
    input("\n✨ Press Enter to exit...")


if __name__ == "__main__":
    main()