import os
import json
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from shapely.geometry import Point, Polygon
from datetime import datetime, timedelta
from shapely.geometry import Polygon


# Utility functions
def point_in_zone(point, zone):
    try:
        return zone.contains(Point(point))
    except Exception as e:
        print(f"Error in point_in_zone for point {point}: {e}")
        return False

def find_zone(point, zone_dict):
    for name, polygon in zone_dict.items():
        if point_in_zone(point, polygon):
            return name
    return None

# Default turn mapping
TURN_MAP = {
    "North_in": {"East_out": "left", "South_out": "straight", "West_out": "right", "North_out": "U"},
    "East_in": {"South_out": "left", "West_out": "straight", "North_out": "right", "East_out": "U"},
    "South_in": {"West_out": "left", "North_out": "straight", "East_out": "right", "South_out": "U"},
    "West_in": {"North_out": "left", "East_out": "straight", "South_out": "right", "West_out": "U"}
}

# Bounding box colors
TURNING_BOUNDING_BOX_COLOR = {
    "left": (0, 255, 0),
    "right": (0, 0, 255),
    "straight": (255, 0, 0),
    "U": (255, 255, 0),
    "unknown": (255, 255, 255)
}

# VehicleTracking class
class VehicleTracking:
    def __init__(self, track_id: int, timestamp: str):
        self.track_id = track_id
        self.entry_timestamp = timestamp
        self.entered_into = None
        self.exited_out = None
        self.turn = None
        self.turn_at = None
        self.path_points = []

    def add_path_point(self, point: tuple):
        self.path_points.append(point)

    def set_entry(self, entry_zone: str):
        self.entered_into = entry_zone

    def set_exit(self, exit_zone: str):
        self.exited_out = exit_zone

    def set_turn(self, turn: str, timestamp: str):
        self.turn = turn
        self.turn_at = timestamp

    def get_vehicle_entered_into(self):
        return self.entered_into

    def get_vehicle_exited_out(self):
        return self.exited_out

    def get_turn_details(self) -> dict:
        if self.turn and self.entered_into and self.exited_out and self.turn_at:
            return {
                "vehicle_id": self.track_id,
                "entry_zone": self.entered_into,
                "exit_zone": self.exited_out,
                "turn_type": self.turn,
                "entry_timestamp": self.entry_timestamp,
                "turn_timestamp": self.turn_at
            }
        return None

# TrafficAnalyzer class
class TrafficAnalyzer:
    def __init__(self):
        self.setup_cuda()
        self.setup_models()
        self.setup_colors()
        self.setup_tracking_data()

    def setup_cuda(self):
        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda_available else 'cpu')
        if self.cuda_available:
            print(f"CUDA available: Using GPU - {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()
        else:
            print("CUDA not available: Using CPU")

    def setup_models(self):
        self.model = YOLO("yolov8m.pt")
        self.model.to(self.device)
        print(f"YOLO model loaded on {'GPU' if self.cuda_available else 'CPU'}")
        self.model.conf = 0.35 
        self.model.iou = 0.45  
        self.CAR_CLASSES = [2]  
        
        self.tracker = DeepSort(
            max_age=100,
            n_init=3,
            max_iou_distance=0.7,
            max_cosine_distance=0.3,
            nn_budget=120
        )

    def setup_colors(self):
        self.turning_bb_color = TURNING_BOUNDING_BOX_COLOR
        self.trajectory_color = (255, 255, 255)
        self.entry_zone_color = (0, 255, 255)
        self.exit_zone_color = (255, 0, 255)

    def setup_tracking_data(self):
        self.vehicle_tracks = {}
        self.turn_counts = {"left": 0, "right": 0, "straight": 0, "U": 0, "unknown": 0}
        self.total_cars = 0
        self.previous_tracks = set()

# VideoProcessor class
class VideoProcessor:
    def __init__(self, analyzer):
        self.analyzer = analyzer

    def get_box_details(self, boxes):
        return boxes.cls, boxes.xyxy, boxes.conf, boxes.xywh

    def draw_label(self, img, text, pos, bg_color, text_color=(255, 255, 255)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        end_x = pos[0] + size[0] + 2
        end_y = pos[1] - size[1] - 2
        cv2.rectangle(img, pos, (end_x, end_y), bg_color, -1)
        cv2.putText(img, text, (pos[0], pos[1] - 2), font, font_scale, text_color, thickness)

    def draw_enhanced_bbox(self, frame, x1, y1, x2, y2, track_id, turn_type=None):
        bb_color = self.analyzer.turning_bb_color.get(turn_type, self.analyzer.turning_bb_color["unknown"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), bb_color, 2)
        self._draw_corner_markers(frame, x1, y1, x2, y2, bb_color)
        
        label_text = f"ID:{track_id}"
        if turn_type:
            label_text += f" ({turn_type})"
        self.draw_label(frame, label_text, (x1, y1-10), bb_color)

    def _draw_corner_markers(self, frame, x1, y1, x2, y2, color):
        corner_length = 15
        corner_thickness = 3
        corners = [
            ((x1, y1), (x1 + corner_length, y1), (x1, y1 + corner_length)),
            ((x2, y1), (x2 - corner_length, y1), (x2, y1 + corner_length)),
            ((x1, y2), (x1 + corner_length, y2), (x1, y2 - corner_length)),
            ((x2, y2), (x2 - corner_length, y2), (x2, y2 - corner_length))
        ]
        for start, end1, end2 in corners:
            cv2.line(frame, start, end1, color, corner_thickness)
            cv2.line(frame, start, end2, color, corner_thickness)

    def draw_zones(self, frame, entry_zones, exit_zones):
        for name, poly in entry_zones.items():
            points = np.array(poly.exterior.coords, np.int32)[:-1]
            cv2.polylines(frame, [points], True, self.analyzer.entry_zone_color, 2)
            cv2.putText(frame, name, (points[0][0], points[0][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.analyzer.entry_zone_color, 2)
        
        for name, poly in exit_zones.items():
            points = np.array(poly.exterior.coords, np.int32)[:-1]
            cv2.polylines(frame, [points], True, self.analyzer.exit_zone_color, 2)
            cv2.putText(frame, name, (points[0][0], points[0][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.analyzer.exit_zone_color, 2)

    def draw_debug_info(self, frame, total_detected, valid_tracked, turn_counts, total_cars):
        cv2.rectangle(frame, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 200), (255, 255, 255), 2)
        
        stats = [
            f"Detected this frame: {total_detected}",
            f"Actively tracked: {valid_tracked}",
            f"Total vehicles: {total_cars}"
        ]
        for i, stat in enumerate(stats):
            cv2.putText(frame, stat, (20, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        y_offset = 105
        cv2.putText(frame, "Turn Counts:", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 25
        for turn_type, count in turn_counts.items():
            if turn_type in self.analyzer.turning_bb_color:
                color = self.analyzer.turning_bb_color[turn_type]
                cv2.putText(frame, f"{turn_type.title()}: {count}", (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                y_offset += 20

        self._draw_color_legend(frame, frame.shape[1])

    def _draw_color_legend(self, frame, frame_width):
        legend_x = frame_width - 200
        legend_y = 20
        legend_height = 120
        cv2.rectangle(frame, (legend_x-10, legend_y-10), (frame_width-10, legend_y+legend_height), (0, 0, 0), -1)
        cv2.putText(frame, "Turn Colors:", (legend_x, legend_y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_pos = legend_y + 25
        for turn_type, color in self.analyzer.turning_bb_color.items():
            if turn_type != 'unknown':
                cv2.rectangle(frame, (legend_x, y_pos), (legend_x+15, y_pos+10), color, -1)
                cv2.putText(frame, turn_type.title(), (legend_x+20, y_pos+8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                y_pos += 15



def load_zones_from_json(zones_data):
    try:
        # Handle case where zones_data is a file path
        if isinstance(zones_data, (str, bytes, os.PathLike)):
            print(f"Loading zones from file: {zones_data}")
            if not os.path.exists(zones_data):
                raise FileNotFoundError(f"Zones file not found: {zones_data}")
            with open(zones_data, 'r') as f:
                data = json.load(f)
        # Handle case where zones_data is a dictionary
        elif isinstance(zones_data, dict):
            print("Using provided zones dictionary")
            data = zones_data
        else:
            raise TypeError(f"Expected str, bytes, os.PathLike, or dict, got {type(zones_data)}")

        entry_zones = {}
        exit_zones = {}
        
        # Process entry zones
        for name, coords in data.get('entry_zones', {}).items():
            try:
                if isinstance(coords, Polygon):
                    # Use Polygon directly
                    entry_zones[name] = coords
                elif isinstance(coords, list) and len(coords) >= 3:
                    # Create Polygon from coordinate list
                    entry_zones[name] = Polygon(coords)
                else:
                    print(f"Warning: Invalid coordinates for entry zone {name}, skipping.")
                    continue
                
                if name not in TURN_MAP:
                    print(f"Warning: Entry zone {name} not found in TURN_MAP.")
            except Exception as e:
                print(f"Error processing entry zone {name}: {e}")
        
        # Process exit zones
        for name, coords in data.get('exit_zones', {}).items():
            try:
                if isinstance(coords, Polygon):
                    # Use Polygon directly
                    exit_zones[name] = coords
                elif isinstance(coords, list) and len(coords) >= 3:
                    # Create Polygon from coordinate list
                    exit_zones[name] = Polygon(coords)
                else:
                    print(f"Warning: Invalid coordinates for exit zone {name}, skipping.")
                    continue
            except Exception as e:
                print(f"Error processing exit zone {name}: {e}")
        
        return entry_zones, exit_zones
    except Exception as e:
        print(f"Error loading zones: {e}")
        return {}, {}    

# Video processing function
def process_video(video_path, output_path, zones_file='zones.json'):
    # Load zones dynamically from zones.json
    entry_zones, exit_zones = load_zones_from_json(zones_file)
    
    if not entry_zones or not exit_zones:
        print("Error: No valid zones loaded from zones.json. Exiting.")
        return None, None
    
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found!")
        return None, None
    
    analyzer = TrafficAnalyzer()
    processor = VideoProcessor(analyzer)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None, None
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {w}x{h} at {fps}FPS, {total_frames} frames")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    if not out.isOpened():
        print("Error: Could not open output video writer")
        cap.release()
        return None, None
    
    frame_count = 0
    video_start_time = datetime.now()
    print("Processing video...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp = str(video_start_time + timedelta(seconds=frame_count / fps))
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%)")
        
        frame = process_frame(frame, analyzer, processor, entry_zones, exit_zones, timestamp)
        out.write(frame)
    
    cap.release()
    out.release()
    
    analytics = generate_analytics(analyzer, frame_count, output_path)
    
    return output_path, analytics

def process_frame(frame, analyzer, processor, entry_zones, exit_zones, timestamp):
    results = analyzer.model(frame, imgsz=1280, conf=0.35, classes=analyzer.CAR_CLASSES)[0]
    
    if results.boxes is None or len(results.boxes) == 0:
        processor.draw_zones(frame, entry_zones, exit_zones)
        processor.draw_debug_info(frame, 0, 0, analyzer.turn_counts, analyzer.total_cars)
        return frame
    
    cls, xyxy, conf, _ = processor.get_box_details(results.boxes)
    detections = []
    total_detected = 0
    
    for i, (box, c) in enumerate(zip(xyxy, cls)):
        class_id = int(c)
        if class_id in analyzer.CAR_CLASSES:
            total_detected += 1
            bbox = box.cpu().numpy()
            x1, y1, x2, y2 = bbox
            conf_val = float(conf[i])
            
            if conf_val > 0.45:
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf_val, "car"))
    
    outputs = analyzer.tracker.update_tracks(detections, frame=frame)
    current_tracks = set()
    valid_tracked = 0
    
    for track in outputs:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        
        valid_tracked += 1
        track_id = track.track_id
        current_tracks.add(track_id)
        
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        center_point = (cx, cy)
        
        if track_id not in analyzer.vehicle_tracks:
            analyzer.total_cars += 1
            analyzer.vehicle_tracks[track_id] = VehicleTracking(track_id, timestamp)
        
        vehicle = analyzer.vehicle_tracks[track_id]
        vehicle.add_path_point(center_point)
        
        if not vehicle.get_vehicle_entered_into():
            entry_zone = find_zone(center_point, entry_zones)
            if entry_zone:
                vehicle.set_entry(entry_zone)
        
        if vehicle.get_vehicle_entered_into() and not vehicle.get_vehicle_exited_out():
            exit_zone = find_zone(center_point, exit_zones)
            if exit_zone:
                vehicle.set_exit(exit_zone)
                turn = TURN_MAP.get(vehicle.get_vehicle_entered_into(), {}).get(exit_zone)
                if turn:
                    vehicle.set_turn(turn, timestamp)
                    analyzer.turn_counts[turn] += 1
        
        turn_type = vehicle.turn if vehicle.turn else "unknown"
        processor.draw_enhanced_bbox(frame, x1, y1, x2, y2, track_id, turn_type)
        
        for i, pt in enumerate(vehicle.path_points):
            alpha = max(0.3, (i / len(vehicle.path_points)))
            point_color = tuple(int(c * alpha) for c in analyzer.trajectory_color)
            cv2.circle(frame, pt, 2, point_color, -1)
    
    analyzer.previous_tracks = current_tracks.copy()
    
    processor.draw_zones(frame, entry_zones, exit_zones)
    processor.draw_debug_info(frame, total_detected, valid_tracked, analyzer.turn_counts, analyzer.total_cars)
    
    return frame

def generate_analytics(analyzer, frame_count, output_path):
    turn_details = []
    for track_id, vehicle in analyzer.vehicle_tracks.items():
        turn_info = vehicle.get_turn_details()
        if turn_info:
            turn_details.append(turn_info)
    
    analytics = {
        "total_cars": analyzer.total_cars,
        "turn_counts": analyzer.turn_counts,
        "turns": turn_details
    }
    
    with open("traffic_analytics.json", "w") as f:
        json.dump(analytics, f, indent=4)
    
    print("Processing complete!")
    print(f"Total vehicles tracked: {analyzer.total_cars}")
    print(f"Turn counts: {analyzer.turn_counts}")
    print(f"Output video saved as: {output_path}")
    print("Analytics saved as: traffic_analytics.json")
    
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"Output video file size: {file_size / (1024*1024):.2f} MB")
    
    return analytics

























