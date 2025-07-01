import cv2
import numpy as np
import os
import yaml
import time
import argparse
from ultralytics import YOLO
from utils import Sort, crop_bbox, generate_colors
from PIL import Image, ImageDraw, ImageFont

# --- Configuration ---
# Default paths
DEFAULT_VIDEO_PATH = 'videos/videos.mp4'
DEFAULT_DETECTION_MODEL_PATH = 'models/best-detect.pt'
DEFAULT_CLASSIFICATION_MODEL_PATH = 'models/best-cls.pt'
DEFAULT_CONFIG_PATH = 'config.yml'
FONT_PATH = 'fonts/arial.ttf'

# UI Constants
SIDEBAR_WIDTH = 400
TRACKBAR_AREA_HEIGHT = 350

# Default parameters
detection_params = {
    'conf': 50, 'iou': 45, 'class_score': 50, 'objectness': 50, 
    'cls_conf': 50, 'frame_skip': 1, 'mode': 1
}

# Names and value ranges for trackbars
trackbar_configs = [
    {'name': 'Confidence', 'key': 'conf', 'max': 100},
    {'name': 'IoU', 'key': 'iou', 'max': 100},
    {'name': 'ClassScore', 'key': 'class_score', 'max': 100},
    {'name': 'Objectness', 'key': 'objectness', 'max': 100},
    {'name': 'ClsConf', 'key': 'cls_conf', 'max': 100},
    {'name': 'FrameSkip', 'key': 'frame_skip', 'max': 10, 'min': 1},
    {'name': 'Mode', 'key': 'mode', 'max': 1}
]

# State definitions
ITEM_STATES = {
    'PREPARING': 'PREPARING',
    'READY_TO_SERVE': 'READY_TO_SERVE',
    'SERVED': 'SERVED',
    'REMOVED': 'REMOVED'
}

# --- Font enhancement utilities ---
def get_bold_font():
    # Try to use Arial Bold if available, else fallback to Hershey Simplex with thick line
    # OpenCV doesn't support custom fonts, so use FONT_HERSHEY_SIMPLEX with high thickness
    return cv2.FONT_HERSHEY_SIMPLEX, 0.7, 3  # font, scale, thickness

def get_text_color_for_background(background_bgr):
    # Automatically choose black/white text color based on background brightness
    b, g, r = background_bgr
    brightness = 0.299*r + 0.587*g + 0.114*b
    return (0,0,0) if brightness > 160 else (255,255,255)

# --- Sidebar with state color ---
def get_state_text_color(state, visualization_config):
    if state == 'PREPARING':
        return tuple(visualization_config.get('preparing_color', [0,255,255]))
    elif state == 'READY_TO_SERVE':
        return tuple(visualization_config.get('ready_color', [255,165,0]))
    elif state == 'SERVED':
        return tuple(visualization_config.get('served_color', [0,0,255]))
    elif state == 'REMOVED':
        return tuple(visualization_config.get('removed_color', [128,128,128]))
    else:
        return tuple(visualization_config.get('text_color', [255,255,255]))

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Kitchen Monitor - Object Detection and Tracking')
    parser.add_argument('--video', '-v', type=str, default=DEFAULT_VIDEO_PATH,
                        help=f'Path to video file (default: {DEFAULT_VIDEO_PATH})')
    parser.add_argument('--config', '-c', type=str, default=DEFAULT_CONFIG_PATH,
                        help=f'Path to config file (default: {DEFAULT_CONFIG_PATH})')
    parser.add_argument('--detection-model', '-d', type=str, default=DEFAULT_DETECTION_MODEL_PATH,
                        help=f'Path to detection model (default: {DEFAULT_DETECTION_MODEL_PATH})')
    parser.add_argument('--classification-model', '-cls', type=str, default=DEFAULT_CLASSIFICATION_MODEL_PATH,
                        help=f'Path to classification model (default: {DEFAULT_CLASSIFICATION_MODEL_PATH})')
    return parser.parse_args()

class KitchenMonitorMenu:
    def __init__(self, video_path=None, config_path=None, detection_model_path=None, classification_model_path=None):
        # Use provided paths or defaults
        self.video_path = video_path or DEFAULT_VIDEO_PATH
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.detection_model_path = detection_model_path or DEFAULT_DETECTION_MODEL_PATH
        self.classification_model_path = classification_model_path or DEFAULT_CLASSIFICATION_MODEL_PATH
        
        self.config = self._load_config()
        self.detection_model = YOLO(self.detection_model_path)
        self.classification_model = YOLO(self.classification_model_path)
        self.classification_names = list(self.classification_model.names.values())
        self.colors = generate_colors(100)
        
        # Load font for PIL
        try:
            self.font_small = ImageFont.truetype(FONT_PATH, 20)
            self.font_medium = ImageFont.truetype(FONT_PATH, 24)
            self.font_large = ImageFont.truetype(FONT_PATH, 32)
        except IOError:
            print(f"Font file not found at {FONT_PATH}. Text rendering will use default cv2 fonts and may have issues.")
            self.font_small = self.font_medium = self.font_large = None

        self.tracker = Sort(
            max_age=self.config.get('tracking', {}).get('max_age', 60),
            min_hits=self.config.get('tracking', {}).get('min_hits', 2),
            iou_threshold=self.config.get('tracking', {}).get('iou_threshold', 0.2)
        )
        
        self.track_history = {}
        self.track_class_history = {}
        self.tracked_items = {}
        
        self.ready_zone = np.array(self.config.get('red_zone', {}).get('points', [[100, 100], [300, 100], [300, 300], [100, 300]]), np.int32)      # Red zone = Ready zone
        self.prepare_zone = np.array(self.config.get('yellow_zone', {}).get('points', [[50, 50], [350, 50], [350, 350], [50, 350]]), np.int32)  # Yellow zone = Prepare zone
        
        self.fps = 30
        self.frame_count = 0
        self.dwell_time_threshold = self.config.get('red_zone', {}).get('dwell_time_threshold', 30)
        
        self.zone_definition_mode = False
        self.current_zone = None
        self.zone_points = []
        self.trackbar_rectangles = {}
        self.dragging_slider = None
        
        # Feedback system variables
        self.feedback_mode = False
        self.feedback_target_id = None
        self.feedback_menu_rectangles = {}
        self.feedback_menu_position = None
        self.pending_feedback = None
        
        print("Kitchen Monitor Menu initialized successfully!")

    def initiate_feedback_process(self, x, y):
        """Initiate feedback process when user right-clicks on an object"""
        # Find the object that was clicked
        clicked_track_id = None
        for track_id, item in self.tracked_items.items():
            if item.get('last_frame') == self.frame_count:  # Only check current frame objects
                bbox = item.get('bbox', None)
                if bbox:
                    # Check if click is within bounding box
                    if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                        clicked_track_id = track_id
                        break
        
        if clicked_track_id:
            self.feedback_mode = True
            self.feedback_target_id = clicked_track_id
            self.feedback_menu_position = (x, y)
            print(f"Feedback mode activated for object ID: {clicked_track_id}")

    def draw_feedback_menu(self, frame):
        """Draw feedback menu when in feedback mode"""
        if not self.feedback_mode or not self.feedback_menu_position:
            return frame
        
        x, y = self.feedback_menu_position
        
        # Menu background
        menu_width = 200
        menu_height = 50 + len(self.classification_names) * 30
        menu_x = min(x, frame.shape[1] - menu_width - 10)
        menu_y = min(y, frame.shape[0] - menu_height - 10)
        
        # Draw background
        cv2.rectangle(frame, (menu_x, menu_y), (menu_x + menu_width, menu_y + menu_height), 
                     (50, 50, 50), -1)
        cv2.rectangle(frame, (menu_x, menu_y), (menu_x + menu_width, menu_y + menu_height), 
                     (255, 255, 255), 2)
        
        # Draw title
        title = "What is the correct label?"
        frame = self.draw_text_with_pil(frame, title, (menu_x + 10, menu_y + 20), 
                                      self.font_small, (255, 255, 255))
        
        # Draw class options
        self.feedback_menu_rectangles = {}
        for i, class_name in enumerate(self.classification_names):
            option_y = menu_y + 50 + i * 30
            option_rect = (menu_x + 10, option_y, menu_x + menu_width - 10, option_y + 25)
            self.feedback_menu_rectangles[class_name] = option_rect
            
            # Highlight current prediction
            current_class = self.tracked_items.get(self.feedback_target_id, {}).get('det_class_name', 'Unknown')
            if class_name == current_class:
                cv2.rectangle(frame, (option_rect[0], option_rect[1]), (option_rect[2], option_rect[3]), 
                             (0, 255, 0), -1)
            
            frame = self.draw_text_with_pil(frame, class_name, (option_rect[0] + 5, option_rect[1] + 5), 
                                          self.font_small, (0, 0, 0))
        
        return frame

    def save_feedback(self, track_id, correct_class_name, current_frame=None):
        """Save feedback data for model improvement"""
        try:
            # Get current frame and object bounding box
            if track_id not in self.tracked_items:
                print(f"Error: Track ID {track_id} not found in tracked items")
                return
            
            item = self.tracked_items[track_id]
            bbox = item.get('bbox', None)
            if not bbox:
                print(f"Error: No bounding box found for track ID {track_id}")
                return
            
            # Crop image from current frame
            if current_frame is not None:
                cropped_image = crop_bbox(current_frame, bbox)
            else:
                # Fallback to placeholder if no frame provided
                cropped_image = np.zeros((100, 100, 3), dtype=np.uint8)
            
            # Create feedback directory structure
            feedback_dir = "feedback_data"
            class_dir = os.path.join(feedback_dir, correct_class_name)
            
            os.makedirs(class_dir, exist_ok=True)
            
            # Save image
            timestamp = int(time.time())
            image_filename = f"{timestamp}_{track_id}.png"
            image_path = os.path.join(class_dir, image_filename)
            cv2.imwrite(image_path, cropped_image)
            
            # Log feedback
            log_file = os.path.join(feedback_dir, "feedback_log.csv")
            log_entry = f"{timestamp},{image_path},{item.get('det_class_name', 'Unknown')},{correct_class_name}\n"
            
            # Create log file with header if it doesn't exist
            if not os.path.exists(log_file):
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write("timestamp,image_path,model_prediction,user_correction\n")
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
            
            print(f"Feedback saved: {item.get('det_class_name', 'Unknown')} -> {correct_class_name}")
            print(f"Image saved: {image_path}")
            
        except Exception as e:
            print(f"Error saving feedback: {e}")

    def draw_text_with_pil(self, image, text, position, font, color=(255, 255, 255)):
        """Helper to draw text using Pillow to support Unicode characters."""
        if self.font_small is None: # Fallback if font file is not found
             cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
             return image
        
        # Convert the image to RGB (PIL uses RGB)
        cv2_im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)
        
        draw = ImageDraw.Draw(pil_im)
        draw.text(position, text, font=font, fill=color)
        
        # Convert back to BGR (OpenCV uses BGR)
        cv2_im_bgr = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
        return cv2_im_bgr

    def _load_config(self):
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            detection_params['conf'] = int(config.get('detection', {}).get('conf_threshold', 0.5) * 100)
            detection_params['iou'] = int(config.get('detection', {}).get('iou_threshold', 0.45) * 100)
            detection_params['objectness'] = int(config.get('detection', {}).get('objectness_threshold', 0.5) * 100)
            detection_params['class_score'] = int(config.get('detection', {}).get('class_score_threshold', 0.5) * 100)
            detection_params['cls_conf'] = int(config.get('detection', {}).get('cls_conf_threshold', 0.5) * 100)
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}

    def is_point_in_zone(self, point, zone):
        return cv2.pointPolygonTest(zone, point, False) >= 0

    def update_tracked_item_state(self, track_id, bounding_box, frame_number):
        if track_id not in self.tracked_items:
            self.tracked_items[track_id] = {
                'state': ITEM_STATES['PREPARING'], 'in_prepare_zone': False, 'in_ready_zone': False,
                'dwell_prepare': 0, 'dwell_ready': 0, 'last_frame': frame_number,
                'det_class_name': 'Unknown', 'status': 'N/A', 'was_in_ready_zone': False,
                'empty_dwell_time': 0,  # Track time for empty objects in ready zone
                'bbox': bounding_box  # Store bounding box for feedback
            }
        
        item = self.tracked_items[track_id]
        # Update bounding box
        item['bbox'] = bounding_box
        
        center = (int((bounding_box[0] + bounding_box[2]) / 2), int((bounding_box[1] + bounding_box[3]) / 2))
        in_prepare = self.is_point_in_zone(center, self.prepare_zone)  # Yellow zone = Prepare zone
        in_ready = self.is_point_in_zone(center, self.ready_zone)     # Red zone = Ready zone
        
        # Track dwell time
        if in_prepare: item['dwell_prepare'] += 1
        if in_ready: item['dwell_ready'] += 1
        
        # Check if object just left ready zone (event trigger for SERVED)
        if item['was_in_ready_zone'] and not in_ready:
            item['state'] = ITEM_STATES['SERVED']
        elif in_prepare and not in_ready:
            item['state'] = ITEM_STATES['PREPARING']
        elif in_ready:
            # NEW RULE: Check classification for ready zone logic
            class_name = item.get('det_class_name', 'Unknown').lower()
            empty_classes = ['dish-empty', 'tray-empty']
            ready_classes = ['dish-not-empty', 'tray-not-empty', 'dish-kakigori', 'tray-kakigori']
            
            if class_name in empty_classes:
                # Empty object in ready zone - track dwell time
                item['empty_dwell_time'] += 1
                empty_dwell_seconds = item['empty_dwell_time'] / self.fps if self.fps > 0 else 0
                
                if empty_dwell_seconds > 5:  # More than 5 seconds
                    item['state'] = ITEM_STATES['PREPARING']  # Keep as preparing
                else:
                    item['state'] = ITEM_STATES['READY_TO_SERVE']  # Allow ready state briefly
            elif class_name in ready_classes:
                # Non-empty object in ready zone - always READY_TO_SERVE
                item['empty_dwell_time'] = 0
                item['state'] = ITEM_STATES['READY_TO_SERVE']
            else:
                # Unknown class - set to preparing state
                item['state'] = ITEM_STATES['PREPARING']
        else:
            item['state'] = ITEM_STATES['REMOVED']
        
        # Update zone status
        item['in_prepare_zone'] = in_prepare
        item['in_ready_zone'] = in_ready
        item['was_in_ready_zone'] = in_ready  # Track for next frame
        item['last_frame'] = frame_number

    def get_alert_color(self, item):
        """Get color based on state and dwell time, using colors from config.yml"""
        state = item.get('state', 'Unknown')
        dwell_ready_seconds = item.get('dwell_ready', 0) / self.fps if self.fps > 0 else 0
        empty_dwell_seconds = item.get('empty_dwell_time', 0) / self.fps if self.fps > 0 else 0
        visualization_config = self.config.get('visualization', {})
        
        if state == ITEM_STATES['PREPARING']:
            return tuple(visualization_config.get('preparing_color', [0, 255, 255]))
        elif state == ITEM_STATES['READY_TO_SERVE']:
            # Check for both regular dwell time and empty object dwell time
            if dwell_ready_seconds > self.dwell_time_threshold:
                return tuple(visualization_config.get('alert_color', [0, 0, 255]))
            elif empty_dwell_seconds > 5:  # Empty object staying too long
                return tuple(visualization_config.get('alert_color', [0, 0, 255]))
            else:
                return tuple(visualization_config.get('ready_color', [255, 165, 0]))
        elif state == ITEM_STATES['SERVED']:
            return tuple(visualization_config.get('served_color', [0, 0, 255]))
        else:
            return tuple(visualization_config.get('removed_color', [128, 128, 128]))

    def draw_zones(self, frame):
        """Draw prepare and ready zones on frame, using colors from config.yml"""
        visualization_config = self.config.get('visualization', {})
        ready_color = tuple(visualization_config.get('red_zone_color', [0, 0, 255]))      # Red zone = Ready zone
        prepare_color = tuple(visualization_config.get('yellow_zone_color', [0, 255, 255]))  # Yellow zone = Prepare zone
        thickness = visualization_config.get('line_thickness', 2)
        cv2.polylines(frame, [self.ready_zone], True, ready_color, thickness, lineType=cv2.LINE_AA)
        cv2.polylines(frame, [self.prepare_zone], True, prepare_color, thickness, lineType=cv2.LINE_AA)

    def draw_zone_definition_feedback(self, frame):
        for point in self.zone_points:
            color = (0, 0, 255) if self.current_zone == "RED" else (0, 255, 255)
            cv2.circle(frame, tuple(point), 8, color, -1, lineType=cv2.LINE_AA)
        if self.current_zone == "YELLOW":
            cv2.polylines(frame, [self.ready_zone], True, (0, 0, 255), 2, lineType=cv2.LINE_AA)

    def process_detection_and_classification(self, frame, current_params):
        detection_results = self.detection_model(frame, conf=current_params['conf'], iou=current_params['iou'], verbose=False)
        detections = []
        if len(detection_results[0].boxes) > 0:
            for box, confidence in zip(detection_results[0].boxes.xyxy.cpu().numpy(), detection_results[0].boxes.conf.cpu().numpy()):
                if confidence >= current_params['objectness'] and confidence >= current_params['class_score']:
                    detections.append([box[0], box[1], box[2], box[3], confidence])
        
        tracks = self.tracker.update(np.array(detections)) if len(detections) > 0 else self.tracker.update(np.empty((0, 5)))
        
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            self.update_tracked_item_state(track_id, [x1, y1, x2, y2], self.frame_count)
            
            cropped = crop_bbox(frame, [x1, y1, x2, y2])
            if cropped.size > 0:
                classification_results = self.classification_model(cropped, conf=current_params['cls_conf'], verbose=False)
                if len(classification_results[0].probs) > 0:
                    probabilities = classification_results[0].probs.data.cpu().numpy()
                    top_index = np.argmax(probabilities)
                    top_confidence = probabilities[top_index]
                    most_common_class = self.classification_names[top_index]
                    
                    if track_id in self.tracked_items:
                        self.tracked_items[track_id]['det_class_name'] = most_common_class
                        self.tracked_items[track_id]['status'] = f"{most_common_class}:{top_confidence:.2f}"
            
            item = self.tracked_items.get(track_id, {})
            color = self.get_alert_color(item)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
            
            # --- NEW LABEL FORMAT and COLOR ---
            label = f"ID: {track_id} | {item.get('state', 'N/A')} | Status: {item.get('status', 'N/A')}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Draw background rectangle for the label
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1, lineType=cv2.LINE_AA)
            # Draw the label text with black color for high contrast
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)

        return frame, tracks

    def process_detection_only(self, frame, current_params):
        results = self.detection_model(frame, conf=current_params['conf'], iou=current_params['iou'], verbose=False)
        annotated_frame = results[0].plot() if results else frame.copy()
        annotated_frame = self.draw_text_with_pil(annotated_frame, "Detection only", (10, 60), self.font_medium, (0, 255, 0))
        return annotated_frame

    def get_current_params(self):
        return {
            'conf': detection_params['conf'] / 100.0, 'iou': detection_params['iou'] / 100.0,
            'class_score': detection_params['class_score'] / 100.0, 'objectness': detection_params['objectness'] / 100.0,
            'cls_conf': detection_params['cls_conf'] / 100.0, 'frame_skip': max(1, detection_params['frame_skip']),
            'mode': detection_params['mode']
        }

    def enter_zone_definition_mode(self):
        self.zone_definition_mode = True
        self.current_zone = "RED"
        self.zone_points = []
        print("Zone definition mode activated. Click 4 points for RED zone.")

    def handle_mouse_events(self, event, x, y, flags, param):
        video_width = param
        
        # Handle feedback mode clicks
        if self.feedback_mode and event == cv2.EVENT_LBUTTONDOWN:
            for class_name, rect in self.feedback_menu_rectangles.items():
                if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                    # We need to pass current frame, but it's not available here
                    # We'll store the feedback request and handle it in the main loop
                    self.pending_feedback = (self.feedback_target_id, class_name)
                    self.feedback_mode = False
                    self.feedback_target_id = None
                    self.feedback_menu_rectangles = {}
                    self.feedback_menu_position = None
                    return
        
        # Handle right-click on video area for feedback initiation
        if x <= video_width and event == cv2.EVENT_RBUTTONDOWN:
            self.initiate_feedback_process(x, y)
            return
        
        # Handle sidebar interactions
        if x > video_width:
            if event == cv2.EVENT_LBUTTONDOWN:
                for key, rect in self.trackbar_rectangles.items():
                    if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                        self.dragging_slider = key
                        break
            if self.dragging_slider and (event == cv2.EVENT_MOUSEMOVE or event == cv2.EVENT_LBUTTONDOWN):
                slider_rect, definition = self.trackbar_rectangles[self.dragging_slider], next(item for item in trackbar_configs if item['key'] == self.dragging_slider)
                track_start_x, track_width = slider_rect[0] + 120, slider_rect[2] - (slider_rect[0] + 120)
                relative_x = x - track_start_x
                min_val, max_val = definition.get('min', 0), definition.get('max', 100)
                new_val = int(np.clip(((relative_x / track_width) * (max_val - min_val) + min_val), min_val, max_val))
                detection_params[self.dragging_slider] = new_val
            if event == cv2.EVENT_LBUTTONUP:
                self.dragging_slider = None
        elif self.zone_definition_mode and event == cv2.EVENT_LBUTTONDOWN:
            self.zone_points.append([x, y])
            if len(self.zone_points) == 4:
                if self.current_zone == "RED":
                    self.ready_zone, self.current_zone, self.zone_points = np.array(self.zone_points, np.int32), "YELLOW", []
                    print("Ready zone defined. Now click 4 points for PREPARE zone.")
                elif self.current_zone == "YELLOW":
                    self.prepare_zone, self.zone_definition_mode, self.zone_points = np.array(self.zone_points, np.int32), False, []
                    print("Prepare zone defined. Zone definition complete!")

    def draw_ui_panel(self, height, width):
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel = self.draw_text_with_pil(panel, "READY ZONE OBJECTS", (10, 15), self.font_medium, (255, 255, 255))
        
        y_offset, ready_zone_count = 60, 0
        for track_id, item in self.tracked_items.items():
            if item.get('in_ready_zone', False) and item.get('state') != ITEM_STATES['REMOVED']:
                ready_zone_count += 1
                color = self.get_alert_color(item)
                dwell_ready_seconds = item.get('dwell_ready', 0) / self.fps if self.fps > 0 else 0
                
                panel = self.draw_text_with_pil(panel, f"ID: {track_id}", (10, y_offset), self.font_small, color)
                y_offset += 20
                info_texts = [f"Class: {item.get('det_class_name', 'Unknown')}", f"State: {item.get('state', 'Unknown')}", f"Dwell: {dwell_ready_seconds:.1f}s"]
                # Add empty dwell time info for empty objects
                class_name = item.get('det_class_name', 'Unknown').lower()
                if class_name in ['dish-empty', 'tray-empty']:
                    empty_dwell_seconds = item.get('empty_dwell_time', 0) / self.fps if self.fps > 0 else 0
                    info_texts.append(f"Empty Dwell: {empty_dwell_seconds:.1f}s")
                elif class_name in ['dish-not-empty', 'tray-not-empty', 'dish-kakigori', 'tray-kakigori']:
                    # Add ready dwell time info for non-empty objects
                    info_texts.append(f"Ready Dwell: {dwell_ready_seconds:.1f}s")
                for text in info_texts:
                    panel = self.draw_text_with_pil(panel, text, (20, y_offset), self.font_small, color)
                    y_offset += 20
                y_offset += 10
        
        if ready_zone_count == 0:
            panel = self.draw_text_with_pil(panel, "No objects in Ready Zone", (10, y_offset), self.font_small, (128, 128, 128))

        trackbar_y_start = height - TRACKBAR_AREA_HEIGHT
        cv2.line(panel, (0, trackbar_y_start - 10), (width, trackbar_y_start - 10), (100, 100, 100), 1, lineType=cv2.LINE_AA)
        
        y_offset = trackbar_y_start + 20
        for definition in trackbar_configs:
            # --- THIS IS THE CORRECTED PART ---
            key = definition['key']
            name = definition['name']
            min_val = definition.get('min', 0)
            max_val = definition.get('max', 100)
            current_val = detection_params[key]
            # --- END OF CORRECTION ---
            
            panel = self.draw_text_with_pil(panel, name, (10, y_offset + 2), self.font_small, (255, 255, 255))
            
            track_x, track_y, track_w, track_h = 130, y_offset, width - 180, 20
            self.trackbar_rectangles[key] = (track_x, track_y, track_x + track_w, track_y + track_h)
            
            cv2.rectangle(panel, (track_x, track_y), (track_x + track_w, track_y + track_h), (50, 50, 50), -1, lineType=cv2.LINE_AA)
            thumb_pos = int(((current_val - min_val) / (max_val - min_val)) * track_w) if (max_val - min_val) != 0 else 0
            cv2.rectangle(panel, (track_x + thumb_pos - 4, track_y), (track_x + thumb_pos + 4, track_y + track_h), (0, 255, 0), -1, lineType=cv2.LINE_AA)
            panel = self.draw_text_with_pil(panel, str(current_val), (track_x + track_w + 10, y_offset + 2), self.font_small, (0, 255, 255))
            y_offset += 40

        status_y = height - 40
        if self.zone_definition_mode:
            zone_name = "READY" if self.current_zone == "RED" else "PREPARE"
            status_text, color = f"Drawing {zone_name} zone, {4 - len(self.zone_points)} points remaining...", (0, 255, 255)
        else:
            status_text, color = "Press 'Z' to draw zones. 'ESC' to exit.", (0, 255, 0)
        panel = self.draw_text_with_pil(panel, status_text, (10, status_y), self.font_small, color)
        return panel
    
    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f'Cannot open video: {self.video_path}'); return
        
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        cv2.namedWindow('Kitchen Monitor', cv2.WINDOW_NORMAL)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cv2.setMouseCallback('Kitchen Monitor', self.handle_mouse_events, param=frame_width)
        
        print("\n--- Controls ---\n- See instruction.md for details\n- Press 'Z' to enter zone definition mode\n- Press 'ESC' to exit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
            
            self.frame_count += 1
            current_params = self.get_current_params()
            
            if self.frame_count % current_params['frame_skip'] != 0: continue
            
            self.draw_zones(frame)
            if self.zone_definition_mode: self.draw_zone_definition_feedback(frame)
            
            processed_frame, _ = (self.process_detection_and_classification(frame, current_params) if current_params['mode'] != 0 
                                  else (self.process_detection_only(frame, current_params), []))
            
            # Draw feedback menu if in feedback mode
            if self.feedback_mode:
                processed_frame = self.draw_feedback_menu(processed_frame)
            
            # Handle pending feedback
            if self.pending_feedback:
                track_id, correct_class_name = self.pending_feedback
                self.save_feedback(track_id, correct_class_name, frame)
                self.pending_feedback = None

            # --- Draw Timestamp using PIL ---
            from datetime import datetime
            timestamp_str = datetime.now().strftime("%d-%m-%Y  %H:%M:%S")
            processed_frame = self.draw_text_with_pil(processed_frame, f"FPS: {self.fps / current_params['frame_skip']:.1f}", (10, 15), self.font_large, (0, 255, 0))
            processed_frame = self.draw_text_with_pil(processed_frame, timestamp_str, (10, 50), self.font_large, (255, 255, 255))
            
            ui_panel = self.draw_ui_panel(processed_frame.shape[0], SIDEBAR_WIDTH)
            final_frame = np.hstack([processed_frame, ui_panel])
            
            cv2.imshow('Kitchen Monitor', final_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break
            elif key in [ord('z'), ord('Z')]: self.enter_zone_definition_mode()
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    args = parse_arguments()
    
    # Validate file paths
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return
    
    if not os.path.exists(args.detection_model):
        print(f"Error: Detection model not found: {args.detection_model}")
        return
    
    if not os.path.exists(args.classification_model):
        print(f"Error: Classification model not found: {args.classification_model}")
        return
    
    print(f"Starting Kitchen Monitor with:")
    print(f"  Video: {args.video}")
    print(f"  Config: {args.config}")
    print(f"  Detection Model: {args.detection_model}")
    print(f"  Classification Model: {args.classification_model}")
    print()
    
    monitor = KitchenMonitorMenu(
        video_path=args.video,
        config_path=args.config,
        detection_model_path=args.detection_model,
        classification_model_path=args.classification_model
    )
    monitor.run()

if __name__ == '__main__':
    main()