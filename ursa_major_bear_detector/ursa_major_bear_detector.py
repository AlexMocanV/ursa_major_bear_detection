# -*- coding: utf-8 -*-
import sys
import cv2
import torch
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QComboBox, 
                             QTextEdit, QFrame, QStatusBar, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont
from ultralytics import YOLO
import threading
import time
import queue
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BearDetector:
    """Advanced bear detection and species classification system"""
    
    def __init__(self):
        self.detection_model = None
        self.species_model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.species_map = {
            0: 'American Black Bear',
            1: 'Brown/Grizzly Bear', 
            2: 'Polar Bear',
            3: 'Asiatic Black Bear',
            4: 'Sloth Bear',
            5: 'Sun Bear'
        }
        self.confidence_threshold = 0.5
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize YOLO models for detection and classification"""
        try:
            # Primary detection model - YOLOv8 nano for speed
            logger.info("Loading YOLO detection model...")
            self.detection_model = YOLO('yolov8n.pt')
            
            # Set model to appropriate device
            if self.device == 'cuda':
                self.detection_model.to('cuda')
                logger.info("Models loaded on GPU")
            else:
                logger.info("Models loaded on CPU")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def detect_bears(self, frame):
        """Detect bears in frame and return results with species classification"""
        try:
            # Run detection
            results = self.detection_model(frame, conf=self.confidence_threshold, verbose=False)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # Get detection data
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu())
                        cls = int(box.cls[0].cpu())
                        
                        # Check if detected object could be a bear (person, animal classes)
                        # COCO classes: 0=person, 14=bird, 15=cat, 16=dog, etc.
                        # We'll classify any detection as potential bear for species analysis
                        class_name = self.detection_model.names[cls]
                        
                        # Extract region for species classification
                        x1, y1, x2, y2 = map(int, xyxy)
                        roi = frame[y1:y2, x1:x2]
                        
                        # Simple species classification based on visual features
                        species, species_confidence = self.classify_bear_species(roi, conf)
                        
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class': class_name,
                            'species': species,
                            'species_confidence': species_confidence,
                            'is_bear': self.is_likely_bear(class_name, conf)
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
    
    def classify_bear_species(self, roi, base_confidence):
        """Classify bear species based on visual features"""
        if roi.size == 0:
            return "Unknown", 0.0
        
        try:
            # Simplified species classification using image properties
            # In production, this would use a trained species classification model
            
            # Basic color and size analysis for species estimation
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            
            # Simple heuristic-based classification
            # This is a placeholder - real implementation would use trained models
            if mean_brightness < 80:
                return "American Black Bear", min(base_confidence * 0.8, 0.95)
            elif mean_brightness > 180:
                return "Polar Bear", min(base_confidence * 0.7, 0.90)
            else:
                return "Brown/Grizzly Bear", min(base_confidence * 0.75, 0.92)
                
        except Exception as e:
            logger.error(f"Species classification error: {e}")
            return "Unknown", 0.0
    
    def is_likely_bear(self, class_name, confidence):
        """Determine if detection is likely a bear"""
        # YOLO detects generic classes, we use heuristics to identify potential bears
        bear_classes = ['person', 'horse', 'cow', 'dog', 'cat', 'bear']  # Classes that might be confused with bears
        return class_name in bear_classes and confidence > 0.3

class CameraManager:
    """Manages USB camera connection and video capture"""
    
    def __init__(self):
        self.cap = None
        self.camera_index = 0
        self.is_connected = False
        
    def find_cameras(self):
        """Find available cameras"""
        cameras = []
        # Only check first 3 indices to avoid the obsensor errors
        for i in range(3):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        cameras.append(i)
                    cap.release()
            except:
                pass
        return cameras
    
    def connect_camera(self, camera_index=0):
        """Connect to specified camera"""
        try:
            if self.cap:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(camera_index)
            
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open camera {camera_index}")
            
            # Set camera properties for optimal performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.camera_index = camera_index
            self.is_connected = True
            logger.info(f"Connected to camera {camera_index}")
            return True
            
        except Exception as e:
            logger.error(f"Camera connection error: {e}")
            self.is_connected = False
            return False
    
    def read_frame(self):
        """Read frame from camera"""
        if not self.is_connected or not self.cap:
            return False, None
        
        ret, frame = self.cap.read()
        if not ret:
            self.is_connected = False
        return ret, frame
    
    def release(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
        self.is_connected = False

class VideoThread(QThread):
    """Thread for handling video processing"""
    
    frame_ready = pyqtSignal(np.ndarray, list)
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.camera_manager = CameraManager()
        self.bear_detector = BearDetector()
        self.running = False
        self.process_every_n_frames = 3  # Process every 3rd frame for performance
        self.frame_count = 0
        
    def set_camera(self, camera_index):
        """Set camera index"""
        return self.camera_manager.connect_camera(camera_index)
    
    def run(self):
        """Main thread loop"""
        self.running = True
        retry_count = 0
        max_retries = 3
        
        while self.running:
            try:
                ret, frame = self.camera_manager.read_frame()
                
                if not ret:
                    if retry_count < max_retries:
                        retry_count += 1
                        self.error_occurred.emit(f"Camera read failed, retrying ({retry_count}/{max_retries})")
                        time.sleep(1)
                        continue
                    else:
                        self.error_occurred.emit("Camera disconnected, attempting reconnection...")
                        if self.camera_manager.connect_camera(self.camera_manager.camera_index):
                            retry_count = 0
                            continue
                        else:
                            self.error_occurred.emit("Failed to reconnect to camera")
                            break
                
                retry_count = 0
                self.frame_count += 1
                
                # Process detection every N frames for performance
                detections = []
                if self.frame_count % self.process_every_n_frames == 0:
                    detections = self.bear_detector.detect_bears(frame)
                
                self.frame_ready.emit(frame, detections)
                
            except Exception as e:
                self.error_occurred.emit(f"Video processing error: {str(e)}")
                time.sleep(0.1)
    
    def stop(self):
        """Stop the thread"""
        self.running = False
        self.camera_manager.release()
        self.wait()

class BearDetectionGUI(QMainWindow):
    """Main GUI application for bear detection system"""
    
    def __init__(self):
        super().__init__()
        self.video_thread = VideoThread()
        self.current_detections = []
        self.detection_history = []
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Setup the user interface"""
        self.setWindowTitle("Advanced Bear Detection System")
        self.setGeometry(100, 100, 1400, 900)
    
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
    
        # Main layout
        main_layout = QHBoxLayout(central_widget)
    
        # Left panel for video display
        left_panel = QVBoxLayout()
    
        # Video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("border: 2px solid gray; background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("Camera feed will appear here")
        left_panel.addWidget(self.video_label)
    
        # Control buttons
        button_layout = QHBoxLayout()
    
        self.start_btn = QPushButton("Start Detection")
        self.start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
    
        self.stop_btn = QPushButton("Stop Detection")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; }")
    
        self.camera_combo = QComboBox()
        self.refresh_cameras()
    
        button_layout.addWidget(QLabel("Camera:"))
        button_layout.addWidget(self.camera_combo)
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addStretch()
    
        left_panel.addLayout(button_layout)
    
        # Right panel for information display - FIX: Create widget first, then set width
        right_widget = QWidget()
        right_widget.setMaximumWidth(400)  # Move this line up - set width on widget, not layout
        right_panel = QVBoxLayout(right_widget)
    
        # Detection info
        detection_frame = QFrame()
        detection_frame.setFrameStyle(QFrame.Box)
        detection_layout = QVBoxLayout(detection_frame)
    
        detection_title = QLabel("Detection Information")
        detection_title.setFont(QFont("Arial", 12, QFont.Bold))
        detection_layout.addWidget(detection_title)
    
        self.detection_info = QTextEdit()
        self.detection_info.setMaximumHeight(300)
        self.detection_info.setReadOnly(True)
        detection_layout.addWidget(self.detection_info)
    
        right_panel.addWidget(detection_frame)
    
        # Species information
        species_frame = QFrame()
        species_frame.setFrameStyle(QFrame.Box)
        species_layout = QVBoxLayout(species_frame)
    
        species_title = QLabel("Bear Species Guide")
        species_title.setFont(QFont("Arial", 12, QFont.Bold))
        species_layout.addWidget(species_title)
    
        species_info = QTextEdit()
        species_info.setReadOnly(True)
        species_info.setHtml("""
        <b>American Black Bear:</b> Smaller, straight facial profile, large pointed ears<br><br>
        <b>Brown/Grizzly Bear:</b> Shoulder hump, concave face, smaller rounded ears<br><br>
        <b>Polar Bear:</b> White/cream colored, long neck, smaller ears<br><br>
        <b>Detection Tips:</b><br>
        - Look for shoulder hump (grizzlies)<br>
        - Ear shape and size<br>
        - Facial profile<br>
        - Body size and color
        """)
        species_layout.addWidget(species_info)
    
        right_panel.addWidget(species_frame)
    
        # System status
        status_frame = QFrame()
        status_frame.setFrameStyle(QFrame.Box)
        status_layout = QVBoxLayout(status_frame)
    
        status_title = QLabel("System Status")
        status_title.setFont(QFont("Arial", 12, QFont.Bold))
        status_layout.addWidget(status_title)
    
        self.status_info = QTextEdit()
        self.status_info.setMaximumHeight(150)
        self.status_info.setReadOnly(True)
        status_layout.addWidget(self.status_info)
    
        right_panel.addWidget(status_frame)
    
        # Add panels to main layout
        main_layout.addLayout(left_panel, 2)
        main_layout.addWidget(right_widget, 1)  # Add the widget, not the layout
    
        # Status bar
        self.statusBar().showMessage("Ready to start detection")
    
        # Update system info
        self.update_system_status()
    
    def setup_connections(self):
        """Setup signal connections"""
        self.start_btn.clicked.connect(self.start_detection)
        self.stop_btn.clicked.connect(self.stop_detection)
        self.video_thread.frame_ready.connect(self.update_video)
        self.video_thread.error_occurred.connect(self.handle_error)
    
    def refresh_cameras(self):
        """Refresh available cameras list"""
        cameras = CameraManager().find_cameras()
        self.camera_combo.clear()
        for cam in cameras:
            self.camera_combo.addItem(f"Camera {cam}", cam)
        
        if not cameras:
            self.camera_combo.addItem("No cameras found", -1)
    
    def start_detection(self):
        """Start bear detection"""
        camera_index = self.camera_combo.currentData()
        
        if camera_index == -1:
            QMessageBox.warning(self, "Warning", "No camera available!")
            return
        
        if self.video_thread.set_camera(camera_index):
            self.video_thread.start()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.statusBar().showMessage("Detection started")
            self.status_info.append(f"Started detection on camera {camera_index}")
        else:
            QMessageBox.critical(self, "Error", "Failed to connect to camera!")
    
    def stop_detection(self):
        """Stop bear detection"""
        self.video_thread.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.statusBar().showMessage("Detection stopped")
        self.status_info.append("Detection stopped")
    
    def update_video(self, frame, detections):
        """Update video display with detections"""
        self.current_detections = detections
        
        # Draw bounding boxes and labels
        display_frame = frame.copy()
        
        for detection in detections:
            if detection['is_bear']:
                x1, y1, x2, y2 = detection['bbox']
                confidence = detection['confidence']
                species = detection['species']
                species_conf = detection['species_confidence']
                
                # Draw bounding box
                color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw labels
                label = f"{species}: {species_conf:.2f}"
                conf_label = f"Detection: {confidence:.2f}"
                
                # Background for text
                cv2.rectangle(display_frame, (x1, y1-40), (x1+300, y1), color, -1)
                cv2.putText(display_frame, label, (x1+5, y1-25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(display_frame, conf_label, (x1+5, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Convert to Qt format and display
        rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
        
        # Update detection info
        self.update_detection_info()
    
    def update_detection_info(self):
        """Update detection information display"""
        info_text = ""
        bear_count = 0
        
        for detection in self.current_detections:
            if detection['is_bear']:
                bear_count += 1
                info_text += f"Bear #{bear_count}:\n"
                info_text += f"  Species: {detection['species']}\n"
                info_text += f"  Confidence: {detection['species_confidence']:.2%}\n"
                info_text += f"  Detection Score: {detection['confidence']:.2%}\n"
                info_text += f"  Location: ({detection['bbox'][0]}, {detection['bbox'][1]})\n\n"
        
        if bear_count == 0:
            info_text = "No bears detected in current frame.\n\n"
        
        info_text += f"Total Bears Detected: {bear_count}\n"
        info_text += f"Frame Processing: Active\n"
        info_text += f"System Status: Running"
        
        self.detection_info.setPlainText(info_text)
    
    def update_system_status(self):
        """Update system status information"""
        device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        status_text = f"Processing Device: {device}\n"
        status_text += f"Detection Model: YOLOv8\n"
        status_text += f"Species Classification: Active\n"
        status_text += f"Real-time Processing: Enabled\n"
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            status_text += f"GPU: {gpu_name}\n"
        
        self.status_info.setPlainText(status_text)
    
    def handle_error(self, error_message):
        """Handle errors from video thread"""
        self.status_info.append(f"ERROR: {error_message}")
        self.statusBar().showMessage(f"Error: {error_message}")
    
    def closeEvent(self, event):
        """Handle application close event"""
        if self.video_thread.isRunning():
            self.video_thread.stop()
        event.accept()

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Check for camera availability
    camera_manager = CameraManager()
    cameras = camera_manager.find_cameras()
    
    if not cameras:
        QMessageBox.warning(None, "Warning", 
            "No cameras detected. Please connect a USB camera and restart the application.")
    
    # Create and show main window
    window = BearDetectionGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()