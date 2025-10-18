"""
Main application for AI-Powered Library Seat Monitoring System
"""
import cv2
import time
import argparse
import sys
import os
from typing import Optional, Dict, List
import threading
import signal

# Import our modules
from detection import YOLODetector
from posture_analysis import PostureAnalyzer
from alert_system import AlertSystem
from visualization import VisualizationSystem
from data_logging import DataLogger

class LibraryMonitoringSystem:
    def __init__(self, 
                 camera_source: int = 0,
                 model_path: str = "yolov8n.pt",
                 enable_dashboard: bool = True,
                 enable_alerts: bool = True,
                 enable_logging: bool = True,
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.3):
        """
        Initialize the library monitoring system
        
        Args:
            camera_source: Camera source (0 for default webcam, or path to video file)
            model_path: Path to YOLO model weights
            enable_dashboard: Enable real-time dashboard
            enable_alerts: Enable alert system
            enable_logging: Enable data logging
            confidence_threshold: Detection confidence threshold
            iou_threshold: IoU threshold for seat-person association
        """
        self.camera_source = camera_source
        self.model_path = model_path
        self.enable_dashboard = enable_dashboard
        self.enable_alerts = enable_alerts
        self.enable_logging = enable_logging
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Initialize components
        self.detector = YOLODetector(model_path)
        self.posture_analyzer = PostureAnalyzer()
        self.alert_system = AlertSystem() if enable_alerts else None
        self.visualizer = VisualizationSystem(show_dashboard=enable_dashboard)
        self.data_logger = DataLogger() if enable_logging else None
        
        # System state
        self.running = False
        self.session_id = None
        self.frame_count = 0
        self.total_detections = 0
        
        # Performance tracking
        self.fps_counter = []
        self.last_fps_time = time.time()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\nShutdown signal received. Stopping system...")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """Start the monitoring system"""
        print("Starting AI-Powered Library Seat Monitoring System...")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_source)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera source {self.camera_source}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Start session
        if self.data_logger:
            self.session_id = self.data_logger.start_session(
                notes=f"Library monitoring session started at {time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            print(f"Session started: {self.session_id}")
        
        self.running = True
        print("System started successfully!")
        
        # Main processing loop
        self._main_loop()
        
        return True
    
    def _main_loop(self):
        """Main processing loop"""
        try:
            while self.running:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                
                # Process frame
                start_time = time.time()
                self._process_frame(frame)
                processing_time = time.time() - start_time
                
                # Update FPS
                self._update_fps()
                
                # Log performance
                if self.data_logger and self.session_id:
                    self.data_logger.log_performance(
                        fps=self._get_current_fps(),
                        detection_time=processing_time,
                        processing_time=processing_time,
                        session_id=self.session_id
                    )
                
                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('s'):  # 's' for screenshot
                    self._save_screenshot(frame)
                elif key == ord('h'):  # 'h' for heatmap
                    self._save_heatmap()
                elif key == ord('e'):  # 'e' for export data
                    self._export_data()
                
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received")
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            self.stop()
    
    def _process_frame(self, frame):
        """Process a single frame"""
        self.frame_count += 1
        
        # Detect chairs and persons
        detections = self.detector.detect(frame, self.confidence_threshold)
        chairs = detections.get('chairs', [])
        persons = detections.get('persons', [])
        
        if chairs:
            self.total_detections += len(chairs)
        
        # Associate persons with chairs
        associated_seats = self.detector.associate_persons_to_chairs(
            chairs, persons, self.iou_threshold
        )
        
        # Analyze posture for each person
        posture_analyses = []
        for seat_info in associated_seats:
            if seat_info['is_occupied'] and seat_info['person']:
                person_bbox = seat_info['person']['bbox']
                person_id = seat_info['seat_id']  # Use seat_id as person_id for simplicity
                
                # Analyze posture
                analysis = self.posture_analyzer.analyze_person(
                    frame, person_bbox, person_id
                )
                posture_analyses.append(analysis)
                
                # Log data
                if self.data_logger and self.session_id:
                    # Log occupancy
                    self.data_logger.log_occupancy(
                        seat_id=seat_info['seat_id'],
                        is_occupied=True,
                        person_id=person_id,
                        confidence=seat_info['person']['confidence'],
                        session_id=self.session_id
                    )
                    
                    # Log focus
                    self.data_logger.log_focus(
                        person_id=person_id,
                        seat_id=seat_info['seat_id'],
                        is_focused=analysis['is_focused'],
                        head_tilt=analysis['features'].get('head_tilt', 0.0),
                        shoulder_angle=analysis['features'].get('shoulder_angle', 0.0),
                        body_lean=analysis['features'].get('body_lean', 0.0),
                        hand_activity=analysis['features'].get('hand_activity', 0.0),
                        confidence=analysis['confidence'],
                        session_id=self.session_id
                    )
                    
                    # Send alerts if needed
                    if (self.alert_system and analysis['should_alert'] and 
                        not analysis['is_focused']):
                        
                        alert_type = 'inactivity' if analysis['inactivity_duration'] > 30 else 'distraction'
                        severity = min(5, max(1, int(analysis['inactivity_duration'] / 10)))
                        
                        self.alert_system.send_alert(
                            person_id=person_id,
                            seat_id=seat_info['seat_id'],
                            alert_type=alert_type,
                            severity=severity,
                            session_id=self.session_id
                        )
                        
                        # Log alert
                        self.data_logger.log_alert(
                            person_id=person_id,
                            seat_id=seat_info['seat_id'],
                            alert_type=alert_type,
                            message=f"Alert for person {person_id}",
                            severity=severity,
                            session_id=self.session_id
                        )
            else:
                # Log empty seat
                if self.data_logger and self.session_id:
                    self.data_logger.log_occupancy(
                        seat_id=seat_info['seat_id'],
                        is_occupied=False,
                        session_id=self.session_id
                    )
        
        # Update visualization
        if self.visualizer:
            # Draw overlays
            frame_with_overlays = self.visualizer.draw_overlays(
                frame, detections, associated_seats, posture_analyses
            )
            
            # Update dashboard
            self.visualizer.update_dashboard(associated_seats, posture_analyses)
            
            # Show frame
            cv2.imshow('Library Seat Monitoring', frame_with_overlays)
        
        # Print status every 100 frames
        if self.frame_count % 100 == 0:
            self._print_status(associated_seats, posture_analyses)
    
    def _update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        self.fps_counter.append(current_time)
        
        # Keep only last 30 frames
        if len(self.fps_counter) > 30:
            self.fps_counter.pop(0)
    
    def _get_current_fps(self) -> float:
        """Get current FPS"""
        if len(self.fps_counter) < 2:
            return 0.0
        
        time_diff = self.fps_counter[-1] - self.fps_counter[0]
        if time_diff > 0:
            return (len(self.fps_counter) - 1) / time_diff
        return 0.0
    
    def _print_status(self, associated_seats: List[Dict], posture_analyses: List[Dict]):
        """Print current status"""
        total_seats = len(associated_seats)
        occupied_seats = sum(1 for seat in associated_seats if seat['is_occupied'])
        focused_seats = sum(1 for analysis in posture_analyses if analysis.get('is_focused', False))
        
        occupancy_rate = (occupied_seats / total_seats * 100) if total_seats > 0 else 0
        focus_rate = (focused_seats / occupied_seats * 100) if occupied_seats > 0 else 0
        
        print(f"Frame {self.frame_count}: "
              f"Seats: {total_seats}, "
              f"Occupied: {occupied_seats} ({occupancy_rate:.1f}%), "
              f"Focused: {focused_seats} ({focus_rate:.1f}%), "
              f"FPS: {self._get_current_fps():.1f}")
    
    def _save_screenshot(self, frame):
        """Save current frame as screenshot"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f'screenshot_{timestamp}.jpg'
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved as {filename}")
    
    def _save_heatmap(self):
        """Save seat utilization heatmap"""
        if self.visualizer:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f'heatmap_{timestamp}.png'
            self.visualizer.save_heatmap(filename)
            print(f"Heatmap saved as {filename}")
    
    def _export_data(self):
        """Export data to CSV"""
        if self.data_logger:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            output_dir = f'export_{timestamp}'
            self.data_logger.export_to_csv(session_id=self.session_id, output_dir=output_dir)
            print(f"Data exported to {output_dir}")
    
    def stop(self):
        """Stop the monitoring system"""
        if not self.running:
            return
        
        print("Stopping monitoring system...")
        self.running = False
        
        # End session
        if self.data_logger and self.session_id:
            avg_fps = self._get_current_fps()
            self.data_logger.end_session(
                session_id=self.session_id,
                total_frames=self.frame_count,
                total_detections=self.total_detections,
                avg_fps=avg_fps
            )
            print(f"Session {self.session_id} ended")
        
        # Cleanup
        if hasattr(self, 'cap'):
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        if self.visualizer:
            self.visualizer.close_dashboard()
        
        print("System stopped successfully!")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='AI-Powered Library Seat Monitoring System')
    parser.add_argument('--camera', type=int, default=0, 
                       help='Camera source (0 for default webcam)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Path to YOLO model weights')
    parser.add_argument('--no-dashboard', action='store_true',
                       help='Disable dashboard')
    parser.add_argument('--no-alerts', action='store_true',
                       help='Disable alert system')
    parser.add_argument('--no-logging', action='store_true',
                       help='Disable data logging')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Detection confidence threshold')
    parser.add_argument('--iou', type=float, default=0.3,
                       help='IoU threshold for seat-person association')
    
    args = parser.parse_args()
    
    # Create and start monitoring system
    system = LibraryMonitoringSystem(
        camera_source=args.camera,
        model_path=args.model,
        enable_dashboard=not args.no_dashboard,
        enable_alerts=not args.no_alerts,
        enable_logging=not args.no_logging,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou
    )
    
    try:
        system.start()
    except Exception as e:
        print(f"Error starting system: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
