"""
Configuration settings for the Library Monitoring System
"""
import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DetectionConfig:
    """Detection system configuration"""
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.3
    max_detections: int = 50
    detection_fps_target: int = 15

@dataclass
class PostureConfig:
    """Posture analysis configuration"""
    inactivity_threshold: float = 30.0  # seconds
    head_tilt_threshold: float = 30.0   # degrees
    movement_threshold: float = 0.1     # normalized
    max_tracking_history: int = 30      # frames
    pose_confidence_threshold: float = 0.5

@dataclass
class AlertConfig:
    """Alert system configuration"""
    enable_audio: bool = True
    enable_visual: bool = True
    enable_notifications: bool = True
    alert_cooldown: float = 60.0  # seconds
    inactivity_alert_threshold: float = 30.0  # seconds
    distraction_alert_threshold: float = 15.0  # seconds

@dataclass
class VisualizationConfig:
    """Visualization system configuration"""
    show_overlays: bool = True
    show_dashboard: bool = True
    dashboard_size: tuple = (800, 600)
    overlay_alpha: float = 0.7
    update_interval: float = 0.1  # seconds

@dataclass
class LoggingConfig:
    """Data logging configuration"""
    enable_logging: bool = True
    db_path: str = "library_monitoring.db"
    log_directory: str = "logs"
    max_log_files: int = 30
    cleanup_days: int = 30
    export_format: str = "csv"  # csv, json, both

@dataclass
class SystemConfig:
    """Main system configuration"""
    camera_source: int = 0  # 0 for default webcam
    frame_width: int = 1280
    frame_height: int = 720
    target_fps: int = 30
    max_persons: int = 40
    performance_targets: Dict[str, float] = None
    
    def __post_init__(self):
        if self.performance_targets is None:
            self.performance_targets = {
                'min_fps': 15.0,
                'max_detection_time': 0.1,
                'max_processing_time': 0.2,
                'min_occupancy_accuracy': 0.9,
                'max_false_alert_rate': 0.1
            }

class Config:
    """Main configuration class"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        
        # Initialize with default values
        self.detection = DetectionConfig()
        self.posture = PostureConfig()
        self.alert = AlertConfig()
        self.visualization = VisualizationConfig()
        self.logging = LoggingConfig()
        self.system = SystemConfig()
        
        # Load from file if exists
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            try:
                import json
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Update configurations
                if 'detection' in config_data:
                    for key, value in config_data['detection'].items():
                        if hasattr(self.detection, key):
                            setattr(self.detection, key, value)
                
                if 'posture' in config_data:
                    for key, value in config_data['posture'].items():
                        if hasattr(self.posture, key):
                            setattr(self.posture, key, value)
                
                if 'alert' in config_data:
                    for key, value in config_data['alert'].items():
                        if hasattr(self.alert, key):
                            setattr(self.alert, key, value)
                
                if 'visualization' in config_data:
                    for key, value in config_data['visualization'].items():
                        if hasattr(self.visualization, key):
                            setattr(self.visualization, key, value)
                
                if 'logging' in config_data:
                    for key, value in config_data['logging'].items():
                        if hasattr(self.logging, key):
                            setattr(self.logging, key, value)
                
                if 'system' in config_data:
                    for key, value in config_data['system'].items():
                        if hasattr(self.system, key):
                            setattr(self.system, key, value)
                
                print(f"Configuration loaded from {self.config_file}")
                
            except Exception as e:
                print(f"Error loading configuration: {e}")
                print("Using default configuration")
    
    def save_config(self):
        """Save configuration to file"""
        try:
            import json
            config_data = {
                'detection': {
                    'model_path': self.detection.model_path,
                    'confidence_threshold': self.detection.confidence_threshold,
                    'iou_threshold': self.detection.iou_threshold,
                    'max_detections': self.detection.max_detections,
                    'detection_fps_target': self.detection.detection_fps_target
                },
                'posture': {
                    'inactivity_threshold': self.posture.inactivity_threshold,
                    'head_tilt_threshold': self.posture.head_tilt_threshold,
                    'movement_threshold': self.posture.movement_threshold,
                    'max_tracking_history': self.posture.max_tracking_history,
                    'pose_confidence_threshold': self.posture.pose_confidence_threshold
                },
                'alert': {
                    'enable_audio': self.alert.enable_audio,
                    'enable_visual': self.alert.enable_visual,
                    'enable_notifications': self.alert.enable_notifications,
                    'alert_cooldown': self.alert.alert_cooldown,
                    'inactivity_alert_threshold': self.alert.inactivity_alert_threshold,
                    'distraction_alert_threshold': self.alert.distraction_alert_threshold
                },
                'visualization': {
                    'show_overlays': self.visualization.show_overlays,
                    'show_dashboard': self.visualization.show_dashboard,
                    'dashboard_size': self.visualization.dashboard_size,
                    'overlay_alpha': self.visualization.overlay_alpha,
                    'update_interval': self.visualization.update_interval
                },
                'logging': {
                    'enable_logging': self.logging.enable_logging,
                    'db_path': self.logging.db_path,
                    'log_directory': self.logging.log_directory,
                    'max_log_files': self.logging.max_log_files,
                    'cleanup_days': self.logging.cleanup_days,
                    'export_format': self.logging.export_format
                },
                'system': {
                    'camera_source': self.system.camera_source,
                    'frame_width': self.system.frame_width,
                    'frame_height': self.system.frame_height,
                    'target_fps': self.system.target_fps,
                    'max_persons': self.system.max_persons,
                    'performance_targets': self.system.performance_targets
                }
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            print(f"Configuration saved to {self.config_file}")
            
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def get_performance_targets(self) -> Dict[str, float]:
        """Get performance targets"""
        return self.system.performance_targets
    
    def validate_config(self) -> bool:
        """Validate configuration values"""
        errors = []
        
        # Validate detection config
        if not 0.0 <= self.detection.confidence_threshold <= 1.0:
            errors.append("Detection confidence threshold must be between 0.0 and 1.0")
        
        if not 0.0 <= self.detection.iou_threshold <= 1.0:
            errors.append("IoU threshold must be between 0.0 and 1.0")
        
        # Validate posture config
        if self.posture.inactivity_threshold <= 0:
            errors.append("Inactivity threshold must be positive")
        
        if not 0.0 <= self.posture.head_tilt_threshold <= 180.0:
            errors.append("Head tilt threshold must be between 0.0 and 180.0 degrees")
        
        # Validate alert config
        if self.alert.alert_cooldown <= 0:
            errors.append("Alert cooldown must be positive")
        
        # Validate system config
        if self.system.target_fps <= 0:
            errors.append("Target FPS must be positive")
        
        if self.system.max_persons <= 0:
            errors.append("Max persons must be positive")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True

# Global configuration instance
config = Config()
