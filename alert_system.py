"""
Alert system for sending gentle reminders when inactivity or distraction is detected
"""
import pygame
import time
import threading
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import json
import os

class AlertType(Enum):
    VISUAL = "visual"
    AUDIO = "audio"
    NOTIFICATION = "notification"

@dataclass
class Alert:
    person_id: int
    seat_id: int
    alert_type: AlertType
    message: str
    timestamp: float
    severity: int  # 1-5 scale
    acknowledged: bool = False

class AlertSystem:
    def __init__(self, 
                 enable_audio: bool = True,
                 enable_visual: bool = True,
                 enable_notifications: bool = True,
                 alert_cooldown: float = 60.0):  # seconds between alerts for same person
        """
        Initialize alert system
        
        Args:
            enable_audio: Enable audio alerts
            enable_visual: Enable visual alerts
            enable_notifications: Enable system notifications
            alert_cooldown: Minimum time between alerts for the same person
        """
        self.enable_audio = enable_audio
        self.enable_visual = enable_visual
        self.enable_notifications = enable_notifications
        self.alert_cooldown = alert_cooldown
        
        # Initialize pygame for audio
        if self.enable_audio:
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                self.audio_initialized = True
            except:
                print("Warning: Could not initialize audio system")
                self.audio_initialized = False
        else:
            self.audio_initialized = False
        
        # Alert history and cooldowns
        self.alert_history = []
        self.person_cooldowns = {}
        self.max_history = 1000
        
        # Alert templates
        self.alert_templates = {
            'inactivity': [
                "Gentle reminder: Take a moment to refocus",
                "You've been inactive for a while. Time to get back to studying!",
                "Focus check: Are you still engaged with your work?"
            ],
            'distraction': [
                "Stay focused! Your study session is important",
                "Gentle nudge: Try to minimize distractions",
                "Focus reminder: You're doing great, keep it up!"
            ],
            'posture': [
                "Posture check: Sit up straight for better focus",
                "Take a moment to adjust your posture",
                "Good posture helps with concentration"
            ]
        }
        
        # Load alert settings
        self.load_settings()
    
    def send_alert(self, person_id: int, seat_id: int, alert_type: str, 
                   severity: int = 2, custom_message: Optional[str] = None) -> bool:
        """
        Send an alert for a specific person
        
        Args:
            person_id: ID of the person
            seat_id: ID of the seat
            alert_type: Type of alert ('inactivity', 'distraction', 'posture')
            severity: Severity level (1-5)
            custom_message: Custom message (overrides template)
            
        Returns:
            True if alert was sent, False if blocked by cooldown
        """
        # Check cooldown
        if self._is_on_cooldown(person_id):
            return False
        
        # Generate message
        if custom_message:
            message = custom_message
        else:
            message = self._get_alert_message(alert_type)
        
        # Create alert
        alert = Alert(
            person_id=person_id,
            seat_id=seat_id,
            alert_type=AlertType.VISUAL,  # Default to visual
            message=message,
            timestamp=time.time(),
            severity=severity
        )
        
        # Send alerts based on enabled types
        sent = False
        
        if self.enable_visual:
            self._send_visual_alert(alert)
            sent = True
        
        if self.enable_audio and self.audio_initialized:
            self._send_audio_alert(alert)
            sent = True
        
        if self.enable_notifications:
            self._send_notification_alert(alert)
            sent = True
        
        if sent:
            # Record alert
            self.alert_history.append(alert)
            if len(self.alert_history) > self.max_history:
                self.alert_history.pop(0)
            
            # Set cooldown
            self.person_cooldowns[person_id] = time.time()
            
            # Log alert
            self._log_alert(alert)
        
        return sent
    
    def _is_on_cooldown(self, person_id: int) -> bool:
        """Check if person is on alert cooldown"""
        if person_id not in self.person_cooldowns:
            return False
        
        last_alert = self.person_cooldowns[person_id]
        return (time.time() - last_alert) < self.alert_cooldown
    
    def _get_alert_message(self, alert_type: str) -> str:
        """Get random alert message for type"""
        import random
        
        if alert_type in self.alert_templates:
            messages = self.alert_templates[alert_type]
            return random.choice(messages)
        
        return "Gentle reminder: Stay focused on your studies!"
    
    def _send_visual_alert(self, alert: Alert):
        """Send visual alert (overlay on screen)"""
        # This would be handled by the visualization system
        print(f"VISUAL ALERT for Person {alert.person_id} at Seat {alert.seat_id}: {alert.message}")
    
    def _send_audio_alert(self, alert: Alert):
        """Send audio alert"""
        try:
            # Generate different tones based on severity
            frequency = 440 + (alert.severity * 100)  # Higher severity = higher pitch
            duration = 0.5 + (alert.severity * 0.2)   # Higher severity = longer duration
            
            # Create a simple beep sound
            sample_rate = 22050
            frames = int(duration * sample_rate)
            arr = []
            
            for i in range(frames):
                time_val = float(i) / sample_rate
                wave = 4096 * (1 if (int(time_val * frequency) % 2) else -1)
                arr.append([int(wave), int(wave)])
            
            sound = pygame.sndarray.make_sound(pygame.array.array('i', arr))
            sound.play()
            
        except Exception as e:
            print(f"Error playing audio alert: {e}")
    
    def _send_notification_alert(self, alert: Alert):
        """Send system notification"""
        try:
            # This would integrate with system notification APIs
            # For now, just print to console
            print(f"NOTIFICATION: {alert.message}")
        except Exception as e:
            print(f"Error sending notification: {e}")
    
    def _log_alert(self, alert: Alert):
        """Log alert to file"""
        log_entry = {
            'timestamp': alert.timestamp,
            'person_id': alert.person_id,
            'seat_id': alert.seat_id,
            'message': alert.message,
            'severity': alert.severity,
            'type': alert.alert_type.value
        }
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Append to daily log file
        date_str = time.strftime('%Y-%m-%d')
        log_file = f'logs/alerts_{date_str}.json'
        
        try:
            # Load existing logs
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            # Add new log entry
            logs.append(log_entry)
            
            # Save back to file
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            print(f"Error logging alert: {e}")
    
    def get_alert_stats(self, hours: int = 24) -> Dict:
        """Get alert statistics for the last N hours"""
        cutoff_time = time.time() - (hours * 3600)
        
        recent_alerts = [alert for alert in self.alert_history 
                        if alert.timestamp > cutoff_time]
        
        if not recent_alerts:
            return {
                'total_alerts': 0,
                'alerts_by_type': {},
                'alerts_by_severity': {},
                'most_alerted_persons': [],
                'average_alerts_per_hour': 0.0
            }
        
        # Count by type
        alerts_by_type = {}
        for alert in recent_alerts:
            alert_type = alert.alert_type.value
            alerts_by_type[alert_type] = alerts_by_type.get(alert_type, 0) + 1
        
        # Count by severity
        alerts_by_severity = {}
        for alert in recent_alerts:
            severity = alert.severity
            alerts_by_severity[severity] = alerts_by_severity.get(severity, 0) + 1
        
        # Most alerted persons
        person_alert_counts = {}
        for alert in recent_alerts:
            person_id = alert.person_id
            person_alert_counts[person_id] = person_alert_counts.get(person_id, 0) + 1
        
        most_alerted = sorted(person_alert_counts.items(), 
                           key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_alerts': len(recent_alerts),
            'alerts_by_type': alerts_by_type,
            'alerts_by_severity': alerts_by_severity,
            'most_alerted_persons': most_alerted,
            'average_alerts_per_hour': len(recent_alerts) / hours
        }
    
    def acknowledge_alert(self, person_id: int, alert_id: Optional[int] = None):
        """Acknowledge an alert (mark as read)"""
        if alert_id is not None:
            # Acknowledge specific alert
            for alert in self.alert_history:
                if (hasattr(alert, 'id') and alert.id == alert_id) or alert.person_id == person_id:
                    alert.acknowledged = True
                    break
        else:
            # Acknowledge all recent alerts for person
            recent_time = time.time() - 300  # Last 5 minutes
            for alert in self.alert_history:
                if (alert.person_id == person_id and 
                    alert.timestamp > recent_time and 
                    not alert.acknowledged):
                    alert.acknowledged = True
    
    def load_settings(self):
        """Load alert settings from file"""
        settings_file = 'alert_settings.json'
        if os.path.exists(settings_file):
            try:
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                    self.enable_audio = settings.get('enable_audio', self.enable_audio)
                    self.enable_visual = settings.get('enable_visual', self.enable_visual)
                    self.enable_notifications = settings.get('enable_notifications', self.enable_notifications)
                    self.alert_cooldown = settings.get('alert_cooldown', self.alert_cooldown)
            except Exception as e:
                print(f"Error loading alert settings: {e}")
    
    def save_settings(self):
        """Save alert settings to file"""
        settings = {
            'enable_audio': self.enable_audio,
            'enable_visual': self.enable_visual,
            'enable_notifications': self.enable_notifications,
            'alert_cooldown': self.alert_cooldown
        }
        
        try:
            with open('alert_settings.json', 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"Error saving alert settings: {e}")
    
    def cleanup_old_alerts(self, max_age_hours: int = 168):  # 1 week default
        """Remove old alerts from history"""
        cutoff_time = time.time() - (max_age_hours * 3600)
        self.alert_history = [alert for alert in self.alert_history 
                             if alert.timestamp > cutoff_time]
        
        # Also cleanup old cooldowns
        current_time = time.time()
        self.person_cooldowns = {person_id: last_alert 
                                for person_id, last_alert in self.person_cooldowns.items()
                                if current_time - last_alert < self.alert_cooldown * 2}
