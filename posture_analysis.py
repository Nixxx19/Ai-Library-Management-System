"""
Posture analysis module using MediaPipe for detecting focused vs distracted behavior
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Tuple, Optional
import time
from collections import deque
import math

class PostureAnalyzer:
    def __init__(self, 
                 inactivity_threshold: float = 30.0,  # seconds
                 head_tilt_threshold: float = 30.0,   # degrees
                 movement_threshold: float = 0.1):    # normalized movement
        """
        Initialize posture analyzer
        
        Args:
            inactivity_threshold: Time in seconds before considering person inactive
            head_tilt_threshold: Maximum head tilt angle in degrees for focused posture
            movement_threshold: Minimum movement threshold to consider activity
        """
        self.mp_pose = mp.solutions.pose
        self.mp_face = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.face = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Analysis parameters
        self.inactivity_threshold = inactivity_threshold
        self.head_tilt_threshold = head_tilt_threshold
        self.movement_threshold = movement_threshold
        
        # Tracking data for each person
        self.person_tracking = {}
        self.max_tracking_history = 30  # frames
        
    def analyze_person(self, frame: np.ndarray, person_bbox: List[int], 
                      person_id: int) -> Dict:
        """
        Analyze posture and activity for a specific person
        
        Args:
            frame: Input frame
            person_bbox: Bounding box [x1, y1, x2, y2] of the person
            person_id: Unique identifier for the person
            
        Returns:
            Dictionary with posture analysis results
        """
        x1, y1, x2, y2 = person_bbox
        person_roi = frame[y1:y2, x1:x2]
        
        if person_roi.size == 0:
            return self._get_default_analysis()
        
        # Convert BGR to RGB for MediaPipe
        person_roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
        
        # Analyze pose
        pose_results = self.pose.process(person_roi_rgb)
        face_results = self.face.process(person_roi_rgb)
        hands_results = self.hands.process(person_roi_rgb)
        
        # Extract features
        features = self._extract_features(pose_results, face_results, hands_results, person_roi)
        
        # Update tracking data
        if person_id not in self.person_tracking:
            self.person_tracking[person_id] = {
                'history': deque(maxlen=self.max_tracking_history),
                'last_activity': time.time(),
                'inactivity_start': None,
                'alert_sent': False
            }
        
        # Add current features to history
        current_time = time.time()
        self.person_tracking[person_id]['history'].append({
            'timestamp': current_time,
            'features': features
        })
        
        # Analyze behavior patterns
        behavior_analysis = self._analyze_behavior_patterns(person_id)
        
        # Determine focus status
        is_focused = self._determine_focus_status(features, behavior_analysis)
        
        # Update activity tracking
        if is_focused:
            self.person_tracking[person_id]['last_activity'] = current_time
            self.person_tracking[person_id]['inactivity_start'] = None
            self.person_tracking[person_id]['alert_sent'] = False
        else:
            if self.person_tracking[person_id]['inactivity_start'] is None:
                self.person_tracking[person_id]['inactivity_start'] = current_time
        
        # Check for alert conditions
        should_alert = self._should_send_alert(person_id, current_time)
        
        return {
            'person_id': person_id,
            'is_focused': is_focused,
            'features': features,
            'behavior_analysis': behavior_analysis,
            'should_alert': should_alert,
            'inactivity_duration': self._get_inactivity_duration(person_id, current_time),
            'confidence': features.get('pose_confidence', 0.0)
        }
    
    def _extract_features(self, pose_results, face_results, hands_results, roi) -> Dict:
        """Extract relevant features from MediaPipe results"""
        features = {
            'pose_confidence': 0.0,
            'head_tilt': 0.0,
            'shoulder_angle': 0.0,
            'body_lean': 0.0,
            'hand_activity': 0.0,
            'eye_gaze': 0.0,
            'movement_magnitude': 0.0
        }
        
        # Pose analysis
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            features['pose_confidence'] = pose_results.pose_landmarks.visibility
            
            # Calculate head tilt
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
            left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR]
            right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR]
            
            if nose.visibility > 0.5 and left_ear.visibility > 0.5 and right_ear.visibility > 0.5:
                # Calculate head tilt angle
                ear_midpoint_x = (left_ear.x + right_ear.x) / 2
                ear_midpoint_y = (left_ear.y + right_ear.y) / 2
                
                head_tilt = math.degrees(math.atan2(
                    nose.x - ear_midpoint_x, 
                    nose.y - ear_midpoint_y
                ))
                features['head_tilt'] = abs(head_tilt)
            
            # Calculate shoulder angle
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
                shoulder_angle = math.degrees(math.atan2(
                    right_shoulder.y - left_shoulder.y,
                    right_shoulder.x - left_shoulder.x
                ))
                features['shoulder_angle'] = abs(shoulder_angle)
            
            # Calculate body lean
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
            
            if (nose.visibility > 0.5 and left_hip.visibility > 0.5 and 
                right_hip.visibility > 0.5):
                hip_midpoint_x = (left_hip.x + right_hip.x) / 2
                body_lean = abs(nose.x - hip_midpoint_x)
                features['body_lean'] = body_lean
        
        # Face analysis
        if face_results.multi_face_landmarks:
            # Simple eye gaze estimation based on face orientation
            face_landmarks = face_results.multi_face_landmarks[0]
            # This is a simplified gaze estimation - in practice, you'd use more sophisticated methods
            features['eye_gaze'] = 0.5  # Placeholder for gaze direction
        
        # Hand analysis
        if hands_results.multi_hand_landmarks:
            # Calculate hand movement/activity
            total_hand_landmarks = 0
            for hand_landmarks in hands_results.multi_hand_landmarks:
                total_hand_landmarks += len(hand_landmarks.landmark)
            features['hand_activity'] = min(total_hand_landmarks / 42, 1.0)  # Normalize
        
        return features
    
    def _analyze_behavior_patterns(self, person_id: int) -> Dict:
        """Analyze behavior patterns from historical data"""
        if person_id not in self.person_tracking:
            return {'movement_trend': 0.0, 'consistency': 0.0}
        
        history = self.person_tracking[person_id]['history']
        if len(history) < 5:
            return {'movement_trend': 0.0, 'consistency': 0.0}
        
        # Calculate movement trend
        recent_movements = [h['features'].get('movement_magnitude', 0.0) for h in list(history)[-5:]]
        movement_trend = np.mean(recent_movements) if recent_movements else 0.0
        
        # Calculate consistency (lower variance = higher consistency)
        head_tilts = [h['features'].get('head_tilt', 0.0) for h in list(history)[-10:]]
        consistency = 1.0 - np.var(head_tilts) if len(head_tilts) > 1 else 0.0
        
        return {
            'movement_trend': movement_trend,
            'consistency': max(0.0, min(1.0, consistency))
        }
    
    def _determine_focus_status(self, features: Dict, behavior_analysis: Dict) -> bool:
        """Determine if person is focused based on features and behavior"""
        # Focus criteria
        head_tilt_ok = features.get('head_tilt', 0) < self.head_tilt_threshold
        pose_confidence_ok = features.get('pose_confidence', 0) > 0.5
        movement_ok = features.get('movement_magnitude', 0) > self.movement_threshold
        consistency_ok = behavior_analysis.get('consistency', 0) > 0.3
        
        # Weighted scoring
        focus_score = 0.0
        if head_tilt_ok:
            focus_score += 0.3
        if pose_confidence_ok:
            focus_score += 0.3
        if movement_ok:
            focus_score += 0.2
        if consistency_ok:
            focus_score += 0.2
        
        return focus_score > 0.6
    
    def _should_send_alert(self, person_id: int, current_time: float) -> bool:
        """Check if alert should be sent for this person"""
        if person_id not in self.person_tracking:
            return False
        
        tracking = self.person_tracking[person_id]
        
        # Don't send multiple alerts for the same inactivity period
        if tracking['alert_sent']:
            return False
        
        # Check inactivity duration
        if tracking['inactivity_start'] is not None:
            inactivity_duration = current_time - tracking['inactivity_start']
            if inactivity_duration > self.inactivity_threshold:
                tracking['alert_sent'] = True
                return True
        
        return False
    
    def _get_inactivity_duration(self, person_id: int, current_time: float) -> float:
        """Get current inactivity duration for person"""
        if person_id not in self.person_tracking:
            return 0.0
        
        tracking = self.person_tracking[person_id]
        if tracking['inactivity_start'] is None:
            return 0.0
        
        return current_time - tracking['inactivity_start']
    
    def _get_default_analysis(self) -> Dict:
        """Return default analysis when person detection fails"""
        return {
            'person_id': -1,
            'is_focused': False,
            'features': {},
            'behavior_analysis': {},
            'should_alert': False,
            'inactivity_duration': 0.0,
            'confidence': 0.0
        }
    
    def get_tracking_stats(self) -> Dict:
        """Get statistics about tracked persons"""
        total_tracked = len(self.person_tracking)
        active_persons = sum(1 for tracking in self.person_tracking.values() 
                           if tracking['inactivity_start'] is None)
        
        return {
            'total_tracked_persons': total_tracked,
            'active_persons': active_persons,
            'inactive_persons': total_tracked - active_persons
        }
    
    def cleanup_old_tracking(self, max_age: float = 300.0):
        """Remove tracking data for persons not seen recently"""
        current_time = time.time()
        to_remove = []
        
        for person_id, tracking in self.person_tracking.items():
            if tracking['history']:
                last_seen = tracking['history'][-1]['timestamp']
                if current_time - last_seen > max_age:
                    to_remove.append(person_id)
        
        for person_id in to_remove:
            del self.person_tracking[person_id]
