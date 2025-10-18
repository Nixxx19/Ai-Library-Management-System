"""
Data logging and analytics storage system
"""
import json
import csv
import sqlite3
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict, deque

class DataLogger:
    def __init__(self, 
                 db_path: str = "library_monitoring.db",
                 log_directory: str = "logs",
                 max_log_files: int = 30):
        """
        Initialize data logging system
        
        Args:
            db_path: Path to SQLite database
            log_directory: Directory for log files
            max_log_files: Maximum number of log files to keep
        """
        self.db_path = db_path
        self.log_directory = log_directory
        self.max_log_files = max_log_files
        
        # Create directories
        os.makedirs(self.log_directory, exist_ok=True)
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # In-memory cache for real-time data
        self.cache = {
            'occupancy_data': deque(maxlen=1000),
            'focus_data': deque(maxlen=1000),
            'alert_data': deque(maxlen=500),
            'performance_data': deque(maxlen=100)
        }
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Occupancy table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS occupancy_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    seat_id INTEGER NOT NULL,
                    is_occupied BOOLEAN NOT NULL,
                    person_id INTEGER,
                    confidence REAL,
                    session_id TEXT
                )
            ''')
            
            # Focus/Posture table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS focus_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    person_id INTEGER NOT NULL,
                    seat_id INTEGER NOT NULL,
                    is_focused BOOLEAN NOT NULL,
                    head_tilt REAL,
                    shoulder_angle REAL,
                    body_lean REAL,
                    hand_activity REAL,
                    confidence REAL,
                    session_id TEXT
                )
            ''')
            
            # Alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alert_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    person_id INTEGER NOT NULL,
                    seat_id INTEGER NOT NULL,
                    alert_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    severity INTEGER NOT NULL,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    session_id TEXT
                )
            ''')
            
            # Performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    fps REAL NOT NULL,
                    detection_time REAL NOT NULL,
                    processing_time REAL NOT NULL,
                    memory_usage REAL,
                    session_id TEXT
                )
            ''')
            
            # Sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    total_frames INTEGER DEFAULT 0,
                    total_detections INTEGER DEFAULT 0,
                    avg_fps REAL,
                    notes TEXT
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_occupancy_timestamp ON occupancy_logs(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_occupancy_seat_id ON occupancy_logs(seat_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_focus_timestamp ON focus_logs(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_focus_person_id ON focus_logs(person_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alert_timestamp ON alert_logs(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_logs(timestamp)')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error initializing database: {e}")
    
    def log_occupancy(self, seat_id: int, is_occupied: bool, 
                     person_id: Optional[int] = None, confidence: float = 0.0,
                     session_id: str = None):
        """Log seat occupancy data"""
        timestamp = time.time()
        
        # Add to cache
        self.cache['occupancy_data'].append({
            'timestamp': timestamp,
            'seat_id': seat_id,
            'is_occupied': is_occupied,
            'person_id': person_id,
            'confidence': confidence,
            'session_id': session_id
        })
        
        # Add to database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO occupancy_logs 
                (timestamp, seat_id, is_occupied, person_id, confidence, session_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (timestamp, seat_id, is_occupied, person_id, confidence, session_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error logging occupancy: {e}")
    
    def log_focus(self, person_id: int, seat_id: int, is_focused: bool,
                 head_tilt: float = 0.0, shoulder_angle: float = 0.0,
                 body_lean: float = 0.0, hand_activity: float = 0.0,
                 confidence: float = 0.0, session_id: str = None):
        """Log focus/posture data"""
        timestamp = time.time()
        
        # Add to cache
        self.cache['focus_data'].append({
            'timestamp': timestamp,
            'person_id': person_id,
            'seat_id': seat_id,
            'is_focused': is_focused,
            'head_tilt': head_tilt,
            'shoulder_angle': shoulder_angle,
            'body_lean': body_lean,
            'hand_activity': hand_activity,
            'confidence': confidence,
            'session_id': session_id
        })
        
        # Add to database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO focus_logs 
                (timestamp, person_id, seat_id, is_focused, head_tilt, shoulder_angle, 
                 body_lean, hand_activity, confidence, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, person_id, seat_id, is_focused, head_tilt, shoulder_angle,
                  body_lean, hand_activity, confidence, session_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error logging focus: {e}")
    
    def log_alert(self, person_id: int, seat_id: int, alert_type: str,
                 message: str, severity: int, session_id: str = None):
        """Log alert data"""
        timestamp = time.time()
        
        # Add to cache
        self.cache['alert_data'].append({
            'timestamp': timestamp,
            'person_id': person_id,
            'seat_id': seat_id,
            'alert_type': alert_type,
            'message': message,
            'severity': severity,
            'session_id': session_id
        })
        
        # Add to database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO alert_logs 
                (timestamp, person_id, seat_id, alert_type, message, severity, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, person_id, seat_id, alert_type, message, severity, session_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error logging alert: {e}")
    
    def log_performance(self, fps: float, detection_time: float, 
                       processing_time: float, memory_usage: float = 0.0,
                       session_id: str = None):
        """Log performance metrics"""
        timestamp = time.time()
        
        # Add to cache
        self.cache['performance_data'].append({
            'timestamp': timestamp,
            'fps': fps,
            'detection_time': detection_time,
            'processing_time': processing_time,
            'memory_usage': memory_usage,
            'session_id': session_id
        })
        
        # Add to database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_logs 
                (timestamp, fps, detection_time, processing_time, memory_usage, session_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (timestamp, fps, detection_time, processing_time, memory_usage, session_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error logging performance: {e}")
    
    def start_session(self, session_id: str = None, notes: str = "") -> str:
        """Start a new monitoring session"""
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        timestamp = time.time()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO sessions 
                (id, start_time, notes)
                VALUES (?, ?, ?)
            ''', (session_id, timestamp, notes))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error starting session: {e}")
        
        return session_id
    
    def end_session(self, session_id: str, total_frames: int = 0, 
                   total_detections: int = 0, avg_fps: float = 0.0):
        """End a monitoring session"""
        timestamp = time.time()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE sessions 
                SET end_time = ?, total_frames = ?, total_detections = ?, avg_fps = ?
                WHERE id = ?
            ''', (timestamp, total_frames, total_detections, avg_fps, session_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error ending session: {e}")
    
    def get_analytics(self, start_time: float = None, end_time: float = None,
                     session_id: str = None) -> Dict:
        """Get analytics data for specified time range"""
        if start_time is None:
            start_time = time.time() - 3600  # Last hour
        if end_time is None:
            end_time = time.time()
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            analytics = {}
            
            # Occupancy analytics
            query = '''
                SELECT 
                    COUNT(*) as total_logs,
                    AVG(CASE WHEN is_occupied THEN 1.0 ELSE 0.0 END) as avg_occupancy_rate,
                    COUNT(DISTINCT seat_id) as unique_seats
                FROM occupancy_logs 
                WHERE timestamp BETWEEN ? AND ?
            '''
            params = [start_time, end_time]
            
            if session_id:
                query += ' AND session_id = ?'
                params.append(session_id)
            
            cursor = conn.cursor()
            cursor.execute(query, params)
            occupancy_stats = cursor.fetchone()
            
            analytics['occupancy'] = {
                'total_logs': occupancy_stats[0],
                'avg_occupancy_rate': occupancy_stats[1] * 100 if occupancy_stats[1] else 0,
                'unique_seats': occupancy_stats[2]
            }
            
            # Focus analytics
            query = '''
                SELECT 
                    COUNT(*) as total_logs,
                    AVG(CASE WHEN is_focused THEN 1.0 ELSE 0.0 END) as avg_focus_rate,
                    AVG(head_tilt) as avg_head_tilt,
                    AVG(hand_activity) as avg_hand_activity
                FROM focus_logs 
                WHERE timestamp BETWEEN ? AND ?
            '''
            params = [start_time, end_time]
            
            if session_id:
                query += ' AND session_id = ?'
                params.append(session_id)
            
            cursor.execute(query, params)
            focus_stats = cursor.fetchone()
            
            analytics['focus'] = {
                'total_logs': focus_stats[0],
                'avg_focus_rate': focus_stats[1] * 100 if focus_stats[1] else 0,
                'avg_head_tilt': focus_stats[2] if focus_stats[2] else 0,
                'avg_hand_activity': focus_stats[3] if focus_stats[3] else 0
            }
            
            # Alert analytics
            query = '''
                SELECT 
                    COUNT(*) as total_alerts,
                    COUNT(DISTINCT person_id) as unique_persons_alerted,
                    AVG(severity) as avg_severity
                FROM alert_logs 
                WHERE timestamp BETWEEN ? AND ?
            '''
            params = [start_time, end_time]
            
            if session_id:
                query += ' AND session_id = ?'
                params.append(session_id)
            
            cursor.execute(query, params)
            alert_stats = cursor.fetchone()
            
            analytics['alerts'] = {
                'total_alerts': alert_stats[0],
                'unique_persons_alerted': alert_stats[1],
                'avg_severity': alert_stats[2] if alert_stats[2] else 0
            }
            
            # Performance analytics
            query = '''
                SELECT 
                    AVG(fps) as avg_fps,
                    AVG(detection_time) as avg_detection_time,
                    AVG(processing_time) as avg_processing_time
                FROM performance_logs 
                WHERE timestamp BETWEEN ? AND ?
            '''
            params = [start_time, end_time]
            
            if session_id:
                query += ' AND session_id = ?'
                params.append(session_id)
            
            cursor.execute(query, params)
            perf_stats = cursor.fetchone()
            
            analytics['performance'] = {
                'avg_fps': perf_stats[0] if perf_stats[0] else 0,
                'avg_detection_time': perf_stats[1] if perf_stats[1] else 0,
                'avg_processing_time': perf_stats[2] if perf_stats[2] else 0
            }
            
            conn.close()
            return analytics
            
        except Exception as e:
            print(f"Error getting analytics: {e}")
            return {}
    
    def export_to_csv(self, start_time: float = None, end_time: float = None,
                     session_id: str = None, output_dir: str = None):
        """Export data to CSV files"""
        if output_dir is None:
            output_dir = self.log_directory
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Export occupancy data
            query = '''
                SELECT * FROM occupancy_logs 
                WHERE timestamp BETWEEN ? AND ?
            '''
            params = [start_time or 0, end_time or time.time()]
            
            if session_id:
                query += ' AND session_id = ?'
                params.append(session_id)
            
            df_occupancy = pd.read_sql_query(query, conn, params=params)
            df_occupancy.to_csv(f'{output_dir}/occupancy_{timestamp}.csv', index=False)
            
            # Export focus data
            query = '''
                SELECT * FROM focus_logs 
                WHERE timestamp BETWEEN ? AND ?
            '''
            params = [start_time or 0, end_time or time.time()]
            
            if session_id:
                query += ' AND session_id = ?'
                params.append(session_id)
            
            df_focus = pd.read_sql_query(query, conn, params=params)
            df_focus.to_csv(f'{output_dir}/focus_{timestamp}.csv', index=False)
            
            # Export alert data
            query = '''
                SELECT * FROM alert_logs 
                WHERE timestamp BETWEEN ? AND ?
            '''
            params = [start_time or 0, end_time or time.time()]
            
            if session_id:
                query += ' AND session_id = ?'
                params.append(session_id)
            
            df_alerts = pd.read_sql_query(query, conn, params=params)
            df_alerts.to_csv(f'{output_dir}/alerts_{timestamp}.csv', index=False)
            
            # Export performance data
            query = '''
                SELECT * FROM performance_logs 
                WHERE timestamp BETWEEN ? AND ?
            '''
            params = [start_time or 0, end_time or time.time()]
            
            if session_id:
                query += ' AND session_id = ?'
                params.append(session_id)
            
            df_performance = pd.read_sql_query(query, conn, params=params)
            df_performance.to_csv(f'{output_dir}/performance_{timestamp}.csv', index=False)
            
            conn.close()
            
            print(f"Data exported to {output_dir}")
            
        except Exception as e:
            print(f"Error exporting data: {e}")
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data to save space"""
        cutoff_time = time.time() - (days_to_keep * 24 * 3600)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete old records
            tables = ['occupancy_logs', 'focus_logs', 'alert_logs', 'performance_logs']
            
            for table in tables:
                cursor.execute(f'DELETE FROM {table} WHERE timestamp < ?', (cutoff_time,))
                print(f"Deleted old records from {table}")
            
            # Vacuum database to reclaim space
            cursor.execute('VACUUM')
            
            conn.commit()
            conn.close()
            
            print(f"Cleanup completed. Kept data from last {days_to_keep} days.")
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def get_realtime_stats(self) -> Dict:
        """Get real-time statistics from cache"""
        current_time = time.time()
        recent_window = 300  # Last 5 minutes
        
        stats = {
            'occupancy_rate': 0.0,
            'focus_rate': 0.0,
            'alert_count': 0,
            'avg_fps': 0.0
        }
        
        # Calculate occupancy rate
        recent_occupancy = [d for d in self.cache['occupancy_data'] 
                           if current_time - d['timestamp'] < recent_window]
        if recent_occupancy:
            occupied_count = sum(1 for d in recent_occupancy if d['is_occupied'])
            stats['occupancy_rate'] = (occupied_count / len(recent_occupancy)) * 100
        
        # Calculate focus rate
        recent_focus = [d for d in self.cache['focus_data'] 
                       if current_time - d['timestamp'] < recent_window]
        if recent_focus:
            focused_count = sum(1 for d in recent_focus if d['is_focused'])
            stats['focus_rate'] = (focused_count / len(recent_focus)) * 100
        
        # Count recent alerts
        recent_alerts = [d for d in self.cache['alert_data'] 
                        if current_time - d['timestamp'] < recent_window]
        stats['alert_count'] = len(recent_alerts)
        
        # Calculate average FPS
        recent_performance = [d for d in self.cache['performance_data'] 
                             if current_time - d['timestamp'] < recent_window]
        if recent_performance:
            stats['avg_fps'] = np.mean([d['fps'] for d in recent_performance])
        
        return stats
