"""
Real-time visualization system with overlays and dashboard
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import time
from collections import defaultdict, deque
import json
import os

class VisualizationSystem:
    def __init__(self, 
                 show_overlays: bool = True,
                 show_dashboard: bool = True,
                 dashboard_size: Tuple[int, int] = (800, 600),
                 overlay_alpha: float = 0.7):
        """
        Initialize visualization system
        
        Args:
            show_overlays: Enable real-time overlays on video
            show_dashboard: Enable separate dashboard window
            dashboard_size: Size of dashboard window (width, height)
            overlay_alpha: Transparency of overlays
        """
        self.show_overlays = show_overlays
        self.show_dashboard = show_dashboard
        self.dashboard_size = dashboard_size
        self.overlay_alpha = overlay_alpha
        
        # Color scheme
        self.colors = {
            'empty_seat': (0, 255, 0),      # Green
            'occupied_seat': (0, 0, 255),   # Red
            'distracted_seat': (0, 255, 255), # Yellow
            'focused_seat': (0, 128, 0),    # Dark green
            'alert_seat': (255, 0, 0),      # Bright red
            'text': (255, 255, 255),        # White
            'background': (0, 0, 0)         # Black
        }
        
        # Dashboard data
        self.dashboard_data = {
            'occupancy_history': deque(maxlen=100),
            'focus_history': deque(maxlen=100),
            'alert_history': deque(maxlen=50),
            'seat_stats': defaultdict(lambda: {'total_time': 0, 'focused_time': 0, 'alerts': 0})
        }
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.last_fps_time = time.time()
        
        # Initialize dashboard if enabled
        if self.show_dashboard:
            self._init_dashboard()
    
    def _init_dashboard(self):
        """Initialize dashboard window"""
        try:
            plt.ion()  # Interactive mode
            self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
            self.fig.suptitle('Library Seat Monitoring Dashboard', fontsize=16)
            plt.tight_layout()
        except Exception as e:
            print(f"Warning: Could not initialize dashboard: {e}")
            self.show_dashboard = False
    
    def draw_overlays(self, frame: np.ndarray, detections: Dict, 
                     associated_seats: List[Dict], posture_analyses: List[Dict]) -> np.ndarray:
        """
        Draw overlays on the frame
        
        Args:
            frame: Input frame
            detections: Detection results
            associated_seats: Seat-person associations
            posture_analyses: Posture analysis results
            
        Returns:
            Frame with overlays
        """
        if not self.show_overlays:
            return frame
        
        overlay_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw seat overlays
        for seat_info in associated_seats:
            chair = seat_info['chair']
            person = seat_info.get('person')
            seat_id = seat_info['seat_id']
            is_occupied = seat_info['is_occupied']
            
            x1, y1, x2, y2 = chair['bbox']
            
            # Determine color based on status
            if not is_occupied:
                color = self.colors['empty_seat']
                status_text = "EMPTY"
            else:
                # Check if person is focused
                person_id = person.get('person_id', -1) if person else -1
                posture_analysis = next((p for p in posture_analyses 
                                       if p.get('person_id') == person_id), None)
                
                if posture_analysis and posture_analysis.get('is_focused', False):
                    color = self.colors['focused_seat']
                    status_text = "FOCUSED"
                else:
                    color = self.colors['distracted_seat']
                    status_text = "DISTRACTED"
            
            # Draw bounding box
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw seat ID and status
            label = f"Seat {seat_id}: {status_text}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background for text
            cv2.rectangle(overlay_frame, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), 
                         color, -1)
            
            # Text
            cv2.putText(overlay_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        # Draw person overlays
        for person in detections.get('persons', []):
            x1, y1, x2, y2 = person['bbox']
            confidence = person['confidence']
            
            # Draw person bounding box
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
            
            # Confidence text
            conf_text = f"Person: {confidence:.2f}"
            cv2.putText(overlay_frame, conf_text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        # Draw statistics overlay
        self._draw_stats_overlay(overlay_frame, associated_seats, posture_analyses)
        
        # Blend with original frame
        result = cv2.addWeighted(frame, 1 - self.overlay_alpha, 
                                overlay_frame, self.overlay_alpha, 0)
        
        return result
    
    def _draw_stats_overlay(self, frame: np.ndarray, associated_seats: List[Dict], 
                           posture_analyses: List[Dict]):
        """Draw statistics overlay on frame"""
        h, w = frame.shape[:2]
        
        # Calculate statistics
        total_seats = len(associated_seats)
        occupied_seats = sum(1 for seat in associated_seats if seat['is_occupied'])
        focused_seats = sum(1 for analysis in posture_analyses 
                          if analysis.get('is_focused', False))
        
        occupancy_rate = (occupied_seats / total_seats * 100) if total_seats > 0 else 0
        focus_rate = (focused_seats / occupied_seats * 100) if occupied_seats > 0 else 0
        
        # Draw stats background
        stats_height = 120
        cv2.rectangle(frame, (10, 10), (300, stats_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, stats_height), (255, 255, 255), 2)
        
        # Draw stats text
        stats_text = [
            f"Total Seats: {total_seats}",
            f"Occupied: {occupied_seats} ({occupancy_rate:.1f}%)",
            f"Focused: {focused_seats} ({focus_rate:.1f}%)",
            f"FPS: {self._get_current_fps():.1f}"
        ]
        
        for i, text in enumerate(stats_text):
            y_pos = 30 + i * 20
            cv2.putText(frame, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
    
    def update_dashboard(self, associated_seats: List[Dict], posture_analyses: List[Dict]):
        """Update dashboard with current data"""
        if not self.show_dashboard:
            return
        
        try:
            current_time = time.time()
            
            # Update occupancy data
            total_seats = len(associated_seats)
            occupied_seats = sum(1 for seat in associated_seats if seat['is_occupied'])
            occupancy_rate = (occupied_seats / total_seats) if total_seats > 0 else 0
            
            self.dashboard_data['occupancy_history'].append({
                'timestamp': current_time,
                'occupancy_rate': occupancy_rate,
                'total_seats': total_seats,
                'occupied_seats': occupied_seats
            })
            
            # Update focus data
            focused_count = sum(1 for analysis in posture_analyses 
                              if analysis.get('is_focused', False))
            focus_rate = (focused_count / occupied_seats) if occupied_seats > 0 else 0
            
            self.dashboard_data['focus_history'].append({
                'timestamp': current_time,
                'focus_rate': focus_rate,
                'focused_count': focused_count
            })
            
            # Update seat statistics
            for i, seat in enumerate(associated_seats):
                if seat['is_occupied']:
                    self.dashboard_data['seat_stats'][i]['total_time'] += 1
                    
                    # Check if focused
                    person_id = seat.get('person', {}).get('person_id', -1)
                    posture_analysis = next((p for p in posture_analyses 
                                           if p.get('person_id') == person_id), None)
                    
                    if posture_analysis and posture_analysis.get('is_focused', False):
                        self.dashboard_data['seat_stats'][i]['focused_time'] += 1
            
            # Update FPS
            self._update_fps()
            
            # Redraw dashboard
            self._redraw_dashboard()
            
        except Exception as e:
            print(f"Error updating dashboard: {e}")
    
    def _redraw_dashboard(self):
        """Redraw the dashboard plots"""
        try:
            # Clear axes
            for ax in self.axes.flat:
                ax.clear()
            
            # Plot 1: Occupancy over time
            if self.dashboard_data['occupancy_history']:
                times = [d['timestamp'] for d in self.dashboard_data['occupancy_history']]
                occupancy_rates = [d['occupancy_rate'] * 100 for d in self.dashboard_data['occupancy_history']]
                
                self.axes[0, 0].plot(times, occupancy_rates, 'b-', linewidth=2)
                self.axes[0, 0].set_title('Occupancy Rate Over Time')
                self.axes[0, 0].set_ylabel('Occupancy %')
                self.axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Focus rate over time
            if self.dashboard_data['focus_history']:
                times = [d['timestamp'] for d in self.dashboard_data['focus_history']]
                focus_rates = [d['focus_rate'] * 100 for d in self.dashboard_data['focus_history']]
                
                self.axes[0, 1].plot(times, focus_rates, 'g-', linewidth=2)
                self.axes[0, 1].set_title('Focus Rate Over Time')
                self.axes[0, 1].set_ylabel('Focus %')
                self.axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Seat utilization heatmap
            if self.dashboard_data['seat_stats']:
                seat_ids = list(self.dashboard_data['seat_stats'].keys())
                utilization_rates = []
                
                for seat_id in seat_ids:
                    stats = self.dashboard_data['seat_stats'][seat_id]
                    if stats['total_time'] > 0:
                        utilization = stats['focused_time'] / stats['total_time']
                    else:
                        utilization = 0
                    utilization_rates.append(utilization)
                
                if utilization_rates:
                    self.axes[1, 0].bar(seat_ids, utilization_rates, color='skyblue', alpha=0.7)
                    self.axes[1, 0].set_title('Seat Utilization Rates')
                    self.axes[1, 0].set_xlabel('Seat ID')
                    self.axes[1, 0].set_ylabel('Utilization Rate')
                    self.axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Current statistics
            current_stats = self._get_current_stats()
            stats_text = [
                f"Total Seats: {current_stats['total_seats']}",
                f"Occupied: {current_stats['occupied_seats']}",
                f"Focused: {current_stats['focused_seats']}",
                f"Occupancy Rate: {current_stats['occupancy_rate']:.1f}%",
                f"Focus Rate: {current_stats['focus_rate']:.1f}%",
                f"FPS: {self._get_current_fps():.1f}"
            ]
            
            self.axes[1, 1].text(0.1, 0.9, '\n'.join(stats_text), 
                               transform=self.axes[1, 1].transAxes,
                               fontsize=12, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            self.axes[1, 1].set_title('Current Statistics')
            self.axes[1, 1].axis('off')
            
            # Update the plot
            plt.draw()
            plt.pause(0.01)
            
        except Exception as e:
            print(f"Error redrawing dashboard: {e}")
    
    def _get_current_stats(self) -> Dict:
        """Get current statistics"""
        if not self.dashboard_data['occupancy_history']:
            return {
                'total_seats': 0,
                'occupied_seats': 0,
                'focused_seats': 0,
                'occupancy_rate': 0.0,
                'focus_rate': 0.0
            }
        
        latest_occupancy = self.dashboard_data['occupancy_history'][-1]
        latest_focus = self.dashboard_data['focus_history'][-1] if self.dashboard_data['focus_history'] else {'focus_rate': 0, 'focused_count': 0}
        
        return {
            'total_seats': latest_occupancy['total_seats'],
            'occupied_seats': latest_occupancy['occupied_seats'],
            'focused_seats': latest_focus['focused_count'],
            'occupancy_rate': latest_occupancy['occupancy_rate'] * 100,
            'focus_rate': latest_focus['focus_rate'] * 100
        }
    
    def _update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        self.fps_counter.append(current_time)
        
        # Keep only last 30 frames
        if len(self.fps_counter) > 30:
            self.fps_counter.popleft()
    
    def _get_current_fps(self) -> float:
        """Get current FPS"""
        if len(self.fps_counter) < 2:
            return 0.0
        
        time_diff = self.fps_counter[-1] - self.fps_counter[0]
        if time_diff > 0:
            return (len(self.fps_counter) - 1) / time_diff
        return 0.0
    
    def save_heatmap(self, filename: str = None):
        """Save seat utilization heatmap"""
        if not self.dashboard_data['seat_stats']:
            return
        
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f'heatmap_{timestamp}.png'
        
        try:
            # Create heatmap data
            seat_ids = list(self.dashboard_data['seat_stats'].keys())
            utilization_rates = []
            
            for seat_id in seat_ids:
                stats = self.dashboard_data['seat_stats'][seat_id]
                if stats['total_time'] > 0:
                    utilization = stats['focused_time'] / stats['total_time']
                else:
                    utilization = 0
                utilization_rates.append(utilization)
            
            # Create heatmap
            plt.figure(figsize=(10, 6))
            sns.heatmap([utilization_rates], 
                       xticklabels=seat_ids,
                       yticklabels=['Utilization Rate'],
                       cmap='RdYlGn',
                       cbar_kws={'label': 'Utilization Rate'})
            plt.title('Seat Utilization Heatmap')
            plt.xlabel('Seat ID')
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Heatmap saved as {filename}")
            
        except Exception as e:
            print(f"Error saving heatmap: {e}")
    
    def export_data(self, filename: str = None):
        """Export dashboard data to JSON"""
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f'dashboard_data_{timestamp}.json'
        
        try:
            export_data = {
                'occupancy_history': list(self.dashboard_data['occupancy_history']),
                'focus_history': list(self.dashboard_data['focus_history']),
                'seat_stats': dict(self.dashboard_data['seat_stats']),
                'export_timestamp': time.time()
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"Data exported to {filename}")
            
        except Exception as e:
            print(f"Error exporting data: {e}")
    
    def close_dashboard(self):
        """Close dashboard window"""
        if self.show_dashboard:
            try:
                plt.close(self.fig)
            except:
                pass
