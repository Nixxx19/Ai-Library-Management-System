"""
Test script for the Library Monitoring System
"""
import cv2
import numpy as np
import time
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detection import YOLODetector
from posture_analysis import PostureAnalyzer
from alert_system import AlertSystem
from visualization import VisualizationSystem
from data_logging import DataLogger
from config import config

def test_detection():
    """Test YOLO detection system"""
    print("Testing YOLO detection system...")
    
    try:
        detector = YOLODetector()
        print("✓ YOLO detector initialized successfully")
        
        # Create a test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test detection
        detections = detector.detect(test_image)
        print(f"✓ Detection test completed - found {len(detections.get('chairs', []))} chairs, {len(detections.get('persons', []))} persons")
        
        return True
        
    except Exception as e:
        print(f"✗ Detection test failed: {e}")
        return False

def test_posture_analysis():
    """Test posture analysis system"""
    print("Testing posture analysis system...")
    
    try:
        analyzer = PostureAnalyzer()
        print("✓ Posture analyzer initialized successfully")
        
        # Create a test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_bbox = [100, 100, 200, 300]
        person_id = 1
        
        # Test analysis
        analysis = analyzer.analyze_person(test_image, test_bbox, person_id)
        print(f"✓ Posture analysis test completed - focus: {analysis.get('is_focused', False)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Posture analysis test failed: {e}")
        return False

def test_alert_system():
    """Test alert system"""
    print("Testing alert system...")
    
    try:
        alert_system = AlertSystem(enable_audio=False)  # Disable audio for testing
        print("✓ Alert system initialized successfully")
        
        # Test alert
        success = alert_system.send_alert(
            person_id=1,
            seat_id=1,
            alert_type='test',
            severity=2,
            custom_message="Test alert"
        )
        
        if success:
            print("✓ Alert system test completed successfully")
        else:
            print("⚠ Alert system test completed (alert blocked by cooldown)")
        
        return True
        
    except Exception as e:
        print(f"✗ Alert system test failed: {e}")
        return False

def test_visualization():
    """Test visualization system"""
    print("Testing visualization system...")
    
    try:
        visualizer = VisualizationSystem(show_dashboard=False)  # Disable dashboard for testing
        print("✓ Visualization system initialized successfully")
        
        # Create test data
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = {'chairs': [], 'persons': []}
        associated_seats = []
        posture_analyses = []
        
        # Test overlay drawing
        result = visualizer.draw_overlays(test_image, detections, associated_seats, posture_analyses)
        print("✓ Visualization overlay test completed")
        
        return True
        
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        return False

def test_data_logging():
    """Test data logging system"""
    print("Testing data logging system...")
    
    try:
        logger = DataLogger(db_path="test_monitoring.db")
        print("✓ Data logger initialized successfully")
        
        # Test session
        session_id = logger.start_session(notes="Test session")
        print(f"✓ Test session started: {session_id}")
        
        # Test logging
        logger.log_occupancy(seat_id=1, is_occupied=True, person_id=1, confidence=0.9, session_id=session_id)
        logger.log_focus(person_id=1, seat_id=1, is_focused=True, session_id=session_id)
        logger.log_alert(person_id=1, seat_id=1, alert_type='test', message='Test alert', severity=2, session_id=session_id)
        logger.log_performance(fps=30.0, detection_time=0.05, processing_time=0.1, session_id=session_id)
        
        print("✓ Data logging test completed")
        
        # End session
        logger.end_session(session_id, total_frames=1, total_detections=1, avg_fps=30.0)
        print("✓ Test session ended")
        
        # Cleanup test database
        if os.path.exists("test_monitoring.db"):
            os.remove("test_monitoring.db")
        
        return True
        
    except Exception as e:
        print(f"✗ Data logging test failed: {e}")
        return False

def test_camera():
    """Test camera access"""
    print("Testing camera access...")
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("⚠ No camera available for testing")
            return True  # Not a failure, just no camera
        
        ret, frame = cap.read()
        if ret:
            print(f"✓ Camera test successful - frame shape: {frame.shape}")
        else:
            print("⚠ Camera test - could not read frame")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def test_dependencies():
    """Test required dependencies"""
    print("Testing dependencies...")
    
    dependencies = [
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('ultralytics', 'Ultralytics YOLO'),
        ('mediapipe', 'MediaPipe'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('pandas', 'Pandas'),
        ('pygame', 'Pygame')
    ]
    
    all_good = True
    
    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print(f"✓ {display_name} available")
        except ImportError:
            print(f"✗ {display_name} not available")
            all_good = False
    
    return all_good

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("AI-Powered Library Seat Monitoring System - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Camera Access", test_camera),
        ("YOLO Detection", test_detection),
        ("Posture Analysis", test_posture_analysis),
        ("Alert System", test_alert_system),
        ("Visualization", test_visualization),
        ("Data Logging", test_data_logging)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        return True
    else:
        print("⚠ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
