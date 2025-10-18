"""
Setup script for the Library Monitoring System
"""
import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False

def download_yolo_model():
    """Download YOLO model if not present"""
    print("Checking for YOLO model...")
    
    model_path = "yolov8n.pt"
    if os.path.exists(model_path):
        print("✓ YOLO model already exists")
        return True
    
    try:
        from ultralytics import YOLO
        print("Downloading YOLO model...")
        model = YOLO('yolov8n.pt')  # This will download the model
        print("✓ YOLO model downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to download YOLO model: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("Creating directories...")
    
    directories = ['logs', 'exports', 'screenshots', 'heatmaps']
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created directory: {directory}")
        except Exception as e:
            print(f"✗ Failed to create directory {directory}: {e}")
            return False
    
    return True

def setup_database():
    """Initialize database"""
    print("Setting up database...")
    
    try:
        from data_logging import DataLogger
        logger = DataLogger()
        print("✓ Database initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to setup database: {e}")
        return False

def main():
    """Main setup function"""
    print("=" * 60)
    print("AI-Powered Library Seat Monitoring System - Setup")
    print("=" * 60)
    
    steps = [
        ("Installing Requirements", install_requirements),
        ("Creating Directories", create_directories),
        ("Downloading YOLO Model", download_yolo_model),
        ("Setting up Database", setup_database)
    ]
    
    success = True
    
    for step_name, step_func in steps:
        print(f"\n{step_name}:")
        print("-" * 40)
        if not step_func():
            success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Setup completed successfully!")
        print("\nTo run the system:")
        print("  python main.py")
        print("\nTo test the system:")
        print("  python test_system.py")
    else:
        print("⚠ Setup completed with errors. Please check the issues above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
