import os
import sys

# Get the backend directory
backend_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Backend directory: {backend_dir}")

# Check if main.py exists
main_path = os.path.join(backend_dir, "main.py")
print(f"Main.py exists: {os.path.exists(main_path)}")

# Try to import
try:
    sys.path.insert(0, backend_dir)
    from main import app
    print("✅ Import successful!")
    print(f"App title: {app.title}")
except Exception as e:
    print(f"❌ Import failed: {e}") 