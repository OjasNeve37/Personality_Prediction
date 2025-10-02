import uvicorn
import os
import sys
from pathlib import Path

def start_server():
    """
    Adds the project root to the Python path and starts the Uvicorn server.
    """
    # Add project root to the Python path
    project_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_root))
    
    print(f"Project root added to path: {project_root}")
    print("Starting FastAPI server...")
    
    # Run the server
    # The app is located at 'web.backend.app.main' and the app instance is named 'app'
    uvicorn.run(
        "web.backend.app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        # Add the backend directory to the reload dirs to watch for changes
        reload_dirs=[str(project_root / "web" / "backend")]
    )

if __name__ == "__main__":
    start_server()
