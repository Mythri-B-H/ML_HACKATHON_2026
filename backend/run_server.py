"""
PathSnap - Backend Server Runner
Runs the PathSnap REST API and web application server.
Supports both Flask (default, reliable synchronous WSGI) and Uvicorn/Starlette.
"""

import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.src.api.flask_app import app

def main():
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    debug = os.environ.get("DEBUG", "true").lower() == "true"
    
    print("=" * 65)
    print("  PathSnap – Intelligent GPS Route Reconstruction Server")
    print(f"  Access Web UI & REST APIs at: http://{host}:{port}/")
    print("=" * 65)
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    main()
