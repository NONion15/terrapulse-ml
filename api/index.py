import sys
from pathlib import Path

# Ensure project root is in Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.server import app

# Vercel serverless entrypoint
app = app

if __name__ == "__main__":
    app.run(port=8000)
