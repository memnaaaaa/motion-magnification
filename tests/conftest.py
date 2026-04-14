"""
tests/conftest.py
-----------------
Pytest configuration: ensure the project root is on sys.path so that
``from src.X import Y`` imports work when running ``python -m pytest tests/``
from the project root.
"""

import sys
from pathlib import Path

# Project root is the parent of the tests/ directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
