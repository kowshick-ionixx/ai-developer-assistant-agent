"""Pytest configuration for calculator_app to allow importing local modules."""

import sys
from pathlib import Path

# Add the calculator_app root directory to sys.path so tests can import calculator
sys.path.insert(0, str(Path(__file__).parent))
