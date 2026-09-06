import sys
import os

# Ensure the generated project directory is in sys.path so tests can import modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
