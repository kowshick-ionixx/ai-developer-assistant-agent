import sys
import os

# Ensure the generated project root is on sys.path so tests can import local modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
