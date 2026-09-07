import os
import sys

# Ensure the generated project root is on sys.path so tests can import modules directly
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
