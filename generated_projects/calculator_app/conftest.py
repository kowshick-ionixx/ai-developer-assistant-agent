import sys
from pathlib import Path

# Add project root to sys.path so tests can import calculator module directly
sys.path.insert(0, str(Path(__file__).parent))
