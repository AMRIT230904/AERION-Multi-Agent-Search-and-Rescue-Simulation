"""
Root conftest.py

Its only job is to make pytest add the project root to sys.path so that
`tasks/` (and the root-level modules like coordinator.py) are importable
from anywhere under tests/, without needing to `pip install -e .`.
"""
