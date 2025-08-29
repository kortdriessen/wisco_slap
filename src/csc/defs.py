import yaml

from pathlib import Path

# Load sub_data_root from DEFS.yaml (resolve path relative to this file)
defs_path = Path(__file__).resolve().parent / 'DEFS.yaml'
with defs_path.open('r') as f:
    DEFS = yaml.safe_load(f)