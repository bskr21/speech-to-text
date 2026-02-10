# Setup some folders for Speech To Text Project

from pathlib import Path

THIS_FOLDER = Path(__file__).parent

folders = [
    "data/input",
    "data/output",
    "data/processed",
    "data/cache",
    "configs/samples",
]

# Create folder based on a variable folders
for folder in folders:
    (THIS_FOLDER / folder).mkdir(parents=True, exist_ok=True)


