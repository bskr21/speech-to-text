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
print('Folders successfully created')

for folder in ["data/input", "data/output", "data/processed", "data/cache"]:
    Path(f"{folder}/.gitkeep").touch(exist_ok=True)
print('.gitkeep files created in folder data')

Path("configs/config.yaml").touch(exist_ok=True)
Path("configs/samples/cpu_offline_id_en.yaml").touch(exist_ok=True)
print('All Directories successsfully created')




