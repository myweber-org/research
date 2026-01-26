
import os
import shutil
from pathlib import Path

def organize_files(directory="."):
    """
    Organizes files in the specified directory by moving them into
    subdirectories named after their file extensions.
    """
    base_path = Path(directory).resolve()

    if not base_path.is_dir():
        print(f"Error: '{directory}' is not a valid directory.")
        return

    for item in base_path.iterdir():
        if item.is_file():
            ext = item.suffix.lower()
            if ext:
                folder_name = ext[1:] if ext.startswith('.') else ext
            else:
                folder_name = "no_extension"

            target_dir = base_path / folder_name
            target_dir.mkdir(exist_ok=True)

            try:
                shutil.move(str(item), str(target_dir / item.name))
                print(f"Moved: {item.name} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {item.name}: {e}")

if __name__ == "__main__":
    organize_files()