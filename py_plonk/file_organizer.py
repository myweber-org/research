
import os
import shutil
from pathlib import Path

def organize_files_by_extension(directory_path):
    """
    Organize files in the specified directory by moving them into
    subfolders based on their file extensions.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a valid directory.")
        return

    base_path = Path(directory_path)
    file_mappings = {}

    for item in base_path.iterdir():
        if item.is_file():
            extension = item.suffix.lower()
            if extension:
                folder_name = extension[1:] + "_files"
            else:
                folder_name = "no_extension_files"

            target_folder = base_path / folder_name
            target_folder.mkdir(exist_ok=True)

            try:
                shutil.move(str(item), str(target_folder / item.name))
                file_mappings.setdefault(folder_name, []).append(item.name)
            except Exception as e:
                print(f"Failed to move {item.name}: {e}")

    for folder, files in file_mappings.items():
        print(f"Moved {len(files)} file(s) to '{folder}'")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files_by_extension(target_directory)