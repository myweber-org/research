
import os
import shutil
from pathlib import Path

def organize_files(directory_path):
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return

    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()
            if file_extension:
                target_folder = os.path.join(directory_path, file_extension[1:])
            else:
                target_folder = os.path.join(directory_path, "no_extension")
            
            os.makedirs(target_folder, exist_ok=True)
            shutil.move(item_path, os.path.join(target_folder, item))
            print(f"Moved {item} to {target_folder}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)