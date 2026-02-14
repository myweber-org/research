
import os
import shutil

def organize_files(directory):
    """
    Organize files in the given directory by moving them into folders
    named after their file extensions.
    """
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            file_ext = filename.split('.')[-1] if '.' in filename else 'no_extension'
            folder_name = file_ext.upper() + "_FILES"
            folder_path = os.path.join(directory, folder_name)

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            try:
                shutil.move(file_path, os.path.join(folder_path, filename))
                print(f"Moved: {filename} -> {folder_name}/")
            except Exception as e:
                print(f"Error moving {filename}: {e}")

if __name__ == "__main__":
    target_dir = input("Enter the directory path to organize: ").strip()
    organize_files(target_dir)
import os
import shutil
from pathlib import Path

def organize_files_by_extension(directory_path):
    """
    Organizes files in the given directory by moving them into
    subfolders named after their file extensions.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: The path '{directory_path}' is not a valid directory.")
        return

    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)

        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()

            if not file_extension:
                folder_name = "no_extension"
            else:
                folder_name = file_extension[1:]

            target_folder = os.path.join(directory_path, folder_name)
            os.makedirs(target_folder, exist_ok=True)

            try:
                shutil.move(item_path, os.path.join(target_folder, item))
                print(f"Moved: {item} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files_by_extension(target_directory)