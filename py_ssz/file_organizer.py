
import os
import shutil

def organize_files(directory):
    """
    Organize files in the given directory by moving them into folders
    based on their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            _, extension = os.path.splitext(filename)
            extension = extension.lower()

            if extension:
                folder_name = extension[1:] + "_files"
            else:
                folder_name = "no_extension_files"

            target_folder = os.path.join(directory, folder_name)

            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            try:
                shutil.move(file_path, os.path.join(target_folder, filename))
                print(f"Moved: {filename} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {filename}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    # Define categories and their associated file extensions
    categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Archives': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv']
    }

    # Create category folders if they don't exist
    for category in categories:
        category_path = os.path.join(directory, category)
        os.makedirs(category_path, exist_ok=True)

    # Iterate over files in the directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        # Skip directories
        if os.path.isdir(item_path):
            continue

        # Get file extension
        file_extension = Path(item).suffix.lower()

        # Determine the category for the file
        target_category = 'Other'
        for category, extensions in categories.items():
            if file_extension in extensions:
                target_category = category
                break

        # Move file to the appropriate category folder
        target_folder = os.path.join(directory, target_category)
        target_path = os.path.join(target_folder, item)

        try:
            shutil.move(item_path, target_path)
            print(f"Moved: {item} -> {target_category}/")
        except Exception as e:
            print(f"Error moving {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)