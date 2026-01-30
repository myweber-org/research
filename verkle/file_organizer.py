
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the specified directory by moving them into
    subfolders named after their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()

            if file_extension:
                folder_name = file_extension[1:] + "_files"
            else:
                folder_name = "no_extension_files"

            target_folder = os.path.join(directory, folder_name)
            os.makedirs(target_folder, exist_ok=True)

            try:
                shutil.move(item_path, os.path.join(target_folder, item))
                print(f"Moved: {item} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(directory_path):
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        return

    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()
            if file_extension:
                folder_name = file_extension[1:] + "_files"
            else:
                folder_name = "no_extension_files"

            target_folder = os.path.join(directory_path, folder_name)
            os.makedirs(target_folder, exist_ok=True)

            try:
                shutil.move(item_path, os.path.join(target_folder, item))
                print(f"Moved: {item} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil

def organize_files(directory):
    """
    Organize files in the specified directory into subfolders based on their extensions.
    """
    # Define file type categories and their associated extensions
    file_categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv'],
        'Archives': ['.zip', '.tar', '.gz', '.rar'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp']
    }

    # Ensure the directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    # Get all files in the directory
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    except PermissionError:
        print(f"Error: Permission denied for directory '{directory}'.")
        return

    moved_count = 0

    for filename in files:
        file_ext = os.path.splitext(filename)[1].lower()

        # Find the category for the file extension
        target_category = None
        for category, extensions in file_categories.items():
            if file_ext in extensions:
                target_category = category
                break

        # If no category matched, put in 'Other'
        if target_category is None:
            target_category = 'Other'

        # Create target directory if it doesn't exist
        target_dir = os.path.join(directory, target_category)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # Move the file
        source_path = os.path.join(directory, filename)
        target_path = os.path.join(target_dir, filename)

        try:
            shutil.move(source_path, target_path)
            moved_count += 1
            print(f"Moved: {filename} -> {target_category}/")
        except Exception as e:
            print(f"Failed to move {filename}: {e}")

    print(f"\nOrganization complete. Moved {moved_count} file(s).")

if __name__ == "__main__":
    # You can change this path to the directory you want to organize
    target_directory = input("Enter the directory path to organize (or press Enter for current directory): ").strip()
    
    if not target_directory:
        target_directory = os.getcwd()
    
    organize_files(target_directory)