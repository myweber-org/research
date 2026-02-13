
import os
import shutil
from pathlib import Path

def organize_files_by_extension(directory_path):
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
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
                print(f"Moved '{item}' to '{folder_name}/'")
            except Exception as e:
                print(f"Error moving '{item}': {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files_by_extension(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organizes files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    # Define categories and their associated file extensions
    categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv'],
        'Archives': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.json']
    }

    # Ensure the directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    # Create category folders if they don't exist
    for category in categories.keys():
        category_path = os.path.join(directory, category)
        os.makedirs(category_path, exist_ok=True)

    # Track moved files and errors
    moved_files = []
    error_files = []

    # Iterate over all items in the directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        # Skip directories
        if os.path.isdir(item_path):
            continue

        # Get file extension
        file_extension = Path(item).suffix.lower()

        # Find the appropriate category
        target_category = 'Other'  # Default category
        for category, extensions in categories.items():
            if file_extension in extensions:
                target_category = category
                break

        # Define target path
        target_folder = os.path.join(directory, target_category)
        target_path = os.path.join(target_folder, item)

        # Move the file
        try:
            shutil.move(item_path, target_path)
            moved_files.append((item, target_category))
            print(f"Moved: {item} -> {target_category}/")
        except Exception as e:
            error_files.append((item, str(e)))
            print(f"Error moving {item}: {e}")

    # Print summary
    print(f"\nOrganization complete!")
    print(f"Files moved: {len(moved_files)}")
    print(f"Errors: {len(error_files)}")

    if moved_files:
        print("\nMoved files summary:")
        for filename, category in moved_files:
            print(f"  {category}: {filename}")

if __name__ == "__main__":
    # Get directory from user input or use current directory
    target_dir = input("Enter directory path to organize (or press Enter for current directory): ").strip()
    if not target_dir:
        target_dir = os.getcwd()

    organize_files(target_dir)