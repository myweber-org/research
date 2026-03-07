
import os
import shutil
from pathlib import Path

def organize_files_by_extension(directory_path):
    """
    Organizes files in the specified directory by moving them into
    subfolders named after their file extensions.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: The path '{directory_path}' is not a valid directory.")
        return

    base_path = Path(directory_path)

    for item in base_path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            if not file_extension:
                file_extension = "no_extension"

            target_folder_name = file_extension[1:] if file_extension.startswith('.') else file_extension
            target_folder = base_path / target_folder_name

            target_folder.mkdir(exist_ok=True)

            try:
                shutil.move(str(item), str(target_folder / item.name))
                print(f"Moved: {item.name} -> {target_folder_name}/")
            except Exception as e:
                print(f"Failed to move {item.name}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files_by_extension(target_directory)
import os
import shutil

def organize_files(directory):
    """
    Organize files in the specified directory by moving them into
    subdirectories named after their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            file_ext = os.path.splitext(filename)[1].lower()

            if file_ext:
                target_dir = os.path.join(directory, file_ext[1:])
            else:
                target_dir = os.path.join(directory, "no_extension")

            os.makedirs(target_dir, exist_ok=True)

            try:
                shutil.move(file_path, os.path.join(target_dir, filename))
                print(f"Moved: {filename} -> {target_dir}")
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
    Organizes files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    # Define categories and their associated file extensions
    categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv'],
        'Archives': ['.zip', '.tar', '.gz', '.rar'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp']
    }

    # Ensure the directory exists
    target_dir = Path(directory)
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"Error: Directory '{directory}' does not exist or is not a directory.")
        return

    # Create category folders if they don't exist
    for category in categories:
        category_path = target_dir / category
        category_path.mkdir(exist_ok=True)

    # Track moved files and errors
    moved_files = []
    errors = []

    # Iterate over all items in the directory
    for item in target_dir.iterdir():
        # Skip directories
        if item.is_dir():
            continue

        # Get file extension
        file_extension = item.suffix.lower()

        # Find the appropriate category
        target_category = None
        for category, extensions in categories.items():
            if file_extension in extensions:
                target_category = category
                break

        # If no category found, place in 'Other'
        if target_category is None:
            target_category = 'Other'
            other_path = target_dir / target_category
            other_path.mkdir(exist_ok=True)

        # Construct destination path
        destination = target_dir / target_category / item.name

        # Move the file
        try:
            # Handle name conflicts by adding a number suffix
            counter = 1
            original_destination = destination
            while destination.exists():
                stem = original_destination.stem
                suffix = original_destination.suffix
                destination = original_destination.parent / f"{stem}_{counter}{suffix}"
                counter += 1

            shutil.move(str(item), str(destination))
            moved_files.append((item.name, target_category))
        except Exception as e:
            errors.append((item.name, str(e)))

    # Print summary
    print(f"\nOrganization complete for '{directory}'")
    print(f"Moved {len(moved_files)} files:")
    for filename, category in moved_files:
        print(f"  {filename} -> {category}/")

    if errors:
        print(f"\nEncountered {len(errors)} errors:")
        for filename, error_msg in errors:
            print(f"  {filename}: {error_msg}")

if __name__ == "__main__":
    # Example usage: organize the current directory
    current_directory = os.getcwd()
    organize_files(current_directory)