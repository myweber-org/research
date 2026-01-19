
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organizes files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    # Define file type categories and their associated extensions
    file_categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv'],
        'Archives': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c']
    }

    # Convert directory to Path object for easier handling
    base_path = Path(directory)

    # Ensure the directory exists
    if not base_path.exists() or not base_path.is_dir():
        print(f"Error: Directory '{directory}' does not exist or is not a directory.")
        return

    # Iterate through all items in the directory
    for item in base_path.iterdir():
        # Skip if it's a directory
        if item.is_dir():
            continue

        # Get the file extension
        file_extension = item.suffix.lower()

        # Find the category for this file extension
        target_category = None
        for category, extensions in file_categories.items():
            if file_extension in extensions:
                target_category = category
                break

        # If no category found, use 'Other'
        if target_category is None:
            target_category = 'Other'

        # Create target directory if it doesn't exist
        target_dir = base_path / target_category
        target_dir.mkdir(exist_ok=True)

        # Move the file to the target directory
        try:
            shutil.move(str(item), str(target_dir / item.name))
            print(f"Moved: {item.name} -> {target_category}/")
        except Exception as e:
            print(f"Failed to move {item.name}: {e}")

if __name__ == "__main__":
    # Get the current working directory as the target
    target_directory = os.getcwd()
    print(f"Organizing files in: {target_directory}")
    organize_files(target_directory)
    print("File organization complete.")