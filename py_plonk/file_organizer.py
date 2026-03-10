
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
import os
import shutil
from pathlib import Path

def organize_files(directory_path):
    """
    Organizes files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    # Define file type categories and their associated extensions
    file_categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Audio': ['.mp3', '.wav', '.aac', '.flac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv'],
        'Archives': ['.zip', '.tar', '.gz', '.rar'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp']
    }

    # Convert to Path object for easier handling
    base_path = Path(directory_path)

    # Check if the directory exists
    if not base_path.exists() or not base_path.is_dir():
        print(f"Error: The directory '{directory_path}' does not exist or is not a directory.")
        return

    # Iterate over all items in the directory
    for item in base_path.iterdir():
        # Skip if it's a directory
        if item.is_dir():
            continue

        # Get the file extension
        file_extension = item.suffix.lower()

        # Determine the target category
        target_category = None
        for category, extensions in file_categories.items():
            if file_extension in extensions:
                target_category = category
                break

        # If no category matches, put it in 'Others'
        if target_category is None:
            target_category = 'Others'

        # Create the target directory if it doesn't exist
        target_dir = base_path / target_category
        target_dir.mkdir(exist_ok=True)

        # Construct the target path
        target_path = target_dir / item.name

        # Check if a file with the same name already exists in the target
        counter = 1
        while target_path.exists():
            # Append a number to the filename to avoid overwriting
            stem = item.stem
            new_name = f"{stem}_{counter}{item.suffix}"
            target_path = target_dir / new_name
            counter += 1

        # Move the file
        try:
            shutil.move(str(item), str(target_path))
            print(f"Moved: {item.name} -> {target_category}/")
        except Exception as e:
            print(f"Failed to move {item.name}: {e}")

    print("File organization complete.")

if __name__ == "__main__":
    # Example usage: organize files in the current directory
    current_directory = os.getcwd()
    organize_files(current_directory)