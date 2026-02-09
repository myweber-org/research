
import os
import shutil
from pathlib import Path

def organize_files(directory_path):
    """
    Organizes files in the specified directory by moving them into
    subfolders based on their file extensions.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: The path '{directory_path}' is not a valid directory.")
        return

    # Define categories and their associated file extensions
    categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Archives': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac'],
        'Video': ['.mp4', '.avi', '.mkv', '.mov', '.wmv'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c'],
        'Executables': ['.exe', '.msi', '.bat', '.sh'],
        'Others': []  # For files with extensions not in the above categories
    }

    # Create a reverse lookup dictionary: extension -> category
    extension_to_category = {}
    for category, extensions in categories.items():
        for ext in extensions:
            extension_to_category[ext.lower()] = category

    # Iterate over all items in the directory
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)

        # Skip if it's a directory
        if os.path.isdir(item_path):
            continue

        # Get the file extension
        file_extension = Path(item).suffix.lower()

        # Determine the category
        category = extension_to_category.get(file_extension, 'Others')

        # Create the category folder if it doesn't exist
        category_folder = os.path.join(directory_path, category)
        os.makedirs(category_folder, exist_ok=True)

        # Move the file to the category folder
        try:
            shutil.move(item_path, os.path.join(category_folder, item))
            print(f"Moved: {item} -> {category}/")
        except Exception as e:
            print(f"Failed to move {item}: {e}")

    print("File organization complete.")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)