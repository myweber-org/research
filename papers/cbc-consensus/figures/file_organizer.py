
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organizes files in the specified directory by moving them into
    subfolders based on their file extensions.
    """
    # Define categories and their associated file extensions
    categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Archives': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp'],
        'Audio': ['.mp3', '.wav', '.aac', '.flac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv']
    }

    # Ensure the directory exists
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"Error: Directory '{directory}' does not exist.")
        return

    # Create category folders if they don't exist
    for category in categories:
        category_path = dir_path / category
        category_path.mkdir(exist_ok=True)

    # Track moved files and extensions not categorized
    moved_files = []
    uncategorized_extensions = set()

    # Iterate over all items in the directory
    for item in dir_path.iterdir():
        # Skip directories
        if item.is_dir():
            continue

        # Get file extension
        extension = item.suffix.lower()

        # Find the appropriate category
        target_category = None
        for category, extensions in categories.items():
            if extension in extensions:
                target_category = category
                break

        # Move file to the corresponding category folder
        if target_category:
            target_path = dir_path / target_category / item.name
            # Handle duplicate filenames
            if target_path.exists():
                base_name = item.stem
                counter = 1
                while target_path.exists():
                    new_name = f"{base_name}_{counter}{item.suffix}"
                    target_path = dir_path / target_category / new_name
                    counter += 1

            try:
                shutil.move(str(item), str(target_path))
                moved_files.append((item.name, target_category))
            except Exception as e:
                print(f"Failed to move {item.name}: {e}")
        else:
            uncategorized_extensions.add(extension)

    # Print summary
    print(f"\nOrganization complete for: {directory}")
    print(f"Files moved: {len(moved_files)}")
    if moved_files:
        print("\nMoved files:")
        for filename, category in moved_files:
            print(f"  {filename} -> {category}/")
    
    if uncategorized_extensions:
        print(f"\nUncategorized extensions found: {sorted(uncategorized_extensions)}")

if __name__ == "__main__":
    # Example usage: organize files in the current directory
    target_dir = input("Enter directory path to organize (or press Enter for current directory): ").strip()
    if not target_dir:
        target_dir = os.getcwd()
    
    organize_files(target_dir)