
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
        'Archives': ['.zip', '.tar', '.gz', '.rar'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp']
    }

    # Ensure the directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    # Create category folders if they don't exist
    for category in file_categories.keys():
        category_path = os.path.join(directory, category)
        if not os.path.exists(category_path):
            os.makedirs(category_path)

    # Track moved files and errors
    moved_files = []
    error_files = []

    # Iterate through all files in the directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        # Skip directories
        if os.path.isdir(item_path):
            continue

        # Get file extension
        file_extension = Path(item).suffix.lower()

        # Find the appropriate category for the file
        target_category = None
        for category, extensions in file_categories.items():
            if file_extension in extensions:
                target_category = category
                break

        # If no category found, move to 'Other'
        if target_category is None:
            target_category = 'Other'
            other_folder = os.path.join(directory, target_category)
            if not os.path.exists(other_folder):
                os.makedirs(other_folder)

        # Construct target path
        target_folder = os.path.join(directory, target_category)
        target_path = os.path.join(target_folder, item)

        # Move the file
        try:
            # Avoid overwriting existing files
            if os.path.exists(target_path):
                base_name = Path(item).stem
                counter = 1
                while os.path.exists(target_path):
                    new_name = f"{base_name}_{counter}{file_extension}"
                    target_path = os.path.join(target_folder, new_name)
                    counter += 1

            shutil.move(item_path, target_path)
            moved_files.append((item, target_category))
        except Exception as e:
            error_files.append((item, str(e)))

    # Print summary
    print(f"\nFile organization complete for: {directory}")
    print(f"Total files moved: {len(moved_files)}")
    
    if moved_files:
        print("\nMoved files:")
        for file_name, category in moved_files:
            print(f"  {file_name} -> {category}/")
    
    if error_files:
        print(f"\nErrors ({len(error_files)} files):")
        for file_name, error_msg in error_files:
            print(f"  {file_name}: {error_msg}")

if __name__ == "__main__":
    # Use current directory if no argument provided
    import sys
    target_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    organize_files(target_dir)