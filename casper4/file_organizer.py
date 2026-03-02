
import os
import shutil
from pathlib import Path

def organize_files(directory="."):
    """
    Organizes files in the specified directory by moving them into
    subdirectories named after their file extensions.
    """
    base_path = Path(directory).resolve()

    if not base_path.is_dir():
        print(f"Error: '{directory}' is not a valid directory.")
        return

    for item in base_path.iterdir():
        if item.is_file():
            ext = item.suffix.lower()
            if ext:
                folder_name = ext[1:] if ext.startswith('.') else ext
            else:
                folder_name = "no_extension"

            target_dir = base_path / folder_name
            target_dir.mkdir(exist_ok=True)

            try:
                shutil.move(str(item), str(target_dir / item.name))
                print(f"Moved: {item.name} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {item.name}: {e}")

if __name__ == "__main__":
    organize_files()
import os
import shutil
from pathlib import Path

def organize_files_by_extension(directory_path):
    """
    Organize files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return
    
    path = Path(directory_path)
    
    # Define categories and their associated extensions
    categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv'],
        'Archives': ['.zip', '.tar', '.gz', '.rar'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp']
    }
    
    # Create category folders if they don't exist
    for category in categories.keys():
        category_path = path / category
        category_path.mkdir(exist_ok=True)
    
    # Track moved files
    moved_files = []
    
    # Iterate through files in the directory
    for item in path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            
            # Find the appropriate category for the file
            target_category = None
            for category, extensions in categories.items():
                if file_extension in extensions:
                    target_category = category
                    break
            
            # If no category matches, move to 'Other'
            if target_category is None:
                target_category = 'Other'
                other_path = path / target_category
                other_path.mkdir(exist_ok=True)
            
            # Move the file
            target_path = path / target_category / item.name
            try:
                shutil.move(str(item), str(target_path))
                moved_files.append((item.name, target_category))
                print(f"Moved: {item.name} -> {target_category}/")
            except Exception as e:
                print(f"Error moving {item.name}: {e}")
    
    # Print summary
    print(f"\nOrganization complete. Moved {len(moved_files)} files.")
    
    if moved_files:
        print("\nSummary of moved files:")
        for filename, category in moved_files:
            print(f"  {filename} -> {category}")

if __name__ == "__main__":
    # Example usage: organize files in the current directory
    current_directory = os.getcwd()
    organize_files_by_extension(current_directory)