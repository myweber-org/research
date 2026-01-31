
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organizes files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    # Define file type categories
    categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Audio': ['.mp3', '.wav', '.aac', '.flac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv'],
        'Archives': ['.zip', '.tar', '.gz', '.rar'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp']
    }
    
    # Ensure the directory exists
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"Directory '{directory}' does not exist.")
        return
    
    # Create category folders if they don't exist
    for category in categories:
        category_path = dir_path / category
        category_path.mkdir(exist_ok=True)
    
    # Track moved files
    moved_files = []
    
    # Iterate over files in the directory
    for item in dir_path.iterdir():
        if item.is_file():
            file_ext = item.suffix.lower()
            moved = False
            
            # Find the appropriate category
            for category, extensions in categories.items():
                if file_ext in extensions:
                    dest_dir = dir_path / category
                    try:
                        shutil.move(str(item), str(dest_dir / item.name))
                        moved_files.append((item.name, category))
                        moved = True
                        break
                    except Exception as e:
                        print(f"Error moving {item.name}: {e}")
            
            # If file doesn't match any category, move to 'Other'
            if not moved:
                other_dir = dir_path / 'Other'
                other_dir.mkdir(exist_ok=True)
                try:
                    shutil.move(str(item), str(other_dir / item.name))
                    moved_files.append((item.name, 'Other'))
                except Exception as e:
                    print(f"Error moving {item.name}: {e}")
    
    # Print summary
    if moved_files:
        print(f"Organized {len(moved_files)} files:")
        for filename, category in moved_files:
            print(f"  {filename} -> {category}/")
    else:
        print("No files were moved.")

if __name__ == "__main__":
    # Get directory from user or use current directory
    target_dir = input("Enter directory path (or press Enter for current directory): ").strip()
    if not target_dir:
        target_dir = os.getcwd()
    
    organize_files(target_dir)