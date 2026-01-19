
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the given directory by moving them into
    subfolders based on their file extensions.
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
    target_dir = Path(directory)
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    # Create category folders if they don't exist
    for category in file_categories.keys():
        category_path = target_dir / category
        category_path.mkdir(exist_ok=True)
    
    # Track moved files
    moved_files = []
    skipped_files = []
    
    # Iterate through files in the directory
    for item in target_dir.iterdir():
        if item.is_file():
            file_ext = item.suffix.lower()
            moved = False
            
            # Find the appropriate category for the file
            for category, extensions in file_categories.items():
                if file_ext in extensions:
                    dest_folder = target_dir / category
                    dest_path = dest_folder / item.name
                    
                    # Handle duplicate filenames
                    counter = 1
                    while dest_path.exists():
                        name_parts = item.stem, item.suffix
                        new_name = f"{name_parts[0]}_{counter}{name_parts[1]}"
                        dest_path = dest_folder / new_name
                        counter += 1
                    
                    # Move the file
                    try:
                        shutil.move(str(item), str(dest_path))
                        moved_files.append((item.name, category))
                        moved = True
                        break
                    except Exception as e:
                        print(f"Error moving {item.name}: {e}")
                        skipped_files.append(item.name)
            
            # If file doesn't match any category, move to 'Other'
            if not moved:
                other_folder = target_dir / 'Other'
                other_folder.mkdir(exist_ok=True)
                dest_path = other_folder / item.name
                
                # Handle duplicate filenames in Other
                counter = 1
                while dest_path.exists():
                    name_parts = item.stem, item.suffix
                    new_name = f"{name_parts[0]}_{counter}{name_parts[1]}"
                    dest_path = other_folder / new_name
                    counter += 1
                
                try:
                    shutil.move(str(item), str(dest_path))
                    moved_files.append((item.name, 'Other'))
                except Exception as e:
                    print(f"Error moving {item.name}: {e}")
                    skipped_files.append(item.name)
    
    # Print summary
    print(f"\nOrganization complete!")
    print(f"Total files processed: {len(moved_files) + len(skipped_files)}")
    print(f"Files moved: {len(moved_files)}")
    print(f"Files skipped: {len(skipped_files)}")
    
    if moved_files:
        print("\nMoved files by category:")
        category_counts = {}
        for _, category in moved_files:
            category_counts[category] = category_counts.get(category, 0) + 1
        
        for category, count in sorted(category_counts.items()):
            print(f"  {category}: {count} file(s)")
    
    if skipped_files:
        print(f"\nSkipped files (could not move):")
        for filename in skipped_files:
            print(f"  {filename}")

if __name__ == "__main__":
    # Example usage: organize files in the current directory
    current_directory = os.getcwd()
    organize_files(current_directory)