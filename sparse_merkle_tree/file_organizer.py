
import os
import shutil
from pathlib import Path

def organize_files(directory="."):
    """
    Organize files in the specified directory by moving them into
    subdirectories based on their file extensions.
    """
    base_path = Path(directory).resolve()
    
    # Define categories and their associated extensions
    categories = {
        "Images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp"],
        "Documents": [".pdf", ".docx", ".txt", ".xlsx", ".pptx", ".md", ".rtf"],
        "Archives": [".zip", ".tar", ".gz", ".7z", ".rar"],
        "Audio": [".mp3", ".wav", ".flac", ".aac", ".ogg"],
        "Video": [".mp4", ".avi", ".mkv", ".mov", ".wmv"],
        "Code": [".py", ".js", ".html", ".css", ".java", ".cpp", ".c", ".json"],
        "Executables": [".exe", ".msi", ".sh", ".bat", ".app"],
    }
    
    # Create category folders if they don't exist
    for category in categories:
        category_path = base_path / category
        category_path.mkdir(exist_ok=True)
    
    # Track files that don't match any category
    other_files = []
    
    # Iterate over all items in the directory
    for item in base_path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            moved = False
            
            # Find the appropriate category for the file
            for category, extensions in categories.items():
                if file_extension in extensions:
                    target_path = base_path / category / item.name
                    # Handle naming conflicts
                    if target_path.exists():
                        counter = 1
                        while target_path.exists():
                            new_name = f"{item.stem}_{counter}{item.suffix}"
                            target_path = base_path / category / new_name
                            counter += 1
                    
                    try:
                        shutil.move(str(item), str(target_path))
                        print(f"Moved: {item.name} -> {category}/")
                        moved = True
                        break
                    except Exception as e:
                        print(f"Error moving {item.name}: {e}")
            
            # If file doesn't match any category, add to other list
            if not moved:
                other_files.append(item.name)
    
    # Create an "Other" folder for uncategorized files
    if other_files:
        other_path = base_path / "Other"
        other_path.mkdir(exist_ok=True)
        print(f"\n{len(other_files)} files moved to 'Other' folder:")
        for file_name in other_files:
            print(f"  - {file_name}")
    
    print("\nFile organization complete.")

if __name__ == "__main__":
    # You can specify a directory as a command line argument
    import sys
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    organize_files(target_dir)