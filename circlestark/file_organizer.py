
import os
import shutil
from pathlib import Path

def organize_files(directory="."):
    """
    Organizes files in the specified directory by moving them into
    subdirectories based on their file extensions.
    """
    base_path = Path(directory).resolve()

    # Define categories and their associated file extensions
    categories = {
        "Images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".tiff"],
        "Documents": [".pdf", ".docx", ".txt", ".xlsx", ".pptx", ".md", ".rtf"],
        "Audio": [".mp3", ".wav", ".flac", ".aac", ".ogg"],
        "Video": [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"],
        "Archives": [".zip", ".rar", ".7z", ".tar", ".gz"],
        "Code": [".py", ".js", ".html", ".css", ".java", ".cpp", ".c", ".json", ".xml"],
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

    # Create an "Other" folder for uncategorized files if any exist
    if other_files:
        other_path = base_path / "Other"
        other_path.mkdir(exist_ok=True)
        for file_name in other_files:
            file_path = base_path / file_name
            target_path = other_path / file_name
            try:
                shutil.move(str(file_path), str(target_path))
                print(f"Moved: {file_name} -> Other/")
            except Exception as e:
                print(f"Error moving {file_name} to Other/: {e}")

    print("\nFile organization complete.")

if __name__ == "__main__":
    # Example: organize files in the current directory
    # You can pass a different path as an argument, e.g., organize_files("/path/to/folder")
    organize_files()