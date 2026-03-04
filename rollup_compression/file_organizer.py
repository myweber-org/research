
import os
import shutil
from pathlib import Path

def organize_files(directory_path):
    """
    Organize files in the given directory by moving them into folders
    named after their file extensions.
    """
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        
        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()
            
            if file_extension:
                folder_name = file_extension[1:] + "_files"
            else:
                folder_name = "no_extension_files"
            
            target_folder = os.path.join(directory_path, folder_name)
            os.makedirs(target_folder, exist_ok=True)
            
            target_path = os.path.join(target_folder, item)
            shutil.move(item_path, target_path)
            print(f"Moved: {item} -> {folder_name}/")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(directory="."):
    """
    Organizes files in the specified directory by moving them into
    subdirectories named after their file extensions.
    """
    base_path = Path(directory)
    
    # Define categories and their associated extensions
    categories = {
        "images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg"],
        "documents": [".pdf", ".docx", ".txt", ".xlsx", ".pptx", ".md"],
        "audio": [".mp3", ".wav", ".flac", ".aac"],
        "video": [".mp4", ".avi", ".mkv", ".mov"],
        "archives": [".zip", ".tar", ".gz", ".rar", ".7z"],
        "code": [".py", ".js", ".html", ".css", ".java", ".cpp", ".c"],
    }
    
    # Create category folders if they don't exist
    for category in categories:
        (base_path / category).mkdir(exist_ok=True)
    
    # Create an 'others' folder for uncategorized files
    others_folder = base_path / "others"
    others_folder.mkdir(exist_ok=True)
    
    # Iterate over files in the directory
    for item in base_path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            moved = False
            
            # Find the appropriate category
            for category, extensions in categories.items():
                if file_extension in extensions:
                    target_folder = base_path / category
                    try:
                        shutil.move(str(item), str(target_folder / item.name))
                        print(f"Moved: {item.name} -> {category}/")
                        moved = True
                        break
                    except Exception as e:
                        print(f"Error moving {item.name}: {e}")
            
            # If file doesn't match any category, move to 'others'
            if not moved:
                try:
                    shutil.move(str(item), str(others_folder / item.name))
                    print(f"Moved: {item.name} -> others/")
                except Exception as e:
                    print(f"Error moving {item.name}: {e}")
    
    print("File organization complete.")

if __name__ == "__main__":
    organize_files()