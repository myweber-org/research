
import os
import shutil
from pathlib import Path

def organize_files_by_extension(directory_path):
    """
    Organize files in the specified directory by moving them into
    subfolders based on their file extensions.
    """
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    path = Path(directory_path)
    
    extension_categories = {
        'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'audio': ['.mp3', '.wav', '.flac', '.aac'],
        'video': ['.mp4', '.avi', '.mov', '.mkv', '.flv'],
        'archives': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c']
    }

    for item in path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            target_folder = None

            for category, extensions in extension_categories.items():
                if file_extension in extensions:
                    target_folder = category
                    break

            if not target_folder:
                target_folder = 'others'

            target_path = path / target_folder
            target_path.mkdir(exist_ok=True)

            try:
                shutil.move(str(item), str(target_path / item.name))
                print(f"Moved: {item.name} -> {target_folder}/")
            except Exception as e:
                print(f"Error moving {item.name}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files_by_extension(target_directory)