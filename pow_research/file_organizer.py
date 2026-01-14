
import os
import shutil

def organize_files(directory):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if os.path.isfile(file_path):
            file_extension = filename.split('.')[-1] if '.' in filename else 'no_extension'
            target_dir = os.path.join(directory, file_extension)
            
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            shutil.move(file_path, os.path.join(target_dir, filename))
            print(f"Moved {filename} to {file_extension}/")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(directory="."):
    """Organize files in the specified directory by their extensions."""
    base_path = Path(directory).resolve()
    
    if not base_path.exists():
        print(f"Directory '{base_path}' does not exist.")
        return
    
    if not base_path.is_dir():
        print(f"'{base_path}' is not a directory.")
        return

    categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Archives': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'Code': ['.py', '.js', '.html', '.css', '.json', '.xml'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac'],
        'Video': ['.mp4', '.avi', '.mkv', '.mov']
    }

    for item in base_path.iterdir():
        if item.is_file():
            file_ext = item.suffix.lower()
            moved = False
            
            for category, extensions in categories.items():
                if file_ext in extensions:
                    target_dir = base_path / category
                    target_dir.mkdir(exist_ok=True)
                    
                    try:
                        shutil.move(str(item), str(target_dir / item.name))
                        print(f"Moved: {item.name} -> {category}/")
                        moved = True
                        break
                    except Exception as e:
                        print(f"Error moving {item.name}: {e}")
            
            if not moved:
                other_dir = base_path / 'Other'
                other_dir.mkdir(exist_ok=True)
                try:
                    shutil.move(str(item), str(other_dir / item.name))
                    print(f"Moved: {item.name} -> Other/")
                except Exception as e:
                    print(f"Error moving {item.name}: {e}")

    print("File organization completed.")

if __name__ == "__main__":
    organize_files()