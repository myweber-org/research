
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()
            
            if file_extension:
                folder_name = file_extension[1:] + "_files"
            else:
                folder_name = "no_extension_files"
            
            target_folder = os.path.join(directory, folder_name)
            os.makedirs(target_folder, exist_ok=True)
            
            try:
                shutil.move(item_path, os.path.join(target_folder, item))
                print(f"Moved: {item} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)
    print("File organization completed.")
import os
import shutil
from pathlib import Path

def organize_files(directory):
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return
    
    extensions_folders = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv'],
        'Archives': ['.zip', '.rar', '.7z', '.tar.gz'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp']
    }
    
    for folder in extensions_folders.keys():
        folder_path = os.path.join(directory, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()
            moved = False
            
            for folder, extensions in extensions_folders.items():
                if file_extension in extensions:
                    dest_folder = os.path.join(directory, folder)
                    try:
                        shutil.move(item_path, dest_folder)
                        print(f"Moved '{item}' to '{folder}'")
                        moved = True
                        break
                    except Exception as e:
                        print(f"Error moving '{item}': {e}")
            
            if not moved:
                other_folder = os.path.join(directory, 'Other')
                if not os.path.exists(other_folder):
                    os.makedirs(other_folder)
                try:
                    shutil.move(item_path, other_folder)
                    print(f"Moved '{item}' to 'Other'")
                except Exception as e:
                    print(f"Error moving '{item}': {e}")
    
    print("File organization completed.")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)