
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
        'Audio': ['.mp3', '.wav', '.aac', '.flac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv'],
        'Archives': ['.zip', '.rar', '.tar', '.gz'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp'],
        'Executables': ['.exe', '.msi', '.sh', '.bat']
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
                        print(f"Moved: {item} -> {folder}/")
                        moved = True
                        break
                    except Exception as e:
                        print(f"Error moving {item}: {e}")

            if not moved:
                other_folder = os.path.join(directory, 'Other')
                if not os.path.exists(other_folder):
                    os.makedirs(other_folder)
                try:
                    shutil.move(item_path, other_folder)
                    print(f"Moved: {item} -> Other/")
                except Exception as e:
                    print(f"Error moving {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)