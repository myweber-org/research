
import os
import shutil

def organize_files(directory):
    """
    Organize files in the given directory by moving them into folders
    named after their file extensions.
    """
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            file_ext = filename.split('.')[-1] if '.' in filename else 'no_extension'
            folder_name = file_ext.upper() + "_FILES"
            folder_path = os.path.join(directory, folder_name)

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            try:
                shutil.move(file_path, os.path.join(folder_path, filename))
                print(f"Moved: {filename} -> {folder_name}/")
            except Exception as e:
                print(f"Error moving {filename}: {e}")

if __name__ == "__main__":
    target_dir = input("Enter the directory path to organize: ").strip()
    organize_files(target_dir)