
import os
import shutil

def organize_files(directory):
    """
    Organize files in the specified directory by moving them into
    subdirectories based on their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            _, extension = os.path.splitext(filename)
            extension = extension.lower()

            if extension:
                target_dir = os.path.join(directory, extension[1:] + "_files")
            else:
                target_dir = os.path.join(directory, "no_extension_files")

            os.makedirs(target_dir, exist_ok=True)
            shutil.move(file_path, os.path.join(target_dir, filename))
            print(f"Moved: {filename} -> {target_dir}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)