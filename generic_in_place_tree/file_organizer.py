
import os
import shutil

def organize_files(directory_path):
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        return

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        if os.path.isfile(file_path):
            file_extension = filename.split('.')[-1] if '.' in filename else 'no_extension'
            extension_folder = os.path.join(directory_path, file_extension)

            if not os.path.exists(extension_folder):
                os.makedirs(extension_folder)

            destination_path = os.path.join(extension_folder, filename)
            shutil.move(file_path, destination_path)
            print(f"Moved: {filename} -> {extension_folder}/")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ")
    organize_files(target_directory)