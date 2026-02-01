import os

def remove_empty_dirs(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for dir_name in dirs:
            full_path = os.path.join(root, dir_name)
            if not os.listdir(full_path):
                os.rmdir(full_path)
                print(f"Removed empty directory: {full_path}")

if __name__ == "__main__":
    target_path = input("Enter directory path to clean: ").strip()
    if os.path.isdir(target_path):
        remove_empty_dirs(target_path)
        print("Cleanup completed.")
    else:
        print("Invalid directory path.")