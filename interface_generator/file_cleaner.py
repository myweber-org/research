import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List

class TempFileCleaner:
    def __init__(self, target_dir: Optional[str] = None):
        self.target_dir = Path(target_dir) if target_dir else Path(tempfile.gettempdir())
        self.removed_files = []
        self.removed_dirs = []

    def scan_and_clean(self, patterns: Optional[List[str]] = None, days_old: int = 7) -> dict:
        if patterns is None:
            patterns = ['*.tmp', '*.temp', '*.log', 'cache*']

        current_time = os.path.getctime(self.target_dir)
        cutoff_time = current_time - (days_old * 86400)

        for pattern in patterns:
            for file_path in self.target_dir.rglob(pattern):
                try:
                    if os.path.getctime(file_path) < cutoff_time:
                        if file_path.is_file():
                            file_path.unlink()
                            self.removed_files.append(str(file_path))
                        elif file_path.is_dir():
                            shutil.rmtree(file_path)
                            self.removed_dirs.append(str(file_path))
                except (OSError, PermissionError):
                    continue

        return {
            'target_directory': str(self.target_dir),
            'files_removed': self.removed_files,
            'directories_removed': self.removed_dirs,
            'total_cleaned': len(self.removed_files) + len(self.removed_dirs)
        }

    def get_stats(self) -> dict:
        total_size = 0
        for file_path in self.removed_files:
            try:
                total_size += os.path.getsize(file_path)
            except OSError:
                continue
        return {
            'files_count': len(self.removed_files),
            'dirs_count': len(self.removed_dirs),
            'estimated_space_freed': total_size
        }

def cleanup_temp_directory(days: int = 7) -> None:
    cleaner = TempFileCleaner()
    result = cleaner.scan_and_clean(days_old=days)
    stats = cleaner.get_stats()
    
    print(f"Cleanup completed in {result['target_directory']}")
    print(f"Files removed: {result['files_removed']}")
    print(f"Directories removed: {result['directories_removed']}")
    print(f"Total items cleaned: {result['total_cleaned']}")
    print(f"Estimated space freed: {stats['estimated_space_freed']} bytes")

if __name__ == "__main__":
    cleanup_temp_directory()import os
import shutil
import sys

def clean_temp_files(directory, extensions=('.tmp', '.temp', '.log')):
    """
    Remove temporary files with specified extensions from a directory.
    
    Args:
        directory (str): Path to the directory to clean.
        extensions (tuple): File extensions to consider as temporary.
    
    Returns:
        int: Number of files removed.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return 0
    
    removed_count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extensions):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
                    removed_count += 1
                except OSError as e:
                    print(f"Error removing {file_path}: {e}")
    
    return removed_count

def clean_empty_directories(directory):
    """
    Remove empty directories recursively.
    
    Args:
        directory (str): Path to the directory to clean.
    
    Returns:
        int: Number of directories removed.
    """
    removed_count = 0
    for root, dirs, files in os.walk(directory, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):
                try:
                    os.rmdir(dir_path)
                    print(f"Removed empty directory: {dir_path}")
                    removed_count += 1
                except OSError as e:
                    print(f"Error removing directory {dir_path}: {e}")
    
    return removed_count

def main():
    if len(sys.argv) < 2:
        print("Usage: python file_cleaner.py <directory_path>")
        sys.exit(1)
    
    target_directory = sys.argv[1]
    
    if not os.path.exists(target_directory):
        print(f"Error: Directory '{target_directory}' does not exist.")
        sys.exit(1)
    
    print(f"Cleaning temporary files in: {target_directory}")
    files_removed = clean_temp_files(target_directory)
    print(f"Removed {files_removed} temporary file(s).")
    
    print(f"Cleaning empty directories in: {target_directory}")
    dirs_removed = clean_empty_directories(target_directory)
    print(f"Removed {dirs_removed} empty directory(ies).")
    
    print("Cleanup completed.")

if __name__ == "__main__":
    main()