
import os
import shutil
import tempfile
from pathlib import Path

def clean_temp_files(directory_path, extensions=None, days_old=7):
    """
    Remove temporary files from a specified directory.
    
    Args:
        directory_path (str or Path): Path to directory to clean
        extensions (list, optional): List of file extensions to target.
                                     Defaults to common temp extensions.
        days_old (int, optional): Only remove files older than this many days.
                                  Defaults to 7.
    
    Returns:
        tuple: (files_removed, total_size_freed)
    """
    if extensions is None:
        extensions = ['.tmp', '.temp', '.bak', '.swp', '.log']
    
    directory = Path(directory_path)
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Invalid directory: {directory_path}")
    
    files_removed = 0
    total_size = 0
    current_time = os.path.getctime(tempfile.gettempdir())
    
    for item in directory.rglob('*'):
        if item.is_file():
            file_age = current_time - item.stat().st_mtime
            age_in_days = file_age / (60 * 60 * 24)
            
            if age_in_days > days_old:
                if any(item.suffix.lower() == ext.lower() for ext in extensions):
                    try:
                        file_size = item.stat().st_size
                        item.unlink()
                        files_removed += 1
                        total_size += file_size
                    except (OSError, PermissionError):
                        continue
    
    return files_removed, total_size

def main():
    """Example usage of the file cleaner utility."""
    import sys
    
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        target_dir = tempfile.gettempdir()
    
    try:
        removed, size_freed = clean_temp_files(target_dir)
        print(f"Cleaned {removed} files, freed {size_freed / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()