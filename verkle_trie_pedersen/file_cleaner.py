import os
import shutil
import tempfile
from pathlib import Path

def clean_temporary_files(directory_path, extensions=('.tmp', '.temp', '.log')):
    """
    Remove temporary files with specified extensions from a directory.
    
    Args:
        directory_path (str or Path): Path to the directory to clean.
        extensions (tuple): File extensions to consider as temporary.
    
    Returns:
        list: Paths of files that were removed.
    """
    removed_files = []
    dir_path = Path(directory_path)
    
    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"Invalid directory path: {directory_path}")
    
    for item in dir_path.iterdir():
        if item.is_file() and item.suffix.lower() in extensions:
            try:
                item.unlink()
                removed_files.append(str(item))
                print(f"Removed: {item}")
            except OSError as e:
                print(f"Failed to remove {item}: {e}")
    
    return removed_files

def create_sample_temporary_files(directory_path, num_files=5):
    """
    Create sample temporary files for testing purposes.
    
    Args:
        directory_path (str or Path): Path to create sample files.
        num_files (int): Number of sample files to create.
    """
    dir_path = Path(directory_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_files):
        temp_file = dir_path / f"temp_file_{i}.tmp"
        temp_file.write_text(f"Temporary content {i}")
        log_file = dir_path / f"app_log_{i}.log"
        log_file.write_text(f"Log entry {i}")
    
    print(f"Created {num_files * 2} sample files in {directory_path}")

if __name__ == "__main__":
    # Example usage
    test_dir = Path(tempfile.gettempdir()) / "test_cleanup"
    
    print("Creating sample temporary files...")
    create_sample_temporary_files(test_dir, 3)
    
    print("\nCleaning temporary files...")
    removed = clean_temporary_files(test_dir, ('.tmp', '.log'))
    
    print(f"\nTotal files removed: {len(removed)}")
    
    # Clean up test directory
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print(f"Cleaned up test directory: {test_dir}")