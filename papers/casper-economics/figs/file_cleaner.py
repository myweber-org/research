import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List

class TempFileCleaner:
    def __init__(self, target_dir: Optional[str] = None):
        self.target_dir = Path(target_dir) if target_dir else Path(tempfile.gettempdir())
        self.safe_patterns = ['.pid', '.lock', '.sock']

    def is_safe_to_delete(self, filepath: Path) -> bool:
        return not any(filepath.name.endswith(pattern) for pattern in self.safe_patterns)

    def get_old_files(self, days_old: int = 7) -> List[Path]:
        from datetime import datetime, timedelta
        cutoff_time = datetime.now() - timedelta(days=days_old)
        old_files = []
        
        for item in self.target_dir.iterdir():
            if item.is_file():
                stat = item.stat()
                last_modified = datetime.fromtimestamp(stat.st_mtime)
                if last_modified < cutoff_time and self.is_safe_to_delete(item):
                    old_files.append(item)
        
        return old_files

    def cleanup(self, days_old: int = 7, dry_run: bool = True) -> dict:
        old_files = self.get_old_files(days_old)
        results = {
            'total_found': len(old_files),
            'deleted': [],
            'skipped': [],
            'errors': []
        }
        
        if dry_run:
            results['deleted'] = [str(f) for f in old_files]
            return results
        
        for filepath in old_files:
            try:
                filepath.unlink()
                results['deleted'].append(str(filepath))
            except PermissionError:
                results['skipped'].append(str(filepath))
            except Exception as e:
                results['errors'].append(f"{filepath}: {str(e)}")
        
        return results

    def cleanup_empty_dirs(self, max_depth: int = 3) -> int:
        removed_count = 0
        for root, dirs, _ in os.walk(self.target_dir, topdown=False):
            if root.count(os.sep) - self.target_dir.as_posix().count(os.sep) >= max_depth:
                continue
                
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                try:
                    if not any(dir_path.iterdir()):
                        dir_path.rmdir()
                        removed_count += 1
                except (OSError, PermissionError):
                    continue
        
        return removed_count

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Clean temporary files older than specified days')
    parser.add_argument('--dir', help='Target directory (default: system temp)')
    parser.add_argument('--days', type=int, default=7, help='Delete files older than N days')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted')
    parser.add_argument('--clean-dirs', action='store_true', help='Remove empty directories')
    
    args = parser.parse_args()
    
    cleaner = TempFileCleaner(args.dir)
    
    print(f"Scanning directory: {cleaner.target_dir}")
    
    if args.clean_dirs:
        removed = cleaner.cleanup_empty_dirs()
        print(f"Removed {removed} empty directories")
    
    results = cleaner.cleanup(args.days, args.dry_run)
    
    print(f"\nFound {results['total_found']} files older than {args.days} days")
    
    if args.dry_run:
        print("Dry run - no files deleted")
        if results['deleted']:
            print("\nFiles that would be deleted:")
            for f in results['deleted']:
                print(f"  {f}")
    else:
        print(f"Deleted {len(results['deleted'])} files")
        if results['skipped']:
            print(f"Skipped {len(results['skipped'])} files (permission denied)")
        if results['errors']:
            print(f"Errors: {len(results['errors'])}")
            for err in results['errors']:
                print(f"  {err}")

if __name__ == '__main__':
    main()import os
import shutil
import tempfile
from pathlib import Path

def clean_temporary_files(directory_path, extensions=None, days_old=7):
    """
    Remove temporary files from a specified directory.
    
    Args:
        directory_path (str or Path): Path to the directory to clean.
        extensions (list, optional): List of file extensions to consider as temporary.
                                     Defaults to common temporary extensions.
        days_old (int, optional): Remove files older than this many days. Defaults to 7.
    """
    if extensions is None:
        extensions = ['.tmp', '.temp', '.bak', '.swp', '.log']
    
    directory = Path(directory_path)
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Invalid directory path: {directory_path}")
    
    current_time = time.time()
    cutoff_time = current_time - (days_old * 24 * 60 * 60)
    
    files_removed = 0
    total_size = 0
    
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            # Check if file matches temporary extensions
            if file_path.suffix.lower() in extensions:
                # Check if file is older than the cutoff
                file_mtime = file_path.stat().st_mtime
                if file_mtime < cutoff_time:
                    try:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        files_removed += 1
                        total_size += file_size
                        print(f"Removed: {file_path} ({file_size} bytes)")
                    except OSError as e:
                        print(f"Error removing {file_path}: {e}")
    
    print(f"\nCleanup completed:")
    print(f"  Files removed: {files_removed}")
    print(f"  Total space freed: {total_size} bytes")
    return files_removed, total_size

def create_test_environment():
    """Create a test environment with temporary files."""
    test_dir = Path(tempfile.mkdtemp(prefix="clean_test_"))
    print(f"Created test directory: {test_dir}")
    
    # Create some test files
    test_files = [
        "document.tmp",
        "backup.bak",
        "important.txt",
        "logfile.log",
        "swapfile.swp",
        "config.temp"
    ]
    
    for filename in test_files:
        file_path = test_dir / filename
        file_path.write_text("Sample content for testing")
        # Make some files older by modifying mtime
        if filename.endswith('.tmp'):
            old_time = time.time() - (10 * 24 * 60 * 60)  # 10 days old
            os.utime(file_path, (old_time, old_time))
    
    return test_dir

if __name__ == "__main__":
    import time
    
    # Create and clean a test environment
    test_directory = create_test_environment()
    
    try:
        print("\nStarting cleanup process...")
        removed, freed = clean_temporary_files(test_directory, days_old=5)
        
        # List remaining files
        print("\nRemaining files:")
        for item in test_directory.iterdir():
            print(f"  {item.name}")
            
    finally:
        # Clean up test directory
        shutil.rmtree(test_directory)
        print(f"\nTest directory removed: {test_directory}")