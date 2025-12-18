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
    cleanup_temp_directory()