#!/usr/bin/env python
"""
Cache Management Utilities

Manage cached data, features, and models
"""

import os
import shutil
from pathlib import Path
from datetime import datetime


class CacheManager:
    """Manage all caches for the stock picker system"""

    def __init__(self, base_dir: str = "./stock_picker_data"):
        self.base_dir = Path(base_dir)
        self.cache_dir = self.base_dir / "cache"
        self.models_dir = self.base_dir / "models"
        self.results_dir = self.base_dir / "results"
        self.backtest_results_dir = self.base_dir / "backtest_results"

    def get_dir_size(self, directory: Path) -> int:
        """Get total size of directory in bytes"""
        if not directory.exists():
            return 0

        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size

    def format_size(self, size_bytes: int) -> str:
        """Format bytes to human-readable string"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"

    def count_files(self, directory: Path) -> int:
        """Count files in directory"""
        if not directory.exists():
            return 0
        return sum(1 for _ in directory.rglob('*') if _.is_file())

    def get_cache_info(self):
        """Display comprehensive cache information"""
        print("\n" + "=" * 80)
        print("💾 CACHE INFORMATION")
        print("=" * 80)

        # BSE Data Cache
        bse_cache = self.cache_dir / "bse_data"
        bse_size = self.get_dir_size(bse_cache)
        bse_files = self.count_files(bse_cache)
        print(f"\n📊 BSE Data Cache:")
        print(f"   Location: {bse_cache}")
        print(f"   Files: {bse_files}")
        print(f"   Size: {self.format_size(bse_size)}")
        print(f"   Contains: Raw BhavCopy data (OHLCV)")

        # Features Cache
        features_cache = self.cache_dir / "features"
        features_size = self.get_dir_size(features_cache)
        features_files = self.count_files(features_cache)
        print(f"\n🔧 Features Cache:")
        print(f"   Location: {features_cache}")
        print(f"   Files: {features_files}")
        print(f"   Size: {self.format_size(features_size)}")
        print(f"   Contains: Computed technical indicators (50+ features)")

        # Models
        models_size = self.get_dir_size(self.models_dir)
        models_files = self.count_files(self.models_dir)
        print(f"\n🤖 Trained Models:")
        print(f"   Location: {self.models_dir}")
        print(f"   Files: {models_files}")
        print(f"   Size: {self.format_size(models_size)}")
        print(f"   Contains: LightGBM models + config + metadata")

        # Results
        results_size = self.get_dir_size(self.results_dir)
        results_files = self.count_files(self.results_dir)
        print(f"\n📋 Daily Results:")
        print(f"   Location: {self.results_dir}")
        print(f"   Files: {results_files}")
        print(f"   Size: {self.format_size(results_size)}")
        print(f"   Contains: Daily stock picks CSV files")

        # Backtest Results
        backtest_size = self.get_dir_size(self.backtest_results_dir)
        backtest_files = self.count_files(self.backtest_results_dir)
        print(f"\n🧪 Backtest Results:")
        print(f"   Location: {self.backtest_results_dir}")
        print(f"   Files: {backtest_files}")
        print(f"   Size: {self.format_size(backtest_size)}")
        print(f"   Contains: Historical backtest results CSV")

        # Total
        total_size = bse_size + features_size + models_size + results_size + backtest_size
        total_files = bse_files + features_files + models_files + results_files + backtest_files
        print(f"\n" + "-" * 80)
        print(f"📦 TOTAL:")
        print(f"   Files: {total_files}")
        print(f"   Size: {self.format_size(total_size)}")
        print("=" * 80)

        # Recommendations
        print(f"\n💡 Recommendations:")
        if bse_size > 500 * 1024 * 1024:  # > 500 MB
            print("   • BSE cache is large (>500MB). Consider clearing old date ranges.")
        if features_size > 1024 * 1024 * 1024:  # > 1 GB
            print("   • Features cache is large (>1GB). Consider clearing if disk space low.")
        if total_size < 100 * 1024 * 1024:  # < 100 MB
            print("   • Cache is small. System will be fast on subsequent runs!")

    def clear_bse_cache(self, confirm: bool = False):
        """Clear BSE data cache"""
        if not confirm:
            print("⚠️  This will delete all cached BSE data.")
            print("   Next run will download data fresh (~5-10 minutes).")
            response = input("   Continue? (yes/no): ")
            if response.lower() != 'yes':
                print("   Cancelled.")
                return

        bse_cache = self.cache_dir / "bse_data"
        if bse_cache.exists():
            shutil.rmtree(bse_cache)
            bse_cache.mkdir(parents=True, exist_ok=True)
            print(f"✅ BSE cache cleared: {bse_cache}")
        else:
            print("   No BSE cache found.")

    def clear_features_cache(self, confirm: bool = False):
        """Clear computed features cache"""
        if not confirm:
            print("⚠️  This will delete all cached features.")
            print("   Next run will recompute features (~10-15 minutes).")
            response = input("   Continue? (yes/no): ")
            if response.lower() != 'yes':
                print("   Cancelled.")
                return

        features_cache = self.cache_dir / "features"
        if features_cache.exists():
            shutil.rmtree(features_cache)
            features_cache.mkdir(parents=True, exist_ok=True)
            print(f"✅ Features cache cleared: {features_cache}")
        else:
            print("   No features cache found.")

    def clear_all_cache(self, confirm: bool = False):
        """Clear ALL caches (BSE data + features)"""
        if not confirm:
            print("⚠️  This will delete ALL cached data!")
            print("   Next run will be slow (~20 minutes).")
            response = input("   Continue? (yes/no): ")
            if response.lower() != 'yes':
                print("   Cancelled.")
                return

        self.clear_bse_cache(confirm=True)
        self.clear_features_cache(confirm=True)
        print("✅ All caches cleared!")

    def list_cached_files(self, cache_type: str = "all"):
        """List all cached files with dates"""
        print("\n" + "=" * 80)
        print(f"📁 CACHED FILES - {cache_type.upper()}")
        print("=" * 80)

        directories = []
        if cache_type in ["all", "bse"]:
            directories.append(("BSE Data", self.cache_dir / "bse_data"))
        if cache_type in ["all", "features"]:
            directories.append(("Features", self.cache_dir / "features"))
        if cache_type in ["all", "models"]:
            directories.append(("Models", self.models_dir))

        for name, directory in directories:
            if not directory.exists():
                print(f"\n{name}: No cache found")
                continue

            files = sorted(directory.rglob('*'), key=lambda p: p.stat().st_mtime, reverse=True)
            files = [f for f in files if f.is_file()]

            print(f"\n{name} ({len(files)} files):")
            for f in files[:10]:  # Show latest 10
                size = os.path.getsize(f)
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                print(f"   • {f.name}")
                print(f"     Size: {self.format_size(size)} | Modified: {mtime.strftime('%Y-%m-%d %H:%M')}")

            if len(files) > 10:
                print(f"   ... and {len(files) - 10} more files")


def main():
    """Interactive cache management"""
    manager = CacheManager()

    print("\n" + "=" * 80)
    print("💾 CACHE MANAGER - 5-Session Stock Picker")
    print("=" * 80)

    while True:
        print("\n📋 Options:")
        print("   1. Show cache info")
        print("   2. List cached files")
        print("   3. Clear BSE cache")
        print("   4. Clear features cache")
        print("   5. Clear ALL caches")
        print("   6. Exit")

        choice = input("\nSelect option (1-6): ").strip()

        if choice == '1':
            manager.get_cache_info()
        elif choice == '2':
            print("\nList which cache?")
            print("   a. All")
            print("   b. BSE data only")
            print("   c. Features only")
            print("   d. Models only")
            sub_choice = input("Select (a-d): ").strip().lower()
            cache_map = {'a': 'all', 'b': 'bse', 'c': 'features', 'd': 'models'}
            manager.list_cached_files(cache_map.get(sub_choice, 'all'))
        elif choice == '3':
            manager.clear_bse_cache()
        elif choice == '4':
            manager.clear_features_cache()
        elif choice == '5':
            manager.clear_all_cache()
        elif choice == '6':
            print("\n👋 Goodbye!")
            break
        else:
            print("   Invalid option!")


if __name__ == "__main__":
    main()
