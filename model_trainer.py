#!/usr/bin/env python
"""
Model Training Tracker & Yearly Training System

Tracks:
- Which dates have been trained on
- Model versions and performance metrics
- Automatically trains on all dates in a year
- Detects model updates and resets training status
"""

import os
import json
import pickle
import hashlib
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set
import pandas as pd
import numpy as np


class ModelTracker:
    """
    Tracks model training progress across dates and years
    Handles model versioning and detects when models are updated
    """

    def __init__(self, base_dir: str = "./stock_picker_data"):
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"
        self.tracker_file = self.models_dir / "training_tracker.json"

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Load or initialize tracker
        self.tracker = self._load_tracker()

    def _load_tracker(self) -> Dict:
        """Load training tracker from disk"""
        if self.tracker_file.exists():
            with open(self.tracker_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'model_version': None,
                'model_hash': None,
                'model_signature': None,
                'trained_dates': {},  # {year: [list of date strings]}
                'training_history': [],  # List of training records
                'last_updated': None
            }

    def _save_tracker(self):
        """Save training tracker to disk"""
        with open(self.tracker_file, 'w') as f:
            json.dump(self.tracker, f, indent=2)

    def _compute_model_signature(self, model_package: Dict) -> str:
        """
        Compute a signature for the model based on its configuration and structure
        This helps detect if the model has been fundamentally changed
        """
        signature_components = {
            'config': model_package.get('config', {}),
            'feature_cols': model_package.get('feature_cols', []),
            'n_features': model_package.get('metadata', {}).get('n_features', 0),
        }

        # Create hash from components
        sig_str = json.dumps(signature_components, sort_keys=True)
        return hashlib.md5(sig_str.encode()).hexdigest()

    def _compute_model_hash(self, model_path: Path) -> str:
        """Compute file hash of the model"""
        if not model_path.exists():
            return None

        with open(model_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def check_model_updated(self, model_path: Path) -> bool:
        """
        Check if the model has been updated/changed
        Returns True if model is different from tracked version
        """
        if not model_path.exists():
            return False

        # Load model package
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)

        # Compute current signature
        current_sig = self._compute_model_signature(model_package)
        current_hash = self._compute_model_hash(model_path)

        # Compare with tracked version
        if self.tracker['model_signature'] is None:
            # First time tracking this model
            return False

        if current_sig != self.tracker['model_signature']:
            print("⚠️  Model structure/config has changed!")
            return True

        if current_hash != self.tracker['model_hash']:
            print("⚠️  Model file has changed!")
            return True

        return False

    def update_model_version(self, model_path: Path):
        """
        Update tracked model version
        Call this after training a new model
        """
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)

        self.tracker['model_signature'] = self._compute_model_signature(model_package)
        self.tracker['model_hash'] = self._compute_model_hash(model_path)
        self.tracker['model_version'] = str(date.today())
        self.tracker['last_updated'] = str(date.today())

        self._save_tracker()

    def reset_training_for_year(self, year: int):
        """
        Reset training status for a specific year
        Use this when model is updated and you want to retrain
        """
        year_key = str(year)
        if year_key in self.tracker['trained_dates']:
            old_count = len(self.tracker['trained_dates'][year_key])
            self.tracker['trained_dates'][year_key] = []
            self._save_tracker()
            print(f"🔄 Reset training for {year} (cleared {old_count} dates)")

    def mark_date_trained(self, training_date: date, metrics: Dict = None):
        """
        Mark a specific date as trained
        """
        year_key = str(training_date.year)
        date_str = str(training_date)

        # Initialize year if needed
        if year_key not in self.tracker['trained_dates']:
            self.tracker['trained_dates'][year_key] = []

        # Add date if not already there
        if date_str not in self.tracker['trained_dates'][year_key]:
            self.tracker['trained_dates'][year_key].append(date_str)

        # Add to history
        history_record = {
            'date': date_str,
            'trained_on': str(date.today()),
            'metrics': metrics or {}
        }
        self.tracker['training_history'].append(history_record)

        self._save_tracker()

    def is_date_trained(self, training_date: date) -> bool:
        """Check if a specific date has been trained"""
        year_key = str(training_date.year)
        date_str = str(training_date)

        if year_key not in self.tracker['trained_dates']:
            return False

        return date_str in self.tracker['trained_dates'][year_key]

    def get_year_progress(self, year: int) -> Dict:
        """
        Get training progress for a specific year
        """
        year_key = str(year)

        # Get all business days in year
        business_days = self._get_business_days_in_year(year)

        # Get trained dates
        trained_dates = set(self.tracker['trained_dates'].get(year_key, []))

        # Convert to date objects for comparison
        trained_date_objs = {date.fromisoformat(d) for d in trained_dates}

        # Find missing dates
        missing = [d for d in business_days if str(d) not in trained_dates]

        return {
            'year': year,
            'total_business_days': len(business_days),
            'trained_days': len(trained_date_objs),
            'missing_days': len(missing),
            'completion_pct': len(trained_date_objs) / len(business_days) * 100 if business_days else 0,
            'missing_dates': missing[:10]  # Show first 10 missing
        }

    def is_year_complete(self, year: int) -> bool:
        """Check if all business days in a year have been trained"""
        progress = self.get_year_progress(year)
        return progress['missing_days'] == 0

    def _get_business_days_in_year(self, year: int) -> List[date]:
        """Get all business days (Mon-Fri) in a year"""
        start_date = date(year, 1, 1)
        end_date = date(year, 12, 31)

        business_days = []
        cur = start_date

        while cur <= end_date:
            if cur.weekday() < 5:  # Mon-Fri
                business_days.append(cur)
            cur += timedelta(days=1)

        return business_days

    def get_untrained_dates(self, year: int) -> List[date]:
        """Get all untrained business days in a year"""
        year_key = str(year)
        trained_dates = set(self.tracker['trained_dates'].get(year_key, []))

        business_days = self._get_business_days_in_year(year)

        untrained = [d for d in business_days if str(d) not in trained_dates]

        return sorted(untrained)

    def display_status(self, years: List[int] = None):
        """Display training status"""
        print("\n" + "=" * 80)
        print("📊 MODEL TRAINING STATUS")
        print("=" * 80)

        # Model info
        print(f"\n🤖 Model Version: {self.tracker.get('model_version', 'Not set')}")
        print(f"   Last Updated: {self.tracker.get('last_updated', 'Never')}")
        print(f"   Signature: {self.tracker.get('model_signature', 'N/A')[:16]}...")

        # Training stats
        if not years:
            years = sorted([int(y) for y in self.tracker['trained_dates'].keys()])

        if years:
            print(f"\n📅 Training Progress by Year:")
            for year in years:
                progress = self.get_year_progress(year)
                print(f"\n   {year}:")
                print(f"      Total business days: {progress['total_business_days']}")
                print(f"      Trained: {progress['trained_days']}")
                print(f"      Missing: {progress['missing_days']}")
                print(f"      Completion: {progress['completion_pct']:.1f}%")

                if progress['missing_days'] > 0 and progress['missing_days'] <= 10:
                    print(f"      Missing dates: {', '.join(str(d) for d in progress['missing_dates'])}")
                elif progress['missing_days'] > 10:
                    print(f"      First 10 missing: {', '.join(str(d) for d in progress['missing_dates'][:10])}")

        # Recent training
        recent = self.tracker['training_history'][-5:]
        if recent:
            print(f"\n🕒 Recent Training Sessions:")
            for record in recent:
                print(f"      {record['date']} (trained on {record['trained_on']})")

        print("=" * 80)


# Quick test
if __name__ == "__main__":
    tracker = ModelTracker()
    tracker.display_status([2023, 2024, 2025])
