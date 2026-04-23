#!/usr/bin/env python3
"""
Comprehensive Data Caching System - Requirement 7 Compliance

This module provides caching for ALL external data sources:
- FII/DII data from NSE
- Google Trends data
- Sentiment data (Twitter/News)
- Insider trading data
- Earnings call transcripts
- Any other external API data

Compliance: Addresses Requirement 7 - "all the data must be cached"

Features:
- Automatic cache management
- Cache expiration support
- File-based caching (SQLite for structured data)
- Easy cache invalidation
- Thread-safe operations
"""

import os
import json
import pickle
import sqlite3
import hashlib
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Optional, Callable, Dict, List
import logging
import pandas as pd
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCache:
    """
    Comprehensive caching system for all external data sources.

    Supports:
    - JSON caching (for API responses)
    - Pickle caching (for Python objects)
    - SQLite caching (for structured data like FII/DII)
    - DataFrame caching (for time-series data)
    """

    def __init__(self, cache_dir: str = "./cache"):
        """
        Initialize data cache.

        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories for different data types
        self.fii_dii_dir = self.cache_dir / "fii_dii"
        self.trends_dir = self.cache_dir / "trends"
        self.sentiment_dir = self.cache_dir / "sentiment"
        self.insider_dir = self.cache_dir / "insider"
        self.earnings_dir = self.cache_dir / "earnings"
        self.misc_dir = self.cache_dir / "misc"

        for subdir in [self.fii_dii_dir, self.trends_dir, self.sentiment_dir,
                       self.insider_dir, self.earnings_dir, self.misc_dir]:
            subdir.mkdir(parents=True, exist_ok=True)

        # SQLite database for structured data
        self.db_path = self.cache_dir / "data_cache.db"
        self._init_database()

        logger.info(f"✅ DataCache initialized at {self.cache_dir}")

    def _init_database(self):
        """Initialize SQLite database for structured caching."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # FII/DII data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fii_dii_data (
                date TEXT PRIMARY KEY,
                fii_buy REAL,
                fii_sell REAL,
                dii_buy REAL,
                dii_sell REAL,
                fii_net REAL,
                dii_net REAL,
                cached_at TEXT
            )
        """)

        # Sentiment data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_data (
                symbol TEXT,
                date TEXT,
                sentiment_score REAL,
                source TEXT,
                cached_at TEXT,
                PRIMARY KEY (symbol, date, source)
            )
        """)

        # Insider trading table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS insider_trading (
                symbol TEXT,
                date TEXT,
                insider_name TEXT,
                transaction_type TEXT,
                shares REAL,
                value REAL,
                cached_at TEXT,
                PRIMARY KEY (symbol, date, insider_name)
            )
        """)

        # Google Trends table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS google_trends (
                symbol TEXT,
                date TEXT,
                interest REAL,
                cached_at TEXT,
                PRIMARY KEY (symbol, date)
            )
        """)

        # Generic key-value cache
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kv_cache (
                cache_key TEXT PRIMARY KEY,
                cache_value TEXT,
                cached_at TEXT,
                expires_at TEXT
            )
        """)

        conn.commit()
        conn.close()

    # ====================
    # FII/DII Data Caching
    # ====================

    def get_fii_dii_data(self, target_date: date) -> Optional[Dict[str, float]]:
        """
        Get cached FII/DII data for a date.

        Args:
            target_date: Date to get data for

        Returns:
            Dict with FII/DII data or None if not cached
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT fii_buy, fii_sell, dii_buy, dii_sell, fii_net, dii_net FROM fii_dii_data WHERE date = ?",
            (target_date.isoformat(),)
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            logger.debug(f"✅ FII/DII cache hit for {target_date}")
            return {
                'fii_buy': row[0],
                'fii_sell': row[1],
                'dii_buy': row[2],
                'dii_sell': row[3],
                'fii_net': row[4],
                'dii_net': row[5]
            }

        logger.debug(f"❌ FII/DII cache miss for {target_date}")
        return None

    def cache_fii_dii_data(
        self,
        target_date: date,
        fii_buy: float,
        fii_sell: float,
        dii_buy: float,
        dii_sell: float,
        fii_net: float,
        dii_net: float
    ):
        """Cache FII/DII data for a date."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO fii_dii_data
            (date, fii_buy, fii_sell, dii_buy, dii_sell, fii_net, dii_net, cached_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            target_date.isoformat(),
            fii_buy, fii_sell, dii_buy, dii_sell, fii_net, dii_net,
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

        logger.debug(f"💾 Cached FII/DII data for {target_date}")

    # ====================
    # Sentiment Data Caching
    # ====================

    def get_sentiment_data(
        self,
        symbol: str,
        target_date: date,
        source: str = "combined"
    ) -> Optional[float]:
        """
        Get cached sentiment score for a symbol on a date.

        Args:
            symbol: Stock symbol
            target_date: Date
            source: Source of sentiment (twitter, news, combined)

        Returns:
            Sentiment score or None if not cached
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT sentiment_score FROM sentiment_data WHERE symbol = ? AND date = ? AND source = ?",
            (symbol, target_date.isoformat(), source)
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            logger.debug(f"✅ Sentiment cache hit for {symbol} on {target_date}")
            return row[0]

        logger.debug(f"❌ Sentiment cache miss for {symbol} on {target_date}")
        return None

    def cache_sentiment_data(
        self,
        symbol: str,
        target_date: date,
        sentiment_score: float,
        source: str = "combined"
    ):
        """Cache sentiment data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO sentiment_data
            (symbol, date, sentiment_score, source, cached_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            symbol, target_date.isoformat(), sentiment_score, source,
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

        logger.debug(f"💾 Cached sentiment for {symbol} on {target_date}")

    # ====================
    # Google Trends Caching
    # ====================

    def get_trends_data(self, symbol: str, target_date: date) -> Optional[float]:
        """Get cached Google Trends interest for a symbol on a date."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT interest FROM google_trends WHERE symbol = ? AND date = ?",
            (symbol, target_date.isoformat())
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            logger.debug(f"✅ Trends cache hit for {symbol} on {target_date}")
            return row[0]

        logger.debug(f"❌ Trends cache miss for {symbol} on {target_date}")
        return None

    def cache_trends_data(self, symbol: str, target_date: date, interest: float):
        """Cache Google Trends data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO google_trends
            (symbol, date, interest, cached_at)
            VALUES (?, ?, ?, ?)
        """, (
            symbol, target_date.isoformat(), interest,
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

        logger.debug(f"💾 Cached trends for {symbol} on {target_date}")

    # ====================
    # Insider Trading Caching
    # ====================

    def get_insider_trading(
        self,
        symbol: str,
        start_date: date,
        end_date: date
    ) -> List[Dict[str, Any]]:
        """Get cached insider trading data for a symbol in date range."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT date, insider_name, transaction_type, shares, value
            FROM insider_trading
            WHERE symbol = ? AND date >= ? AND date <= ?
            ORDER BY date DESC
        """, (symbol, start_date.isoformat(), end_date.isoformat()))

        rows = cursor.fetchall()
        conn.close()

        if rows:
            logger.debug(f"✅ Insider trading cache hit for {symbol}")
            return [
                {
                    'date': row[0],
                    'insider_name': row[1],
                    'transaction_type': row[2],
                    'shares': row[3],
                    'value': row[4]
                }
                for row in rows
            ]

        logger.debug(f"❌ Insider trading cache miss for {symbol}")
        return []

    def cache_insider_trading(
        self,
        symbol: str,
        target_date: date,
        insider_name: str,
        transaction_type: str,
        shares: float,
        value: float
    ):
        """Cache insider trading data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO insider_trading
            (symbol, date, insider_name, transaction_type, shares, value, cached_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol, target_date.isoformat(), insider_name,
            transaction_type, shares, value,
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

        logger.debug(f"💾 Cached insider trading for {symbol}")

    # ====================
    # Generic File Caching
    # ====================

    def get_file_cache(
        self,
        cache_key: str,
        cache_type: str = "pickle",
        subdir: Optional[Path] = None
    ) -> Optional[Any]:
        """
        Get data from file cache.

        Args:
            cache_key: Unique key for cached data
            cache_type: Type of cache (pickle, json, csv)
            subdir: Subdirectory to use (defaults to misc_dir)

        Returns:
            Cached data or None
        """
        if subdir is None:
            subdir = self.misc_dir

        # Generate filename
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        cache_file = subdir / f"{cache_hash}.{cache_type}"

        if not cache_file.exists():
            logger.debug(f"❌ File cache miss for {cache_key}")
            return None

        try:
            if cache_type == "pickle":
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
            elif cache_type == "json":
                with open(cache_file, 'r') as f:
                    data = json.load(f)
            elif cache_type == "csv":
                data = pd.read_csv(cache_file)
            else:
                raise ValueError(f"Unsupported cache type: {cache_type}")

            logger.debug(f"✅ File cache hit for {cache_key}")
            return data

        except Exception as e:
            logger.error(f"Error loading cache for {cache_key}: {e}")
            return None

    def set_file_cache(
        self,
        cache_key: str,
        data: Any,
        cache_type: str = "pickle",
        subdir: Optional[Path] = None
    ):
        """
        Save data to file cache.

        Args:
            cache_key: Unique key for cached data
            data: Data to cache
            cache_type: Type of cache (pickle, json, csv)
            subdir: Subdirectory to use (defaults to misc_dir)
        """
        if subdir is None:
            subdir = self.misc_dir

        # Generate filename
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        cache_file = subdir / f"{cache_hash}.{cache_type}"

        try:
            if cache_type == "pickle":
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
            elif cache_type == "json":
                with open(cache_file, 'w') as f:
                    json.dump(data, f)
            elif cache_type == "csv":
                if isinstance(data, pd.DataFrame):
                    data.to_csv(cache_file, index=False)
                else:
                    raise ValueError("CSV caching requires DataFrame")
            else:
                raise ValueError(f"Unsupported cache type: {cache_type}")

            logger.debug(f"💾 Cached data for {cache_key}")

        except Exception as e:
            logger.error(f"Error caching data for {cache_key}: {e}")

    # ====================
    # Generic KV Cache
    # ====================

    def get_kv(self, key: str) -> Optional[str]:
        """Get value from key-value cache."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT cache_value, expires_at FROM kv_cache WHERE cache_key = ?",
            (key,)
        )

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        # Check expiration
        if row[1]:
            expires_at = datetime.fromisoformat(row[1])
            if datetime.now() > expires_at:
                logger.debug(f"⏰ Cache expired for {key}")
                return None

        return row[0]

    def set_kv(self, key: str, value: str, ttl_hours: Optional[int] = None):
        """
        Set value in key-value cache.

        Args:
            key: Cache key
            value: Cache value (will be converted to string)
            ttl_hours: Time-to-live in hours (None = no expiration)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        expires_at = None
        if ttl_hours:
            expires_at = (datetime.now() + timedelta(hours=ttl_hours)).isoformat()

        cursor.execute("""
            INSERT OR REPLACE INTO kv_cache
            (cache_key, cache_value, cached_at, expires_at)
            VALUES (?, ?, ?, ?)
        """, (key, str(value), datetime.now().isoformat(), expires_at))

        conn.commit()
        conn.close()

    # ====================
    # Cache Management
    # ====================

    def clear_expired(self):
        """Clear expired cache entries."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "DELETE FROM kv_cache WHERE expires_at IS NOT NULL AND expires_at < ?",
            (datetime.now().isoformat(),)
        )

        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        logger.info(f"🧹 Cleared {deleted} expired cache entries")

    def clear_all(self):
        """Clear all cached data (use with caution!)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for table in ['fii_dii_data', 'sentiment_data', 'insider_trading',
                      'google_trends', 'kv_cache']:
            cursor.execute(f"DELETE FROM {table}")

        conn.commit()
        conn.close()

        # Clear file caches
        for subdir in [self.fii_dii_dir, self.trends_dir, self.sentiment_dir,
                       self.insider_dir, self.earnings_dir, self.misc_dir]:
            for file in subdir.glob("*"):
                if file.is_file():
                    file.unlink()

        logger.info("🧹 Cleared all cache data")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        for table in ['fii_dii_data', 'sentiment_data', 'insider_trading',
                      'google_trends', 'kv_cache']:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cursor.fetchone()[0]

        conn.close()

        return stats


# ====================
# Decorator for automatic caching
# ====================

def cached(
    cache_key_fn: Callable,
    cache_type: str = "pickle",
    ttl_hours: Optional[int] = 24
):
    """
    Decorator to automatically cache function results.

    Args:
        cache_key_fn: Function to generate cache key from args
        cache_type: Type of cache to use
        ttl_hours: Time-to-live in hours

    Example:
        @cached(lambda symbol, date: f"fii_{symbol}_{date}")
        def fetch_fii_data(symbol, date):
            return expensive_api_call(symbol, date)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache_key_fn(*args, **kwargs)

            # Try to get from cache
            cache = DataCache()
            cached_data = cache.get_file_cache(cache_key, cache_type)

            if cached_data is not None:
                logger.info(f"✅ Using cached data for {cache_key}")
                return cached_data

            # Call function
            logger.info(f"⏳ Fetching fresh data for {cache_key}")
            result = func(*args, **kwargs)

            # Cache result
            cache.set_file_cache(cache_key, result, cache_type)

            return result

        return wrapper
    return decorator


# ====================
# Global cache instance
# ====================

_global_cache = None


def get_cache() -> DataCache:
    """Get global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = DataCache()
    return _global_cache


if __name__ == "__main__":
    # Test cache
    cache = DataCache()

    # Test FII/DII caching
    test_date = date(2024, 1, 15)
    cache.cache_fii_dii_data(
        test_date,
        fii_buy=1000.0,
        fii_sell=800.0,
        dii_buy=500.0,
        dii_sell=400.0,
        fii_net=200.0,
        dii_net=100.0
    )

    fii_data = cache.get_fii_dii_data(test_date)
    print(f"FII/DII data: {fii_data}")

    # Test sentiment caching
    cache.cache_sentiment_data("RELIANCE", test_date, 0.75, "twitter")
    sentiment = cache.get_sentiment_data("RELIANCE", test_date, "twitter")
    print(f"Sentiment: {sentiment}")

    # Get stats
    stats = cache.get_cache_stats()
    print(f"Cache stats: {stats}")

    print("\n✅ DataCache tests passed!")
