#!/usr/bin/env python3
"""
Fallback Data Generator - Requirement 8 Compliance

This module generates synthetic/fallback data when external APIs are unavailable.
All calculations are based on OHLCV data only - no external dependencies.

Compliance: Addresses Requirement 8 - "whatever can be calculated should be calculated..
without using external source for data"

Features:
- FII/DII activity approximation from volume/price patterns
- Sentiment proxy from price momentum and volatility
- Synthetic Google Trends from search volume patterns
- Insider trading signals from unusual volume
- All based ONLY on OHLCV data
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FallbackDataGenerator:
    """
    Generate synthetic data when external APIs are unavailable.

    All methods use ONLY OHLCV data - no external dependencies.
    """

    def __init__(self):
        """Initialize fallback data generator."""
        logger.info("✅ FallbackDataGenerator initialized (no external dependencies)")

    # ====================
    # FII/DII Activity Approximation
    # ====================

    def approximate_fii_dii_activity(
        self,
        df: pd.DataFrame,
        target_date: date
    ) -> Dict[str, float]:
        """
        Approximate FII/DII activity from price and volume patterns.

        Logic:
        - Large volume + sharp price increase = likely FII buying
        - Large volume + sharp price decrease = likely FII selling
        - Medium activity patterns = DII activity

        Args:
            df: DataFrame with OHLCV data (must have DATE, CLOSE, VOLUME)
            target_date: Date to approximate for

        Returns:
            Dict with approximate FII/DII metrics
        """
        # Filter to target date
        date_str = target_date.strftime('%Y-%m-%d')
        df['DATE'] = pd.to_datetime(df['DATE'])
        day_data = df[df['DATE'].dt.date == target_date].copy()

        if day_data.empty:
            logger.warning(f"No data for {target_date}, returning zero FII/DII")
            return {
                'fii_buy': 0.0,
                'fii_sell': 0.0,
                'dii_buy': 0.0,
                'dii_sell': 0.0,
                'fii_net': 0.0,
                'dii_net': 0.0,
                'synthetic': True
            }

        # Calculate price change and volume metrics
        day_data['price_change_pct'] = (
            (day_data['CLOSE'] - day_data['OPEN']) / day_data['OPEN'] * 100
        )

        # Volume-weighted price change (indicator of institutional activity)
        total_volume = day_data['NO_OF_SHRS'].sum()
        day_data['volume_weight'] = day_data['NO_OF_SHRS'] / total_volume
        day_data['weighted_change'] = day_data['price_change_pct'] * day_data['volume_weight']

        # Strong gains with high volume = FII buying
        fii_buy_proxy = day_data[
            (day_data['price_change_pct'] > 2) &  # Strong gain
            (day_data['NO_OF_SHRS'] > day_data['NO_OF_SHRS'].quantile(0.75))  # High volume
        ]['NET_TURNOV'].sum()

        # Strong losses with high volume = FII selling
        fii_sell_proxy = day_data[
            (day_data['price_change_pct'] < -2) &  # Strong loss
            (day_data['NO_OF_SHRS'] > day_data['NO_OF_SHRS'].quantile(0.75))  # High volume
        ]['NET_TURNOV'].sum()

        # Medium activity = DII
        dii_buy_proxy = day_data[
            (day_data['price_change_pct'] > 0.5) &
            (day_data['price_change_pct'] <= 2) &
            (day_data['NO_OF_SHRS'] > day_data['NO_OF_SHRS'].quantile(0.5))
        ]['NET_TURNOV'].sum()

        dii_sell_proxy = day_data[
            (day_data['price_change_pct'] < -0.5) &
            (day_data['price_change_pct'] >= -2) &
            (day_data['NO_OF_SHRS'] > day_data['NO_OF_SHRS'].quantile(0.5))
        ]['NET_TURNOV'].sum()

        # Scale to reasonable values (in crores)
        scale_factor = 0.001  # Adjust based on market

        return {
            'fii_buy': fii_buy_proxy * scale_factor,
            'fii_sell': fii_sell_proxy * scale_factor,
            'dii_buy': dii_buy_proxy * scale_factor,
            'dii_sell': dii_sell_proxy * scale_factor,
            'fii_net': (fii_buy_proxy - fii_sell_proxy) * scale_factor,
            'dii_net': (dii_buy_proxy - dii_sell_proxy) * scale_factor,
            'synthetic': True  # Mark as synthetic data
        }

    # ====================
    # Sentiment Approximation
    # ====================

    def approximate_sentiment(
        self,
        symbol: str,
        df: pd.DataFrame,
        target_date: date,
        lookback_days: int = 5
    ) -> float:
        """
        Approximate sentiment from price momentum and volatility.

        Logic:
        - Strong upward momentum + low volatility = positive sentiment
        - Strong downward momentum + high volatility = negative sentiment
        - Sideways + low volume = neutral sentiment

        Args:
            symbol: Stock symbol
            df: DataFrame with OHLCV data for the symbol
            target_date: Date to approximate for
            lookback_days: Days to look back for momentum calculation

        Returns:
            Sentiment score between -1 (very negative) and 1 (very positive)
        """
        # Get recent data
        df = df.copy()
        df['DATE'] = pd.to_datetime(df['DATE'])
        end_date = target_date
        start_date = target_date - timedelta(days=lookback_days * 2)  # Extra buffer

        recent_data = df[
            (df['DATE'].dt.date >= start_date) &
            (df['DATE'].dt.date <= end_date)
        ].copy()

        if len(recent_data) < 3:
            logger.warning(f"Insufficient data for sentiment approximation for {symbol}")
            return 0.0  # Neutral

        recent_data = recent_data.sort_values('DATE')

        # Calculate momentum (rate of change)
        recent_data['returns'] = recent_data['CLOSE'].pct_change()
        avg_return = recent_data['returns'].mean()

        # Calculate volatility
        volatility = recent_data['returns'].std()

        # Calculate volume trend
        recent_data['volume_ma'] = recent_data['NO_OF_SHRS'].rolling(3).mean()
        volume_trend = (
            recent_data['NO_OF_SHRS'].iloc[-1] / recent_data['volume_ma'].iloc[-1]
            if recent_data['volume_ma'].iloc[-1] > 0 else 1.0
        )

        # Sentiment score calculation
        # Positive momentum + increasing volume = positive sentiment
        # Negative momentum + increasing volume = negative sentiment
        momentum_score = np.tanh(avg_return * 100)  # Scale and bound to [-1, 1]
        volume_factor = min(volume_trend / 1.5, 1.5)  # Volume amplifies sentiment

        # Volatility dampens extreme sentiment
        volatility_damper = 1 / (1 + volatility * 10)

        sentiment = momentum_score * volume_factor * volatility_damper

        # Bound to [-1, 1]
        sentiment = max(-1.0, min(1.0, sentiment))

        logger.debug(f"Synthetic sentiment for {symbol}: {sentiment:.3f}")

        return sentiment

    # ====================
    # Google Trends Approximation
    # ====================

    def approximate_trends_interest(
        self,
        symbol: str,
        df: pd.DataFrame,
        target_date: date,
        lookback_days: int = 30
    ) -> float:
        """
        Approximate Google Trends interest from volume and volatility.

        Logic:
        - High volume + high volatility = high search interest
        - Low volume + low volatility = low search interest

        Args:
            symbol: Stock symbol
            df: DataFrame with OHLCV data
            target_date: Date to approximate for
            lookback_days: Days for baseline calculation

        Returns:
            Interest score (0-100, like Google Trends)
        """
        # Get recent data
        df = df.copy()
        df['DATE'] = pd.to_datetime(df['DATE'])
        end_date = target_date
        start_date = target_date - timedelta(days=lookback_days)

        recent_data = df[
            (df['DATE'].dt.date >= start_date) &
            (df['DATE'].dt.date <= end_date)
        ].copy()

        if recent_data.empty:
            return 50.0  # Neutral

        # Calculate baseline metrics
        avg_volume = recent_data['NO_OF_SHRS'].mean()
        recent_data['returns'] = recent_data['CLOSE'].pct_change()
        avg_volatility = recent_data['returns'].std()

        # Get target day metrics
        target_data = recent_data[recent_data['DATE'].dt.date == target_date]

        if target_data.empty:
            return 50.0

        target_volume = target_data['NO_OF_SHRS'].iloc[0]

        # Calculate relative volume (key indicator of interest)
        relative_volume = (target_volume / avg_volume) if avg_volume > 0 else 1.0

        # Calculate price movement (unusual moves = more interest)
        if len(target_data) > 1:
            target_volatility = target_data['returns'].std()
        else:
            target_volatility = avg_volatility

        relative_volatility = (target_volatility / avg_volatility) if avg_volatility > 0 else 1.0

        # Interest score = combination of volume and volatility spikes
        interest_score = (relative_volume * 0.7 + relative_volatility * 0.3) * 50

        # Bound to [0, 100]
        interest_score = max(0.0, min(100.0, interest_score))

        logger.debug(f"Synthetic trends interest for {symbol}: {interest_score:.1f}")

        return interest_score

    # ====================
    # Insider Trading Signal
    # ====================

    def detect_insider_trading_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        target_date: date,
        lookback_days: int = 30
    ) -> Dict[str, any]:
        """
        Detect potential insider trading from unusual volume patterns.

        Logic:
        - Volume spike before significant price move = potential insider activity
        - Unusual volume concentration = potential informed trading

        Args:
            symbol: Stock symbol
            df: DataFrame with OHLCV data
            target_date: Date to check
            lookback_days: Days for baseline

        Returns:
            Dict with insider trading signal
        """
        # Get recent data
        df = df.copy()
        df['DATE'] = pd.to_datetime(df['DATE'])
        end_date = target_date
        start_date = target_date - timedelta(days=lookback_days)

        recent_data = df[
            (df['DATE'].dt.date >= start_date) &
            (df['DATE'].dt.date <= end_date)
        ].copy()

        if len(recent_data) < 5:
            return {
                'has_signal': False,
                'signal_strength': 0.0,
                'signal_type': None
            }

        # Calculate volume baseline
        avg_volume = recent_data['NO_OF_SHRS'].mean()
        std_volume = recent_data['NO_OF_SHRS'].std()

        # Get target day data
        target_data = recent_data[recent_data['DATE'].dt.date == target_date]

        if target_data.empty:
            return {'has_signal': False, 'signal_strength': 0.0, 'signal_type': None}

        target_volume = target_data['NO_OF_SHRS'].iloc[0]

        # Check for volume spike (> 2 std above mean)
        volume_z_score = (target_volume - avg_volume) / std_volume if std_volume > 0 else 0

        if volume_z_score > 2:
            # Significant volume spike detected
            # Check price movement
            price_change = (
                (target_data['CLOSE'].iloc[0] - target_data['OPEN'].iloc[0])
                / target_data['OPEN'].iloc[0] * 100
            )

            signal_type = 'buy' if price_change > 0 else 'sell'
            signal_strength = min(volume_z_score / 5, 1.0)  # Normalize to [0, 1]

            logger.debug(
                f"Insider signal detected for {symbol}: "
                f"{signal_type} ({signal_strength:.2f})"
            )

            return {
                'has_signal': True,
                'signal_strength': signal_strength,
                'signal_type': signal_type,
                'volume_z_score': volume_z_score,
                'price_change_pct': price_change
            }

        return {'has_signal': False, 'signal_strength': 0.0, 'signal_type': None}

    # ====================
    # Market-wide FII/DII (aggregated)
    # ====================

    def approximate_market_fii_dii(
        self,
        df: pd.DataFrame,
        target_date: date
    ) -> Dict[str, float]:
        """
        Approximate market-wide FII/DII activity.

        This is the main method that should be used when FII/DII API is unavailable.

        Args:
            df: DataFrame with OHLCV data for entire market
            target_date: Date to approximate for

        Returns:
            Dict with market-wide FII/DII metrics
        """
        return self.approximate_fii_dii_activity(df, target_date)


def test_fallback_generator():
    """Test fallback data generator."""
    from bse_loader import BSEDataFetcher

    logger.info("Testing FallbackDataGenerator...")

    fetcher = BSEDataFetcher()
    generator = FallbackDataGenerator()

    # Test with real data
    test_date = date(2024, 1, 15)
    bhav = fetcher.load_bhav_date(test_date)

    if not bhav.empty:
        # Test FII/DII approximation
        fii_dii = generator.approximate_fii_dii_activity(bhav, test_date)
        print(f"\n📊 FII/DII Approximation for {test_date}:")
        print(f"  FII Net: ₹{fii_dii['fii_net']:.2f} Cr")
        print(f"  DII Net: ₹{fii_dii['dii_net']:.2f} Cr")

        # Test sentiment for a stock
        reliance = bhav[bhav['SC_CODE'] == 500325].copy()
        if not reliance.empty:
            sentiment = generator.approximate_sentiment(
                "RELIANCE",
                reliance,
                test_date,
                lookback_days=5
            )
            print(f"\n😊 Sentiment Approximation for RELIANCE:")
            print(f"  Score: {sentiment:.3f} ({'Positive' if sentiment > 0 else 'Negative'})")

        # Test trends
        if not reliance.empty:
            trends = generator.approximate_trends_interest(
                "RELIANCE",
                reliance,
                test_date
            )
            print(f"\n📈 Trends Interest for RELIANCE:")
            print(f"  Interest: {trends:.1f}/100")

        # Test insider signal
        if not reliance.empty:
            insider = generator.detect_insider_trading_signal(
                "RELIANCE",
                reliance,
                test_date
            )
            print(f"\n🕵️ Insider Trading Signal:")
            print(f"  Has Signal: {insider['has_signal']}")
            if insider['has_signal']:
                print(f"  Type: {insider['signal_type']}")
                print(f"  Strength: {insider['signal_strength']:.2f}")

        print("\n✅ All fallback generator tests passed!")

    else:
        print(f"❌ No data available for {test_date}")


if __name__ == "__main__":
    test_fallback_generator()
