#!/usr/bin/env python
"""
Simple backtest demo - shows the concept with minimal data
This demonstrates EXACTLY what you asked for:
"backtest on particular data from previous and see if it closed positive in 5 sessions"
"""

import pandas as pd
from datetime import date, timedelta

print("=" * 80)
print("🎯 BACKTEST DEMO - Does it close positive in 5 sessions?")
print("=" * 80)
print()
print("What this test does:")
print("  1. Simulate running on Oct 1, 2024")
print("  2. Get the picks (stocks system recommends)")
print("  3. Look forward 5 trading sessions")
print("  4. Check: Did they close positive? ✅ or negative? ❌")
print()
print("=" * 80)
print()

# The concept (simplified example to show the idea):
print("EXAMPLE (Simplified to show concept):")
print()
print("Signal Date: 2024-10-01 (Tuesday)")
print("Buy Date: 2024-10-02 (Wednesday morning)")
print("Check Date: 2024-10-08 (Tuesday, 5 sessions later)")
print()

# Demo data structure
demo_picks = pd.DataFrame({
    'Stock': ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICI'],
    'Price_Oct1': [2456.50, 3678.20, 1523.40, 2789.45, 1234.50],
    'Probability': [0.78, 0.72, 0.68, 0.65, 0.62],
    'Price_Oct8': [2534.80, 3612.50, 1567.30, 2812.30, 1198.20],
})

demo_picks['Return_5d'] = ((demo_picks['Price_Oct8'] / demo_picks['Price_Oct1']) - 1) * 100
demo_picks['Outcome'] = demo_picks['Return_5d'].apply(lambda x: '✅ Positive' if x > 0 else '❌ Negative')

print("PICKS from Oct 1:")
print(demo_picks.to_string(index=False))
print()

positive = (demo_picks['Return_5d'] > 0).sum()
total = len(demo_picks)
win_rate = positive / total * 100

print(f"📊 RESULTS:")
print(f"   Total picks: {total}")
print(f"   Positive in 5 sessions: {positive} ✅")
print(f"   Negative in 5 sessions: {total - positive} ❌")
print(f"   Win Rate: {win_rate:.1f}%")
print(f"   Average Return: {demo_picks['Return_5d'].mean():.2f}%")
print()

print("=" * 80)
print("📝 This is exactly what the full backtest does, but with:")
print("   • Real historical NSE/BSE data (not demo data)")
print("   • Multiple signal dates (not just Oct 1)")
print("   • ML model trained on each date (not hardcoded picks)")
print("   • 50-500 stocks (not just 5)")
print("=" * 80)
print()
print("To run the FULL backtest with real data:")
print("   python run_backtest.py --start 2024-10-01 --end 2024-10-30 --stocks 100")
print()
print("⏱️  First run: ~15 min (downloads data)")
print("⏱️  Subsequent runs: ~3 min (uses cached data)")
