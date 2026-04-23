"""
Backtesting Module with Realistic Indian Trading Costs

Includes:
- STT (Securities Transaction Tax)
- Brokerage
- Exchange charges (NSE/BSE)
- SEBI turnover charges
- GST on brokerage
- Stamp duty
"""

from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class IndianTradingCosts:
    """
    Realistic Indian trading costs (Intraday + Delivery)
    Based on typical discount broker rates
    """
    # STT (Securities Transaction Tax)
    stt_delivery_buy = 0.0  # No STT on delivery buy
    stt_delivery_sell = 0.001  # 0.1% on delivery sell
    stt_intraday = 0.00025  # 0.025% on intraday (both buy/sell)

    # Brokerage (per trade)
    brokerage_pct = 0.0003  # 0.03% or ₹20 per trade, whichever is lower
    brokerage_max = 20  # ₹20 max per trade

    # Exchange charges
    nse_charges = 0.0000325  # 0.00325% of turnover
    bse_charges = 0.0000375  # 0.00375% of turnover

    # SEBI charges
    sebi_charges = 0.0000001  # ₹10 per crore

    # Stamp duty
    stamp_duty = 0.00015  # 0.015% on buy side (delivery)
    stamp_duty_intraday = 0.00003  # 0.003% on intraday buy

    # GST on brokerage and other charges
    gst_rate = 0.18  # 18% GST

    def calculate_delivery_buy_cost(self, price: float, quantity: int) -> float:
        """Calculate total cost for delivery BUY"""
        turnover = price * quantity

        # Brokerage
        brokerage = min(turnover * self.brokerage_pct, self.brokerage_max)

        # Exchange charges (NSE)
        exchange_charges = turnover * self.nse_charges

        # SEBI charges
        sebi = turnover * self.sebi_charges

        # Stamp duty
        stamp = turnover * self.stamp_duty

        # GST on brokerage + exchange + sebi
        gst = (brokerage + exchange_charges + sebi) * self.gst_rate

        total_cost = brokerage + exchange_charges + sebi + stamp + gst
        return total_cost

    def calculate_delivery_sell_cost(self, price: float, quantity: int) -> float:
        """Calculate total cost for delivery SELL"""
        turnover = price * quantity

        # STT (0.1% on sell)
        stt = turnover * self.stt_delivery_sell

        # Brokerage
        brokerage = min(turnover * self.brokerage_pct, self.brokerage_max)

        # Exchange charges (NSE)
        exchange_charges = turnover * self.nse_charges

        # SEBI charges
        sebi = turnover * self.sebi_charges

        # GST on brokerage + exchange + sebi
        gst = (brokerage + exchange_charges + sebi) * self.gst_rate

        total_cost = stt + brokerage + exchange_charges + sebi + gst
        return total_cost

    def calculate_roundtrip_cost(self, price: float, quantity: int) -> float:
        """Calculate total cost for buy + sell roundtrip"""
        buy_cost = self.calculate_delivery_buy_cost(price, quantity)
        sell_cost = self.calculate_delivery_sell_cost(price, quantity)
        return buy_cost + sell_cost


class Backtester:
    """
    Backtest 5-session stock picker with realistic Indian costs
    """

    def __init__(self, costs: IndianTradingCosts = None):
        self.costs = costs or IndianTradingCosts()

    def backtest_picks(
        self,
        picks: pd.DataFrame,
        stock_data: Dict[str, pd.DataFrame],
        holding_period: int = 5,
        initial_capital: float = 100000
    ) -> Dict:
        """
        Backtest a set of picks

        Args:
            picks: DataFrame with columns ['symbol', 'probability', 'date', 'price']
            stock_data: Dict of symbol -> DataFrame with price data
            holding_period: Number of sessions to hold
            initial_capital: Starting capital in ₹

        Returns:
            Dict with backtest results
        """
        results = []
        capital = initial_capital
        total_trades = 0
        winning_trades = 0
        losing_trades = 0

        for _, row in picks.iterrows():
            symbol = row['symbol']
            entry_date = pd.to_datetime(row['date'])
            entry_price = row['price']
            probability = row['probability']

            if symbol not in stock_data:
                continue

            df = stock_data[symbol]
            df['date'] = pd.to_datetime(df['date'])

            # Find entry point
            entry_idx = df[df['date'] == entry_date].index
            if len(entry_idx) == 0:
                continue

            entry_idx = entry_idx[0]

            # Find exit point (after holding_period sessions)
            if entry_idx + holding_period >= len(df):
                continue

            exit_idx = entry_idx + holding_period
            exit_price = df.loc[exit_idx, 'close']
            exit_date = df.loc[exit_idx, 'date']

            # Calculate position size (equal weight, 1/15 of capital per stock)
            position_value = capital / 15  # Assuming top 15 picks
            quantity = int(position_value / entry_price)

            if quantity == 0:
                continue

            actual_position_value = entry_price * quantity

            # Calculate costs
            buy_cost = self.costs.calculate_delivery_buy_cost(entry_price, quantity)
            sell_cost = self.costs.calculate_delivery_sell_cost(exit_price, quantity)
            total_cost = buy_cost + sell_cost

            # Calculate P&L
            gross_pnl = (exit_price - entry_price) * quantity
            net_pnl = gross_pnl - total_cost

            # Calculate returns
            gross_return_pct = ((exit_price / entry_price) - 1) * 100
            net_return_pct = (net_pnl / actual_position_value) * 100

            # Update stats
            total_trades += 1
            if net_pnl > 0:
                winning_trades += 1
            else:
                losing_trades += 1

            # Store result
            results.append({
                'symbol': symbol,
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': quantity,
                'position_value': actual_position_value,
                'gross_pnl': gross_pnl,
                'total_cost': total_cost,
                'net_pnl': net_pnl,
                'gross_return_%': gross_return_pct,
                'net_return_%': net_return_pct,
                'probability': probability
            })

        # Calculate summary statistics
        results_df = pd.DataFrame(results)

        if len(results_df) == 0:
            return {
                'trades': 0,
                'message': 'No trades executed in backtest period'
            }

        summary = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades * 100 if total_trades > 0 else 0,

            'total_gross_pnl': results_df['gross_pnl'].sum(),
            'total_net_pnl': results_df['net_pnl'].sum(),
            'total_costs': results_df['total_cost'].sum(),

            'avg_gross_return': results_df['gross_return_%'].mean(),
            'avg_net_return': results_df['net_return_%'].mean(),

            'best_trade': results_df['net_return_%'].max(),
            'worst_trade': results_df['net_return_%'].min(),

            'total_return_%': (results_df['net_pnl'].sum() / initial_capital) * 100,

            'results_df': results_df
        }

        return summary

    def display_backtest_results(self, summary: Dict):
        """Display backtest results in a formatted way"""
        if 'message' in summary:
            print(summary['message'])
            return

        print("="*80)
        print("BACKTEST RESULTS - 5-SESSION STOCK PICKER")
        print("="*80)

        print(f"\n📊 Trade Statistics:")
        print(f"   • Total trades: {summary['total_trades']}")
        print(f"   • Winning trades: {summary['winning_trades']}")
        print(f"   • Losing trades: {summary['losing_trades']}")
        print(f"   • Win rate: {summary['win_rate']:.2f}%")

        print(f"\n💰 Returns:")
        print(f"   • Total gross P&L: ₹{summary['total_gross_pnl']:,.2f}")
        print(f"   • Total trading costs: ₹{summary['total_costs']:,.2f}")
        print(f"   • Total net P&L: ₹{summary['total_net_pnl']:,.2f}")
        print(f"   • Total return: {summary['total_return_%']:.2f}%")

        print(f"\n📈 Average Returns:")
        print(f"   • Avg gross return: {summary['avg_gross_return']:.2f}%")
        print(f"   • Avg net return: {summary['avg_net_return']:.2f}%")
        print(f"   • Cost impact: {summary['avg_gross_return'] - summary['avg_net_return']:.2f}%")

        print(f"\n🎯 Best/Worst:")
        print(f"   • Best trade: {summary['best_trade']:.2f}%")
        print(f"   • Worst trade: {summary['worst_trade']:.2f}%")

        print("\n" + "="*80)

        # Show top 10 trades
        results_df = summary['results_df']
        print("\n📋 Top 10 Trades by Net Return:")
        print("-"*80)
        top_10 = results_df.nlargest(10, 'net_return_%')
        for idx, row in top_10.iterrows():
            print(f"{row['symbol']:<15} Entry:₹{row['entry_price']:>8.2f} → Exit:₹{row['exit_price']:>8.2f} | "
                  f"Net Return: {row['net_return_%']:>6.2f}% | P&L:₹{row['net_pnl']:>8,.0f}")

        print("\n📋 Bottom 10 Trades by Net Return:")
        print("-"*80)
        bottom_10 = results_df.nsmallest(10, 'net_return_%')
        for idx, row in bottom_10.iterrows():
            print(f"{row['symbol']:<15} Entry:₹{row['entry_price']:>8.2f} → Exit:₹{row['exit_price']:>8.2f} | "
                  f"Net Return: {row['net_return_%']:>6.2f}% | P&L:₹{row['net_pnl']:>8,.0f}")


def example_usage():
    """Example of how to use the backtester"""
    # Example picks
    picks = pd.DataFrame({
        'symbol': ['RELIANCE.NS', 'TCS.NS', 'INFY.NS'],
        'date': ['2025-01-01', '2025-01-01', '2025-01-01'],
        'price': [2500, 3500, 1500],
        'probability': [0.72, 0.68, 0.65]
    })

    # Example stock data (would come from your data loader)
    stock_data = {
        'RELIANCE.NS': pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=10),
            'close': [2500, 2520, 2510, 2530, 2550, 2570, 2560, 2580, 2590, 2600]
        })
    }

    # Run backtest
    backtester = Backtester()
    results = backtester.backtest_picks(picks, stock_data, holding_period=5)
    backtester.display_backtest_results(results)


if __name__ == '__main__':
    print("Backtesting Module - Realistic Indian Trading Costs")
    print("="*80)
    print("\nCost structure:")
    costs = IndianTradingCosts()
    print(f"  STT (delivery sell): {costs.stt_delivery_sell*100}%")
    print(f"  Brokerage: {costs.brokerage_pct*100}% (max ₹{costs.brokerage_max})")
    print(f"  GST: {costs.gst_rate*100}%")
    print(f"  Stamp duty: {costs.stamp_duty*100}%")
    print("\nExample: ₹100,000 roundtrip at ₹1000/share (100 shares)")
    roundtrip = costs.calculate_roundtrip_cost(1000, 100)
    print(f"  Total cost: ₹{roundtrip:.2f} ({roundtrip/100000*100:.3f}%)")
