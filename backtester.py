"""
Vectorized Backtesting Module for Trading Bot
Uses vectorbt library for high-performance backtesting
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import bot components
from Bot_Trading_Swing import EnhancedTradingBot, EnhancedDataManager

class VectorizedBacktester:
    """
    High-performance vectorized backtesting using vectorbt
    """
    
    def __init__(self, symbols, start_date=None, end_date=None):
        self.symbols = symbols
        self.start_date = start_date or (datetime.now() - timedelta(days=365*2))
        self.end_date = end_date or datetime.now()
        
        # Initialize data manager
        self.data_manager = EnhancedDataManager()
        
        # Initialize bot for signal generation
        self.bot = None
        
        # Results storage
        self.results = {}
        self.portfolios = {}
        
        print(f"🚀 VectorizedBacktester initialized for {len(symbols)} symbols")
        print(f"   Period: {self.start_date.date()} to {self.end_date.date()}")
    
    def initialize_bot(self):
        """Initialize the trading bot for signal generation"""
        try:
            self.bot = EnhancedTradingBot()
            print("✅ Trading bot initialized for backtesting")
        except Exception as e:
            print(f"❌ Failed to initialize trading bot: {e}")
            return False
        return True
    
    def fetch_historical_data(self):
        """Fetch historical data for all symbols"""
        print("📊 Fetching historical data...")
        
        all_data = {}
        for symbol in self.symbols:
            try:
                # Fetch data using the bot's data manager
                df = self.data_manager.fetch_multi_timeframe_data(
                    symbol, 
                    count=1000,  # Get enough data for backtesting
                    timeframes_to_use=['D1']
                ).get('D1')
                
                if df is not None and not df.empty:
                    # Filter by date range
                    df.index = pd.to_datetime(df.index)
                    df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
                    
                    if len(df) > 100:  # Ensure sufficient data
                        all_data[symbol] = df
                        print(f"   ✅ {symbol}: {len(df)} candles")
                    else:
                        print(f"   ⚠️ {symbol}: Insufficient data ({len(df)} candles)")
                else:
                    print(f"   ❌ {symbol}: No data available")
                    
            except Exception as e:
                print(f"   ❌ {symbol}: Error fetching data - {e}")
        
        print(f"📊 Data fetch complete: {len(all_data)} symbols with sufficient data")
        return all_data
    
    def generate_signals(self, historical_data):
        """Generate trading signals using the bot's logic"""
        print("🎯 Generating trading signals...")
        
        if not self.bot:
            print("❌ Bot not initialized. Cannot generate signals.")
            return {}
        
        signals = {}
        
        for symbol, df in historical_data.items():
            try:
                print(f"   Processing {symbol}...")
                
                # Create enhanced features
                df_features = self.bot.data_manager.create_enhanced_features(symbol)
                
                if df_features is None or len(df_features) < 100:
                    print(f"   ⚠️ {symbol}: Insufficient features data")
                    continue
                
                # Generate signals for each time point
                entries = []
                exits = []
                
                for i in range(100, len(df_features)):  # Start after 100 periods for stability
                    try:
                        # Get signal using bot's logic
                        signal, confidence, _ = self.bot.get_enhanced_signal(
                            symbol, 
                            df_features=df_features.iloc[:i+1]
                        )
                        
                        if signal == "BUY":
                            entries.append(True)
                            exits.append(False)
                        elif signal == "SELL":
                            entries.append(False)
                            exits.append(True)
                        else:  # HOLD
                            entries.append(False)
                            exits.append(False)
                            
                    except Exception as e:
                        # Default to HOLD on error
                        entries.append(False)
                        exits.append(False)
                
                # Align signals with price data
                if len(entries) > 0:
                    # Create signal series aligned with price data
                    signal_index = df_features.index[100:100+len(entries)]
                    
                    # Align with historical data index
                    entries_series = pd.Series(entries, index=signal_index)
                    exits_series = pd.Series(exits, index=signal_index)
                    
                    # Reindex to match historical data
                    entries_aligned = entries_series.reindex(df.index, fill_value=False)
                    exits_aligned = exits_series.reindex(df.index, fill_value=False)
                    
                    signals[symbol] = {
                        'entries': entries_aligned,
                        'exits': exits_aligned
                    }
                    
                    print(f"   ✅ {symbol}: {sum(entries)} entries, {sum(exits)} exits")
                else:
                    print(f"   ⚠️ {symbol}: No signals generated")
                    
            except Exception as e:
                print(f"   ❌ {symbol}: Error generating signals - {e}")
        
        print(f"🎯 Signal generation complete: {len(signals)} symbols")
        return signals
    
    def run_backtest(self, historical_data, signals):
        """Run vectorized backtest using vectorbt"""
        print("⚡ Running vectorized backtest...")
        
        try:
            # Prepare data for vectorbt
            prices = pd.DataFrame()
            entries = pd.DataFrame()
            exits = pd.DataFrame()
            
            for symbol in signals.keys():
                if symbol in historical_data:
                    df = historical_data[symbol]
                    prices[symbol] = df['close']
                    entries[symbol] = signals[symbol]['entries']
                    exits[symbol] = signals[symbol]['exits']
            
            if prices.empty:
                print("❌ No price data available for backtesting")
                return None
            
            print(f"   📈 Price data: {prices.shape}")
            print(f"   🎯 Entry signals: {entries.sum().sum()} total")
            print(f"   🚪 Exit signals: {exits.sum().sum()} total")
            
            # Create portfolio using vectorbt
            portfolio = vbt.Portfolio.from_signals(
                prices,
                entries,
                exits,
                fees=0.001,  # 0.1% transaction fees
                freq='D'     # Daily frequency
            )
            
            self.portfolios = {symbol: portfolio}
            
            print("✅ Vectorized backtest completed successfully")
            return portfolio
            
        except Exception as e:
            print(f"❌ Backtest failed: {e}")
            return None
    
    def generate_report(self, portfolio):
        """Generate comprehensive backtest report"""
        if portfolio is None:
            print("❌ No portfolio data available for report")
            return
        
        print("\n" + "="*60)
        print("📊 BACKTEST REPORT")
        print("="*60)
        
        try:
            # Basic statistics
            stats = portfolio.stats()
            print("\n📈 PERFORMANCE STATISTICS:")
            print("-" * 40)
            
            key_metrics = [
                'Start', 'End', 'Period', 'Start Value', 'End Value',
                'Total Return [%]', 'Buy & Hold Return [%]', 'Max. Drawdown [%]',
                'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',
                'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]',
                'Avg. Trade [%]', 'Max. Trade Duration', 'Avg. Trade Duration',
                'Trades', 'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]'
            ]
            
            for metric in key_metrics:
                if metric in stats.index:
                    value = stats[metric]
                    if isinstance(value, (int, float)):
                        if 'Return' in metric or 'Rate' in metric or 'Ratio' in metric:
                            print(f"{metric:25}: {value:8.2f}%")
                        else:
                            print(f"{metric:25}: {value:8.2f}")
                    else:
                        print(f"{metric:25}: {value}")
            
            # Trade analysis
            trades = portfolio.trades.records_readable
            if len(trades) > 0:
                print(f"\n💰 TRADE ANALYSIS:")
                print("-" * 40)
                print(f"Total Trades: {len(trades)}")
                print(f"Winning Trades: {len(trades[trades['PnL'] > 0])}")
                print(f"Losing Trades: {len(trades[trades['PnL'] < 0])}")
                print(f"Average PnL: {trades['PnL'].mean():.2f}%")
                print(f"Best Trade: {trades['PnL'].max():.2f}%")
                print(f"Worst Trade: {trades['PnL'].min():.2f}%")
            
            # Risk metrics
            print(f"\n⚠️ RISK METRICS:")
            print("-" * 40)
            returns = portfolio.returns()
            if len(returns) > 0:
                print(f"Volatility: {returns.std() * np.sqrt(252):.2f}%")
                print(f"VaR (95%): {returns.quantile(0.05):.2f}%")
                print(f"CVaR (95%): {returns[returns <= returns.quantile(0.05)].mean():.2f}%")
            
            print("\n" + "="*60)
            
        except Exception as e:
            print(f"❌ Error generating report: {e}")
    
    def plot_results(self, portfolio):
        """Plot backtest results"""
        if portfolio is None:
            print("❌ No portfolio data available for plotting")
            return
        
        try:
            print("📊 Generating plots...")
            
            # Plot portfolio value
            portfolio.plot().show()
            
            # Plot trades
            if len(portfolio.trades.records_readable) > 0:
                portfolio.trades.plot().show()
            
            print("✅ Plots generated successfully")
            
        except Exception as e:
            print(f"❌ Error generating plots: {e}")
    
    def run_full_backtest(self):
        """Run complete backtesting workflow"""
        print("🚀 Starting full backtesting workflow...")
        
        # Step 1: Initialize bot
        if not self.initialize_bot():
            return None
        
        # Step 2: Fetch historical data
        historical_data = self.fetch_historical_data()
        if not historical_data:
            print("❌ No historical data available")
            return None
        
        # Step 3: Generate signals
        signals = self.generate_signals(historical_data)
        if not signals:
            print("❌ No signals generated")
            return None
        
        # Step 4: Run backtest
        portfolio = self.run_backtest(historical_data, signals)
        if portfolio is None:
            return None
        
        # Step 5: Generate report
        self.generate_report(portfolio)
        
        # Step 6: Plot results
        self.plot_results(portfolio)
        
        return portfolio

def main():
    """Main function for running backtests"""
    # Define symbols to test
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
    
    # Create backtester
    backtester = VectorizedBacktester(
        symbols=symbols,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2024, 1, 1)
    )
    
    # Run full backtest
    portfolio = backtester.run_full_backtest()
    
    if portfolio:
        print("🎉 Backtesting completed successfully!")
    else:
        print("❌ Backtesting failed!")

if __name__ == "__main__":
    main()