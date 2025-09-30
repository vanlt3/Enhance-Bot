"""
Helper Functions - Utility functions for refactored trading bot
Refactored from the original Bot-Trading_Swing.py
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import asyncio

# Import configuration and constants
from Bot_Trading_Swing_Refactored import (
    BOT_LOGGERS, SYMBOLS, SYMBOL_ALLOCATION, RISK_MANAGEMENT
)


class HelperFunctions:
    """
    Helper Functions - Collection of utility functions
    Refactored to be more modular and maintainable
    """

    def __init__(self):
        """Initialize Helper Functions"""
        self.logger = BOT_LOGGERS['Observability']
        self.logger.info("🔧 [HelperFunctions] Initializing Helper Functions...")

    @staticmethod
    def calculate_pips(price1: float, price2: float, symbol: str) -> float:
        """
        Calculate pips between two prices
        
        Args:
            price1: First price
            price2: Second price
            symbol: Symbol name
            
        Returns:
            Pips difference
        """
        try:
            # Determine pip value based on symbol
            if 'JPY' in symbol:
                pip_value = 0.01
            else:
                pip_value = 0.0001
            
            pips = abs(price1 - price2) / pip_value
            return pips
            
        except Exception as e:
            logging.error(f"Error calculating pips: {e}")
            return 0.0

    @staticmethod
    def calculate_position_size(account_balance: float, risk_percent: float, 
                              stop_loss_pips: float, symbol: str) -> float:
        """
        Calculate position size based on risk management
        
        Args:
            account_balance: Account balance
            risk_percent: Risk percentage per trade
            stop_loss_pips: Stop loss in pips
            symbol: Symbol name
            
        Returns:
            Position size in lots
        """
        try:
            # Calculate risk amount
            risk_amount = account_balance * risk_percent
            
            # Determine pip value
            if 'JPY' in symbol:
                pip_value = 0.01
            else:
                pip_value = 0.0001
            
            # Calculate position size
            if stop_loss_pips > 0:
                position_size = risk_amount / (stop_loss_pips * pip_value)
                return min(position_size, 10.0)  # Max 10 lots
            
            return 0.01  # Default minimum position
            
        except Exception as e:
            logging.error(f"Error calculating position size: {e}")
            return 0.01

    @staticmethod
    def calculate_stop_loss_take_profit(entry_price: float, direction: str, 
                                      atr: float, risk_reward_ratio: float = 2.0) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels
        
        Args:
            entry_price: Entry price
            direction: Trade direction ('buy' or 'sell')
            atr: Average True Range
            risk_reward_ratio: Risk-reward ratio
            
        Returns:
            Tuple of (stop_loss, take_profit)
        """
        try:
            if direction.lower() == 'buy':
                stop_loss = entry_price - (atr * 2.0)
                take_profit = entry_price + (atr * 2.0 * risk_reward_ratio)
            else:  # sell
                stop_loss = entry_price + (atr * 2.0)
                take_profit = entry_price - (atr * 2.0 * risk_reward_ratio)
            
            return stop_loss, take_profit
            
        except Exception as e:
            logging.error(f"Error calculating SL/TP: {e}")
            return entry_price, entry_price

    @staticmethod
    def validate_trading_conditions(symbol: str, market_data: Dict[str, Any]) -> bool:
        """
        Validate trading conditions
        
        Args:
            symbol: Symbol to validate
            market_data: Market data dictionary
            
        Returns:
            True if conditions are valid for trading
        """
        try:
            # Check if symbol is active
            if symbol not in SYMBOL_ALLOCATION:
                return False
            
            # Check if market data is available
            if not market_data or 'ohlcv' not in market_data:
                return False
            
            ohlcv = market_data['ohlcv']
            if ohlcv.empty or len(ohlcv) < 50:
                return False
            
            # Check if market is open (simplified)
            current_time = datetime.utcnow()
            if current_time.weekday() >= 5:  # Weekend
                return False
            
            # Check if price data is recent
            last_price_time = ohlcv.index[-1]
            if (current_time - last_price_time).total_seconds() > 3600:  # 1 hour
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error validating trading conditions: {e}")
            return False

    @staticmethod
    def calculate_technical_indicators(data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators
        
        Args:
            data: OHLCV data
            
        Returns:
            Dictionary of technical indicators
        """
        try:
            indicators = {}
            
            if data.empty or len(data) < 20:
                return indicators
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
            
            # Moving Averages
            indicators['sma_20'] = data['close'].rolling(window=20).mean()
            indicators['sma_50'] = data['close'].rolling(window=50).mean()
            indicators['ema_20'] = data['close'].ewm(span=20).mean()
            indicators['ema_50'] = data['close'].ewm(span=50).mean()
            
            # Bollinger Bands
            sma_20 = indicators['sma_20']
            std_20 = data['close'].rolling(window=20).std()
            indicators['bb_upper'] = sma_20 + (std_20 * 2)
            indicators['bb_lower'] = sma_20 - (std_20 * 2)
            indicators['bb_middle'] = sma_20
            
            # MACD
            ema_12 = data['close'].ewm(span=12).mean()
            ema_26 = data['close'].ewm(span=26).mean()
            indicators['macd'] = ema_12 - ema_26
            indicators['macd_signal'] = indicators['macd'].ewm(span=9).mean()
            indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
            
            # ATR
            high = data['high']
            low = data['low']
            close = data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            indicators['atr'] = true_range.rolling(window=14).mean()
            
            return indicators
            
        except Exception as e:
            logging.error(f"Error calculating technical indicators: {e}")
            return {}

    @staticmethod
    def detect_support_resistance(data: pd.DataFrame, window: int = 20) -> Dict[str, float]:
        """
        Detect support and resistance levels
        
        Args:
            data: OHLCV data
            window: Window for detection
            
        Returns:
            Dictionary with support and resistance levels
        """
        try:
            if data.empty or len(data) < window:
                return {'support': 0, 'resistance': 0}
            
            # Find local minima and maxima
            highs = data['high'].rolling(window=window, center=True).max()
            lows = data['low'].rolling(window=window, center=True).min()
            
            # Identify support and resistance levels
            resistance = highs.max()
            support = lows.min()
            
            return {
                'support': support,
                'resistance': resistance,
                'current_price': data['close'].iloc[-1]
            }
            
        except Exception as e:
            logging.error(f"Error detecting support/resistance: {e}")
            return {'support': 0, 'resistance': 0}

    @staticmethod
    def calculate_volatility(data: pd.DataFrame, window: int = 20) -> float:
        """
        Calculate volatility
        
        Args:
            data: OHLCV data
            window: Window for calculation
            
        Returns:
            Volatility value
        """
        try:
            if data.empty or len(data) < window:
                return 0.0
            
            returns = data['close'].pct_change().dropna()
            volatility = returns.rolling(window=window).std().iloc[-1]
            
            return volatility * np.sqrt(252)  # Annualized
            
        except Exception as e:
            logging.error(f"Error calculating volatility: {e}")
            return 0.0

    @staticmethod
    def calculate_correlation(symbol1_data: pd.DataFrame, symbol2_data: pd.DataFrame) -> float:
        """
        Calculate correlation between two symbols
        
        Args:
            symbol1_data: First symbol data
            symbol2_data: Second symbol data
            
        Returns:
            Correlation coefficient
        """
        try:
            if symbol1_data.empty or symbol2_data.empty:
                return 0.0
            
            # Align data by index
            common_index = symbol1_data.index.intersection(symbol2_data.index)
            if len(common_index) < 10:
                return 0.0
            
            aligned_data1 = symbol1_data.loc[common_index]
            aligned_data2 = symbol2_data.loc[common_index]
            
            # Calculate correlation
            correlation = aligned_data1['close'].corr(aligned_data2['close'])
            
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logging.error(f"Error calculating correlation: {e}")
            return 0.0

    @staticmethod
    def format_currency(amount: float, currency: str = 'USD') -> str:
        """
        Format currency amount
        
        Args:
            amount: Amount to format
            currency: Currency code
            
        Returns:
            Formatted currency string
        """
        try:
            if currency == 'USD':
                return f"${amount:,.2f}"
            elif currency == 'EUR':
                return f"€{amount:,.2f}"
            elif currency == 'GBP':
                return f"£{amount:,.2f}"
            else:
                return f"{amount:,.2f} {currency}"
                
        except Exception as e:
            logging.error(f"Error formatting currency: {e}")
            return str(amount)

    @staticmethod
    def format_percentage(value: float, decimals: int = 2) -> str:
        """
        Format percentage value
        
        Args:
            value: Value to format
            decimals: Number of decimal places
            
        Returns:
            Formatted percentage string
        """
        try:
            return f"{value * 100:.{decimals}f}%"
        except Exception as e:
            logging.error(f"Error formatting percentage: {e}")
            return str(value)

    @staticmethod
    def calculate_drawdown(equity_curve: List[float]) -> Dict[str, float]:
        """
        Calculate drawdown metrics
        
        Args:
            equity_curve: List of equity values
            
        Returns:
            Dictionary with drawdown metrics
        """
        try:
            if not equity_curve or len(equity_curve) < 2:
                return {'max_drawdown': 0.0, 'current_drawdown': 0.0}
            
            equity_array = np.array(equity_curve)
            running_max = np.maximum.accumulate(equity_array)
            drawdowns = (equity_array - running_max) / running_max
            
            max_drawdown = abs(min(drawdowns))
            current_drawdown = abs(drawdowns[-1])
            
            return {
                'max_drawdown': max_drawdown,
                'current_drawdown': current_drawdown
            }
            
        except Exception as e:
            logging.error(f"Error calculating drawdown: {e}")
            return {'max_drawdown': 0.0, 'current_drawdown': 0.0}

    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            returns: List of returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Sharpe ratio
        """
        try:
            if not returns or len(returns) < 2:
                return 0.0
            
            returns_array = np.array(returns)
            excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate
            
            if np.std(excess_returns) == 0:
                return 0.0
            
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            return sharpe_ratio
            
        except Exception as e:
            logging.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    @staticmethod
    def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino ratio
        
        Args:
            returns: List of returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Sortino ratio
        """
        try:
            if not returns or len(returns) < 2:
                return 0.0
            
            returns_array = np.array(returns)
            excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate
            
            # Calculate downside deviation
            negative_returns = excess_returns[excess_returns < 0]
            if len(negative_returns) == 0:
                return 0.0
            
            downside_deviation = np.std(negative_returns)
            if downside_deviation == 0:
                return 0.0
            
            sortino_ratio = np.mean(excess_returns) / downside_deviation * np.sqrt(252)
            return sortino_ratio
            
        except Exception as e:
            logging.error(f"Error calculating Sortino ratio: {e}")
            return 0.0

    @staticmethod
    def calculate_calmar_ratio(returns: List[float], max_drawdown: float) -> float:
        """
        Calculate Calmar ratio
        
        Args:
            returns: List of returns
            max_drawdown: Maximum drawdown
            
        Returns:
            Calmar ratio
        """
        try:
            if not returns or len(returns) < 2 or max_drawdown == 0:
                return 0.0
            
            annual_return = np.mean(returns) * 252
            calmar_ratio = annual_return / max_drawdown
            
            return calmar_ratio
            
        except Exception as e:
            logging.error(f"Error calculating Calmar ratio: {e}")
            return 0.0

    @staticmethod
    def calculate_win_rate(trades: List[Dict[str, Any]]) -> float:
        """
        Calculate win rate
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Win rate as percentage
        """
        try:
            if not trades:
                return 0.0
            
            winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
            total_trades = len(trades)
            
            return winning_trades / total_trades if total_trades > 0 else 0.0
            
        except Exception as e:
            logging.error(f"Error calculating win rate: {e}")
            return 0.0

    @staticmethod
    def calculate_profit_factor(trades: List[Dict[str, Any]]) -> float:
        """
        Calculate profit factor
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Profit factor
        """
        try:
            if not trades:
                return 0.0
            
            total_profit = sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) > 0)
            total_loss = abs(sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) < 0))
            
            return total_profit / total_loss if total_loss > 0 else 0.0
            
        except Exception as e:
            logging.error(f"Error calculating profit factor: {e}")
            return 0.0

    @staticmethod
    def calculate_average_trade(trades: List[Dict[str, Any]]) -> float:
        """
        Calculate average trade P&L
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Average trade P&L
        """
        try:
            if not trades:
                return 0.0
            
            total_pnl = sum(trade.get('pnl', 0) for trade in trades)
            return total_pnl / len(trades)
            
        except Exception as e:
            logging.error(f"Error calculating average trade: {e}")
            return 0.0

    @staticmethod
    def calculate_consecutive_losses(trades: List[Dict[str, Any]]) -> int:
        """
        Calculate consecutive losses
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Number of consecutive losses
        """
        try:
            if not trades:
                return 0
            
            consecutive_losses = 0
            max_consecutive_losses = 0
            
            for trade in trades:
                pnl = trade.get('pnl', 0)
                if pnl < 0:
                    consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                else:
                    consecutive_losses = 0
            
            return max_consecutive_losses
            
        except Exception as e:
            logging.error(f"Error calculating consecutive losses: {e}")
            return 0

    @staticmethod
    def validate_signal_quality(signal: Dict[str, Any]) -> bool:
        """
        Validate signal quality
        
        Args:
            signal: Trading signal dictionary
            
        Returns:
            True if signal is valid
        """
        try:
            required_fields = ['symbol', 'side', 'entry_price', 'stop_loss', 'take_profit', 'confidence']
            
            # Check required fields
            for field in required_fields:
                if field not in signal:
                    return False
            
            # Validate values
            if signal['confidence'] < 0 or signal['confidence'] > 1:
                return False
            
            if signal['entry_price'] <= 0:
                return False
            
            if signal['stop_loss'] <= 0 or signal['take_profit'] <= 0:
                return False
            
            # Validate side
            if signal['side'] not in ['buy', 'sell']:
                return False
            
            # Validate SL/TP logic
            if signal['side'] == 'buy':
                if signal['stop_loss'] >= signal['entry_price'] or signal['take_profit'] <= signal['entry_price']:
                    return False
            else:  # sell
                if signal['stop_loss'] <= signal['entry_price'] or signal['take_profit'] >= signal['entry_price']:
                    return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error validating signal quality: {e}")
            return False

    @staticmethod
    def calculate_risk_reward_ratio(entry_price: float, stop_loss: float, take_profit: float) -> float:
        """
        Calculate risk-reward ratio
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Risk-reward ratio
        """
        try:
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            
            return reward / risk if risk > 0 else 0.0
            
        except Exception as e:
            logging.error(f"Error calculating risk-reward ratio: {e}")
            return 0.0

    @staticmethod
    def calculate_position_value(position_size: float, entry_price: float, current_price: float) -> float:
        """
        Calculate position value
        
        Args:
            position_size: Position size in lots
            entry_price: Entry price
            current_price: Current price
            
        Returns:
            Position value
        """
        try:
            return position_size * current_price
            
        except Exception as e:
            logging.error(f"Error calculating position value: {e}")
            return 0.0

    @staticmethod
    def calculate_unrealized_pnl(position_size: float, entry_price: float, current_price: float, side: str) -> float:
        """
        Calculate unrealized P&L
        
        Args:
            position_size: Position size in lots
            entry_price: Entry price
            current_price: Current price
            side: Position side ('buy' or 'sell')
            
        Returns:
            Unrealized P&L
        """
        try:
            if side.lower() == 'buy':
                pnl = (current_price - entry_price) * position_size
            else:  # sell
                pnl = (entry_price - current_price) * position_size
            
            return pnl
            
        except Exception as e:
            logging.error(f"Error calculating unrealized P&L: {e}")
            return 0.0

    @staticmethod
    def format_timestamp(timestamp: datetime) -> str:
        """
        Format timestamp for display
        
        Args:
            timestamp: Timestamp to format
            
        Returns:
            Formatted timestamp string
        """
        try:
            return timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')
        except Exception as e:
            logging.error(f"Error formatting timestamp: {e}")
            return str(timestamp)

    @staticmethod
    def calculate_time_difference(start_time: datetime, end_time: datetime) -> str:
        """
        Calculate time difference in human-readable format
        
        Args:
            start_time: Start time
            end_time: End time
            
        Returns:
            Human-readable time difference
        """
        try:
            diff = end_time - start_time
            
            if diff.days > 0:
                return f"{diff.days} days, {diff.seconds // 3600} hours"
            elif diff.seconds > 3600:
                return f"{diff.seconds // 3600} hours, {(diff.seconds % 3600) // 60} minutes"
            elif diff.seconds > 60:
                return f"{diff.seconds // 60} minutes, {diff.seconds % 60} seconds"
            else:
                return f"{diff.seconds} seconds"
                
        except Exception as e:
            logging.error(f"Error calculating time difference: {e}")
            return "Unknown"

    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """
        Safe division with default value
        
        Args:
            numerator: Numerator
            denominator: Denominator
            default: Default value if division by zero
            
        Returns:
            Division result or default value
        """
        try:
            return numerator / denominator if denominator != 0 else default
        except Exception as e:
            logging.error(f"Error in safe division: {e}")
            return default

    @staticmethod
    def clamp_value(value: float, min_value: float, max_value: float) -> float:
        """
        Clamp value between min and max
        
        Args:
            value: Value to clamp
            min_value: Minimum value
            max_value: Maximum value
            
        Returns:
            Clamped value
        """
        try:
            return max(min_value, min(value, max_value))
        except Exception as e:
            logging.error(f"Error clamping value: {e}")
            return value

    @staticmethod
    def is_market_open() -> bool:
        """
        Check if market is open (simplified)
        
        Returns:
            True if market is open
        """
        try:
            now = datetime.utcnow()
            
            # Check if it's weekend
            if now.weekday() >= 5:  # Saturday, Sunday
                return False
            
            # Check if it's Friday after market close
            if now.weekday() == 4 and now.hour >= 21:  # Friday after 21:00 UTC
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error checking market status: {e}")
            return False

    @staticmethod
    def get_market_session() -> str:
        """
        Get current market session
        
        Returns:
            Market session name
        """
        try:
            now = datetime.utcnow()
            hour = now.hour
            
            if 0 <= hour < 8:
                return 'asian'
            elif 8 <= hour < 16:
                return 'european'
            elif 16 <= hour < 24:
                return 'american'
            else:
                return 'unknown'
                
        except Exception as e:
            logging.error(f"Error getting market session: {e}")
            return 'unknown'