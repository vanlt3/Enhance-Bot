"""
Enhanced Data Manager - Data management and processing
Refactored from the original Bot-Trading_Swing.py
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

# Import configuration and constants
from Bot_Trading_Swing_Refactored import (
    SYMBOLS, PRIMARY_TIMEFRAME, FeatureConstants, BOT_LOGGERS
)


class EnhancedDataManager:
    """
    Enhanced Data Manager - Handles all data operations
    Refactored to be more modular and maintainable
    """

    def __init__(self, news_manager=None):
        """Initialize the Enhanced Data Manager"""
        self.logger = BOT_LOGGERS['DataManager']
        self.logger.info("📊 [DataManager] Initializing Enhanced Data Manager...")
        
        self.news_manager = news_manager
        self.symbol_data = {}
        self.data_cache = {}
        self.cache_timeout = 300  # 5 minutes
        
        # Initialize data storage
        self._initialize_data_storage()
        
        self.logger.info("✅ [DataManager] Enhanced Data Manager initialized successfully")

    def _initialize_data_storage(self):
        """Initialize data storage structures"""
        try:
            for symbol in SYMBOLS:
                self.symbol_data[symbol] = {
                    'ohlcv': pd.DataFrame(),
                    'indicators': {},
                    'features': {},
                    'last_update': None,
                    'is_stale': False
                }
            
            self.logger.info("✅ [DataManager] Data storage initialized")
            
        except Exception as e:
            self.logger.error(f"❌ [DataManager] Error initializing data storage: {e}")
            raise

    async def update_symbol_data(self, symbol: str) -> bool:
        """Update data for a specific symbol"""
        try:
            self.logger.debug(f"📊 [DataManager] Updating data for {symbol}")
            
            # Check if data is stale
            if self._is_data_stale(symbol):
                self.logger.warning(f"⚠️ [DataManager] Data for {symbol} is stale")
                self.symbol_data[symbol]['is_stale'] = True
            
            # Fetch new data
            new_data = await self._fetch_symbol_data(symbol)
            
            if new_data is not None and not new_data.empty:
                # Update stored data
                self.symbol_data[symbol]['ohlcv'] = new_data
                self.symbol_data[symbol]['last_update'] = datetime.utcnow()
                self.symbol_data[symbol]['is_stale'] = False
                
                # Calculate indicators
                await self._calculate_indicators(symbol)
                
                # Calculate features
                await self._calculate_features(symbol)
                
                self.logger.debug(f"✅ [DataManager] Data updated for {symbol}")
                return True
            else:
                self.logger.warning(f"⚠️ [DataManager] No new data received for {symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ [DataManager] Error updating data for {symbol}: {e}")
            return False

    def _is_data_stale(self, symbol: str) -> bool:
        """Check if data for symbol is stale"""
        try:
            last_update = self.symbol_data[symbol]['last_update']
            if last_update is None:
                return True
            
            time_diff = datetime.utcnow() - last_update
            return time_diff.total_seconds() > self.cache_timeout
            
        except Exception as e:
            self.logger.error(f"❌ [DataManager] Error checking data staleness for {symbol}: {e}")
            return True

    async def _fetch_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch data for a symbol from external source"""
        try:
            # This is a placeholder - in real implementation, you would fetch from API
            # For now, return mock data
            dates = pd.date_range(end=datetime.utcnow(), periods=100, freq='H')
            data = pd.DataFrame({
                'timestamp': dates,
                'open': np.random.randn(100).cumsum() + 100,
                'high': np.random.randn(100).cumsum() + 105,
                'low': np.random.randn(100).cumsum() + 95,
                'close': np.random.randn(100).cumsum() + 100,
                'volume': np.random.randint(1000, 10000, 100)
            })
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ [DataManager] Error fetching data for {symbol}: {e}")
            return None

    async def _calculate_indicators(self, symbol: str):
        """Calculate technical indicators for a symbol"""
        try:
            data = self.symbol_data[symbol]['ohlcv']
            if data.empty:
                return
            
            indicators = {}
            
            # RSI
            indicators['rsi'] = self._calculate_rsi(data['close'])
            
            # MACD
            macd_data = self._calculate_macd(data['close'])
            indicators['macd'] = macd_data['macd']
            indicators['macd_signal'] = macd_data['signal']
            indicators['macd_histogram'] = macd_data['histogram']
            
            # Bollinger Bands
            bb_data = self._calculate_bollinger_bands(data['close'])
            indicators['bb_upper'] = bb_data['upper']
            indicators['bb_middle'] = bb_data['middle']
            indicators['bb_lower'] = bb_data['lower']
            
            # Moving Averages
            for period in FeatureConstants.EMA_PERIODS:
                indicators[f'ema_{period}'] = self._calculate_ema(data['close'], period)
            
            # ATR
            indicators['atr'] = self._calculate_atr(data)
            
            # Store indicators
            self.symbol_data[symbol]['indicators'] = indicators
            
            self.logger.debug(f"✅ [DataManager] Indicators calculated for {symbol}")
            
        except Exception as e:
            self.logger.error(f"❌ [DataManager] Error calculating indicators for {symbol}: {e}")

    def _calculate_rsi(self, prices: pd.Series, period: int = None) -> pd.Series:
        """Calculate RSI indicator"""
        if period is None:
            period = FeatureConstants.RSI_PERIOD
        
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            self.logger.error(f"❌ [DataManager] Error calculating RSI: {e}")
            return pd.Series()

    def _calculate_macd(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate MACD indicator"""
        try:
            ema_fast = self._calculate_ema(prices, FeatureConstants.MACD_FAST)
            ema_slow = self._calculate_ema(prices, FeatureConstants.MACD_SLOW)
            
            macd = ema_fast - ema_slow
            signal = self._calculate_ema(macd, FeatureConstants.MACD_SIGNAL)
            histogram = macd - signal
            
            return {
                'macd': macd,
                'signal': signal,
                'histogram': histogram
            }
            
        except Exception as e:
            self.logger.error(f"❌ [DataManager] Error calculating MACD: {e}")
            return {'macd': pd.Series(), 'signal': pd.Series(), 'histogram': pd.Series()}

    def _calculate_bollinger_bands(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        try:
            period = FeatureConstants.BB_PERIOD
            std = FeatureConstants.BB_STD
            
            middle = prices.rolling(window=period).mean()
            std_dev = prices.rolling(window=period).std()
            
            upper = middle + (std_dev * std)
            lower = middle - (std_dev * std)
            
            return {
                'upper': upper,
                'middle': middle,
                'lower': lower
            }
            
        except Exception as e:
            self.logger.error(f"❌ [DataManager] Error calculating Bollinger Bands: {e}")
            return {'upper': pd.Series(), 'middle': pd.Series(), 'lower': pd.Series()}

    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        try:
            return prices.ewm(span=period).mean()
        except Exception as e:
            self.logger.error(f"❌ [DataManager] Error calculating EMA: {e}")
            return pd.Series()

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            return atr
            
        except Exception as e:
            self.logger.error(f"❌ [DataManager] Error calculating ATR: {e}")
            return pd.Series()

    async def _calculate_features(self, symbol: str):
        """Calculate advanced features for a symbol"""
        try:
            data = self.symbol_data[symbol]['ohlcv']
            indicators = self.symbol_data[symbol]['indicators']
            
            if data.empty or not indicators:
                return
            
            features = {}
            
            # Price-based features
            features['price_change'] = data['close'].pct_change()
            features['price_range'] = (data['high'] - data['low']) / data['close']
            features['volume_change'] = data['volume'].pct_change()
            
            # Technical indicator features
            if 'rsi' in indicators:
                features['rsi_overbought'] = (indicators['rsi'] > 70).astype(int)
                features['rsi_oversold'] = (indicators['rsi'] < 30).astype(int)
            
            if 'macd' in indicators:
                features['macd_bullish'] = (indicators['macd'] > indicators['macd_signal']).astype(int)
                features['macd_bearish'] = (indicators['macd'] < indicators['macd_signal']).astype(int)
            
            # Volatility features
            if 'atr' in indicators:
                features['atr_normalized'] = indicators['atr'] / data['close']
                features['volatility_regime'] = self._classify_volatility_regime(indicators['atr'])
            
            # Trend features
            features['trend_direction'] = self._classify_trend_direction(data['close'])
            
            # Store features
            self.symbol_data[symbol]['features'] = features
            
            self.logger.debug(f"✅ [DataManager] Features calculated for {symbol}")
            
        except Exception as e:
            self.logger.error(f"❌ [DataManager] Error calculating features for {symbol}: {e}")

    def _classify_volatility_regime(self, atr: pd.Series) -> pd.Series:
        """Classify volatility regime based on ATR"""
        try:
            # Simple classification based on ATR percentiles
            low_threshold = atr.quantile(0.33)
            high_threshold = atr.quantile(0.67)
            
            regime = pd.Series(index=atr.index, dtype=int)
            regime[atr <= low_threshold] = 0  # Low volatility
            regime[(atr > low_threshold) & (atr <= high_threshold)] = 1  # Medium volatility
            regime[atr > high_threshold] = 2  # High volatility
            
            return regime
            
        except Exception as e:
            self.logger.error(f"❌ [DataManager] Error classifying volatility regime: {e}")
            return pd.Series()

    def _classify_trend_direction(self, prices: pd.Series) -> pd.Series:
        """Classify trend direction"""
        try:
            # Simple trend classification based on moving averages
            short_ma = prices.rolling(window=20).mean()
            long_ma = prices.rolling(window=50).mean()
            
            trend = pd.Series(index=prices.index, dtype=int)
            trend[short_ma > long_ma] = 1  # Uptrend
            trend[short_ma < long_ma] = -1  # Downtrend
            trend[short_ma == long_ma] = 0  # Sideways
            
            return trend
            
        except Exception as e:
            self.logger.error(f"❌ [DataManager] Error classifying trend direction: {e}")
            return pd.Series()

    async def get_symbol_data(self, symbol: str) -> Dict[str, Any]:
        """Get complete data for a symbol"""
        try:
            if symbol not in self.symbol_data:
                self.logger.warning(f"⚠️ [DataManager] Symbol {symbol} not found in data")
                return {}
            
            return {
                'ohlcv': self.symbol_data[symbol]['ohlcv'],
                'indicators': self.symbol_data[symbol]['indicators'],
                'features': self.symbol_data[symbol]['features'],
                'last_update': self.symbol_data[symbol]['last_update'],
                'is_stale': self.symbol_data[symbol]['is_stale']
            }
            
        except Exception as e:
            self.logger.error(f"❌ [DataManager] Error getting data for {symbol}: {e}")
            return {}

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            if symbol not in self.symbol_data:
                return None
            
            data = self.symbol_data[symbol]['ohlcv']
            if data.empty:
                return None
            
            return float(data['close'].iloc[-1])
            
        except Exception as e:
            self.logger.error(f"❌ [DataManager] Error getting current price for {symbol}: {e}")
            return None

    def get_stale_symbols(self) -> List[str]:
        """Get list of symbols with stale data"""
        try:
            stale_symbols = []
            for symbol, data in self.symbol_data.items():
                if data['is_stale']:
                    stale_symbols.append(symbol)
            
            return stale_symbols
            
        except Exception as e:
            self.logger.error(f"❌ [DataManager] Error getting stale symbols: {e}")
            return []

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of all data"""
        try:
            summary = {}
            for symbol, data in self.symbol_data.items():
                summary[symbol] = {
                    'last_update': data['last_update'],
                    'is_stale': data['is_stale'],
                    'data_points': len(data['ohlcv']),
                    'has_indicators': bool(data['indicators']),
                    'has_features': bool(data['features'])
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ [DataManager] Error getting data summary: {e}")
            return {}