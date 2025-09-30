"""
Advanced Feature Engineer - Feature engineering and preprocessing
Refactored from the original Bot-Trading_Swing.py
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Import configuration and constants
from Bot_Trading_Swing_Refactored import (
    FeatureConstants, BOT_LOGGERS, ML_CONFIG
)


class AdvancedFeatureEngineer:
    """
    Advanced Feature Engineer - Handles feature engineering and preprocessing
    Refactored to be more modular and maintainable
    """

    def __init__(self):
        """Initialize the Advanced Feature Engineer"""
        self.logger = BOT_LOGGERS['DataManager']
        self.logger.info("🔧 [FeatureEngineer] Initializing Advanced Feature Engineer...")
        
        # Feature engineering parameters
        self.feature_config = {
            'rsi_periods': [14, 21, 34],
            'macd_periods': [(12, 26, 9), (5, 35, 5)],
            'bb_periods': [20, 50],
            'ema_periods': [20, 50, 100, 200],
            'atr_periods': [14, 21],
            'volume_periods': [20, 50],
            'lookback_periods': [5, 10, 20, 50]
        }
        
        # Scalers for different feature types
        self.scalers = {
            'price': RobustScaler(),
            'volume': StandardScaler(),
            'indicators': StandardScaler(),
            'features': StandardScaler()
        }
        
        # Feature selection
        self.feature_selector = None
        self.selected_features = None
        
        self.logger.info("✅ [FeatureEngineer] Advanced Feature Engineer initialized successfully")

    def engineer_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Engineer features for a given dataset
        
        Args:
            data: OHLCV data
            symbol: Symbol name
            
        Returns:
            DataFrame with engineered features
        """
        try:
            self.logger.debug(f"🔧 [FeatureEngineer] Engineering features for {symbol}")
            
            if data.empty:
                self.logger.warning(f"⚠️ [FeatureEngineer] Empty data for {symbol}")
                return pd.DataFrame()
            
            # Create features DataFrame
            features_df = pd.DataFrame(index=data.index)
            
            # Basic price features
            features_df = self._add_price_features(features_df, data)
            
            # Technical indicators
            features_df = self._add_technical_indicators(features_df, data)
            
            # Volume features
            features_df = self._add_volume_features(features_df, data)
            
            # Volatility features
            features_df = self._add_volatility_features(features_df, data)
            
            # Momentum features
            features_df = self._add_momentum_features(features_df, data)
            
            # Pattern recognition features
            features_df = self._add_pattern_features(features_df, data)
            
            # Time-based features
            features_df = self._add_time_features(features_df, data)
            
            # Market regime features
            features_df = self._add_regime_features(features_df, data)
            
            # Clean and validate features
            features_df = self._clean_features(features_df)
            
            self.logger.debug(f"✅ [FeatureEngineer] Features engineered for {symbol}: {len(features_df.columns)} features")
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"❌ [FeatureEngineer] Error engineering features for {symbol}: {e}")
            return pd.DataFrame()

    def _add_price_features(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-based features"""
        try:
            # Price changes
            features_df['price_change_1'] = data['close'].pct_change(1)
            features_df['price_change_5'] = data['close'].pct_change(5)
            features_df['price_change_10'] = data['close'].pct_change(10)
            features_df['price_change_20'] = data['close'].pct_change(20)
            
            # Price ranges
            features_df['daily_range'] = (data['high'] - data['low']) / data['close']
            features_df['body_size'] = abs(data['close'] - data['open']) / data['close']
            features_df['upper_shadow'] = (data['high'] - np.maximum(data['open'], data['close'])) / data['close']
            features_df['lower_shadow'] = (np.minimum(data['open'], data['close']) - data['low']) / data['close']
            
            # Price position within range
            features_df['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
            
            # Gap features
            features_df['gap_up'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
            features_df['gap_down'] = (data['close'].shift(1) - data['open']) / data['close'].shift(1)
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"❌ [FeatureEngineer] Error adding price features: {e}")
            return features_df

    def _add_technical_indicators(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features"""
        try:
            # RSI variations
            for period in self.feature_config['rsi_periods']:
                rsi = self._calculate_rsi(data['close'], period)
                features_df[f'rsi_{period}'] = rsi
                features_df[f'rsi_{period}_overbought'] = (rsi > 70).astype(int)
                features_df[f'rsi_{period}_oversold'] = (rsi < 30).astype(int)
                features_df[f'rsi_{period}_divergence'] = self._detect_rsi_divergence(data['close'], rsi)
            
            # MACD variations
            for fast, slow, signal in self.feature_config['macd_periods']:
                macd_data = self._calculate_macd(data['close'], fast, slow, signal)
                features_df[f'macd_{fast}_{slow}'] = macd_data['macd']
                features_df[f'macd_signal_{fast}_{slow}'] = macd_data['signal']
                features_df[f'macd_histogram_{fast}_{slow}'] = macd_data['histogram']
                features_df[f'macd_bullish_{fast}_{slow}'] = (macd_data['macd'] > macd_data['signal']).astype(int)
            
            # Bollinger Bands
            for period in self.feature_config['bb_periods']:
                bb_data = self._calculate_bollinger_bands(data['close'], period)
                features_df[f'bb_upper_{period}'] = bb_data['upper']
                features_df[f'bb_middle_{period}'] = bb_data['middle']
                features_df[f'bb_lower_{period}'] = bb_data['lower']
                features_df[f'bb_width_{period}'] = (bb_data['upper'] - bb_data['lower']) / bb_data['middle']
                features_df[f'bb_position_{period}'] = (data['close'] - bb_data['lower']) / (bb_data['upper'] - bb_data['lower'])
                features_df[f'bb_squeeze_{period}'] = (features_df[f'bb_width_{period}'] < features_df[f'bb_width_{period}'].rolling(20).mean()).astype(int)
            
            # Moving Averages
            for period in self.feature_config['ema_periods']:
                ema = self._calculate_ema(data['close'], period)
                features_df[f'ema_{period}'] = ema
                features_df[f'price_vs_ema_{period}'] = (data['close'] - ema) / ema
                features_df[f'ema_slope_{period}'] = ema.diff(5) / ema.shift(5)
            
            # ATR
            for period in self.feature_config['atr_periods']:
                atr = self._calculate_atr(data, period)
                features_df[f'atr_{period}'] = atr
                features_df[f'atr_normalized_{period}'] = atr / data['close']
                features_df[f'atr_ratio_{period}'] = atr / atr.rolling(20).mean()
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"❌ [FeatureEngineer] Error adding technical indicators: {e}")
            return features_df

    def _add_volume_features(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        try:
            # Volume changes
            features_df['volume_change_1'] = data['volume'].pct_change(1)
            features_df['volume_change_5'] = data['volume'].pct_change(5)
            features_df['volume_change_10'] = data['volume'].pct_change(10)
            
            # Volume moving averages
            for period in self.feature_config['volume_periods']:
                vol_ma = data['volume'].rolling(period).mean()
                features_df[f'volume_ma_{period}'] = vol_ma
                features_df[f'volume_ratio_{period}'] = data['volume'] / vol_ma
                features_df[f'volume_spike_{period}'] = (data['volume'] > vol_ma * 2).astype(int)
            
            # Volume-price relationship
            features_df['volume_price_trend'] = (data['volume'] * data['close'].pct_change()).rolling(10).sum()
            features_df['volume_weighted_price'] = (data['volume'] * data['close']).rolling(20).sum() / data['volume'].rolling(20).sum()
            features_df['price_volume_divergence'] = self._detect_price_volume_divergence(data['close'], data['volume'])
            
            # On-Balance Volume
            features_df['obv'] = self._calculate_obv(data)
            features_df['obv_ma'] = features_df['obv'].rolling(20).mean()
            features_df['obv_signal'] = (features_df['obv'] > features_df['obv_ma']).astype(int)
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"❌ [FeatureEngineer] Error adding volume features: {e}")
            return features_df

    def _add_volatility_features(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features"""
        try:
            # Historical volatility
            for period in [10, 20, 50]:
                returns = data['close'].pct_change()
                features_df[f'volatility_{period}'] = returns.rolling(period).std() * np.sqrt(252)
                features_df[f'volatility_ratio_{period}'] = features_df[f'volatility_{period}'] / features_df[f'volatility_{period}'].rolling(50).mean()
            
            # GARCH-like features
            features_df['volatility_clustering'] = self._detect_volatility_clustering(data['close'])
            features_df['volatility_regime'] = self._classify_volatility_regime(features_df['volatility_20'])
            
            # Parkinson volatility
            features_df['parkinson_vol'] = np.sqrt(0.361 * np.log(data['high'] / data['low'])**2)
            
            # Garman-Klass volatility
            features_df['gk_vol'] = np.sqrt(0.5 * np.log(data['high'] / data['low'])**2 - (2*np.log(2)-1) * np.log(data['close'] / data['open'])**2)
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"❌ [FeatureEngineer] Error adding volatility features: {e}")
            return features_df

    def _add_momentum_features(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features"""
        try:
            # Rate of Change
            for period in [5, 10, 20]:
                features_df[f'roc_{period}'] = (data['close'] / data['close'].shift(period) - 1) * 100
            
            # Momentum
            for period in [5, 10, 20]:
                features_df[f'momentum_{period}'] = data['close'] - data['close'].shift(period)
            
            # Stochastic Oscillator
            for period in [14, 21]:
                stoch = self._calculate_stochastic(data, period)
                features_df[f'stoch_k_{period}'] = stoch['k']
                features_df[f'stoch_d_{period}'] = stoch['d']
                features_df[f'stoch_overbought_{period}'] = (stoch['k'] > 80).astype(int)
                features_df[f'stoch_oversold_{period}'] = (stoch['k'] < 20).astype(int)
            
            # Williams %R
            for period in [14, 21]:
                williams_r = self._calculate_williams_r(data, period)
                features_df[f'williams_r_{period}'] = williams_r
            
            # Commodity Channel Index
            for period in [14, 21]:
                cci = self._calculate_cci(data, period)
                features_df[f'cci_{period}'] = cci
                features_df[f'cci_overbought_{period}'] = (cci > 100).astype(int)
                features_df[f'cci_oversold_{period}'] = (cci < -100).astype(int)
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"❌ [FeatureEngineer] Error adding momentum features: {e}")
            return features_df

    def _add_pattern_features(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add pattern recognition features"""
        try:
            # Candlestick patterns
            features_df['doji'] = self._detect_doji(data)
            features_df['hammer'] = self._detect_hammer(data)
            features_df['shooting_star'] = self._detect_shooting_star(data)
            features_df['engulfing'] = self._detect_engulfing(data)
            features_df['harami'] = self._detect_harami(data)
            
            # Chart patterns
            features_df['double_top'] = self._detect_double_top(data)
            features_df['double_bottom'] = self._detect_double_bottom(data)
            features_df['head_shoulders'] = self._detect_head_shoulders(data)
            features_df['triangle'] = self._detect_triangle(data)
            
            # Support and resistance
            features_df['support_level'] = self._find_support_level(data)
            features_df['resistance_level'] = self._find_resistance_level(data)
            features_df['near_support'] = (data['close'] <= features_df['support_level'] * 1.02).astype(int)
            features_df['near_resistance'] = (data['close'] >= features_df['resistance_level'] * 0.98).astype(int)
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"❌ [FeatureEngineer] Error adding pattern features: {e}")
            return features_df

    def _add_time_features(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        try:
            # Extract time components
            features_df['hour'] = data.index.hour
            features_df['day_of_week'] = data.index.dayofweek
            features_df['day_of_month'] = data.index.day
            features_df['month'] = data.index.month
            features_df['quarter'] = data.index.quarter
            
            # Cyclical encoding
            features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
            features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
            features_df['day_sin'] = np.sin(2 * np.pi * features_df['day_of_week'] / 7)
            features_df['day_cos'] = np.cos(2 * np.pi * features_df['day_of_week'] / 7)
            features_df['month_sin'] = np.sin(2 * np.pi * features_df['month'] / 12)
            features_df['month_cos'] = np.cos(2 * np.pi * features_df['month'] / 12)
            
            # Trading session features
            features_df['asian_session'] = ((features_df['hour'] >= 0) & (features_df['hour'] < 8)).astype(int)
            features_df['european_session'] = ((features_df['hour'] >= 8) & (features_df['hour'] < 16)).astype(int)
            features_df['american_session'] = ((features_df['hour'] >= 16) & (features_df['hour'] < 24)).astype(int)
            
            # Weekend effect
            features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
            
            # Month-end effect
            features_df['is_month_end'] = (features_df['day_of_month'] >= 28).astype(int)
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"❌ [FeatureEngineer] Error adding time features: {e}")
            return features_df

    def _add_regime_features(self, features_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add market regime features"""
        try:
            # Trend regime
            features_df['trend_regime'] = self._classify_trend_regime(data['close'])
            
            # Volatility regime
            features_df['volatility_regime'] = self._classify_volatility_regime(features_df.get('volatility_20', pd.Series()))
            
            # Market state
            features_df['market_state'] = self._classify_market_state(data)
            
            # Regime changes
            features_df['trend_regime_change'] = features_df['trend_regime'].diff().fillna(0)
            features_df['volatility_regime_change'] = features_df['volatility_regime'].diff().fillna(0)
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"❌ [FeatureEngineer] Error adding regime features: {e}")
            return features_df

    def _clean_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features"""
        try:
            # Remove infinite values
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            
            # Forward fill missing values
            features_df = features_df.fillna(method='ffill')
            
            # Drop columns with too many missing values
            missing_threshold = 0.5
            features_df = features_df.dropna(axis=1, thresh=int(len(features_df) * (1 - missing_threshold)))
            
            # Remove constant columns
            features_df = features_df.loc[:, (features_df != features_df.iloc[0]).any()]
            
            # Remove highly correlated columns
            features_df = self._remove_correlated_features(features_df)
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"❌ [FeatureEngineer] Error cleaning features: {e}")
            return features_df

    def _remove_correlated_features(self, features_df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """Remove highly correlated features"""
        try:
            # Calculate correlation matrix
            corr_matrix = features_df.corr().abs()
            
            # Find pairs of highly correlated features
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Find features to drop
            to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
            
            # Drop highly correlated features
            features_df = features_df.drop(columns=to_drop)
            
            if to_drop:
                self.logger.info(f"🔧 [FeatureEngineer] Dropped {len(to_drop)} highly correlated features")
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"❌ [FeatureEngineer] Error removing correlated features: {e}")
            return features_df

    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int = None) -> pd.DataFrame:
        """Select top K features using statistical tests"""
        try:
            if k is None:
                k = ML_CONFIG.get('FEATURE_SELECTION_TOP_K', 50)
            
            # Remove non-numeric columns
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            X_numeric = X[numeric_columns]
            
            # Handle missing values
            X_numeric = X_numeric.fillna(X_numeric.median())
            
            # Select features
            self.feature_selector = SelectKBest(score_func=f_classif, k=min(k, len(X_numeric.columns)))
            X_selected = self.feature_selector.fit_transform(X_numeric, y)
            
            # Get selected feature names
            self.selected_features = X_numeric.columns[self.feature_selector.get_support()].tolist()
            
            # Create DataFrame with selected features
            X_selected_df = pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
            
            self.logger.info(f"🔧 [FeatureEngineer] Selected {len(self.selected_features)} features out of {len(X_numeric.columns)}")
            
            return X_selected_df
            
        except Exception as e:
            self.logger.error(f"❌ [FeatureEngineer] Error selecting features: {e}")
            return X

    def transform_features(self, X: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """Transform features using scalers"""
        try:
            X_transformed = X.copy()
            
            # Define feature groups
            price_features = [col for col in X.columns if any(keyword in col.lower() for keyword in ['price', 'close', 'open', 'high', 'low'])]
            volume_features = [col for col in X.columns if 'volume' in col.lower()]
            indicator_features = [col for col in X.columns if any(keyword in col.lower() for keyword in ['rsi', 'macd', 'bb', 'ema', 'atr'])]
            other_features = [col for col in X.columns if col not in price_features + volume_features + indicator_features]
            
            # Scale features
            for feature_group, features in [('price', price_features), ('volume', volume_features), 
                                          ('indicators', indicator_features), ('features', other_features)]:
                if features:
                    if fit_scaler:
                        X_transformed[features] = self.scalers[feature_group].fit_transform(X[features])
                    else:
                        X_transformed[features] = self.scalers[feature_group].transform(X[features])
            
            return X_transformed
            
        except Exception as e:
            self.logger.error(f"❌ [FeatureEngineer] Error transforming features: {e}")
            return X

    # Helper methods for technical indicators
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series()

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal).mean()
            histogram = macd - signal_line
            return {'macd': macd, 'signal': signal_line, 'histogram': histogram}
        except:
            return {'macd': pd.Series(), 'signal': pd.Series(), 'histogram': pd.Series()}

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        try:
            middle = prices.rolling(window=period).mean()
            std_dev = prices.rolling(window=period).std()
            upper = middle + (std_dev * std)
            lower = middle - (std_dev * std)
            return {'upper': upper, 'middle': middle, 'lower': lower}
        except:
            return {'upper': pd.Series(), 'middle': pd.Series(), 'lower': pd.Series()}

    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate EMA"""
        try:
            return prices.ewm(span=period).mean()
        except:
            return pd.Series()

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR"""
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
        except:
            return pd.Series()

    def _calculate_stochastic(self, data: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator"""
        try:
            low_min = data['low'].rolling(window=period).min()
            high_max = data['high'].rolling(window=period).max()
            k_percent = 100 * ((data['close'] - low_min) / (high_max - low_min))
            d_percent = k_percent.rolling(window=3).mean()
            return {'k': k_percent, 'd': d_percent}
        except:
            return {'k': pd.Series(), 'd': pd.Series()}

    def _calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        try:
            high_max = data['high'].rolling(window=period).max()
            low_min = data['low'].rolling(window=period).min()
            williams_r = -100 * ((high_max - data['close']) / (high_max - low_min))
            return williams_r
        except:
            return pd.Series()

    def _calculate_cci(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Commodity Channel Index"""
        try:
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            sma = typical_price.rolling(window=period).mean()
            mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (typical_price - sma) / (0.015 * mad)
            return cci
        except:
            return pd.Series()

    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        try:
            obv = pd.Series(index=data.index, dtype=float)
            obv.iloc[0] = data['volume'].iloc[0]
            
            for i in range(1, len(data)):
                if data['close'].iloc[i] > data['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + data['volume'].iloc[i]
                elif data['close'].iloc[i] < data['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - data['volume'].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            return obv
        except:
            return pd.Series()

    # Pattern detection methods (simplified implementations)
    def _detect_doji(self, data: pd.DataFrame) -> pd.Series:
        """Detect Doji candlestick pattern"""
        try:
            body_size = abs(data['close'] - data['open'])
            range_size = data['high'] - data['low']
            return (body_size <= range_size * 0.1).astype(int)
        except:
            return pd.Series()

    def _detect_hammer(self, data: pd.DataFrame) -> pd.Series:
        """Detect Hammer candlestick pattern"""
        try:
            body_size = abs(data['close'] - data['open'])
            lower_shadow = np.minimum(data['open'], data['close']) - data['low']
            upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
            return ((lower_shadow > body_size * 2) & (upper_shadow < body_size * 0.5)).astype(int)
        except:
            return pd.Series()

    def _detect_shooting_star(self, data: pd.DataFrame) -> pd.Series:
        """Detect Shooting Star candlestick pattern"""
        try:
            body_size = abs(data['close'] - data['open'])
            lower_shadow = np.minimum(data['open'], data['close']) - data['low']
            upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
            return ((upper_shadow > body_size * 2) & (lower_shadow < body_size * 0.5)).astype(int)
        except:
            return pd.Series()

    def _detect_engulfing(self, data: pd.DataFrame) -> pd.Series:
        """Detect Engulfing candlestick pattern"""
        try:
            prev_body = abs(data['close'].shift(1) - data['open'].shift(1))
            curr_body = abs(data['close'] - data['open'])
            prev_bullish = data['close'].shift(1) > data['open'].shift(1)
            curr_bullish = data['close'] > data['open']
            
            bullish_engulfing = (~prev_bullish & curr_bullish & 
                               (data['open'] < data['close'].shift(1)) & 
                               (data['close'] > data['open'].shift(1)))
            
            bearish_engulfing = (prev_bullish & ~curr_bullish & 
                               (data['open'] > data['close'].shift(1)) & 
                               (data['close'] < data['open'].shift(1)))
            
            return (bullish_engulfing | bearish_engulfing).astype(int)
        except:
            return pd.Series()

    def _detect_harami(self, data: pd.DataFrame) -> pd.Series:
        """Detect Harami candlestick pattern"""
        try:
            prev_body = abs(data['close'].shift(1) - data['open'].shift(1))
            curr_body = abs(data['close'] - data['open'])
            prev_bullish = data['close'].shift(1) > data['open'].shift(1)
            curr_bullish = data['close'] > data['open']
            
            bullish_harami = (prev_bullish & ~curr_bullish & 
                            (data['open'] > data['close'].shift(1)) & 
                            (data['close'] < data['open'].shift(1)) &
                            (curr_body < prev_body * 0.5))
            
            bearish_harami = (~prev_bullish & curr_bullish & 
                            (data['open'] < data['close'].shift(1)) & 
                            (data['close'] > data['open'].shift(1)) &
                            (curr_body < prev_body * 0.5))
            
            return (bullish_harami | bearish_harami).astype(int)
        except:
            return pd.Series()

    def _detect_double_top(self, data: pd.DataFrame) -> pd.Series:
        """Detect Double Top pattern"""
        try:
            # Simplified implementation
            highs = data['high'].rolling(window=20).max()
            return (data['high'] == highs).astype(int)
        except:
            return pd.Series()

    def _detect_double_bottom(self, data: pd.DataFrame) -> pd.Series:
        """Detect Double Bottom pattern"""
        try:
            # Simplified implementation
            lows = data['low'].rolling(window=20).min()
            return (data['low'] == lows).astype(int)
        except:
            return pd.Series()

    def _detect_head_shoulders(self, data: pd.DataFrame) -> pd.Series:
        """Detect Head and Shoulders pattern"""
        try:
            # Simplified implementation
            return pd.Series(0, index=data.index)
        except:
            return pd.Series()

    def _detect_triangle(self, data: pd.DataFrame) -> pd.Series:
        """Detect Triangle pattern"""
        try:
            # Simplified implementation
            return pd.Series(0, index=data.index)
        except:
            return pd.Series()

    def _find_support_level(self, data: pd.DataFrame) -> pd.Series:
        """Find support levels"""
        try:
            return data['low'].rolling(window=20).min()
        except:
            return pd.Series()

    def _find_resistance_level(self, data: pd.DataFrame) -> pd.Series:
        """Find resistance levels"""
        try:
            return data['high'].rolling(window=20).max()
        except:
            return pd.Series()

    def _detect_rsi_divergence(self, prices: pd.Series, rsi: pd.Series) -> pd.Series:
        """Detect RSI divergence"""
        try:
            # Simplified implementation
            return pd.Series(0, index=prices.index)
        except:
            return pd.Series()

    def _detect_price_volume_divergence(self, prices: pd.Series, volume: pd.Series) -> pd.Series:
        """Detect price-volume divergence"""
        try:
            # Simplified implementation
            return pd.Series(0, index=prices.index)
        except:
            return pd.Series()

    def _detect_volatility_clustering(self, prices: pd.Series) -> pd.Series:
        """Detect volatility clustering"""
        try:
            returns = prices.pct_change()
            volatility = returns.rolling(window=20).std()
            return (volatility > volatility.rolling(window=50).mean() * 1.5).astype(int)
        except:
            return pd.Series()

    def _classify_volatility_regime(self, volatility: pd.Series) -> pd.Series:
        """Classify volatility regime"""
        try:
            if volatility.empty:
                return pd.Series()
            
            low_threshold = volatility.quantile(0.33)
            high_threshold = volatility.quantile(0.67)
            
            regime = pd.Series(index=volatility.index, dtype=int)
            regime[volatility <= low_threshold] = 0  # Low volatility
            regime[(volatility > low_threshold) & (volatility <= high_threshold)] = 1  # Medium volatility
            regime[volatility > high_threshold] = 2  # High volatility
            
            return regime
        except:
            return pd.Series()

    def _classify_trend_regime(self, prices: pd.Series) -> pd.Series:
        """Classify trend regime"""
        try:
            short_ma = prices.rolling(window=20).mean()
            long_ma = prices.rolling(window=50).mean()
            
            regime = pd.Series(index=prices.index, dtype=int)
            regime[short_ma > long_ma * 1.02] = 1  # Strong uptrend
            regime[(short_ma > long_ma) & (short_ma <= long_ma * 1.02)] = 0  # Weak uptrend
            regime[short_ma < long_ma * 0.98] = -1  # Strong downtrend
            regime[(short_ma < long_ma) & (short_ma >= long_ma * 0.98)] = 0  # Weak downtrend
            regime[short_ma == long_ma] = 0  # Sideways
            
            return regime
        except:
            return pd.Series()

    def _classify_market_state(self, data: pd.DataFrame) -> pd.Series:
        """Classify market state"""
        try:
            # Simplified implementation
            return pd.Series(0, index=data.index)
        except:
            return pd.Series()