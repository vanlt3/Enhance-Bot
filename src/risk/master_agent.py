"""
Master Agent - Risk management and decision coordination
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
    BOT_LOGGERS, SYMBOL_ALLOCATION, RISK_MANAGEMENT, 
    TradingConstants, FeatureConstants
)


class MasterAgent:
    """
    Master Agent - Coordinates trading decisions and risk management
    Refactored to be more modular and maintainable
    """

    def __init__(self, news_manager=None):
        """Initialize the Master Agent"""
        self.logger = BOT_LOGGERS['RiskManager']
        self.news_manager = news_manager
        self.logger.info("🎯 [MasterAgent] Initializing Master Agent...")
        
        # Risk management parameters
        self.risk_config = {
            'max_risk_per_trade': RISK_MANAGEMENT['MAX_RISK_PER_TRADE'],
            'max_portfolio_risk': RISK_MANAGEMENT['MAX_PORTFOLIO_RISK'],
            'max_open_positions': RISK_MANAGEMENT['MAX_OPEN_POSITIONS'],
            'volatility_lookback': RISK_MANAGEMENT['VOLATILITY_LOOKBACK'],
            'sl_atr_multiplier': RISK_MANAGEMENT['SL_ATR_MULTIPLIER'],
            'base_rr_ratio': RISK_MANAGEMENT['BASE_RR_RATIO']
        }
        
        # Decision thresholds
        self.decision_thresholds = {
            'min_confidence': TradingConstants.MIN_CONFIDENCE_THRESHOLD,
            'max_confidence': TradingConstants.MAX_CONFIDENCE_THRESHOLD,
            'min_volume_ratio': 1.2,
            'max_spread_ratio': 0.001,
            'min_atr_ratio': 0.5
        }
        
        # Performance tracking
        self.decision_history = []
        self.performance_metrics = {
            'total_decisions': 0,
            'successful_decisions': 0,
            'failed_decisions': 0,
            'avg_confidence': 0.0,
            'risk_adjusted_returns': 0.0
        }
        
        # Market regime detection
        self.current_market_regime = 'neutral'
        self.regime_history = []
        
        self.logger.info("✅ [MasterAgent] Master Agent initialized successfully")

    async def get_trading_signal(self, symbol: str, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get trading signal for a symbol
        
        Args:
            symbol: Symbol to analyze
            market_data: Market data for the symbol
            
        Returns:
            Trading signal dictionary or None
        """
        try:
            self.logger.debug(f"🎯 [MasterAgent] Analyzing {symbol} for trading signal...")
            
            # Validate input data
            if not self._validate_market_data(market_data):
                self.logger.warning(f"⚠️ [MasterAgent] Invalid market data for {symbol}")
                return None
            
            # Analyze market conditions
            market_analysis = await self._analyze_market_conditions(symbol, market_data)
            
            # Check risk conditions
            risk_assessment = await self._assess_risk_conditions(symbol, market_data)
            
            # Generate signal
            signal = await self._generate_trading_signal(symbol, market_data, market_analysis, risk_assessment)
            
            # Log decision
            if signal:
                self._log_decision(symbol, signal, market_analysis, risk_assessment)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ [MasterAgent] Error getting trading signal for {symbol}: {e}")
            return None

    def _validate_market_data(self, market_data: Dict[str, Any]) -> bool:
        """Validate market data structure and content"""
        try:
            required_keys = ['ohlcv', 'indicators', 'features']
            
            for key in required_keys:
                if key not in market_data:
                    return False
            
            # Check if data is not empty
            ohlcv = market_data['ohlcv']
            if ohlcv.empty or len(ohlcv) < 50:
                return False
            
            # Check if indicators are available
            indicators = market_data['indicators']
            if not indicators:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ [MasterAgent] Error validating market data: {e}")
            return False

    async def _analyze_market_conditions(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current market conditions"""
        try:
            ohlcv = market_data['ohlcv']
            indicators = market_data['indicators']
            features = market_data['features']
            
            analysis = {
                'trend_direction': self._analyze_trend(ohlcv, indicators),
                'volatility_regime': self._analyze_volatility(ohlcv, indicators),
                'momentum_strength': self._analyze_momentum(indicators),
                'volume_analysis': self._analyze_volume(ohlcv, features),
                'support_resistance': self._analyze_support_resistance(ohlcv),
                'market_regime': self._detect_market_regime(ohlcv, indicators)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ [MasterAgent] Error analyzing market conditions: {e}")
            return {}

    def _analyze_trend(self, ohlcv: pd.DataFrame, indicators: Dict[str, Any]) -> str:
        """Analyze trend direction"""
        try:
            if ohlcv.empty:
                return 'neutral'
            
            # Get recent prices
            recent_prices = ohlcv['close'].tail(20)
            
            # Calculate trend indicators
            price_change = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
            
            # Check moving averages
            if 'ema_20' in indicators and 'ema_50' in indicators:
                ema_20 = indicators['ema_20'].iloc[-1] if not indicators['ema_20'].empty else 0
                ema_50 = indicators['ema_50'].iloc[-1] if not indicators['ema_50'].empty else 0
                
                if ema_20 > ema_50 * 1.02:
                    return 'bullish'
                elif ema_20 < ema_50 * 0.98:
                    return 'bearish'
            
            # Price-based trend
            if price_change > 0.02:
                return 'bullish'
            elif price_change < -0.02:
                return 'bearish'
            
            return 'neutral'
            
        except Exception as e:
            self.logger.error(f"❌ [MasterAgent] Error analyzing trend: {e}")
            return 'neutral'

    def _analyze_volatility(self, ohlcv: pd.DataFrame, indicators: Dict[str, Any]) -> str:
        """Analyze volatility regime"""
        try:
            if ohlcv.empty:
                return 'medium'
            
            # Calculate recent volatility
            recent_returns = ohlcv['close'].pct_change().tail(20)
            volatility = recent_returns.std() * np.sqrt(252)
            
            # Use ATR if available
            if 'atr' in indicators and not indicators['atr'].empty:
                atr = indicators['atr'].iloc[-1]
                current_price = ohlcv['close'].iloc[-1]
                atr_ratio = atr / current_price
                
                if atr_ratio > 0.03:
                    return 'high'
                elif atr_ratio < 0.01:
                    return 'low'
            
            # Volatility-based classification
            if volatility > 0.25:
                return 'high'
            elif volatility < 0.10:
                return 'low'
            
            return 'medium'
            
        except Exception as e:
            self.logger.error(f"❌ [MasterAgent] Error analyzing volatility: {e}")
            return 'medium'

    def _analyze_momentum(self, indicators: Dict[str, Any]) -> str:
        """Analyze momentum strength"""
        try:
            momentum_score = 0
            
            # RSI analysis
            if 'rsi' in indicators and not indicators['rsi'].empty:
                rsi = indicators['rsi'].iloc[-1]
                if rsi > 70:
                    momentum_score -= 1
                elif rsi < 30:
                    momentum_score += 1
            
            # MACD analysis
            if 'macd' in indicators and 'macd_signal' in indicators:
                macd = indicators['macd'].iloc[-1] if not indicators['macd'].empty else 0
                macd_signal = indicators['macd_signal'].iloc[-1] if not indicators['macd_signal'].empty else 0
                
                if macd > macd_signal:
                    momentum_score += 1
                else:
                    momentum_score -= 1
            
            # Classify momentum
            if momentum_score > 0:
                return 'bullish'
            elif momentum_score < 0:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            self.logger.error(f"❌ [MasterAgent] Error analyzing momentum: {e}")
            return 'neutral'

    def _analyze_volume(self, ohlcv: pd.DataFrame, features: Dict[str, Any]) -> str:
        """Analyze volume conditions"""
        try:
            if ohlcv.empty or 'volume' not in ohlcv.columns:
                return 'normal'
            
            # Calculate volume ratio
            recent_volume = ohlcv['volume'].tail(20)
            avg_volume = recent_volume.mean()
            current_volume = recent_volume.iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Classify volume
            if volume_ratio > 1.5:
                return 'high'
            elif volume_ratio < 0.5:
                return 'low'
            else:
                return 'normal'
                
        except Exception as e:
            self.logger.error(f"❌ [MasterAgent] Error analyzing volume: {e}")
            return 'normal'

    def _analyze_support_resistance(self, ohlcv: pd.DataFrame) -> Dict[str, float]:
        """Analyze support and resistance levels"""
        try:
            if ohlcv.empty:
                return {'support': 0, 'resistance': 0, 'current_position': 0}
            
            # Calculate support and resistance
            recent_highs = ohlcv['high'].tail(50)
            recent_lows = ohlcv['low'].tail(50)
            current_price = ohlcv['close'].iloc[-1]
            
            resistance = recent_highs.max()
            support = recent_lows.min()
            
            # Calculate position within range
            if resistance > support:
                position = (current_price - support) / (resistance - support)
            else:
                position = 0.5
            
            return {
                'support': support,
                'resistance': resistance,
                'current_position': position
            }
            
        except Exception as e:
            self.logger.error(f"❌ [MasterAgent] Error analyzing support/resistance: {e}")
            return {'support': 0, 'resistance': 0, 'current_position': 0}

    def _detect_market_regime(self, ohlcv: pd.DataFrame, indicators: Dict[str, Any]) -> str:
        """Detect current market regime"""
        try:
            # Combine multiple indicators
            trend = self._analyze_trend(ohlcv, indicators)
            volatility = self._analyze_volatility(ohlcv, indicators)
            momentum = self._analyze_momentum(indicators)
            
            # Regime classification
            if trend == 'bullish' and momentum == 'bullish' and volatility == 'medium':
                return 'trending_bull'
            elif trend == 'bearish' and momentum == 'bearish' and volatility == 'medium':
                return 'trending_bear'
            elif volatility == 'high':
                return 'volatile'
            elif volatility == 'low' and trend == 'neutral':
                return 'ranging'
            else:
                return 'mixed'
                
        except Exception as e:
            self.logger.error(f"❌ [MasterAgent] Error detecting market regime: {e}")
            return 'unknown'

    async def _assess_risk_conditions(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk conditions for trading"""
        try:
            assessment = {
                'risk_level': 'medium',
                'position_size_multiplier': 1.0,
                'stop_loss_multiplier': 1.0,
                'take_profit_multiplier': 1.0,
                'risk_factors': []
            }
            
            # Check symbol allocation
            symbol_config = SYMBOL_ALLOCATION.get(symbol, {})
            max_exposure = symbol_config.get('max_exposure', 0.1)
            risk_multiplier = symbol_config.get('risk_multiplier', 1.0)
            
            # Adjust risk based on symbol characteristics
            assessment['position_size_multiplier'] *= risk_multiplier
            
            # Check volatility
            ohlcv = market_data['ohlcv']
            if not ohlcv.empty:
                recent_returns = ohlcv['close'].pct_change().tail(20)
                volatility = recent_returns.std()
                
                if volatility > 0.02:  # High volatility
                    assessment['risk_level'] = 'high'
                    assessment['stop_loss_multiplier'] *= 1.5
                    assessment['risk_factors'].append('high_volatility')
                elif volatility < 0.005:  # Low volatility
                    assessment['risk_level'] = 'low'
                    assessment['stop_loss_multiplier'] *= 0.8
                    assessment['risk_factors'].append('low_volatility')
            
            # Check market regime
            market_regime = self._detect_market_regime(ohlcv, market_data['indicators'])
            if market_regime == 'volatile':
                assessment['risk_level'] = 'high'
                assessment['position_size_multiplier'] *= 0.7
                assessment['risk_factors'].append('volatile_market')
            elif market_regime == 'ranging':
                assessment['risk_level'] = 'low'
                assessment['position_size_multiplier'] *= 1.2
                assessment['risk_factors'].append('ranging_market')
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"❌ [MasterAgent] Error assessing risk conditions: {e}")
            return {'risk_level': 'high', 'position_size_multiplier': 0.5, 'risk_factors': ['error']}

    async def _generate_trading_signal(self, symbol: str, market_data: Dict[str, Any], 
                                     market_analysis: Dict[str, Any], risk_assessment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate trading signal based on analysis"""
        try:
            # Check if risk conditions allow trading
            if risk_assessment['risk_level'] == 'high' and len(risk_assessment['risk_factors']) > 2:
                self.logger.debug(f"🚫 [MasterAgent] Risk too high for {symbol}")
                return None
            
            # Get current price and indicators
            ohlcv = market_data['ohlcv']
            indicators = market_data['indicators']
            
            if ohlcv.empty:
                return None
            
            current_price = ohlcv['close'].iloc[-1]
            
            # Calculate signal strength
            signal_strength = self._calculate_signal_strength(market_analysis, indicators)
            
            # Check minimum confidence threshold
            if signal_strength < self.decision_thresholds['min_confidence']:
                return None
            
            # Determine signal direction
            signal_direction = self._determine_signal_direction(market_analysis, indicators)
            
            if signal_direction == 'neutral':
                return None
            
            # Calculate entry, stop loss, and take profit
            entry_price = current_price
            stop_loss, take_profit = self._calculate_sl_tp(
                symbol, entry_price, signal_direction, market_data, risk_assessment
            )
            
            # Calculate position size
            position_size = self._calculate_position_size(
                symbol, entry_price, stop_loss, risk_assessment
            )
            
            # Create signal
            signal = {
                'symbol': symbol,
                'side': signal_direction,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size,
                'confidence': signal_strength,
                'risk_level': risk_assessment['risk_level'],
                'market_regime': market_analysis['market_regime'],
                'timestamp': datetime.utcnow(),
                'trailing_stop': False,
                'risk_factors': risk_assessment['risk_factors']
            }
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ [MasterAgent] Error generating trading signal: {e}")
            return None

    def _calculate_signal_strength(self, market_analysis: Dict[str, Any], indicators: Dict[str, Any]) -> float:
        """Calculate signal strength/confidence"""
        try:
            confidence = 0.5  # Base confidence
            
            # Trend strength
            trend = market_analysis.get('trend_direction', 'neutral')
            if trend == 'bullish':
                confidence += 0.2
            elif trend == 'bearish':
                confidence += 0.2
            
            # Momentum strength
            momentum = market_analysis.get('momentum_strength', 'neutral')
            if momentum == 'bullish':
                confidence += 0.15
            elif momentum == 'bearish':
                confidence += 0.15
            
            # Volume confirmation
            volume = market_analysis.get('volume_analysis', 'normal')
            if volume == 'high':
                confidence += 0.1
            
            # RSI confirmation
            if 'rsi' in indicators and not indicators['rsi'].empty:
                rsi = indicators['rsi'].iloc[-1]
                if 30 < rsi < 70:  # Not overbought/oversold
                    confidence += 0.05
            
            # MACD confirmation
            if 'macd' in indicators and 'macd_signal' in indicators:
                macd = indicators['macd'].iloc[-1] if not indicators['macd'].empty else 0
                macd_signal = indicators['macd_signal'].iloc[-1] if not indicators['macd_signal'].empty else 0
                
                if (macd > macd_signal and trend == 'bullish') or (macd < macd_signal and trend == 'bearish'):
                    confidence += 0.1
            
            return min(confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"❌ [MasterAgent] Error calculating signal strength: {e}")
            return 0.0

    def _determine_signal_direction(self, market_analysis: Dict[str, Any], indicators: Dict[str, Any]) -> str:
        """Determine signal direction"""
        try:
            trend = market_analysis.get('trend_direction', 'neutral')
            momentum = market_analysis.get('momentum_strength', 'neutral')
            volume = market_analysis.get('volume_analysis', 'normal')
            
            # Strong bullish signal
            if trend == 'bullish' and momentum == 'bullish' and volume == 'high':
                return 'buy'
            
            # Strong bearish signal
            if trend == 'bearish' and momentum == 'bearish' and volume == 'high':
                return 'sell'
            
            # Moderate signals
            if trend == 'bullish' and momentum == 'bullish':
                return 'buy'
            elif trend == 'bearish' and momentum == 'bearish':
                return 'sell'
            
            # RSI-based signals
            if 'rsi' in indicators and not indicators['rsi'].empty:
                rsi = indicators['rsi'].iloc[-1]
                if rsi < 30 and trend == 'bullish':
                    return 'buy'
                elif rsi > 70 and trend == 'bearish':
                    return 'sell'
            
            return 'neutral'
            
        except Exception as e:
            self.logger.error(f"❌ [MasterAgent] Error determining signal direction: {e}")
            return 'neutral'

    def _calculate_sl_tp(self, symbol: str, entry_price: float, signal_direction: str, 
                        market_data: Dict[str, Any], risk_assessment: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        try:
            ohlcv = market_data['ohlcv']
            indicators = market_data['indicators']
            
            # Get ATR for volatility-based SL/TP
            atr = 0.01  # Default 1%
            if 'atr' in indicators and not indicators['atr'].empty:
                atr = indicators['atr'].iloc[-1] / entry_price
            
            # Apply risk multipliers
            sl_multiplier = risk_assessment.get('stop_loss_multiplier', 1.0)
            tp_multiplier = risk_assessment.get('take_profit_multiplier', 1.0)
            
            # Calculate SL and TP
            if signal_direction == 'buy':
                stop_loss = entry_price * (1 - atr * self.risk_config['sl_atr_multiplier'] * sl_multiplier)
                take_profit = entry_price * (1 + atr * self.risk_config['sl_atr_multiplier'] * self.risk_config['base_rr_ratio'] * tp_multiplier)
            else:  # sell
                stop_loss = entry_price * (1 + atr * self.risk_config['sl_atr_multiplier'] * sl_multiplier)
                take_profit = entry_price * (1 - atr * self.risk_config['sl_atr_multiplier'] * self.risk_config['base_rr_ratio'] * tp_multiplier)
            
            return stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"❌ [MasterAgent] Error calculating SL/TP: {e}")
            # Return conservative SL/TP
            if signal_direction == 'buy':
                return entry_price * 0.98, entry_price * 1.04
            else:
                return entry_price * 1.02, entry_price * 0.96

    def _calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float, 
                               risk_assessment: Dict[str, Any]) -> float:
        """Calculate position size based on risk management"""
        try:
            # Get symbol allocation
            symbol_config = SYMBOL_ALLOCATION.get(symbol, {})
            max_exposure = symbol_config.get('max_exposure', 0.1)
            
            # Calculate risk per trade
            risk_per_trade = self.risk_config['max_risk_per_trade']
            position_size_multiplier = risk_assessment.get('position_size_multiplier', 1.0)
            
            # Calculate stop loss distance
            stop_loss_distance = abs(entry_price - stop_loss) / entry_price
            
            if stop_loss_distance > 0:
                # Calculate position size
                position_size = (risk_per_trade * max_exposure * position_size_multiplier) / stop_loss_distance
                return min(position_size, max_exposure)
            
            return 0.01  # Default small position
            
        except Exception as e:
            self.logger.error(f"❌ [MasterAgent] Error calculating position size: {e}")
            return 0.01

    def _log_decision(self, symbol: str, signal: Dict[str, Any], market_analysis: Dict[str, Any], risk_assessment: Dict[str, Any]):
        """Log trading decision"""
        try:
            decision = {
                'timestamp': datetime.utcnow(),
                'symbol': symbol,
                'signal': signal,
                'market_analysis': market_analysis,
                'risk_assessment': risk_assessment
            }
            
            self.decision_history.append(decision)
            
            # Update performance metrics
            self.performance_metrics['total_decisions'] += 1
            self.performance_metrics['avg_confidence'] = (
                (self.performance_metrics['avg_confidence'] * (self.performance_metrics['total_decisions'] - 1) + signal['confidence']) /
                self.performance_metrics['total_decisions']
            )
            
            self.logger.info(f"🎯 [MasterAgent] Signal generated for {symbol}: {signal['side']} at {signal['entry_price']:.4f} (confidence: {signal['confidence']:.3f})")
            
        except Exception as e:
            self.logger.error(f"❌ [MasterAgent] Error logging decision: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        try:
            return {
                'total_decisions': self.performance_metrics['total_decisions'],
                'avg_confidence': self.performance_metrics['avg_confidence'],
                'current_market_regime': self.current_market_regime,
                'recent_decisions': len([d for d in self.decision_history if (datetime.utcnow() - d['timestamp']).days < 7])
            }
            
        except Exception as e:
            self.logger.error(f"❌ [MasterAgent] Error getting performance summary: {e}")
            return {}