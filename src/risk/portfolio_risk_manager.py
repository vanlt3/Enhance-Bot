"""
Portfolio Risk Manager - Portfolio-level risk management
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
    TradingConstants, SYMBOLS
)


class PortfolioRiskManager:
    """
    Portfolio Risk Manager - Handles portfolio-level risk management
    Refactored to be more modular and maintainable
    """

    def __init__(self):
        """Initialize the Portfolio Risk Manager"""
        self.logger = BOT_LOGGERS['RiskManager']
        self.logger.info("🛡️ [PortfolioRiskManager] Initializing Portfolio Risk Manager...")
        
        # Risk management parameters
        self.risk_config = {
            'max_risk_per_trade': RISK_MANAGEMENT['MAX_RISK_PER_TRADE'],
            'max_portfolio_risk': RISK_MANAGEMENT['MAX_PORTFOLIO_RISK'],
            'max_open_positions': RISK_MANAGEMENT['MAX_OPEN_POSITIONS'],
            'volatility_lookback': RISK_MANAGEMENT['VOLATILITY_LOOKBACK'],
            'sl_atr_multiplier': RISK_MANAGEMENT['SL_ATR_MULTIPLIER'],
            'base_rr_ratio': RISK_MANAGEMENT['BASE_RR_RATIO']
        }
        
        # Portfolio state
        self.open_positions = {}
        self.portfolio_value = 10000.0  # Initial portfolio value
        self.cash_balance = 10000.0
        self.total_exposure = 0.0
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        
        # Risk metrics
        self.risk_metrics = {
            'var_95': 0.0,
            'expected_shortfall': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'calmar_ratio': 0.0,
            'portfolio_beta': 0.0,
            'correlation_risk': 0.0
        }
        
        # Performance tracking
        self.performance_history = []
        self.risk_alerts = []
        self.circuit_breaker_active = False
        
        # Data manager reference
        self.data_manager = None
        
        self.logger.info("✅ [PortfolioRiskManager] Portfolio Risk Manager initialized successfully")

    def set_data_manager(self, data_manager):
        """Set data manager reference"""
        self.data_manager = data_manager

    async def assess_portfolio_risk(self) -> Dict[str, Any]:
        """
        Assess overall portfolio risk
        
        Returns:
            Risk assessment dictionary
        """
        try:
            self.logger.debug("🛡️ [PortfolioRiskManager] Assessing portfolio risk...")
            
            # Calculate current portfolio metrics
            portfolio_metrics = await self._calculate_portfolio_metrics()
            
            # Assess concentration risk
            concentration_risk = self._assess_concentration_risk()
            
            # Assess correlation risk
            correlation_risk = await self._assess_correlation_risk()
            
            # Assess volatility risk
            volatility_risk = await self._assess_volatility_risk()
            
            # Assess liquidity risk
            liquidity_risk = self._assess_liquidity_risk()
            
            # Calculate overall risk score
            overall_risk_score = self._calculate_overall_risk_score(
                portfolio_metrics, concentration_risk, correlation_risk, 
                volatility_risk, liquidity_risk
            )
            
            # Generate risk assessment
            risk_assessment = {
                'overall_risk_score': overall_risk_score,
                'risk_level': self._classify_risk_level(overall_risk_score),
                'portfolio_metrics': portfolio_metrics,
                'concentration_risk': concentration_risk,
                'correlation_risk': correlation_risk,
                'volatility_risk': volatility_risk,
                'liquidity_risk': liquidity_risk,
                'risk_alerts': self._generate_risk_alerts(overall_risk_score),
                'recommendations': self._generate_risk_recommendations(overall_risk_score),
                'timestamp': datetime.utcnow()
            }
            
            # Check for circuit breaker conditions
            if overall_risk_score > 0.8:
                await self._activate_circuit_breaker(risk_assessment)
            
            return risk_assessment
            
        except Exception as e:
            self.logger.error(f"❌ [PortfolioRiskManager] Error assessing portfolio risk: {e}")
            return {'overall_risk_score': 1.0, 'risk_level': 'critical', 'error': str(e)}

    async def _calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """Calculate portfolio performance metrics"""
        try:
            # Calculate current portfolio value
            current_value = self.cash_balance
            total_exposure = 0.0
            
            for symbol, position in self.open_positions.items():
                if self.data_manager:
                    current_price = await self.data_manager.get_current_price(symbol)
                    if current_price:
                        position_value = position['size'] * current_price
                        current_value += position_value
                        total_exposure += abs(position_value)
            
            # Update portfolio value
            self.portfolio_value = current_value
            self.total_exposure = total_exposure
            
            # Calculate returns
            if len(self.performance_history) > 0:
                initial_value = self.performance_history[0]['portfolio_value']
                total_return = (current_value - initial_value) / initial_value
            else:
                total_return = 0.0
            
            # Calculate daily return
            if len(self.performance_history) > 0:
                previous_value = self.performance_history[-1]['portfolio_value']
                daily_return = (current_value - previous_value) / previous_value
            else:
                daily_return = 0.0
            
            # Calculate risk metrics
            var_95 = self._calculate_var_95()
            expected_shortfall = self._calculate_expected_shortfall()
            max_drawdown = self._calculate_max_drawdown()
            sharpe_ratio = self._calculate_sharpe_ratio()
            calmar_ratio = self._calculate_calmar_ratio()
            
            metrics = {
                'portfolio_value': current_value,
                'cash_balance': self.cash_balance,
                'total_exposure': total_exposure,
                'exposure_ratio': total_exposure / current_value if current_value > 0 else 0,
                'total_return': total_return,
                'daily_return': daily_return,
                'var_95': var_95,
                'expected_shortfall': expected_shortfall,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'calmar_ratio': calmar_ratio,
                'num_positions': len(self.open_positions)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ [PortfolioRiskManager] Error calculating portfolio metrics: {e}")
            return {}

    def _assess_concentration_risk(self) -> Dict[str, Any]:
        """Assess concentration risk in portfolio"""
        try:
            if not self.open_positions:
                return {'risk_score': 0.0, 'risk_level': 'low', 'details': 'No positions'}
            
            # Calculate position sizes
            position_sizes = []
            total_value = 0.0
            
            for symbol, position in self.open_positions.items():
                position_value = position['size'] * position.get('current_price', 0)
                position_sizes.append(position_value)
                total_value += position_value
            
            if total_value == 0:
                return {'risk_score': 0.0, 'risk_level': 'low', 'details': 'No position value'}
            
            # Calculate concentration metrics
            position_weights = [size / total_value for size in position_sizes]
            max_weight = max(position_weights)
            herfindahl_index = sum(w**2 for w in position_weights)
            
            # Assess risk
            if max_weight > 0.3 or herfindahl_index > 0.25:
                risk_level = 'high'
                risk_score = 0.8
            elif max_weight > 0.2 or herfindahl_index > 0.15:
                risk_level = 'medium'
                risk_score = 0.5
            else:
                risk_level = 'low'
                risk_score = 0.2
            
            return {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'max_position_weight': max_weight,
                'herfindahl_index': herfindahl_index,
                'num_positions': len(self.open_positions),
                'details': f'Max position: {max_weight:.1%}, HHI: {herfindahl_index:.3f}'
            }
            
        except Exception as e:
            self.logger.error(f"❌ [PortfolioRiskManager] Error assessing concentration risk: {e}")
            return {'risk_score': 0.5, 'risk_level': 'medium', 'details': 'Error in calculation'}

    async def _assess_correlation_risk(self) -> Dict[str, Any]:
        """Assess correlation risk between positions"""
        try:
            if len(self.open_positions) < 2:
                return {'risk_score': 0.0, 'risk_level': 'low', 'details': 'Insufficient positions'}
            
            # Get price data for correlation analysis
            if not self.data_manager:
                return {'risk_score': 0.5, 'risk_level': 'medium', 'details': 'No data manager'}
            
            # Calculate correlations between positions
            correlations = []
            symbols = list(self.open_positions.keys())
            
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    try:
                        symbol1, symbol2 = symbols[i], symbols[j]
                        data1 = await self.data_manager.get_symbol_data(symbol1)
                        data2 = await self.data_manager.get_symbol_data(symbol2)
                        
                        if data1 and data2 and 'ohlcv' in data1 and 'ohlcv' in data2:
                            prices1 = data1['ohlcv']['close'].tail(50)
                            prices2 = data2['ohlcv']['close'].tail(50)
                            
                            if len(prices1) == len(prices2) and len(prices1) > 10:
                                correlation = prices1.corr(prices2)
                                if not np.isnan(correlation):
                                    correlations.append(abs(correlation))
                    except Exception as e:
                        self.logger.warning(f"⚠️ [PortfolioRiskManager] Error calculating correlation: {e}")
                        continue
            
            if not correlations:
                return {'risk_score': 0.5, 'risk_level': 'medium', 'details': 'Unable to calculate correlations'}
            
            # Assess correlation risk
            avg_correlation = np.mean(correlations)
            max_correlation = max(correlations)
            
            if avg_correlation > 0.7 or max_correlation > 0.9:
                risk_level = 'high'
                risk_score = 0.8
            elif avg_correlation > 0.5 or max_correlation > 0.7:
                risk_level = 'medium'
                risk_score = 0.5
            else:
                risk_level = 'low'
                risk_score = 0.2
            
            return {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'avg_correlation': avg_correlation,
                'max_correlation': max_correlation,
                'num_correlations': len(correlations),
                'details': f'Avg correlation: {avg_correlation:.3f}, Max: {max_correlation:.3f}'
            }
            
        except Exception as e:
            self.logger.error(f"❌ [PortfolioRiskManager] Error assessing correlation risk: {e}")
            return {'risk_score': 0.5, 'risk_level': 'medium', 'details': 'Error in calculation'}

    async def _assess_volatility_risk(self) -> Dict[str, Any]:
        """Assess volatility risk in portfolio"""
        try:
            if not self.open_positions:
                return {'risk_score': 0.0, 'risk_level': 'low', 'details': 'No positions'}
            
            # Calculate portfolio volatility
            if not self.data_manager:
                return {'risk_score': 0.5, 'risk_level': 'medium', 'details': 'No data manager'}
            
            # Get returns for each position
            returns = []
            for symbol, position in self.open_positions.items():
                try:
                    data = await self.data_manager.get_symbol_data(symbol)
                    if data and 'ohlcv' in data:
                        prices = data['ohlcv']['close'].tail(50)
                        if len(prices) > 10:
                            position_returns = prices.pct_change().dropna()
                            returns.extend(position_returns.tolist())
                except Exception as e:
                    self.logger.warning(f"⚠️ [PortfolioRiskManager] Error getting returns for {symbol}: {e}")
                    continue
            
            if not returns:
                return {'risk_score': 0.5, 'risk_level': 'medium', 'details': 'Unable to calculate returns'}
            
            # Calculate volatility metrics
            returns_array = np.array(returns)
            volatility = np.std(returns_array) * np.sqrt(252)  # Annualized
            var_95 = np.percentile(returns_array, 5)
            expected_shortfall = np.mean(returns_array[returns_array <= var_95])
            
            # Assess volatility risk
            if volatility > 0.3 or var_95 < -0.05:
                risk_level = 'high'
                risk_score = 0.8
            elif volatility > 0.2 or var_95 < -0.03:
                risk_level = 'medium'
                risk_score = 0.5
            else:
                risk_level = 'low'
                risk_score = 0.2
            
            return {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'volatility': volatility,
                'var_95': var_95,
                'expected_shortfall': expected_shortfall,
                'details': f'Volatility: {volatility:.1%}, VaR 95%: {var_95:.1%}'
            }
            
        except Exception as e:
            self.logger.error(f"❌ [PortfolioRiskManager] Error assessing volatility risk: {e}")
            return {'risk_score': 0.5, 'risk_level': 'medium', 'details': 'Error in calculation'}

    def _assess_liquidity_risk(self) -> Dict[str, Any]:
        """Assess liquidity risk in portfolio"""
        try:
            if not self.open_positions:
                return {'risk_score': 0.0, 'risk_level': 'low', 'details': 'No positions'}
            
            # Calculate liquidity metrics
            total_position_value = sum(
                position['size'] * position.get('current_price', 0) 
                for position in self.open_positions.values()
            )
            
            liquidity_ratio = self.cash_balance / (self.cash_balance + total_position_value)
            
            # Assess liquidity risk
            if liquidity_ratio < 0.1:
                risk_level = 'high'
                risk_score = 0.8
            elif liquidity_ratio < 0.2:
                risk_level = 'medium'
                risk_score = 0.5
            else:
                risk_level = 'low'
                risk_score = 0.2
            
            return {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'liquidity_ratio': liquidity_ratio,
                'cash_balance': self.cash_balance,
                'total_position_value': total_position_value,
                'details': f'Liquidity ratio: {liquidity_ratio:.1%}'
            }
            
        except Exception as e:
            self.logger.error(f"❌ [PortfolioRiskManager] Error assessing liquidity risk: {e}")
            return {'risk_score': 0.5, 'risk_level': 'medium', 'details': 'Error in calculation'}

    def _calculate_overall_risk_score(self, portfolio_metrics: Dict[str, Any], 
                                    concentration_risk: Dict[str, Any], 
                                    correlation_risk: Dict[str, Any], 
                                    volatility_risk: Dict[str, Any], 
                                    liquidity_risk: Dict[str, Any]) -> float:
        """Calculate overall portfolio risk score"""
        try:
            # Weighted average of risk components
            weights = {
                'concentration': 0.25,
                'correlation': 0.25,
                'volatility': 0.30,
                'liquidity': 0.20
            }
            
            overall_score = (
                concentration_risk.get('risk_score', 0.5) * weights['concentration'] +
                correlation_risk.get('risk_score', 0.5) * weights['correlation'] +
                volatility_risk.get('risk_score', 0.5) * weights['volatility'] +
                liquidity_risk.get('risk_score', 0.5) * weights['liquidity']
            )
            
            # Adjust for portfolio size
            num_positions = portfolio_metrics.get('num_positions', 0)
            if num_positions > self.risk_config['max_open_positions']:
                overall_score += 0.2
            
            # Adjust for exposure ratio
            exposure_ratio = portfolio_metrics.get('exposure_ratio', 0)
            if exposure_ratio > 0.8:
                overall_score += 0.2
            
            return min(overall_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"❌ [PortfolioRiskManager] Error calculating overall risk score: {e}")
            return 0.5

    def _classify_risk_level(self, risk_score: float) -> str:
        """Classify risk level based on score"""
        if risk_score >= 0.8:
            return 'critical'
        elif risk_score >= 0.6:
            return 'high'
        elif risk_score >= 0.4:
            return 'medium'
        elif risk_score >= 0.2:
            return 'low'
        else:
            return 'minimal'

    def _generate_risk_alerts(self, risk_score: float) -> List[str]:
        """Generate risk alerts based on risk score"""
        alerts = []
        
        if risk_score >= 0.8:
            alerts.append("CRITICAL: Portfolio risk is extremely high")
            alerts.append("Consider reducing positions immediately")
        elif risk_score >= 0.6:
            alerts.append("HIGH: Portfolio risk is elevated")
            alerts.append("Monitor positions closely")
        elif risk_score >= 0.4:
            alerts.append("MEDIUM: Portfolio risk is moderate")
        
        return alerts

    def _generate_risk_recommendations(self, risk_score: float) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        if risk_score >= 0.8:
            recommendations.append("Reduce position sizes")
            recommendations.append("Increase cash allocation")
            recommendations.append("Consider hedging strategies")
        elif risk_score >= 0.6:
            recommendations.append("Monitor correlation between positions")
            recommendations.append("Consider position diversification")
        elif risk_score >= 0.4:
            recommendations.append("Maintain current risk levels")
            recommendations.append("Continue monitoring")
        
        return recommendations

    async def _activate_circuit_breaker(self, risk_assessment: Dict[str, Any]):
        """Activate circuit breaker for high risk"""
        try:
            self.circuit_breaker_active = True
            
            alert = {
                'timestamp': datetime.utcnow(),
                'type': 'circuit_breaker',
                'message': 'Circuit breaker activated due to high portfolio risk',
                'risk_score': risk_assessment['overall_risk_score'],
                'details': risk_assessment
            }
            
            self.risk_alerts.append(alert)
            
            self.logger.warning("🚨 [PortfolioRiskManager] Circuit breaker activated!")
            
            # Here you would implement actual circuit breaker actions
            # such as closing positions, reducing exposure, etc.
            
        except Exception as e:
            self.logger.error(f"❌ [PortfolioRiskManager] Error activating circuit breaker: {e}")

    def _calculate_var_95(self) -> float:
        """Calculate 95% Value at Risk"""
        try:
            if len(self.performance_history) < 20:
                return 0.0
            
            # Get recent returns
            returns = []
            for i in range(1, min(len(self.performance_history), 50)):
                prev_value = self.performance_history[i-1]['portfolio_value']
                curr_value = self.performance_history[i]['portfolio_value']
                if prev_value > 0:
                    returns.append((curr_value - prev_value) / prev_value)
            
            if not returns:
                return 0.0
            
            # Calculate VaR
            var_95 = np.percentile(returns, 5)
            return abs(var_95)
            
        except Exception as e:
            self.logger.error(f"❌ [PortfolioRiskManager] Error calculating VaR: {e}")
            return 0.0

    def _calculate_expected_shortfall(self) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        try:
            if len(self.performance_history) < 20:
                return 0.0
            
            # Get recent returns
            returns = []
            for i in range(1, min(len(self.performance_history), 50)):
                prev_value = self.performance_history[i-1]['portfolio_value']
                curr_value = self.performance_history[i]['portfolio_value']
                if prev_value > 0:
                    returns.append((curr_value - prev_value) / prev_value)
            
            if not returns:
                return 0.0
            
            # Calculate Expected Shortfall
            var_95 = np.percentile(returns, 5)
            tail_returns = [r for r in returns if r <= var_95]
            
            if tail_returns:
                expected_shortfall = np.mean(tail_returns)
                return abs(expected_shortfall)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"❌ [PortfolioRiskManager] Error calculating Expected Shortfall: {e}")
            return 0.0

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        try:
            if len(self.performance_history) < 2:
                return 0.0
            
            # Get portfolio values
            values = [entry['portfolio_value'] for entry in self.performance_history]
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(values)
            
            # Calculate drawdowns
            drawdowns = (values - running_max) / running_max
            
            # Return maximum drawdown
            return abs(min(drawdowns))
            
        except Exception as e:
            self.logger.error(f"❌ [PortfolioRiskManager] Error calculating max drawdown: {e}")
            return 0.0

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(self.performance_history) < 20:
                return 0.0
            
            # Get returns
            returns = []
            for i in range(1, len(self.performance_history)):
                prev_value = self.performance_history[i-1]['portfolio_value']
                curr_value = self.performance_history[i]['portfolio_value']
                if prev_value > 0:
                    returns.append((curr_value - prev_value) / prev_value)
            
            if not returns:
                return 0.0
            
            # Calculate Sharpe ratio
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return > 0:
                sharpe_ratio = mean_return / std_return * np.sqrt(252)  # Annualized
                return sharpe_ratio
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"❌ [PortfolioRiskManager] Error calculating Sharpe ratio: {e}")
            return 0.0

    def _calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio"""
        try:
            if len(self.performance_history) < 20:
                return 0.0
            
            # Get returns
            returns = []
            for i in range(1, len(self.performance_history)):
                prev_value = self.performance_history[i-1]['portfolio_value']
                curr_value = self.performance_history[i]['portfolio_value']
                if prev_value > 0:
                    returns.append((curr_value - prev_value) / prev_value)
            
            if not returns:
                return 0.0
            
            # Calculate Calmar ratio
            mean_return = np.mean(returns) * 252  # Annualized
            max_drawdown = self._calculate_max_drawdown()
            
            if max_drawdown > 0:
                calmar_ratio = mean_return / max_drawdown
                return calmar_ratio
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"❌ [PortfolioRiskManager] Error calculating Calmar ratio: {e}")
            return 0.0

    async def add_position(self, symbol: str, position: Dict[str, Any]) -> bool:
        """Add a new position to the portfolio"""
        try:
            # Check if we can add the position
            if len(self.open_positions) >= self.risk_config['max_open_positions']:
                self.logger.warning(f"⚠️ [PortfolioRiskManager] Cannot add position for {symbol}: maximum positions reached")
                return False
            
            # Check exposure limits
            position_value = position['size'] * position['entry_price']
            if position_value > self.cash_balance:
                self.logger.warning(f"⚠️ [PortfolioRiskManager] Cannot add position for {symbol}: insufficient cash")
                return False
            
            # Add position
            self.open_positions[symbol] = position
            self.cash_balance -= position_value
            
            self.logger.info(f"✅ [PortfolioRiskManager] Added position for {symbol}: {position['size']} units at {position['entry_price']}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ [PortfolioRiskManager] Error adding position for {symbol}: {e}")
            return False

    async def remove_position(self, symbol: str) -> bool:
        """Remove a position from the portfolio"""
        try:
            if symbol not in self.open_positions:
                self.logger.warning(f"⚠️ [PortfolioRiskManager] Position for {symbol} not found")
                return False
            
            position = self.open_positions[symbol]
            
            # Calculate P&L
            if self.data_manager:
                current_price = await self.data_manager.get_current_price(symbol)
                if current_price:
                    pnl = (current_price - position['entry_price']) * position['size']
                    self.total_pnl += pnl
                    self.daily_pnl += pnl
                    
                    # Update cash balance
                    self.cash_balance += current_price * position['size']
            
            # Remove position
            del self.open_positions[symbol]
            
            self.logger.info(f"✅ [PortfolioRiskManager] Removed position for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ [PortfolioRiskManager] Error removing position for {symbol}: {e}")
            return False

    def update_performance_history(self):
        """Update performance history"""
        try:
            performance_entry = {
                'timestamp': datetime.utcnow(),
                'portfolio_value': self.portfolio_value,
                'cash_balance': self.cash_balance,
                'total_exposure': self.total_exposure,
                'num_positions': len(self.open_positions),
                'daily_pnl': self.daily_pnl,
                'total_pnl': self.total_pnl
            }
            
            self.performance_history.append(performance_entry)
            
            # Keep only last 1000 entries
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
            
            # Reset daily P&L
            self.daily_pnl = 0.0
            
        except Exception as e:
            self.logger.error(f"❌ [PortfolioRiskManager] Error updating performance history: {e}")

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        try:
            return {
                'portfolio_value': self.portfolio_value,
                'cash_balance': self.cash_balance,
                'total_exposure': self.total_exposure,
                'num_positions': len(self.open_positions),
                'total_pnl': self.total_pnl,
                'circuit_breaker_active': self.circuit_breaker_active,
                'risk_metrics': self.risk_metrics,
                'performance_history_length': len(self.performance_history)
            }
            
        except Exception as e:
            self.logger.error(f"❌ [PortfolioRiskManager] Error getting portfolio summary: {e}")
            return {}