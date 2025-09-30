# Standard library imports
print("🚀 [Bot] Starting imports...")

# ==================================================
# ENCODING FIX FOR GOOGLE COLAB
# ==================================================
import sys
import os

# Ensure UTF-8 encoding for stdout and stderr
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Set environment variables for UTF-8
os.environ['PYTHONIOENCODING'] = 'utf-8'

print("🔧 [Encoding] UTF-8 encoding configured successfully")

import asyncio
import copy
import json
import logging
import re
import sqlite3
import time
import threading
import warnings
print("✅ [Bot] Basic imports completed")
import glob
import shutil
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from collections import deque
from datetime import datetime, timedelta

# ==================================================
# CONFIGURATION & CONSTANTS
# ==================================================

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple
import functools
from typing import Callable, Any, Optional

# Feature flags
ENABLE_HISTORICAL_SL_CHECK = False  # Set to False to disable historical SL checking
SEND_HISTORICAL_SL_ALERTS = False   # Set to True to send Discord alerts for historical SL (only if ENABLE_HISTORICAL_SL_CHECK is True)

class SymbolType(Enum):
    """Enum for symbol types"""
    CRYPTO = "crypto"
    FOREX = "forex"
    EQUITY = "equity"
    COMMODITY = "commodity"
    UNKNOWN = "unknown"

class TimeFrame(Enum):
    """Enum for timeframes"""
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"

@dataclass
class APIConfig:
    """API configuration dataclass"""
    api_key: str
    base_url: str
    rate_limit: int
    timeout: int = 30
    retry_attempts: int = 3

@dataclass
class SymbolConfig:
    """Symbol configuration dataclass"""
    symbol: str
    symbol_type: SymbolType
    primary_timeframe: TimeFrame
    weight: float
    max_exposure: float
    risk_multiplier: float
    is_active: bool = True

# ==================================================
# API CONFIGURATIONS
# ==================================================

# Refactored API configuration
API_CONFIGS: Dict[str, APIConfig] = {
    'FINHUB': APIConfig(
        api_key='d1b3ichr01qjhvtsbj8g',
        base_url='https://finhub.io/api/v1',
        rate_limit=60
    ),
    'MARKETAUX': APIConfig(
        api_key='CkuQmx9sPsjw0FRDeSkoO8U3O9Jj3HWnUYMJNEql',
        base_url='https://api.marketaux.com/v1',
        rate_limit=100
    ),
    'NEWSAPI': APIConfig(
        api_key='abd8f43b808f42fdb8d28fb1c429af72',
        base_url='https://newsapi.org/v2',
        rate_limit=1000
    ),
    'EODHD': APIConfig(
        api_key='68bafd7d44a7f0.25202650',
        base_url='https://eodhistoricaldata.com/api',
        rate_limit=20
    ),
    'ALPHA_VANTAGE': APIConfig(
        api_key='FK3YQ1IKSC4E1AL5',
        base_url='https://www.alphavantage.co/query',
        rate_limit=5
    )
}

# Legacy API configuration (for backward compatibility)
API_KEYS = {name: config.api_key for name, config in API_CONFIGS.items()}
API_ENDPOINTS = {
    'FINHUB': {'base_url': API_CONFIGS['FINHUB'].base_url, 'quote': '/quote', 'news': '/company-news', 'sentiment': '/news-sentiment'},
    'MARKETAUX': {'base_url': 'https://api.marketaux.com/v1', 'news': '/news/all', 'news_intraday': '/news/intraday'},
    'NEWSAPI': {'base_url': 'https://newsapi.org/v2', 'everything': '/everything', 'top_headlines': '/top-headlines'},
    'EODHD': {'base_url': 'https://eodhistoricaldata.com/api', 'eod': '/eod', 'real_time': '/real-time', 'fundamentals': '/fundamentals'}
}
RATE_LIMITS = {name: config.rate_limit for name, config in API_CONFIGS.items()}

# Additional API keys
GOOGLE_AI_API_KEY = "AIzaSyA66wFiXm5cvxtPZM3wIX0HRSvK64TdU34"
TRADING_ECONOMICS_API_KEY = "a284ad0cdba547c:p5oyv77j6kovqhv"
OANDA_API_KEY = "814bb04d60580a8a9b0ce5542f70d5f7-b33dbed32efba816c1d16c393369ec8d"
OANDA_URL = "https://api-fxtrade.oanda.com/v3"

# ==================================================
# TRADING CONSTANTS
# ==================================================

class TradingConstants:
    """Trading-related constants"""
    MIN_CONFIDENCE_THRESHOLD = 0.5
    MAX_CONFIDENCE_THRESHOLD = 0.95
    CONFIDENCE_SMOOTHING_FACTOR = 0.1
    TRAILING_STOP_MULTIPLIER = 0.5
    POSITION_SIZE_MULTIPLIER = 1.0
    SPREAD_COST_PIPS = 2.0
    SLIPPAGE_PIPS = 1.0
    MAX_RISK_PER_TRADE = 0.02
    MAX_DAILY_RISK = 0.05
    MAX_PORTFOLIO_RISK = 0.10

class FeatureConstants:
    """Feature engineering constants"""
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BB_PERIOD = 20
    BB_STD = 2
    EMA_PERIODS = [20, 50, 200]
    VOLUME_PERIOD = 20
    MIN_CANDLES_FOR_ANALYSIS = 100
    MAX_STALE_MINUTES = 30

# Symbol classifications
CRYPTO_SYMBOLS = {'BTCUSD', 'ETHUSD', 'XRPUSD', 'LTCUSD', 'ADAUSD'}
EQUITY_INDICES = {'SPX500', 'NAS100', 'US30', 'DE40', 'UK100', 'FR40', 'JP225', 'AU200'}
FOREX_PAIRS = {'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD', 'AUDNZD'}
COMMODITIES = {'XAUUSD', 'XAGUSD', 'USOIL', 'UKOIL', 'NATGAS'}

# ==================================================
# RISK MANAGEMENT CONFIGURATION
# ==================================================

RISK_MANAGEMENT = {
    "MAX_RISK_PER_TRADE": 0.02,  # 2% max risk per trade
    "MAX_PORTFOLIO_RISK": 0.10,  # 10% max total portfolio risk
    "MAX_OPEN_POSITIONS": 5,     # Maximum number of open positions
    "VOLATILITY_LOOKBACK": 20,   # ATR lookback period
    "SL_ATR_MULTIPLIER": 2.0,    # Stop loss ATR multiplier
    "BASE_RR_RATIO": 2.0,        # Base risk-reward ratio
    "TRAILING_STOP_MULTIPLIER": 1.0,  # Trailing stop multiplier
}

# Enhanced risk management configuration by asset class
RISK_CONFIG_BY_ASSET_CLASS = {
    "equity_index": {
        "max_position_size": 0.25,  # 25% max per equity index
        "correlation_threshold": 0.7,
        "var_multiplier": 1.0,
        "stop_loss_atr": 2.0,
        "take_profit_atr": 4.0,
        "max_daily_loss": 0.02,
        "session_risk_adjustment": True,
        "gap_risk_factor": 1.5
    },
    "commodity": {
        "max_position_size": 0.20,  # 20% max per commodity
        "correlation_threshold": 0.6,
        "var_multiplier": 1.2,
        "stop_loss_atr": 2.5,
        "take_profit_atr": 5.0,
        "max_daily_loss": 0.025,
        "session_risk_adjustment": False,
        "gap_risk_factor": 1.0
    },
    "forex": {
        "max_position_size": 0.15,  # 15% max per forex pair
        "correlation_threshold": 0.8,
        "var_multiplier": 0.8,
        "stop_loss_atr": 1.5,
        "take_profit_atr": 3.0,
        "max_daily_loss": 0.015,
        "session_risk_adjustment": True,
        "gap_risk_factor": 1.2
    },
    "crypto": {
        "max_position_size": 0.10,  # 10% max per crypto
        "correlation_threshold": 0.5,
        "var_multiplier": 1.5,
        "stop_loss_atr": 3.0,
        "take_profit_atr": 6.0,
        "max_daily_loss": 0.03,
        "session_risk_adjustment": False,
        "gap_risk_factor": 2.0
    }
}

# ==================================================
# MARKET TIMING CONSTANTS
# ==================================================

# Market timing constants
FOREX_MARKET_CLOSE_HOUR_UTC = 21  # Friday close time
WEEKEND_FRIDAY = 4
WEEKEND_SATURDAY = 5
WEEKEND_SUNDAY = 6

# Cache timeout constants (seconds)
NEWS_CACHE_TIMEOUT = 3600  # 1 hour
ECONOMIC_CALENDAR_CACHE_TIMEOUT = 86400  # 24 hours

# Trading constants
DEFAULT_CANDLE_WAIT_MINUTES = 60  # H1 default
CRYPTO_PREFIXES = ["BTC", "ETH"]

# Symbol alias mapping for different broker naming conventions
SYMBOL_ALIAS = {
    "BTCUSD": ["BTC-USD", "BTCUSDT", "XBTUSD"],
    "ETHUSD": ["ETH-USD", "ETHUSDT"]
}

# Feature engineering constants
DEFAULT_ATR_MULTIPLIER = 2.0
DEFAULT_RANGE_WINDOW = 50

# Wyckoff Configuration for different symbols
WYCKOFF_CONFIG = {
    "DEFAULT":  {"range_window": 50, "atr_multiplier": 1.2, "volume_ema": 50},
    "XAUUSD":   {"range_window": 60, "atr_multiplier": 1.2, "volume_ema": 40},
    "SPX500":   {"range_window": 75, "atr_multiplier": 1.5, "volume_ema": 30},
    "BTCUSD":   {"range_window": 40, "atr_multiplier": 1.8, "volume_ema": 20},
    "ETHUSD":   {"range_window": 45, "atr_multiplier": 2.0, "volume_ema": 20}
}
DEFAULT_EMA_PERIOD = 200
DEFAULT_ADX_THRESHOLD = 25

# ==================================================
# DISCORD CONFIGURATION
# ==================================================

# DISCORD WEBHOOK - UPDATE THIS WITH YOUR WEBHOOK URL
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1410915486551511062/pzCWm4gbe0w-xFyI0pKbsy417sbsYwjwjg-iWMLhccIGRJR2FqJ4kUwlzIZAyw3C2Fhq"

# DISCORD ALERT CONFIGURATION
DISCORD_CONFIG = {
    "RESEND_SIGNALS_ON_STARTUP": False,  # Enable resending active signals when starting bot
    "RESEND_TRAILING_STOPS": True,       # Include trailing stops
    "RESEND_POSITIONS": True,            # Resend current position information
    "RESEND_PERFORMANCE": False,          # Resend performance information
    "MAX_RESEND_MESSAGES": 10,           # Limit number of resend messages
    "RESEND_DELAY_SECONDS": 2            # Delay between messages to avoid spam
}

# ==================================================
# SYMBOL ALLOCATION & CONFIGURATION
# ==================================================

# Optimized symbol configuration with asset class metadata
# Symbol allocation based on forward test results - EXPANDED VERSION
SYMBOL_ALLOCATION = {
    # === COMMODITIES (Global Trading) ===
    "XAUUSD": {"weight": 0.08, "max_exposure": 0.05, "risk_multiplier": 0.8},  # Gold
    "USOIL": {"weight": 0.06, "max_exposure": 0.04, "risk_multiplier": 0.9},   # Crude Oil

    # === EQUITY INDICES (Session-based Trading) ===
    "SPX500": {"weight": 0.08, "max_exposure": 0.05, "risk_multiplier": 0.8},  # S&P 500
    "DE40": {"weight": 0.05, "max_exposure": 0.03, "risk_multiplier": 0.8},     # DAX 40

    # === FOREX MAJOR PAIRS ===
    "EURUSD": {"weight": 0.06, "max_exposure": 0.04, "risk_multiplier": 0.75},
    "AUDUSD": {"weight": 0.05, "max_exposure": 0.03, "risk_multiplier": 0.75},  # Australian Dollar
    "AUDNZD": {"weight": 0.04, "max_exposure": 0.025, "risk_multiplier": 0.8},  # AUD/NZD cross pair

    # === CRYPTOCURRENCIES (24/7 Trading) ===
    "BTCUSD": {"weight": 0.05, "max_exposure": 0.03, "risk_multiplier": 1.0},  # Bitcoin
    "ETHUSD": {"weight": 0.04, "max_exposure": 0.025, "risk_multiplier": 1.0},  # Ethereum
}

# Active symbols list (all symbols in SYMBOL_ALLOCATION are considered active)
# Ensure consistent order for RL model to work correctly
SYMBOLS = ["BTCUSD", "ETHUSD", "XAUUSD", "USOIL", "SPX500", "DE40", "EURUSD", "AUDUSD", "AUDNZD"]

# Enhanced Entry/TP/SL Configuration for all symbols - EXPANDED VERSION
ENTRY_TP_SL_CONFIG = {
    # === COMMODITIES (Global Trading) ===
    "XAUUSD": {
        "entry_method": "fibonacci_confluence",
        "atr_multiplier_sl": 2.5,
        "atr_multiplier_tp": 4.0,
        "min_rr_ratio": 1.6,
        "max_rr_ratio": 3.5,
        "support_resistance_weight": 0.4,
        "volume_confirmation": False,
        "session_filter": False,
        "volatility_adjustment": True,
        "fibonacci_levels": [0.236, 0.382, 0.5, 0.618, 0.786]
    },
    "USOIL": {
        "entry_method": "trend_following",
        "atr_multiplier_sl": 2.0,
        "atr_multiplier_tp": 3.5,
        "min_rr_ratio": 1.5,
        "max_rr_ratio": 3.2,
        "support_resistance_weight": 0.3,
        "volume_confirmation": True,
        "session_filter": False,
        "volatility_adjustment": True
    },

    # === EQUITY INDICES (Session-based Trading) ===
    "SPX500": {
        "entry_method": "breakout_confirmation",
        "atr_multiplier_sl": 2.0,
        "atr_multiplier_tp": 3.0,
        "min_rr_ratio": 1.5,
        "max_rr_ratio": 3.0,
        "support_resistance_weight": 0.3,
        "volume_confirmation": True,
        "session_filter": True,
        "volatility_adjustment": True
    },
    "DE40": {
        "entry_method": "european_session_breakout",
        "atr_multiplier_sl": 1.8,
        "atr_multiplier_tp": 2.8,
        "min_rr_ratio": 1.4,
        "max_rr_ratio": 2.8,
        "support_resistance_weight": 0.25,
        "volume_confirmation": True,
        "session_filter": True,
        "volatility_adjustment": True
    },

    # === FOREX MAJOR PAIRS ===
    "EURUSD": {
        "entry_method": "london_session_breakout",
        "atr_multiplier_sl": 1.5,
        "atr_multiplier_tp": 2.5,
        "min_rr_ratio": 1.4,
        "max_rr_ratio": 2.5,
        "support_resistance_weight": 0.2,
        "volume_confirmation": False,
        "session_filter": True,
        "volatility_adjustment": True
    },
    "AUDUSD": {
        "entry_method": "asian_session_breakout",
        "atr_multiplier_sl": 1.6,
        "atr_multiplier_tp": 2.6,
        "min_rr_ratio": 1.4,
        "max_rr_ratio": 2.6,
        "support_resistance_weight": 0.2,
        "volume_confirmation": False,
        "session_filter": True,
        "volatility_adjustment": True
    },
    "AUDNZD": {
        "entry_method": "cross_pair_breakout",
        "atr_multiplier_sl": 1.8,
        "atr_multiplier_tp": 3.0,
        "min_rr_ratio": 1.5,
        "max_rr_ratio": 3.0,
        "support_resistance_weight": 0.3,
        "volume_confirmation": False,
        "session_filter": True,
        "volatility_adjustment": True
    },

    # === CRYPTOCURRENCIES (24/7 Trading) ===
    "BTCUSD": {
        "entry_method": "volatility_breakout",
        "atr_multiplier_sl": 3.0,
        "atr_multiplier_tp": 6.0,
        "min_rr_ratio": 1.8,
        "max_rr_ratio": 4.0,
        "support_resistance_weight": 0.4,
        "volume_confirmation": True,
        "session_filter": False,
        "volatility_adjustment": True
    },
    "ETHUSD": {
        "entry_method": "volatility_breakout",
        "atr_multiplier_sl": 3.2,
        "atr_multiplier_tp": 6.5,
        "min_rr_ratio": 1.8,
        "max_rr_ratio": 4.2,
        "support_resistance_weight": 0.4,
        "volume_confirmation": True,
        "session_filter": False,
        "volatility_adjustment": True
    }
}

# ==================================================
# MACHINE LEARNING CONFIGURATION
# ==================================================

# Advanced ML constants
STACKING_CV_FOLDS = 5
CALIBRATION_CV_FOLDS = 3
MIN_SAMPLES_LEAF = 50
OPTUNA_N_TRIALS = 100
OPTUNA_TIMEOUT = 3600  # 1 hour

# Quality gates thresholds
MIN_SHARPE_RATIO = 1.2
MAX_DRAWDOWN_THRESHOLD = 0.15
MIN_CALMAR_RATIO = 0.8
MIN_INFORMATION_RATIO = 0.5

# Primary timeframe configuration
PRIMARY_TIMEFRAME = "H4"

# ML Configuration thresholds
MIN_F1_SCORE_GATE = 0.35       # Minimum F1-Score threshold (reduced for more symbols)
MAX_STD_F1_GATE = 0.20         # Maximum F1 standard deviation threshold (increased)
MIN_ACCURACY_GATE = 0.40       # Minimum win rate threshold (reduced for more symbols)
MIN_SAMPLES_GATE = 100         # Minimum sample count threshold for training
MAX_RETRAIN_ATTEMPTS = 3

ML_CONFIG = {
    "MIN_F1_SCORE": 0.35,  # Giảm từ 0.5 để linh hoạt hơn
    "MIN_ACCURACY": 0.40,  # Giảm từ 0.6 để linh hoạt hơn
    "MAX_STD_F1": 0.20,    # Tăng từ 0.1 để linh hoạt hơn
    "CV_N_SPLITS": 5,      # Giảm từ 10 để tăng tốc độ
    "CONFIDENCE_THRESHOLD": 0.6,  # Restored to original
    "MIN_CONFIDENCE_TRADE": 0.50,  # Restored to original
    "CLOSE_ON_CONFIDENCE_DROP_THRESHOLD": 0.3,  # Restored to reasonable value
    "MIN_SAMPLES_FOR_TRAINING": 100,  # Giảm từ 300 để linh hoạt hơn
    "MAX_CORRELATION_THRESHOLD": 0.85,  # Giảm từ 0.9 để chặt chẽ hơn
    "EARLY_STOPPING_PATIENCE": 3,  # Ultra-strict: test results
    "REGULARIZATION_STRENGTH": 0.15,  # Ultra-strong: test results
    "DROPOUT_RATE": 0.7,  # Ultra-high: test results
    "BATCH_NORMALIZATION": True,  # Bật batch normalization
    "DATA_AUGMENTATION": True,  # Bật data augmentation
    "CROSS_VALIDATION_FOLDS": 10,  # Test results
    "OUT_OF_SAMPLE_TESTING": True,  # Bật out-of-sample testing
    "MAX_DEPTH_LIMIT": 4,  # Ultra-shallow: test results
    "MIN_SAMPLES_SPLIT_LIMIT": 50,  # Ultra-high: test results
    "MIN_SAMPLES_LEAF_LIMIT": 25,  # Ultra-high: test results
    "MAX_FEATURES_LIMIT": 0.3,  # Ultra-few: test results
    "CPU_THRESHOLD": 80,  # Test results
    "MEMORY_THRESHOLD": 85,  # Test results
    "ALERT_LIMIT": 5,  # Test results
    "OVERFITTING_DETECTION": True,  # Bật overfitting detection
    "VALIDATION_REQUIREMENTS": "STRICT",  # Strict validation mode
    "VALIDATION_THRESHOLD": 0.02,  # Ultra-strict validation threshold
    "GENERALIZATION_GAP_THRESHOLD": 0.02,  # Ultra-strict generalization gap
    "STABILITY_THRESHOLD": 0.02,  # Ultra-strict stability requirement

    # Feature selection and model configuration
    "FEATURE_SELECTION_TOP_K": 50,  # Top K features to select
    "FEATURE_IMPORTANCE_THRESHOLD": 0.01,  # Minimum feature importance
    "ENSEMBLE_MODELS": ["rf", "xgb", "lgb", "lstm"],  # Ensemble models
    "MODEL_STACKING_ENABLED": True,  # Enable model stacking

    # LSTM specific configuration
    "L2_REGULARIZATION": 0.001,  # L2 regularization strength
    "ATTENTION_MECHANISM": True,  # Enable attention mechanism
    "GRADIENT_CLIPPING": True,  # Enable gradient clipping
    "LEARNING_RATE_DECAY": 0.95,  # Learning rate decay factor
    "BATCH_SIZE": 64,  # Batch size for training
    "NOISE_INJECTION": 0.01  # Noise injection level for regularization
}

# ==================================================
# TRADE FILTERS & PERFORMANCE THRESHOLDS
# ==================================================

# Trade filters
TRADE_FILTERS = {
    "SKIP_NEAR_HIGH_IMPACT_EVENTS": True,  # skip ±2h around major news
    "EVENT_BUFFER_HOURS": 2,
    "AVOID_WEEKEND": True,                  # don't open new trades near weekend
    "SEND_PRE_CHECK_STATUS_ALERT": True
}

# Performance thresholds for symbol selection
PERFORMANCE_THRESHOLDS = {
    "excellent": {
        "win_rate": 0.60,
        "profit_factor": 1.8,
        "sharpe_ratio": 1.0,
        "calmar_ratio": 0.8,
        "sortino_ratio": 1.2,
        "max_drawdown": 0.10,
        "avg_trade": 0.005,
        "consecutive_losses": 5,
        "recovery_factor": 1.2
    },
    "good": {
        "win_rate": 0.50,
        "profit_factor": 1.4,
        "sharpe_ratio": 0.7,
        "calmar_ratio": 0.5,
        "sortino_ratio": 0.8,
        "max_drawdown": 0.15,
        "avg_trade": 0.003,
        "consecutive_losses": 7,
        "recovery_factor": 1.0
    },
    "acceptable": {
        "win_rate": 0.45,
        "profit_factor": 1.2,
        "sharpe_ratio": 0.5,
        "calmar_ratio": 0.3,
        "sortino_ratio": 0.6,
        "max_drawdown": 0.20,
        "avg_trade": 0.002,
        "consecutive_losses": 10,
        "recovery_factor": 0.8
    }
}

# ==================================================
# DECORATORS AND MIXINS
# ==================================================

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator to retry function execution on failure
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logging.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logging.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            
            raise last_exception
        return wrapper
    return decorator

def log_execution_time(func: Callable) -> Callable:
    """Decorator to log function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        logging.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
        return result
    return wrapper

class LoggingMixin:
    """Mixin class for logging functionality"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)

    def log_info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, **kwargs)

    def log_warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, **kwargs)

    def log_error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, **kwargs)

class ValidationMixin:
    """Mixin class for validation functionality"""
    def validate_not_none(self, value: Any, name: str) -> None:
        """Validate that value is not None"""
        if value is None:
            raise ValueError(f"{name} cannot be None")

    def validate_positive(self, value: float, name: str) -> None:
        """Validate that value is positive"""
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    def validate_range(self, value: float, min_val: float, max_val: float, name: str) -> None:
        """Validate that value is within range"""
        if not (min_val <= value <= max_val):
            raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")

# ==================================================
# LOGGING SETUP
# ==================================================

def setup_detailed_logging() -> Dict[str, logging.Logger]:
    """Setup detailed logging configuration"""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        encoding='utf-8',
        force=True
    )
    
    # Create specialized loggers
    loggers = {}
    logger_configs = {
        'TradingBot': {'level': logging.INFO, 'file': 'logs/trading_bot.log'},
        'DataManager': {'level': logging.INFO, 'file': 'logs/data_manager.log'},
        'MLModels': {'level': logging.INFO, 'file': 'logs/ml_models.log'},
        'RiskManager': {'level': logging.INFO, 'file': 'logs/risk_manager.log'},
        'APIManager': {'level': logging.INFO, 'file': 'logs/api_manager.log'},
        'RLStrategy': {'level': logging.INFO, 'file': 'logs/rl_strategy.log'},
        'NewsManager': {'level': logging.INFO, 'file': 'logs/news_manager.log'},
        'Observability': {'level': logging.INFO, 'file': 'logs/observability.log'}
    }
    
    for name, config in logger_configs.items():
        logger = logging.getLogger(name)
        logger.setLevel(config['level'])
        
        # File handler
        file_handler = logging.FileHandler(config['file'], encoding='utf-8')
        file_handler.setLevel(config['level'])
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        loggers[name] = logger
    
    return loggers

# Initialize loggers
BOT_LOGGERS = setup_detailed_logging()

# ==================================================
# IMPORTS FROM REFACTORED MODULES
# ==================================================

# Import all the refactored classes from their respective modules
from src.core.enhanced_trading_bot import EnhancedTradingBot
from src.data.enhanced_data_manager import EnhancedDataManager
from src.data.advanced_feature_engineer import AdvancedFeatureEngineer
from src.models.enhanced_ensemble_model import EnhancedEnsembleModel
from src.models.rl_agent import RLAgent
from src.models.lstm_model import LSTMModel
from src.risk.master_agent import MasterAgent
from src.risk.portfolio_risk_manager import PortfolioRiskManager
from src.utils.api_manager import APIManager
from src.utils.advanced_observability import AdvancedObservability
from src.utils.log_manager import LogManager
from src.utils.helper_functions import HelperFunctions

# ==================================================
# MAIN EXECUTION
# ==================================================

async def main():
    """Main execution function for the Enhanced Trading Bot"""
    print("🚀 [Main] Starting Enhanced Trading Bot...")
    
    try:
        # Initialize the trading bot
        trading_bot = EnhancedTradingBot()
        
        # Start the bot
        await trading_bot.run()
        
    except KeyboardInterrupt:
        print("🛑 [Main] Bot stopped by user")
    except Exception as e:
        print(f"❌ [Main] Error starting bot: {e}")
        logging.error(f"Main execution error: {e}")
        raise

def run_bot():
    """Synchronous wrapper for running the bot"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("🛑 [Main] Bot stopped by user")
    except Exception as e:
        print(f"❌ [Main] Error running bot: {e}")
        raise

if __name__ == "__main__":
    # Run the main function
    run_bot()