"""
Log Manager - Centralized logging management
Refactored from the original Bot-Trading_Swing.py
"""

import logging
import os
import time
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import deque
import threading

# Import configuration and constants
from Bot_Trading_Swing_Refactored import BOT_LOGGERS


class LogManager:
    """
    Log Manager - Centralized logging management system
    Refactored to be more modular and maintainable
    """

    def __init__(self):
        """Initialize the Log Manager"""
        self.logger = BOT_LOGGERS['Observability']
        self.logger.info("📝 [LogManager] Initializing Log Manager...")
        
        # Log storage
        self.log_entries = deque(maxlen=10000)
        self.error_logs = deque(maxlen=1000)
        self.warning_logs = deque(maxlen=1000)
        self.info_logs = deque(maxlen=5000)
        
        # Log configuration
        self.log_levels = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        
        # Log categories
        self.categories = {
            'trading': ['trade', 'position', 'signal', 'order'],
            'system': ['system', 'performance', 'memory', 'cpu'],
            'api': ['api', 'request', 'response', 'error'],
            'data': ['data', 'market', 'price', 'indicator'],
            'risk': ['risk', 'portfolio', 'exposure', 'drawdown'],
            'ml': ['model', 'prediction', 'training', 'evaluation']
        }
        
        # Performance tracking
        self.performance_stats = {
            'total_logs': 0,
            'logs_per_second': 0,
            'error_rate': 0,
            'last_calculation': time.time()
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Log file management
        self.log_files = {}
        self.max_log_file_size = 10 * 1024 * 1024  # 10MB
        self.log_retention_days = 7
        
        self.logger.info("✅ [LogManager] Log Manager initialized successfully")

    def log(self, level: str, message: str, category: str = 'general', 
           extra_data: Dict[str, Any] = None, source: str = None):
        """
        Log a message with specified level and category
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Log message
            category: Log category
            extra_data: Additional data to include
            source: Source of the log message
        """
        try:
            timestamp = datetime.utcnow()
            
            # Create log entry
            log_entry = {
                'timestamp': timestamp,
                'level': level,
                'message': message,
                'category': category,
                'source': source or 'unknown',
                'extra_data': extra_data or {},
                'thread_id': threading.get_ident()
            }
            
            # Store log entry
            with self._lock:
                self.log_entries.append(log_entry)
                
                # Store in category-specific queues
                if level == 'ERROR' or level == 'CRITICAL':
                    self.error_logs.append(log_entry)
                elif level == 'WARNING':
                    self.warning_logs.append(log_entry)
                elif level == 'INFO':
                    self.info_logs.append(log_entry)
                
                # Update performance stats
                self.performance_stats['total_logs'] += 1
            
            # Log to standard logger
            logger = logging.getLogger(source or 'LogManager')
            log_level = self.log_levels.get(level, logging.INFO)
            logger.log(log_level, f"[{category}] {message}")
            
        except Exception as e:
            # Fallback to standard logging
            logging.error(f"Error in LogManager.log: {e}")

    def log_trading_event(self, event_type: str, symbol: str, data: Dict[str, Any]):
        """Log a trading event"""
        try:
            message = f"Trading event: {event_type} for {symbol}"
            self.log('INFO', message, 'trading', data, 'TradingBot')
        except Exception as e:
            logging.error(f"Error logging trading event: {e}")

    def log_system_event(self, event_type: str, data: Dict[str, Any]):
        """Log a system event"""
        try:
            message = f"System event: {event_type}"
            self.log('INFO', message, 'system', data, 'System')
        except Exception as e:
            logging.error(f"Error logging system event: {e}")

    def log_api_event(self, api_name: str, event_type: str, data: Dict[str, Any]):
        """Log an API event"""
        try:
            message = f"API event: {event_type} for {api_name}"
            self.log('INFO', message, 'api', data, 'APIManager')
        except Exception as e:
            logging.error(f"Error logging API event: {e}")

    def log_data_event(self, event_type: str, symbol: str, data: Dict[str, Any]):
        """Log a data event"""
        try:
            message = f"Data event: {event_type} for {symbol}"
            self.log('INFO', message, 'data', data, 'DataManager')
        except Exception as e:
            logging.error(f"Error logging data event: {e}")

    def log_risk_event(self, event_type: str, data: Dict[str, Any]):
        """Log a risk event"""
        try:
            message = f"Risk event: {event_type}"
            self.log('WARNING', message, 'risk', data, 'RiskManager')
        except Exception as e:
            logging.error(f"Error logging risk event: {e}")

    def log_ml_event(self, event_type: str, model_name: str, data: Dict[str, Any]):
        """Log an ML event"""
        try:
            message = f"ML event: {event_type} for {model_name}"
            self.log('INFO', message, 'ml', data, 'MLModels')
        except Exception as e:
            logging.error(f"Error logging ML event: {e}")

    def get_logs(self, level: str = None, category: str = None, 
                source: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get logs with optional filtering
        
        Args:
            level: Filter by log level
            category: Filter by category
            source: Filter by source
            limit: Maximum number of logs to return
            
        Returns:
            List of log entries
        """
        try:
            with self._lock:
                logs = list(self.log_entries)
            
            # Apply filters
            filtered_logs = []
            for log in logs:
                if level and log['level'] != level:
                    continue
                if category and log['category'] != category:
                    continue
                if source and log['source'] != source:
                    continue
                
                filtered_logs.append(log)
            
            # Return limited results
            return filtered_logs[-limit:] if limit else filtered_logs
            
        except Exception as e:
            logging.error(f"Error getting logs: {e}")
            return []

    def get_error_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get error logs"""
        try:
            with self._lock:
                return list(self.error_logs)[-limit:]
        except Exception as e:
            logging.error(f"Error getting error logs: {e}")
            return []

    def get_warning_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get warning logs"""
        try:
            with self._lock:
                return list(self.warning_logs)[-limit:]
        except Exception as e:
            logging.error(f"Error getting warning logs: {e}")
            return []

    def get_info_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get info logs"""
        try:
            with self._lock:
                return list(self.info_logs)[-limit:]
        except Exception as e:
            logging.error(f"Error getting info logs: {e}")
            return []

    def get_logs_by_time_range(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Get logs within a time range"""
        try:
            with self._lock:
                logs = list(self.log_entries)
            
            filtered_logs = []
            for log in logs:
                if start_time <= log['timestamp'] <= end_time:
                    filtered_logs.append(log)
            
            return filtered_logs
            
        except Exception as e:
            logging.error(f"Error getting logs by time range: {e}")
            return []

    def get_logs_by_category(self, category: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get logs by category"""
        try:
            return self.get_logs(category=category, limit=limit)
        except Exception as e:
            logging.error(f"Error getting logs by category: {e}")
            return []

    def get_logs_by_source(self, source: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get logs by source"""
        try:
            return self.get_logs(source=source, limit=limit)
        except Exception as e:
            logging.error(f"Error getting logs by source: {e}")
            return []

    def get_log_statistics(self) -> Dict[str, Any]:
        """Get log statistics"""
        try:
            with self._lock:
                total_logs = len(self.log_entries)
                error_count = len(self.error_logs)
                warning_count = len(self.warning_logs)
                info_count = len(self.info_logs)
                
                # Calculate error rate
                error_rate = (error_count / total_logs * 100) if total_logs > 0 else 0
                
                # Calculate logs per second
                current_time = time.time()
                time_diff = current_time - self.performance_stats['last_calculation']
                if time_diff > 0:
                    logs_per_second = total_logs / time_diff
                else:
                    logs_per_second = 0
                
                # Count by level
                level_counts = {}
                for log in self.log_entries:
                    level = log['level']
                    level_counts[level] = level_counts.get(level, 0) + 1
                
                # Count by category
                category_counts = {}
                for log in self.log_entries:
                    category = log['category']
                    category_counts[category] = category_counts.get(category, 0) + 1
                
                # Count by source
                source_counts = {}
                for log in self.log_entries:
                    source = log['source']
                    source_counts[source] = source_counts.get(source, 0) + 1
                
                return {
                    'total_logs': total_logs,
                    'error_count': error_count,
                    'warning_count': warning_count,
                    'info_count': info_count,
                    'error_rate': error_rate,
                    'logs_per_second': logs_per_second,
                    'level_breakdown': level_counts,
                    'category_breakdown': category_counts,
                    'source_breakdown': source_counts,
                    'last_update': datetime.utcnow().isoformat()
                }
            
        except Exception as e:
            logging.error(f"Error getting log statistics: {e}")
            return {}

    def get_recent_errors(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent errors within specified hours"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            with self._lock:
                recent_errors = []
                for log in self.error_logs:
                    if log['timestamp'] >= cutoff_time:
                        recent_errors.append(log)
                
                return recent_errors
            
        except Exception as e:
            logging.error(f"Error getting recent errors: {e}")
            return []

    def get_recent_warnings(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent warnings within specified hours"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            with self._lock:
                recent_warnings = []
                for log in self.warning_logs:
                    if log['timestamp'] >= cutoff_time:
                        recent_warnings.append(log)
                
                return recent_warnings
            
        except Exception as e:
            logging.error(f"Error getting recent warnings: {e}")
            return []

    def search_logs(self, search_term: str, level: str = None, 
                   category: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search logs by text content
        
        Args:
            search_term: Text to search for
            level: Filter by log level
            category: Filter by category
            limit: Maximum number of results
            
        Returns:
            List of matching log entries
        """
        try:
            with self._lock:
                logs = list(self.log_entries)
            
            # Apply filters and search
            matching_logs = []
            search_term_lower = search_term.lower()
            
            for log in logs:
                # Apply level filter
                if level and log['level'] != level:
                    continue
                
                # Apply category filter
                if category and log['category'] != category:
                    continue
                
                # Search in message and extra data
                if (search_term_lower in log['message'].lower() or
                    any(search_term_lower in str(value).lower() 
                        for value in log['extra_data'].values())):
                    matching_logs.append(log)
            
            # Return limited results
            return matching_logs[-limit:] if limit else matching_logs
            
        except Exception as e:
            logging.error(f"Error searching logs: {e}")
            return []

    def export_logs(self, filepath: str = None, level: str = None, 
                   category: str = None, hours: int = 24) -> str:
        """
        Export logs to file
        
        Args:
            filepath: Output file path
            level: Filter by log level
            category: Filter by category
            hours: Number of hours to export
            
        Returns:
            Path to exported file
        """
        try:
            if filepath is None:
                filepath = f"logs_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Get logs to export
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            logs_to_export = []
            
            with self._lock:
                for log in self.log_entries:
                    if log['timestamp'] >= cutoff_time:
                        # Apply filters
                        if level and log['level'] != level:
                            continue
                        if category and log['category'] != category:
                            continue
                        
                        logs_to_export.append(log)
            
            # Export to file
            export_data = {
                'export_timestamp': datetime.utcnow().isoformat(),
                'filters': {
                    'level': level,
                    'category': category,
                    'hours': hours
                },
                'total_logs': len(logs_to_export),
                'logs': logs_to_export
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"📁 [LogManager] Logs exported to {filepath}")
            return filepath
            
        except Exception as e:
            logging.error(f"Error exporting logs: {e}")
            return ""

    def clear_logs(self, level: str = None, category: str = None, 
                  older_than_hours: int = None):
        """
        Clear logs based on criteria
        
        Args:
            level: Clear logs of specific level
            category: Clear logs of specific category
            older_than_hours: Clear logs older than specified hours
        """
        try:
            with self._lock:
                if older_than_hours:
                    cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
                    
                    # Clear from main log entries
                    logs_to_keep = []
                    for log in self.log_entries:
                        if log['timestamp'] >= cutoff_time:
                            # Apply other filters
                            if level and log['level'] != level:
                                continue
                            if category and log['category'] != category:
                                continue
                            logs_to_keep.append(log)
                    
                    self.log_entries.clear()
                    self.log_entries.extend(logs_to_keep)
                    
                    # Clear from category-specific queues
                    self.error_logs.clear()
                    self.warning_logs.clear()
                    self.info_logs.clear()
                    
                    # Rebuild category-specific queues
                    for log in self.log_entries:
                        if log['level'] in ['ERROR', 'CRITICAL']:
                            self.error_logs.append(log)
                        elif log['level'] == 'WARNING':
                            self.warning_logs.append(log)
                        elif log['level'] == 'INFO':
                            self.info_logs.append(log)
                else:
                    # Clear all logs
                    self.log_entries.clear()
                    self.error_logs.clear()
                    self.warning_logs.clear()
                    self.info_logs.clear()
            
            self.logger.info(f"🧹 [LogManager] Logs cleared - Level: {level}, Category: {category}, Hours: {older_than_hours}")
            
        except Exception as e:
            logging.error(f"Error clearing logs: {e}")

    def get_log_summary(self) -> Dict[str, Any]:
        """Get log summary information"""
        try:
            stats = self.get_log_statistics()
            recent_errors = self.get_recent_errors(24)
            recent_warnings = self.get_recent_warnings(24)
            
            return {
                'statistics': stats,
                'recent_errors_24h': len(recent_errors),
                'recent_warnings_24h': len(recent_warnings),
                'log_retention_days': self.log_retention_days,
                'max_log_entries': self.log_entries.maxlen,
                'categories': list(self.categories.keys()),
                'levels': list(self.log_levels.keys())
            }
            
        except Exception as e:
            logging.error(f"Error getting log summary: {e}")
            return {}