"""
Advanced Observability - Monitoring and observability system
Refactored from the original Bot-Trading_Swing.py
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import threading

# Import configuration and constants
from Bot_Trading_Swing_Refactored import (
    BOT_LOGGERS, SYMBOLS, SYMBOL_ALLOCATION
)


class AdvancedObservability:
    """
    Advanced Observability - Comprehensive monitoring and observability system
    Refactored to be more modular and maintainable
    """

    def __init__(self):
        """Initialize the Advanced Observability system"""
        self.logger = BOT_LOGGERS['Observability']
        self.logger.info("📊 [Observability] Initializing Advanced Observability...")
        
        # Metrics storage
        self.metrics = defaultdict(list)
        self.performance_metrics = {}
        self.system_metrics = {}
        self.trading_metrics = {}
        
        # Alerting system
        self.alert_thresholds = {
            'error_rate': 0.05,  # 5% error rate
            'latency_p95': 5.0,  # 5 seconds
            'memory_usage': 0.8,  # 80% memory usage
            'cpu_usage': 0.8,     # 80% CPU usage
            'drawdown': 0.1,      # 10% drawdown
            'consecutive_losses': 5
        }
        
        self.active_alerts = []
        self.alert_history = []
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.system_history = deque(maxlen=1000)
        self.trading_history = deque(maxlen=1000)
        
        # Monitoring intervals
        self.monitoring_interval = 60  # 1 minute
        self.metrics_retention_days = 7
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Start monitoring
        self._monitoring_active = False
        self._monitoring_task = None
        
        self.logger.info("✅ [Observability] Advanced Observability initialized successfully")

    async def start_monitoring(self):
        """Start the monitoring system"""
        try:
            if self._monitoring_active:
                self.logger.warning("⚠️ [Observability] Monitoring already active")
                return
            
            self._monitoring_active = True
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info("🚀 [Observability] Monitoring started")
            
        except Exception as e:
            self.logger.error(f"❌ [Observability] Error starting monitoring: {e}")

    async def stop_monitoring(self):
        """Stop the monitoring system"""
        try:
            self._monitoring_active = False
            
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
                self._monitoring_task = None
            
            self.logger.info("🛑 [Observability] Monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"❌ [Observability] Error stopping monitoring: {e}")

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while self._monitoring_active:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Collect performance metrics
                await self._collect_performance_metrics()
                
                # Check for alerts
                await self._check_alerts()
                
                # Cleanup old data
                await self._cleanup_old_data()
                
                # Wait for next interval
                await asyncio.sleep(self.monitoring_interval)
                
        except asyncio.CancelledError:
            self.logger.info("🛑 [Observability] Monitoring loop cancelled")
        except Exception as e:
            self.logger.error(f"❌ [Observability] Error in monitoring loop: {e}")

    async def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            import psutil
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available
            memory_total = memory.total
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free = disk.free
            disk_total = disk.total
            
            # Network metrics
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info().rss
            process_cpu = process.cpu_percent()
            
            system_metrics = {
                'timestamp': datetime.utcnow(),
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'memory_percent': memory_percent,
                'memory_available': memory_available,
                'memory_total': memory_total,
                'disk_percent': disk_percent,
                'disk_free': disk_free,
                'disk_total': disk_total,
                'network_bytes_sent': network_bytes_sent,
                'network_bytes_recv': network_bytes_recv,
                'process_memory': process_memory,
                'process_cpu': process_cpu
            }
            
            with self._lock:
                self.system_history.append(system_metrics)
                self.system_metrics.update(system_metrics)
            
            self.logger.debug(f"📊 [Observability] System metrics collected - CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%")
            
        except ImportError:
            self.logger.warning("⚠️ [Observability] psutil not available, skipping system metrics")
        except Exception as e:
            self.logger.error(f"❌ [Observability] Error collecting system metrics: {e}")

    async def _collect_performance_metrics(self):
        """Collect trading performance metrics"""
        try:
            # This would collect metrics from the trading bot
            # For now, we'll create placeholder metrics
            
            performance_metrics = {
                'timestamp': datetime.utcnow(),
                'active_positions': 0,
                'total_pnl': 0.0,
                'daily_pnl': 0.0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0
            }
            
            with self._lock:
                self.performance_history.append(performance_metrics)
                self.performance_metrics.update(performance_metrics)
            
            self.logger.debug("📊 [Observability] Performance metrics collected")
            
        except Exception as e:
            self.logger.error(f"❌ [Observability] Error collecting performance metrics: {e}")

    async def _check_alerts(self):
        """Check for alert conditions"""
        try:
            current_time = datetime.utcnow()
            
            # Check system alerts
            await self._check_system_alerts()
            
            # Check performance alerts
            await self._check_performance_alerts()
            
            # Check trading alerts
            await self._check_trading_alerts()
            
        except Exception as e:
            self.logger.error(f"❌ [Observability] Error checking alerts: {e}")

    async def _check_system_alerts(self):
        """Check system-related alerts"""
        try:
            if not self.system_metrics:
                return
            
            # CPU usage alert
            cpu_percent = self.system_metrics.get('cpu_percent', 0)
            if cpu_percent > self.alert_thresholds['cpu_usage'] * 100:
                await self._create_alert(
                    'system', 'high_cpu_usage',
                    f"CPU usage is {cpu_percent:.1f}% (threshold: {self.alert_thresholds['cpu_usage'] * 100:.1f}%)",
                    'warning'
                )
            
            # Memory usage alert
            memory_percent = self.system_metrics.get('memory_percent', 0)
            if memory_percent > self.alert_thresholds['memory_usage'] * 100:
                await self._create_alert(
                    'system', 'high_memory_usage',
                    f"Memory usage is {memory_percent:.1f}% (threshold: {self.alert_thresholds['memory_usage'] * 100:.1f}%)",
                    'warning'
                )
            
            # Disk usage alert
            disk_percent = self.system_metrics.get('disk_percent', 0)
            if disk_percent > 90:
                await self._create_alert(
                    'system', 'high_disk_usage',
                    f"Disk usage is {disk_percent:.1f}%",
                    'warning'
                )
            
        except Exception as e:
            self.logger.error(f"❌ [Observability] Error checking system alerts: {e}")

    async def _check_performance_alerts(self):
        """Check performance-related alerts"""
        try:
            if not self.performance_metrics:
                return
            
            # Drawdown alert
            max_drawdown = self.performance_metrics.get('max_drawdown', 0)
            if max_drawdown > self.alert_thresholds['drawdown']:
                await self._create_alert(
                    'performance', 'high_drawdown',
                    f"Maximum drawdown is {max_drawdown:.1%} (threshold: {self.alert_thresholds['drawdown']:.1%})",
                    'critical'
                )
            
            # Consecutive losses alert
            losing_trades = self.performance_metrics.get('losing_trades', 0)
            if losing_trades >= self.alert_thresholds['consecutive_losses']:
                await self._create_alert(
                    'performance', 'consecutive_losses',
                    f"Consecutive losses: {losing_trades} (threshold: {self.alert_thresholds['consecutive_losses']})",
                    'warning'
                )
            
        except Exception as e:
            self.logger.error(f"❌ [Observability] Error checking performance alerts: {e}")

    async def _check_trading_alerts(self):
        """Check trading-related alerts"""
        try:
            # This would check for trading-specific alerts
            # such as unusual trading patterns, high volatility, etc.
            pass
            
        except Exception as e:
            self.logger.error(f"❌ [Observability] Error checking trading alerts: {e}")

    async def _create_alert(self, category: str, alert_type: str, message: str, severity: str):
        """Create a new alert"""
        try:
            alert = {
                'id': f"{category}_{alert_type}_{int(time.time())}",
                'category': category,
                'type': alert_type,
                'message': message,
                'severity': severity,
                'timestamp': datetime.utcnow(),
                'acknowledged': False,
                'resolved': False
            }
            
            with self._lock:
                self.active_alerts.append(alert)
                self.alert_history.append(alert)
            
            # Log alert
            if severity == 'critical':
                self.logger.critical(f"🚨 [Alert] {message}")
            elif severity == 'warning':
                self.logger.warning(f"⚠️ [Alert] {message}")
            else:
                self.logger.info(f"ℹ️ [Alert] {message}")
            
        except Exception as e:
            self.logger.error(f"❌ [Observability] Error creating alert: {e}")

    async def _cleanup_old_data(self):
        """Cleanup old metrics and alerts"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=self.metrics_retention_days)
            
            with self._lock:
                # Cleanup performance history
                while self.performance_history and self.performance_history[0]['timestamp'] < cutoff_time:
                    self.performance_history.popleft()
                
                # Cleanup system history
                while self.system_history and self.system_history[0]['timestamp'] < cutoff_time:
                    self.system_history.popleft()
                
                # Cleanup trading history
                while self.trading_history and self.trading_history[0]['timestamp'] < cutoff_time:
                    self.trading_history.popleft()
                
                # Cleanup alert history
                self.alert_history = [
                    alert for alert in self.alert_history 
                    if alert['timestamp'] > cutoff_time
                ]
            
            self.logger.debug("🧹 [Observability] Old data cleaned up")
            
        except Exception as e:
            self.logger.error(f"❌ [Observability] Error cleaning up old data: {e}")

    def record_metric(self, name: str, value: Any, tags: Dict[str, str] = None):
        """Record a custom metric"""
        try:
            metric = {
                'name': name,
                'value': value,
                'tags': tags or {},
                'timestamp': datetime.utcnow()
            }
            
            with self._lock:
                self.metrics[name].append(metric)
            
            self.logger.debug(f"📊 [Observability] Metric recorded: {name} = {value}")
            
        except Exception as e:
            self.logger.error(f"❌ [Observability] Error recording metric {name}: {e}")

    def record_trading_event(self, event_type: str, symbol: str, data: Dict[str, Any]):
        """Record a trading event"""
        try:
            event = {
                'event_type': event_type,
                'symbol': symbol,
                'data': data,
                'timestamp': datetime.utcnow()
            }
            
            with self._lock:
                self.trading_history.append(event)
            
            self.logger.debug(f"📊 [Observability] Trading event recorded: {event_type} for {symbol}")
            
        except Exception as e:
            self.logger.error(f"❌ [Observability] Error recording trading event: {e}")

    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        try:
            with self._lock:
                for alert in self.active_alerts:
                    if alert['id'] == alert_id:
                        alert['acknowledged'] = True
                        self.logger.info(f"✅ [Observability] Alert acknowledged: {alert_id}")
                        return True
            
            self.logger.warning(f"⚠️ [Observability] Alert not found: {alert_id}")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ [Observability] Error acknowledging alert {alert_id}: {e}")
            return False

    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        try:
            with self._lock:
                for alert in self.active_alerts:
                    if alert['id'] == alert_id:
                        alert['resolved'] = True
                        alert['resolved_at'] = datetime.utcnow()
                        self.active_alerts.remove(alert)
                        self.logger.info(f"✅ [Observability] Alert resolved: {alert_id}")
                        return True
            
            self.logger.warning(f"⚠️ [Observability] Alert not found: {alert_id}")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ [Observability] Error resolving alert {alert_id}: {e}")
            return False

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        try:
            with self._lock:
                return {
                    'system_metrics': self.system_metrics,
                    'performance_metrics': self.performance_metrics,
                    'active_alerts': len(self.active_alerts),
                    'total_alerts': len(self.alert_history),
                    'performance_history_length': len(self.performance_history),
                    'system_history_length': len(self.system_history),
                    'trading_history_length': len(self.trading_history),
                    'monitoring_active': self._monitoring_active
                }
            
        except Exception as e:
            self.logger.error(f"❌ [Observability] Error getting metrics summary: {e}")
            return {}

    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get performance dashboard data"""
        try:
            with self._lock:
                # Convert history to DataFrames for analysis
                if self.performance_history:
                    perf_df = pd.DataFrame(list(self.performance_history))
                    
                    # Calculate trends
                    if len(perf_df) > 1:
                        pnl_trend = perf_df['total_pnl'].iloc[-1] - perf_df['total_pnl'].iloc[0]
                        win_rate_trend = perf_df['win_rate'].iloc[-1] - perf_df['win_rate'].iloc[0]
                    else:
                        pnl_trend = 0
                        win_rate_trend = 0
                else:
                    perf_df = pd.DataFrame()
                    pnl_trend = 0
                    win_rate_trend = 0
                
                if self.system_history:
                    sys_df = pd.DataFrame(list(self.system_history))
                    
                    # Calculate system trends
                    if len(sys_df) > 1:
                        cpu_trend = sys_df['cpu_percent'].iloc[-1] - sys_df['cpu_percent'].iloc[0]
                        memory_trend = sys_df['memory_percent'].iloc[-1] - sys_df['memory_percent'].iloc[0]
                    else:
                        cpu_trend = 0
                        memory_trend = 0
                else:
                    sys_df = pd.DataFrame()
                    cpu_trend = 0
                    memory_trend = 0
                
                return {
                    'performance': {
                        'current': self.performance_metrics,
                        'trends': {
                            'pnl_trend': pnl_trend,
                            'win_rate_trend': win_rate_trend
                        },
                        'history_length': len(perf_df)
                    },
                    'system': {
                        'current': self.system_metrics,
                        'trends': {
                            'cpu_trend': cpu_trend,
                            'memory_trend': memory_trend
                        },
                        'history_length': len(sys_df)
                    },
                    'alerts': {
                        'active': len(self.active_alerts),
                        'recent': len([a for a in self.alert_history if (datetime.utcnow() - a['timestamp']).hours < 24])
                    },
                    'monitoring': {
                        'active': self._monitoring_active,
                        'uptime': time.time() - (self.system_history[0]['timestamp'].timestamp() if self.system_history else time.time())
                    }
                }
            
        except Exception as e:
            self.logger.error(f"❌ [Observability] Error getting performance dashboard: {e}")
            return {}

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary"""
        try:
            with self._lock:
                # Count alerts by severity
                severity_counts = defaultdict(int)
                category_counts = defaultdict(int)
                
                for alert in self.active_alerts:
                    severity_counts[alert['severity']] += 1
                    category_counts[alert['category']] += 1
                
                return {
                    'active_alerts': len(self.active_alerts),
                    'severity_breakdown': dict(severity_counts),
                    'category_breakdown': dict(category_counts),
                    'recent_alerts': len([a for a in self.alert_history if (datetime.utcnow() - a['timestamp']).hours < 24]),
                    'alerts': self.active_alerts[-10:]  # Last 10 alerts
                }
            
        except Exception as e:
            self.logger.error(f"❌ [Observability] Error getting alert summary: {e}")
            return {}

    def export_metrics(self, filepath: str = None) -> str:
        """Export metrics to file"""
        try:
            if filepath is None:
                filepath = f"observability_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            
            export_data = {
                'export_timestamp': datetime.utcnow().isoformat(),
                'metrics_summary': self.get_metrics_summary(),
                'performance_dashboard': self.get_performance_dashboard(),
                'alert_summary': self.get_alert_summary(),
                'system_metrics': list(self.system_history),
                'performance_metrics': list(self.performance_history),
                'trading_events': list(self.trading_history),
                'alert_history': self.alert_history
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"📁 [Observability] Metrics exported to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"❌ [Observability] Error exporting metrics: {e}")
            return ""

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        try:
            health_score = 100
            issues = []
            
            # Check system health
            if self.system_metrics:
                cpu_percent = self.system_metrics.get('cpu_percent', 0)
                memory_percent = self.system_metrics.get('memory_percent', 0)
                
                if cpu_percent > 80:
                    health_score -= 20
                    issues.append(f"High CPU usage: {cpu_percent:.1f}%")
                
                if memory_percent > 80:
                    health_score -= 20
                    issues.append(f"High memory usage: {memory_percent:.1f}%")
            
            # Check alert health
            critical_alerts = len([a for a in self.active_alerts if a['severity'] == 'critical'])
            warning_alerts = len([a for a in self.active_alerts if a['severity'] == 'warning'])
            
            if critical_alerts > 0:
                health_score -= 30
                issues.append(f"{critical_alerts} critical alerts active")
            
            if warning_alerts > 3:
                health_score -= 10
                issues.append(f"{warning_alerts} warning alerts active")
            
            # Check monitoring health
            if not self._monitoring_active:
                health_score -= 50
                issues.append("Monitoring system not active")
            
            # Determine health status
            if health_score >= 90:
                status = 'healthy'
            elif health_score >= 70:
                status = 'warning'
            elif health_score >= 50:
                status = 'degraded'
            else:
                status = 'critical'
            
            return {
                'status': status,
                'health_score': max(health_score, 0),
                'issues': issues,
                'monitoring_active': self._monitoring_active,
                'active_alerts': len(self.active_alerts),
                'last_update': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ [Observability] Error getting health status: {e}")
            return {'status': 'error', 'health_score': 0, 'issues': [str(e)]}