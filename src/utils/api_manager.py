"""
API Manager - External API management and rate limiting
Refactored from the original Bot-Trading_Swing.py
"""

import asyncio
import aiohttp
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import deque
import json

# Import configuration and constants
from Bot_Trading_Swing_Refactored import (
    API_CONFIGS, API_KEYS, API_ENDPOINTS, RATE_LIMITS, BOT_LOGGERS
)


class APIManager:
    """
    API Manager - Handles external API calls with rate limiting and error handling
    Refactored to be more modular and maintainable
    """

    def __init__(self):
        """Initialize the API Manager"""
        self.logger = BOT_LOGGERS['APIManager']
        self.logger.info("🔌 [APIManager] Initializing API Manager...")
        
        # Rate limiting
        self.rate_limiters = {}
        self.last_request_time = {}
        
        # Initialize rate limiters for each API
        for api_name, config in API_CONFIGS.items():
            self.rate_limiters[api_name] = deque()
            self.last_request_time[api_name] = 0
        
        # Session management
        self.session = None
        self.session_timeout = 30
        
        # Request tracking
        self.request_count = {api_name: 0 for api_name in API_CONFIGS.keys()}
        self.error_count = {api_name: 0 for api_name in API_CONFIGS.keys()}
        self.last_error_time = {api_name: 0 for api_name in API_CONFIGS.keys()}
        
        # Circuit breaker
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 300  # 5 minutes
        self.circuit_breaker_active = {api_name: False for api_name in API_CONFIGS.keys()}
        
        self.logger.info("✅ [APIManager] API Manager initialized successfully")

    async def __aenter__(self):
        """Async context manager entry"""
        await self._create_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self._close_session()

    async def _create_session(self):
        """Create aiohttp session"""
        try:
            if self.session is None:
                timeout = aiohttp.ClientTimeout(total=self.session_timeout)
                self.session = aiohttp.ClientSession(timeout=timeout)
                self.logger.debug("🔌 [APIManager] Created aiohttp session")
        except Exception as e:
            self.logger.error(f"❌ [APIManager] Error creating session: {e}")

    async def _close_session(self):
        """Close aiohttp session"""
        try:
            if self.session:
                await self.session.close()
                self.session = None
                self.logger.debug("🔌 [APIManager] Closed aiohttp session")
        except Exception as e:
            self.logger.error(f"❌ [APIManager] Error closing session: {e}")

    def _check_rate_limit(self, api_name: str) -> bool:
        """
        Check rate limit for API
        
        Args:
            api_name: Name of the API service
            
        Returns:
            True if request is allowed, False if rate limited
        """
        try:
            if api_name not in self.rate_limiters:
                self.logger.error(f"❌ [APIManager] Unknown API name: {api_name}")
                return False
            
            if api_name not in API_CONFIGS:
                self.logger.error(f"⚠️ [APIManager] No configuration found for {api_name}")
                return False
                
            current_time = time.time()
            rate_limit = API_CONFIGS[api_name].rate_limit
            
            # Remove requests older than 1 minute
            while (self.rate_limiters[api_name] and 
                   current_time - self.rate_limiters[api_name][0] > 60):
                self.rate_limiters[api_name].popleft()
            
            # Check if exceeding rate limit
            if len(self.rate_limiters[api_name]) >= rate_limit:
                self.logger.warning(f"⚠️ [{api_name}] Rate limit exceeded ({len(self.rate_limiters[api_name])}/{rate_limit})")
                return False
            
            # Add current request
            self.rate_limiters[api_name].append(current_time)
            return True
            
        except Exception as e:
            self.logger.error(f"❌ [APIManager] Error checking rate limit for {api_name}: {e}")
            return False

    def _check_circuit_breaker(self, api_name: str) -> bool:
        """Check if circuit breaker is active for API"""
        try:
            if self.circuit_breaker_active[api_name]:
                current_time = time.time()
                if current_time - self.last_error_time[api_name] > self.circuit_breaker_timeout:
                    # Reset circuit breaker
                    self.circuit_breaker_active[api_name] = False
                    self.error_count[api_name] = 0
                    self.logger.info(f"🔄 [{api_name}] Circuit breaker reset")
                    return True
                else:
                    self.logger.warning(f"🚫 [{api_name}] Circuit breaker active")
                    return False
            return True
        except Exception as e:
            self.logger.error(f"❌ [APIManager] Error checking circuit breaker for {api_name}: {e}")
            return True

    async def _make_request(self, url: str, params: dict = None, headers: dict = None, 
                          timeout: int = 10) -> Optional[dict]:
        """
        Make HTTP request with error handling
        
        Args:
            url: Request URL
            params: Query parameters
            headers: Request headers
            timeout: Request timeout
            
        Returns:
            Response data or None
        """
        try:
            if not self.session:
                await self._create_session()
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    self.logger.warning(f"⚠️ [APIManager] HTTP {response.status} for {url}")
                    return None
                    
        except asyncio.TimeoutError:
            self.logger.warning(f"⏰ [APIManager] Timeout for {url}")
            return None
        except Exception as e:
            self.logger.error(f"❌ [APIManager] Error making request to {url}: {e}")
            return None

    async def get_finhub_quote(self, symbol: str) -> Optional[dict]:
        """Get quote from Finnhub API"""
        try:
            api_name = 'FINHUB'
            
            # Check rate limit and circuit breaker
            if not self._check_rate_limit(api_name) or not self._check_circuit_breaker(api_name):
                return None
            
            # Make request
            url = f"{API_CONFIGS[api_name].base_url}/quote"
            params = {
                'symbol': symbol,
                'token': API_CONFIGS[api_name].api_key
            }
            
            data = await self._make_request(url, params)
            
            if data:
                self.request_count[api_name] += 1
                self.logger.debug(f"✅ [{api_name}] Quote retrieved for {symbol}")
                return data
            else:
                self.error_count[api_name] += 1
                self.last_error_time[api_name] = time.time()
                
                # Check circuit breaker
                if self.error_count[api_name] >= self.circuit_breaker_threshold:
                    self.circuit_breaker_active[api_name] = True
                    self.logger.warning(f"🚫 [{api_name}] Circuit breaker activated")
                
                return None
                
        except Exception as e:
            self.logger.error(f"❌ [APIManager] Error getting Finnhub quote for {symbol}: {e}")
            return None

    async def get_finhub_news(self, symbol: str, from_date: str = None, to_date: str = None) -> Optional[List[dict]]:
        """Get news from Finnhub API"""
        try:
            api_name = 'FINHUB'
            
            # Check rate limit and circuit breaker
            if not self._check_rate_limit(api_name) or not self._check_circuit_breaker(api_name):
                return None
            
            # Set default dates
            if not from_date:
                from_date = (datetime.utcnow() - timedelta(days=7)).strftime('%Y-%m-%d')
            if not to_date:
                to_date = datetime.utcnow().strftime('%Y-%m-%d')
            
            # Make request
            url = f"{API_CONFIGS[api_name].base_url}/company-news"
            params = {
                'symbol': symbol,
                'from': from_date,
                'to': to_date,
                'token': API_CONFIGS[api_name].api_key
            }
            
            data = await self._make_request(url, params)
            
            if data:
                self.request_count[api_name] += 1
                self.logger.debug(f"✅ [{api_name}] News retrieved for {symbol}: {len(data)} articles")
                return data
            else:
                self.error_count[api_name] += 1
                self.last_error_time[api_name] = time.time()
                
                # Check circuit breaker
                if self.error_count[api_name] >= self.circuit_breaker_threshold:
                    self.circuit_breaker_active[api_name] = True
                    self.logger.warning(f"🚫 [{api_name}] Circuit breaker activated")
                
                return None
                
        except Exception as e:
            self.logger.error(f"❌ [APIManager] Error getting Finnhub news for {symbol}: {e}")
            return None

    async def get_marketaux_news(self, symbols: List[str] = None, limit: int = 100) -> Optional[List[dict]]:
        """Get news from Marketaux API"""
        try:
            api_name = 'MARKETAUX'
            
            # Check rate limit and circuit breaker
            if not self._check_rate_limit(api_name) or not self._check_circuit_breaker(api_name):
                return None
            
            # Make request
            url = f"{API_CONFIGS[api_name].base_url}/news/all"
            params = {
                'api_token': API_CONFIGS[api_name].api_key,
                'limit': limit,
                'language': 'en'
            }
            
            if symbols:
                params['symbols'] = ','.join(symbols)
            
            data = await self._make_request(url, params)
            
            if data and 'data' in data:
                self.request_count[api_name] += 1
                self.logger.debug(f"✅ [{api_name}] News retrieved: {len(data['data'])} articles")
                return data['data']
            else:
                self.error_count[api_name] += 1
                self.last_error_time[api_name] = time.time()
                
                # Check circuit breaker
                if self.error_count[api_name] >= self.circuit_breaker_threshold:
                    self.circuit_breaker_active[api_name] = True
                    self.logger.warning(f"🚫 [{api_name}] Circuit breaker activated")
                
                return None
                
        except Exception as e:
            self.logger.error(f"❌ [APIManager] Error getting Marketaux news: {e}")
            return None

    async def get_newsapi_news(self, query: str = None, category: str = None, 
                             country: str = 'us', page_size: int = 100) -> Optional[List[dict]]:
        """Get news from NewsAPI"""
        try:
            api_name = 'NEWSAPI'
            
            # Check rate limit and circuit breaker
            if not self._check_rate_limit(api_name) or not self._check_circuit_breaker(api_name):
                return None
            
            # Make request
            url = f"{API_CONFIGS[api_name].base_url}/top-headlines"
            params = {
                'apiKey': API_CONFIGS[api_name].api_key,
                'pageSize': page_size,
                'country': country
            }
            
            if query:
                params['q'] = query
            if category:
                params['category'] = category
            
            data = await self._make_request(url, params)
            
            if data and 'articles' in data:
                self.request_count[api_name] += 1
                self.logger.debug(f"✅ [{api_name}] News retrieved: {len(data['articles'])} articles")
                return data['articles']
            else:
                self.error_count[api_name] += 1
                self.last_error_time[api_name] = time.time()
                
                # Check circuit breaker
                if self.error_count[api_name] >= self.circuit_breaker_threshold:
                    self.circuit_breaker_active[api_name] = True
                    self.logger.warning(f"🚫 [{api_name}] Circuit breaker activated")
                
                return None
                
        except Exception as e:
            self.logger.error(f"❌ [APIManager] Error getting NewsAPI news: {e}")
            return None

    async def get_eodhd_data(self, symbol: str, from_date: str = None, to_date: str = None) -> Optional[List[dict]]:
        """Get historical data from EODHD API"""
        try:
            api_name = 'EODHD'
            
            # Check rate limit and circuit breaker
            if not self._check_rate_limit(api_name) or not self._check_circuit_breaker(api_name):
                return None
            
            # Set default dates
            if not from_date:
                from_date = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d')
            if not to_date:
                to_date = datetime.utcnow().strftime('%Y-%m-%d')
            
            # Make request
            url = f"{API_CONFIGS[api_name].base_url}/eod/{symbol}"
            params = {
                'api_token': API_CONFIGS[api_name].api_key,
                'from': from_date,
                'to': to_date,
                'fmt': 'json'
            }
            
            data = await self._make_request(url, params)
            
            if data:
                self.request_count[api_name] += 1
                self.logger.debug(f"✅ [{api_name}] Data retrieved for {symbol}: {len(data)} records")
                return data
            else:
                self.error_count[api_name] += 1
                self.last_error_time[api_name] = time.time()
                
                # Check circuit breaker
                if self.error_count[api_name] >= self.circuit_breaker_threshold:
                    self.circuit_breaker_active[api_name] = True
                    self.logger.warning(f"🚫 [{api_name}] Circuit breaker activated")
                
                return None
                
        except Exception as e:
            self.logger.error(f"❌ [APIManager] Error getting EODHD data for {symbol}: {e}")
            return None

    async def get_alpha_vantage_data(self, symbol: str, function: str = 'TIME_SERIES_DAILY') -> Optional[dict]:
        """Get data from Alpha Vantage API"""
        try:
            api_name = 'ALPHA_VANTAGE'
            
            # Check rate limit and circuit breaker
            if not self._check_rate_limit(api_name) or not self._check_circuit_breaker(api_name):
                return None
            
            # Make request
            url = API_CONFIGS[api_name].base_url
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': API_CONFIGS[api_name].api_key,
                'outputsize': 'compact'
            }
            
            data = await self._make_request(url, params)
            
            if data:
                self.request_count[api_name] += 1
                self.logger.debug(f"✅ [{api_name}] Data retrieved for {symbol}")
                return data
            else:
                self.error_count[api_name] += 1
                self.last_error_time[api_name] = time.time()
                
                # Check circuit breaker
                if self.error_count[api_name] >= self.circuit_breaker_threshold:
                    self.circuit_breaker_active[api_name] = True
                    self.logger.warning(f"🚫 [{api_name}] Circuit breaker activated")
                
                return None
                
        except Exception as e:
            self.logger.error(f"❌ [APIManager] Error getting Alpha Vantage data for {symbol}: {e}")
            return None

    def get_api_status(self) -> Dict[str, Any]:
        """Get status of all APIs"""
        try:
            status = {}
            
            for api_name in API_CONFIGS.keys():
                current_time = time.time()
                
                # Calculate rate limit status
                rate_limit_status = "OK"
                if api_name in self.rate_limiters:
                    current_requests = len(self.rate_limiters[api_name])
                    max_requests = API_CONFIGS[api_name].rate_limit
                    if current_requests >= max_requests:
                        rate_limit_status = "LIMITED"
                
                # Circuit breaker status
                circuit_breaker_status = "OK"
                if self.circuit_breaker_active[api_name]:
                    circuit_breaker_status = "ACTIVE"
                
                status[api_name] = {
                    'rate_limit_status': rate_limit_status,
                    'circuit_breaker_status': circuit_breaker_status,
                    'request_count': self.request_count[api_name],
                    'error_count': self.error_count[api_name],
                    'last_error_time': self.last_error_time[api_name],
                    'current_requests': len(self.rate_limiters.get(api_name, [])),
                    'max_requests': API_CONFIGS[api_name].rate_limit
                }
            
            return status
            
        except Exception as e:
            self.logger.error(f"❌ [APIManager] Error getting API status: {e}")
            return {}

    def reset_circuit_breaker(self, api_name: str):
        """Reset circuit breaker for specific API"""
        try:
            if api_name in self.circuit_breaker_active:
                self.circuit_breaker_active[api_name] = False
                self.error_count[api_name] = 0
                self.last_error_time[api_name] = 0
                self.logger.info(f"🔄 [{api_name}] Circuit breaker reset")
        except Exception as e:
            self.logger.error(f"❌ [APIManager] Error resetting circuit breaker for {api_name}: {e}")

    def reset_all_circuit_breakers(self):
        """Reset all circuit breakers"""
        try:
            for api_name in self.circuit_breaker_active.keys():
                self.reset_circuit_breaker(api_name)
            self.logger.info("🔄 [APIManager] All circuit breakers reset")
        except Exception as e:
            self.logger.error(f"❌ [APIManager] Error resetting all circuit breakers: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get API performance summary"""
        try:
            total_requests = sum(self.request_count.values())
            total_errors = sum(self.error_count.values())
            
            success_rate = (total_requests - total_errors) / total_requests if total_requests > 0 else 0
            
            active_circuit_breakers = sum(1 for active in self.circuit_breaker_active.values() if active)
            
            return {
                'total_requests': total_requests,
                'total_errors': total_errors,
                'success_rate': success_rate,
                'active_circuit_breakers': active_circuit_breakers,
                'api_status': self.get_api_status()
            }
            
        except Exception as e:
            self.logger.error(f"❌ [APIManager] Error getting performance summary: {e}")
            return {}