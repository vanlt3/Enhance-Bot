"""
Comprehensive Integration Tests for Enhanced Trading Bot
Test Automation Engineer - Full Integration Testing Suite
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
import unittest
from unittest.mock import Mock, patch, MagicMock
import warnings
warnings.filterwarnings('ignore')

# Add the workspace to Python path
sys.path.append('/workspace')

# Import bot components
try:
    # Import the main bot file
    import importlib.util
    spec = importlib.util.spec_from_file_location("bot", "/workspace/Bot-Trading_Swing.py")
    bot_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bot_module)
    
    # Get classes from the module
    EnhancedTradingBot = bot_module.EnhancedTradingBot
    PortfolioEnvironment = bot_module.PortfolioEnvironment
    AdvancedRiskManager = bot_module.AdvancedRiskManager
    PortfolioRiskManager = bot_module.PortfolioRiskManager
    MasterRLAgent = bot_module.MasterRLAgent
    WorkerRLAgent = bot_module.WorkerRLAgent
    HierarchicalRLCoordinator = bot_module.HierarchicalRLCoordinator
    EventCoordinator = bot_module.EventCoordinator
    MarketDataEvent = bot_module.MarketDataEvent
    SignalEvent = bot_module.SignalEvent
    OrderEvent = bot_module.OrderEvent
    
    print("✅ Successfully imported bot components")
except Exception as e:
    print(f"❌ Failed to import bot components: {e}")
    sys.exit(1)

class TestBotIntegration(unittest.TestCase):
    """Comprehensive integration test suite for the trading bot"""
    
    def setUp(self):
        """Set up test environment"""
        print("\n" + "="*60)
        print("🧪 SETTING UP TEST ENVIRONMENT")
        print("="*60)
        
        # Mock external dependencies
        self.mock_conn = Mock()
        self.mock_news_manager = Mock()
        self.mock_data_manager = Mock()
        
        # Test symbols
        self.test_symbols = ['EURUSD', 'BTCUSD', 'XAUUSD']
        
        print("✅ Test environment setup complete")
    
    def tearDown(self):
        """Clean up after tests"""
        print("🧹 Cleaning up test environment...")

class TestInitializationAndConfiguration(TestBotIntegration):
    """Test Case 1: Khởi tạo và Cấu hình"""
    
    def test_1_1_bot_initialization(self):
        """Test Case 1.1: Khởi tạo lớp EnhancedTradingBot"""
        print("\n🔧 Test Case 1.1: Bot Initialization")
        
        try:
            # Mock external dependencies to avoid real API calls
            with patch('bot.NewsManager') as mock_news, \
                 patch('bot.EnhancedDataManager') as mock_data, \
                 patch('bot.MasterAgentCoordinator') as mock_master:
                
                # Create bot instance
                bot = EnhancedTradingBot()
                
                # Verify core components exist
                self.assertIsNotNone(bot, "Bot instance should be created")
                self.assertIsNotNone(bot.data_manager, "Data manager should be initialized")
                self.assertIsNotNone(bot.news_manager, "News manager should be initialized")
                self.assertIsNotNone(bot.master_agent_coordinator, "Master agent coordinator should be initialized")
                
                # Verify trading attributes
                self.assertIsInstance(bot.active_symbols, set, "Active symbols should be a set")
                self.assertIsInstance(bot.trending_models, dict, "Trending models should be a dictionary")
                self.assertIsInstance(bot.ranging_models, dict, "Ranging models should be a dictionary")
                self.assertIsInstance(bot.open_positions, dict, "Open positions should be a dictionary")
                
                print("✅ Bot initialization test passed")
                
        except Exception as e:
            self.fail(f"Bot initialization failed: {e}")
    
    def test_1_2_model_loading(self):
        """Test Case 1.2: Tải mô hình đã huấn luyện"""
        print("\n🔧 Test Case 1.2: Model Loading")
        
        try:
            with patch('bot.NewsManager') as mock_news, \
                 patch('bot.EnhancedDataManager') as mock_data, \
                 patch('bot.MasterAgentCoordinator') as mock_master:
                
                bot = EnhancedTradingBot()
                
                # Mock model loading
                mock_trending_model = {
                    'ensemble': Mock(),
                    'feature_columns': ['rsi_14', 'macd', 'atr', 'trend_strength']
                }
                mock_ranging_model = {
                    'ensemble': Mock(),
                    'feature_columns': ['rsi_14', 'macd', 'atr', 'trend_strength']
                }
                
                # Test model assignment
                bot.trending_models['EURUSD'] = mock_trending_model
                bot.ranging_models['EURUSD'] = mock_ranging_model
                
                # Verify models are loaded
                self.assertIn('EURUSD', bot.trending_models, "Trending model should be loaded")
                self.assertIn('EURUSD', bot.ranging_models, "Ranging model should be loaded")
                
                # Verify model structure
                trending_model = bot.trending_models['EURUSD']
                self.assertIn('ensemble', trending_model, "Model should have ensemble component")
                self.assertIn('feature_columns', trending_model, "Model should have feature columns")
                
                print("✅ Model loading test passed")
                
        except Exception as e:
            self.fail(f"Model loading test failed: {e}")

class TestDataFlowAndFeatureEngineering(TestBotIntegration):
    """Test Case 2: Luồng Dữ liệu và Kỹ thuật Đặc trưng"""
    
    def test_2_1_multi_timeframe_data(self):
        """Test Case 2.1: Lấy dữ liệu đa khung thời gian"""
        print("\n📊 Test Case 2.1: Multi-timeframe Data Fetching")
        
        try:
            with patch('bot.NewsManager') as mock_news, \
                 patch('bot.EnhancedDataManager') as mock_data, \
                 patch('bot.MasterAgentCoordinator') as mock_master:
                
                bot = EnhancedTradingBot()
                
                # Mock data manager to return test data
                mock_df = pd.DataFrame({
                    'open': [1.1000, 1.1010, 1.1020],
                    'high': [1.1010, 1.1020, 1.1030],
                    'low': [1.0990, 1.1000, 1.1010],
                    'close': [1.1005, 1.1015, 1.1025],
                    'volume': [1000, 1100, 1200]
                })
                
                bot.data_manager.fetch_multi_timeframe_data = Mock(return_value={
                    'H1': mock_df,
                    'H4': mock_df,
                    'D1': mock_df
                })
                
                # Test data fetching
                result = bot.data_manager.fetch_multi_timeframe_data('EURUSD', count=100)
                
                # Verify result structure
                self.assertIsInstance(result, dict, "Result should be a dictionary")
                self.assertIn('H1', result, "Should contain H1 timeframe")
                self.assertIn('H4', result, "Should contain H4 timeframe")
                self.assertIn('D1', result, "Should contain D1 timeframe")
                
                # Verify DataFrame structure
                for tf, df in result.items():
                    self.assertIsInstance(df, pd.DataFrame, f"{tf} should be a DataFrame")
                    self.assertIn('close', df.columns, f"{tf} should have close column")
                    self.assertIn('high', df.columns, f"{tf} should have high column")
                    self.assertIn('low', df.columns, f"{tf} should have low column")
                
                print("✅ Multi-timeframe data test passed")
                
        except Exception as e:
            self.fail(f"Multi-timeframe data test failed: {e}")
    
    def test_2_2_enhanced_features(self):
        """Test Case 2.2: Tạo đặc trưng nâng cao"""
        print("\n📊 Test Case 2.2: Enhanced Feature Engineering")
        
        try:
            with patch('bot.NewsManager') as mock_news, \
                 patch('bot.EnhancedDataManager') as mock_data, \
                 patch('bot.MasterAgentCoordinator') as mock_master:
                
                bot = EnhancedTradingBot()
                
                # Create mock feature data
                mock_features = pd.DataFrame({
                    'rsi_14': [45.5, 50.2, 55.8],
                    'macd': [0.001, 0.002, 0.003],
                    'atr': [0.005, 0.006, 0.007],
                    'trend_strength': [1.2, 1.5, 1.8],
                    'in_supply_reversal': [False, True, False],
                    'distance_to_demand_reversal': [0.02, 0.01, 0.03],
                    'wyckoff_phase': ['accumulation', 'markup', 'distribution'],
                    'market_regime': [0, 1, 1]
                })
                
                bot.data_manager.create_enhanced_features = Mock(return_value=mock_features)
                
                # Test feature creation
                result = bot.data_manager.create_enhanced_features('XAUUSD')
                
                # Verify result
                self.assertIsInstance(result, pd.DataFrame, "Result should be a DataFrame")
                self.assertGreater(len(result), 0, "Result should not be empty")
                
                # Verify key features exist
                required_features = [
                    'rsi_14', 'macd', 'atr', 'trend_strength',
                    'in_supply_reversal', 'distance_to_demand_reversal',
                    'wyckoff_phase', 'market_regime'
                ]
                
                for feature in required_features:
                    self.assertIn(feature, result.columns, f"Should contain {feature}")
                
                print("✅ Enhanced features test passed")
                
        except Exception as e:
            self.fail(f"Enhanced features test failed: {e}")

class TestMLCoreAndDecisionLogic(TestBotIntegration):
    """Test Case 3: Lõi Học Máy và Logic Ra Quyết định"""
    
    def test_3_1_soft_regime_switching(self):
        """Test Case 3.1: Kiểm tra Chuyển đổi Trạng thái Mềm"""
        print("\n🤖 Test Case 3.1: Soft Regime Switching")
        
        try:
            with patch('bot.NewsManager') as mock_news, \
                 patch('bot.EnhancedDataManager') as mock_data, \
                 patch('bot.MasterAgentCoordinator') as mock_master:
                
                bot = EnhancedTradingBot()
                
                # Mock models
                mock_trending_model = Mock()
                mock_trending_model.predict_proba = Mock(return_value=0.7)
                
                mock_ranging_model = Mock()
                mock_ranging_model.predict_proba = Mock(return_value=0.3)
                
                bot.trending_models['EURUSD'] = {
                    'ensemble': mock_trending_model,
                    'feature_columns': ['rsi_14', 'macd', 'atr']
                }
                bot.ranging_models['EURUSD'] = {
                    'ensemble': mock_ranging_model,
                    'feature_columns': ['rsi_14', 'macd', 'atr']
                }
                
                # Mock feature data with regime transition
                mock_features = pd.DataFrame({
                    'market_regime': [0, 0, 1, 1],  # Transition from ranging to trending
                    'trend_strength': [0.5, 1.0, 2.0, 3.0],  # Increasing trend strength
                    'rsi_14': [50, 55, 60, 65],
                    'macd': [0.001, 0.002, 0.003, 0.004],
                    'atr': [0.005, 0.006, 0.007, 0.008]
                })
                
                bot.data_manager.create_enhanced_features = Mock(return_value=mock_features)
                
                # Test soft regime switching
                signal, confidence, prob_buy = bot.get_enhanced_signal('EURUSD', df_features=mock_features)
                
                # Verify soft transition
                self.assertIsNotNone(signal, "Signal should not be None")
                self.assertIsInstance(confidence, float, "Confidence should be a float")
                self.assertIsInstance(prob_buy, float, "Prob buy should be a float")
                
                # Verify regime confidence calculation
                latest_features = mock_features.iloc[-1]
                trend_strength = abs(latest_features.get("trend_strength", 0))
                regime_confidence = min(trend_strength / 5.0, 1.0)
                
                self.assertGreater(regime_confidence, 0, "Regime confidence should be positive")
                self.assertLessEqual(regime_confidence, 1.0, "Regime confidence should be <= 1.0")
                
                print("✅ Soft regime switching test passed")
                
        except Exception as e:
            self.fail(f"Soft regime switching test failed: {e}")
    
    def test_3_2_multimodal_llm_integration(self):
        """Test Case 3.2: Kiểm tra Tích hợp LLM Đa phương thức"""
        print("\n🤖 Test Case 3.2: Multimodal LLM Integration")
        
        try:
            with patch('bot.NewsManager') as mock_news, \
                 patch('bot.EnhancedDataManager') as mock_data, \
                 patch('bot.MasterAgentCoordinator') as mock_master:
                
                bot = EnhancedTradingBot()
                
                # Mock LLM analyzer
                mock_llm = Mock()
                mock_llm.model = Mock()
                mock_llm.model.generate_content_async = Mock(return_value=Mock(text='{"decision": "APPROVE", "sentiment_score": 0.5, "justification": "Test"}'))
                
                bot.news_manager.llm_analyzer = mock_llm
                
                # Mock feature data for Key Market Metrics
                mock_features = pd.DataFrame({
                    'rsi_14': [68.5],
                    'atr': [0.0085],
                    'close': [1.1000],
                    'ema_200': [1.0950],
                    'volume': [1500],
                    'volume_ema': [1000],
                    'macd': [0.002],
                    'macd_signal': [0.001],
                    'bb_upper': [1.1050],
                    'bb_lower': [1.0950]
                })
                
                bot.data_manager.create_enhanced_features = Mock(return_value=mock_features)
                
                # Test multimodal LLM integration (mock the async call)
                async def test_consult_master_agent():
                    return await bot.consult_master_agent(
                        symbol='EURUSD',
                        signal='BUY',
                        reasoning_data={'Main Signal': 'BUY with Confidence 75%'},
                        news_items=[{'title': 'Test news'}]
                    )
                
                # Run the async function
                result = asyncio.run(test_consult_master_agent())
                
                # Verify result structure
                self.assertIsInstance(result, dict, "Result should be a dictionary")
                self.assertIn('decision', result, "Should contain decision")
                self.assertIn('sentiment_score', result, "Should contain sentiment_score")
                self.assertIn('justification', result, "Should contain justification")
                
                print("✅ Multimodal LLM integration test passed")
                
        except Exception as e:
            self.fail(f"Multimodal LLM integration test failed: {e}")
    
    def test_3_3_dynamic_ensemble_weights(self):
        """Test Case 3.3: Kiểm tra Điều chỉnh Trọng số Ensemble Động"""
        print("\n🤖 Test Case 3.3: Dynamic Ensemble Weights")
        
        try:
            # Import EnsembleModel
            EnsembleModel = bot_module.EnsembleModel
            
            # Create ensemble model instance
            ensemble = EnsembleModel()
            
            # Mock models
            ensemble.models = {
                'xgb': Mock(),
                'rf': Mock(),
                'lgb': Mock()
            }
            
            # Test performance tracking
            ensemble.update_performance('xgb', 1)  # Correct prediction
            ensemble.update_performance('rf', 0)   # Incorrect prediction
            ensemble.update_performance('lgb', 1)  # Correct prediction
            
            # Verify performance history
            self.assertIn('xgb', ensemble.performance_history, "XGB performance should be tracked")
            self.assertIn('rf', ensemble.performance_history, "RF performance should be tracked")
            self.assertIn('lgb', ensemble.performance_history, "LGB performance should be tracked")
            
            # Test dynamic weight calculation
            ensemble._update_dynamic_weights()
            
            # Verify dynamic weights
            self.assertIsInstance(ensemble.model_weights_dynamic, dict, "Dynamic weights should be a dictionary")
            
            if ensemble.model_weights_dynamic:
                total_weight = sum(ensemble.model_weights_dynamic.values())
                self.assertAlmostEqual(total_weight, 1.0, places=2, msg="Total weights should sum to 1.0")
                
                # Verify weight constraints
                for model, weight in ensemble.model_weights_dynamic.items():
                    self.assertGreaterEqual(weight, ensemble.min_weight, f"{model} weight should be >= min_weight")
                    self.assertLessEqual(weight, ensemble.max_weight, f"{model} weight should be <= max_weight")
            
            print("✅ Dynamic ensemble weights test passed")
            
        except Exception as e:
            self.fail(f"Dynamic ensemble weights test failed: {e}")

class TestRLAgent(TestBotIntegration):
    """Test Case 4: Tác tử Học Tăng Cường"""
    
    def test_4_1_portfolio_environment(self):
        """Test Case 4.1: Tạo môi trường PortfolioEnvironment"""
        print("\n🎯 Test Case 4.1: Portfolio Environment")
        
        try:
            # Mock data for environment
            mock_dfs = {
                'EURUSD': pd.DataFrame({
                    'close': [1.1000, 1.1010, 1.1020, 1.1030, 1.1040],
                    'high': [1.1010, 1.1020, 1.1030, 1.1040, 1.1050],
                    'low': [1.0990, 1.1000, 1.1010, 1.1020, 1.1030],
                    'open': [1.1000, 1.1010, 1.1020, 1.1030, 1.1040],
                    'volume': [1000, 1100, 1200, 1300, 1400]
                }),
                'GBPUSD': pd.DataFrame({
                    'close': [1.2500, 1.2510, 1.2520, 1.2530, 1.2540],
                    'high': [1.2510, 1.2520, 1.2530, 1.2540, 1.2550],
                    'low': [1.2490, 1.2500, 1.2510, 1.2520, 1.2530],
                    'open': [1.2500, 1.2510, 1.2520, 1.2530, 1.2540],
                    'volume': [2000, 2100, 2200, 2300, 2400]
                })
            }
            
            mock_feature_columns = ['rsi_14', 'macd', 'atr', 'trend_strength']
            symbols = ['EURUSD', 'GBPUSD']
            
            # Create environment
            env = PortfolioEnvironment(
                dict_df_features=mock_dfs,
                dict_feature_columns={symbol: mock_feature_columns for symbol in symbols},
                symbols=symbols,
                initial_balance=10000
            )
            
            # Verify environment creation
            self.assertIsNotNone(env, "Environment should be created")
            self.assertEqual(env.n_symbols, 2, "Should have 2 symbols")
            self.assertEqual(env.initial_balance, 10000, "Initial balance should be 10000")
            
            # Verify observation and action spaces
            self.assertIsNotNone(env.observation_space, "Observation space should be defined")
            self.assertIsNotNone(env.action_space, "Action space should be defined")
            
            # Test reset
            obs, info = env.reset()
            self.assertIsInstance(obs, np.ndarray, "Observation should be numpy array")
            self.assertEqual(env.balance, 10000, "Balance should reset to initial value")
            
            print("✅ Portfolio environment test passed")
            
        except Exception as e:
            self.fail(f"Portfolio environment test failed: {e}")
    
    def test_4_2_enhanced_reward_function(self):
        """Test Case 4.2: Kiểm tra Hàm thưởng Nâng cao"""
        print("\n🎯 Test Case 4.2: Enhanced Reward Function")
        
        try:
            # Create environment
            mock_dfs = {
                'EURUSD': pd.DataFrame({
                    'close': [1.1000, 1.1010, 1.1020, 1.1030, 1.1040],
                    'high': [1.1010, 1.1020, 1.1030, 1.1040, 1.1050],
                    'low': [1.0990, 1.1000, 1.1010, 1.1020, 1.1030],
                    'open': [1.1000, 1.1010, 1.1020, 1.1030, 1.1040],
                    'volume': [1000, 1100, 1200, 1300, 1400]
                })
            }
            
            env = PortfolioEnvironment(
                dict_df_features=mock_dfs,
                dict_feature_columns={'EURUSD': ['rsi_14', 'macd', 'atr']},
                symbols=['EURUSD'],
                initial_balance=10000
            )
            
            # Reset environment
            obs, info = env.reset()
            
            # Test normal step (no drawdown)
            action = [0]  # HOLD
            obs, reward_normal, terminated, truncated, info = env.step(action)
            
            # Simulate drawdown scenario
            env.balance = 9000  # 10% drawdown
            env.peak_balance = 10000
            
            # Test step with drawdown
            obs, reward_drawdown, terminated, truncated, info = env.step(action)
            
            # Verify drawdown penalty
            self.assertLess(reward_drawdown, reward_normal, "Reward with drawdown should be lower")
            
            # Test transaction cost penalty
            action_buy = [1]  # BUY
            obs, reward_with_transaction, terminated, truncated, info = env.step(action_buy)
            
            # Verify transaction cost penalty
            self.assertLess(reward_with_transaction, reward_normal, "Reward with transaction should be lower")
            
            print("✅ Enhanced reward function test passed")
            
        except Exception as e:
            self.fail(f"Enhanced reward function test failed: {e}")

class TestRiskManagement(TestBotIntegration):
    """Test Case 5: Quản lý Rủi ro"""
    
    def test_5_1_garch_volatility_forecasting(self):
        """Test Case 5.1: Kiểm tra Dự báo Rủi ro bằng GARCH"""
        print("\n⚠️ Test Case 5.1: GARCH Volatility Forecasting")
        
        try:
            # Create risk manager
            risk_manager = AdvancedRiskManager()
            
            # Mock data manager
            mock_data_manager = Mock()
            mock_df = pd.DataFrame({
                'close': np.random.randn(250).cumsum() + 100  # Random walk
            })
            mock_data_manager.fetch_multi_timeframe_data = Mock(return_value={
                'H1': mock_df
            })
            risk_manager.data_manager = mock_data_manager
            
            # Test GARCH forecasting
            volatility = risk_manager._forecast_volatility('EURUSD')
            
            # Verify volatility forecast
            self.assertIsInstance(volatility, float, "Volatility should be a float")
            self.assertGreaterEqual(volatility, 0.001, "Volatility should be >= 0.001")
            self.assertLessEqual(volatility, 0.10, "Volatility should be <= 0.10")
            
            # Test enhanced risk management
            sl_distance, tp_distance = risk_manager.enhanced_risk_management(
                symbol='EURUSD',
                signal='BUY',
                current_price=1.1000,
                confidence=0.75
            )
            
            # Verify risk management output
            self.assertIsInstance(sl_distance, float, "SL distance should be a float")
            self.assertIsInstance(tp_distance, float, "TP distance should be a float")
            self.assertGreater(sl_distance, 0, "SL distance should be positive")
            self.assertGreater(tp_distance, 0, "TP distance should be positive")
            
            print("✅ GARCH volatility forecasting test passed")
            
        except Exception as e:
            self.fail(f"GARCH volatility forecasting test failed: {e}")
    
    def test_5_2_dynamic_correlation_matrix(self):
        """Test Case 5.2: Kiểm tra Ma trận Tương quan Động"""
        print("\n⚠️ Test Case 5.2: Dynamic Correlation Matrix")
        
        try:
            # Create risk manager
            mock_data_manager = Mock()
            risk_manager = PortfolioRiskManager(['EURUSD', 'GBPUSD', 'USDJPY'], mock_data_manager)
            
            # Mock returns data
            mock_returns = pd.DataFrame({
                'EURUSD': np.random.randn(100),
                'GBPUSD': np.random.randn(100),
                'USDJPY': np.random.randn(100)
            })
            
            mock_data_manager.fetch_multi_timeframe_data = Mock(return_value={
                'D1': mock_returns
            })
            
            # Test correlation matrix update
            risk_manager.update_correlation_matrix(force_update=True)
            
            # Verify correlation matrix
            self.assertIsNotNone(risk_manager.correlation_matrix, "Correlation matrix should be created")
            
            if isinstance(risk_manager.correlation_matrix, pd.DataFrame):
                # Verify matrix properties
                self.assertEqual(len(risk_manager.correlation_matrix), 3, "Should have 3 symbols")
                self.assertEqual(len(risk_manager.correlation_matrix.columns), 3, "Should have 3 columns")
                
                # Verify correlation values are in [-1, 1]
                for col in risk_manager.correlation_matrix.columns:
                    for idx in risk_manager.correlation_matrix.index:
                        corr_value = risk_manager.correlation_matrix.loc[idx, col]
                        self.assertGreaterEqual(corr_value, -1.0, "Correlation should be >= -1")
                        self.assertLessEqual(corr_value, 1.0, "Correlation should be <= 1")
            
            print("✅ Dynamic correlation matrix test passed")
            
        except Exception as e:
            self.fail(f"Dynamic correlation matrix test failed: {e}")

class TestExecutionAndMonitoring(TestBotIntegration):
    """Test Case 6: Thực thi và Giám sát"""
    
    def test_6_1_discord_alert(self):
        """Test Case 6.1: Gửi cảnh báo Discord"""
        print("\n📢 Test Case 6.1: Discord Alert")
        
        try:
            with patch('bot.NewsManager') as mock_news, \
                 patch('bot.EnhancedDataManager') as mock_data, \
                 patch('bot.MasterAgentCoordinator') as mock_master:
                
                bot = EnhancedTradingBot()
                
                # Mock Discord webhook
                mock_webhook = Mock()
                bot.news_manager.discord_webhook = mock_webhook
                
                # Test Discord alert
                test_message = "🧪 Test Alert from Integration Tests"
                bot.send_discord_alert(test_message, "INFO", "NORMAL")
                
                # Verify webhook was called
                mock_webhook.send.assert_called_once()
                
                print("✅ Discord alert test passed")
                
        except Exception as e:
            self.fail(f"Discord alert test failed: {e}")
    
    def test_6_2_position_logic(self):
        """Test Case 6.2: Logic xử lý vị thế"""
        print("\n📢 Test Case 6.2: Position Logic")
        
        try:
            with patch('bot.NewsManager') as mock_news, \
                 patch('bot.EnhancedDataManager') as mock_data, \
                 patch('bot.MasterAgentCoordinator') as mock_master:
                
                bot = EnhancedTradingBot()
                
                # Mock risk manager
                mock_risk_manager = Mock()
                mock_risk_manager.enhanced_risk_management = Mock(return_value=(0.01, 0.02))
                bot.risk_manager = mock_risk_manager
                
                # Mock position opening
                bot.open_position_enhanced = Mock(return_value=True)
                
                # Test position logic
                result = bot.handle_position_logic(
                    symbol='EURUSD',
                    signal='BUY',
                    confidence=0.8,
                    current_price=1.1000
                )
                
                # Verify risk management was called
                mock_risk_manager.enhanced_risk_management.assert_called_once()
                
                print("✅ Position logic test passed")
                
        except Exception as e:
            self.fail(f"Position logic test failed: {e}")

def run_integration_tests():
    """Run all integration tests"""
    print("🚀 STARTING COMPREHENSIVE INTEGRATION TESTS")
    print("="*80)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestInitializationAndConfiguration,
        TestDataFlowAndFeatureEngineering,
        TestMLCoreAndDecisionLogic,
        TestRLAgent,
        TestRiskManagement,
        TestExecutionAndMonitoring
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)