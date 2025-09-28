"""
Simplified Integration Test for Enhanced Trading Bot
Focus on core functionality validation
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add the workspace to Python path
sys.path.append('/workspace')

def test_bot_import():
    """Test if bot can be imported successfully"""
    print("🔧 Testing bot import...")
    try:
        # Import the main bot file
        import importlib.util
        spec = importlib.util.spec_from_file_location("bot", "/workspace/Bot-Trading_Swing.py")
        bot_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bot_module)
        
        print("✅ Bot imported successfully")
        return True, bot_module
    except Exception as e:
        print(f"❌ Bot import failed: {e}")
        return False, None

def test_portfolio_environment():
    """Test PortfolioEnvironment creation"""
    print("\n🎯 Testing PortfolioEnvironment...")
    try:
        # Import bot module
        success, bot_module = test_bot_import()
        if not success:
            return False
        
        # Create mock data with proper format including features
        mock_dfs = {
            'EURUSD': pd.DataFrame({
                'close': [1.1000, 1.1010, 1.1020, 1.1030, 1.1040],
                'high': [1.1010, 1.1020, 1.1030, 1.1040, 1.1050],
                'low': [1.0990, 1.1000, 1.1010, 1.1020, 1.1030],
                'open': [1.1000, 1.1010, 1.1020, 1.1030, 1.1040],
                'volume': [1000, 1100, 1200, 1300, 1400],
                'rsi_14': [45.5, 50.2, 55.8, 60.1, 65.3],
                'macd': [0.001, 0.002, 0.003, 0.004, 0.005],
                'atr': [0.005, 0.006, 0.007, 0.008, 0.009],
                'trend_strength': [1.2, 1.5, 1.8, 2.1, 2.4]
            }),
            'GBPUSD': pd.DataFrame({
                'close': [1.2500, 1.2510, 1.2520, 1.2530, 1.2540],
                'high': [1.2510, 1.2520, 1.2530, 1.2540, 1.2550],
                'low': [1.2490, 1.2500, 1.2510, 1.2520, 1.2530],
                'open': [1.2500, 1.2510, 1.2520, 1.2530, 1.2540],
                'volume': [2000, 2100, 2200, 2300, 2400],
                'rsi_14': [48.2, 52.1, 56.7, 61.3, 66.8],
                'macd': [0.002, 0.003, 0.004, 0.005, 0.006],
                'atr': [0.006, 0.007, 0.008, 0.009, 0.010],
                'trend_strength': [1.3, 1.6, 1.9, 2.2, 2.5]
            })
        }
        
        mock_feature_columns = ['rsi_14', 'macd', 'atr', 'trend_strength']
        symbols = ['EURUSD', 'GBPUSD']
        
        # Create environment
        env = bot_module.PortfolioEnvironment(
            dict_df_features=mock_dfs,
            dict_feature_columns={symbol: mock_feature_columns for symbol in symbols},
            symbols=symbols,
            initial_balance=10000
        )
        
        # Test reset
        obs, info = env.reset()
        
        # Test step
        action = [0, 0]  # HOLD for both symbols
        obs, reward, terminated, truncated, info = env.step(action)
        
        print("✅ PortfolioEnvironment test passed")
        print(f"   - Observation shape: {obs.shape}")
        print(f"   - Reward: {reward:.4f}")
        print(f"   - Balance: {env.balance}")
        
        return True
        
    except Exception as e:
        print(f"❌ PortfolioEnvironment test failed: {e}")
        return False

def test_garch_volatility():
    """Test GARCH volatility forecasting"""
    print("\n⚠️ Testing GARCH volatility forecasting...")
    try:
        # Import bot module
        success, bot_module = test_bot_import()
        if not success:
            return False
        
        # Create risk manager
        risk_manager = bot_module.AdvancedRiskManager()
        
        # Mock data manager
        mock_data_manager = type('MockDataManager', (), {})()
        mock_df = pd.DataFrame({
            'close': np.random.randn(250).cumsum() + 100  # Random walk
        })
        mock_data_manager.fetch_multi_timeframe_data = lambda *args, **kwargs: {'H1': mock_df}
        risk_manager.data_manager = mock_data_manager
        
        # Test GARCH forecasting
        volatility = risk_manager._forecast_volatility('EURUSD')
        
        print(f"✅ GARCH volatility test passed")
        print(f"   - Forecasted volatility: {volatility:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ GARCH volatility test failed: {e}")
        return False

def test_dynamic_correlation():
    """Test dynamic correlation matrix"""
    print("\n⚠️ Testing dynamic correlation matrix...")
    try:
        # Import bot module
        success, bot_module = test_bot_import()
        if not success:
            return False
        
        # Create risk manager
        mock_data_manager = type('MockDataManager', (), {})()
        risk_manager = bot_module.PortfolioRiskManager(['EURUSD', 'GBPUSD', 'USDJPY'], mock_data_manager)
        
        # Mock returns data
        mock_returns = pd.DataFrame({
            'EURUSD': np.random.randn(100),
            'GBPUSD': np.random.randn(100),
            'USDJPY': np.random.randn(100)
        })
        
        mock_data_manager.fetch_multi_timeframe_data = lambda *args, **kwargs: {'D1': mock_returns}
        
        # Test correlation matrix update
        risk_manager.update_correlation_matrix(force_update=True)
        
        print("✅ Dynamic correlation test passed")
        print(f"   - Correlation matrix shape: {risk_manager.correlation_matrix.shape if hasattr(risk_manager.correlation_matrix, 'shape') else 'N/A'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dynamic correlation test failed: {e}")
        return False

def test_ensemble_model():
    """Test EnsembleModel with dynamic weights"""
    print("\n🤖 Testing EnsembleModel...")
    try:
        # Import bot module
        success, bot_module = test_bot_import()
        if not success:
            return False
        
        # Create ensemble model
        ensemble = bot_module.EnsembleModel()
        
        # Mock models
        ensemble.models = {
            'xgb': type('MockModel', (), {'predict_proba': lambda x: np.array([[0.3, 0.7]])})(),
            'rf': type('MockModel', (), {'predict_proba': lambda x: np.array([[0.4, 0.6]])})(),
            'lgb': type('MockModel', (), {'predict_proba': lambda x: np.array([[0.5, 0.5]])})()
        }
        
        # Test performance tracking
        ensemble.update_performance('xgb', 1)  # Correct prediction
        ensemble.update_performance('rf', 0)   # Incorrect prediction
        ensemble.update_performance('lgb', 1)  # Correct prediction
        
        # Test dynamic weight calculation
        ensemble._update_dynamic_weights()
        
        print("✅ EnsembleModel test passed")
        print(f"   - Dynamic weights: {ensemble.model_weights_dynamic}")
        
        return True
        
    except Exception as e:
        print(f"❌ EnsembleModel test failed: {e}")
        return False

def test_hierarchical_rl():
    """Test Hierarchical RL components"""
    print("\n🎯 Testing Hierarchical RL...")
    try:
        # Import bot module
        success, bot_module = test_bot_import()
        if not success:
            return False
        
        # Test Master Agent
        master_agent = bot_module.MasterRLAgent()
        risk_level = master_agent.predict_risk_level([1.0, 0.5, 0.02, 0.6])
        
        # Test Worker Agent
        worker_agent = bot_module.WorkerRLAgent('EURUSD')
        action = worker_agent.predict_action([0.5, 0.6, 0.7], 0.6)
        
        # Test Coordinator
        coordinator = bot_module.HierarchicalRLCoordinator(['EURUSD', 'GBPUSD'])
        
        print("✅ Hierarchical RL test passed")
        print(f"   - Master risk level: {risk_level}")
        print(f"   - Worker action: {action}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hierarchical RL test failed: {e}")
        return False

def main():
    """Run all simplified tests"""
    print("🚀 STARTING SIMPLIFIED INTEGRATION TESTS")
    print("="*60)
    
    tests = [
        ("Bot Import", test_bot_import),
        ("Portfolio Environment", test_portfolio_environment),
        ("GARCH Volatility", test_garch_volatility),
        ("Dynamic Correlation", test_dynamic_correlation),
        ("Ensemble Model", test_ensemble_model),
        ("Hierarchical RL", test_hierarchical_rl)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_name == "Bot Import":
                success, _ = test_func()
            else:
                success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:25}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Bot is ready for production.")
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)