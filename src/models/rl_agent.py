"""
RL Agent - Reinforcement Learning agent implementation
Refactored from the original Bot-Trading_Swing.py
"""

import numpy as np
import pandas as pd
import logging
import gym
from gym import spaces
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import joblib
import os

# Import configuration and constants
from Bot_Trading_Swing_Refactored import (
    ML_CONFIG, BOT_LOGGERS, SYMBOLS, SYMBOL_ALLOCATION
)


class PortfolioEnvironment(gym.Env):
    """
    Portfolio Environment for Reinforcement Learning
    Custom gym environment for trading portfolio management
    """

    def __init__(self, symbols: List[str], initial_balance: float = 10000.0):
        """Initialize the portfolio environment"""
        super(PortfolioEnvironment, self).__init__()
        
        self.symbols = symbols
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions = {symbol: 0.0 for symbol in symbols}
        self.prices = {symbol: 1.0 for symbol in symbols}
        
        # Action space: [hold, buy, sell] for each symbol
        self.action_space = spaces.MultiDiscrete([3] * len(symbols))
        
        # Observation space: [balance, positions, prices, technical indicators]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(len(symbols) * 10 + 1,), dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.max_steps = 1000
        self.episode_reward = 0.0
        self.episode_history = []

    def reset(self):
        """Reset the environment"""
        self.current_balance = self.initial_balance
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_history = []
        
        return self._get_observation()

    def step(self, action):
        """Execute one step in the environment"""
        # Execute actions
        reward = self._execute_actions(action)
        
        # Update state
        self.current_step += 1
        self.episode_reward += reward
        
        # Check if episode is done
        done = self.current_step >= self.max_steps or self.current_balance <= 0
        
        # Get next observation
        observation = self._get_observation()
        
        # Store step info
        step_info = {
            'step': self.current_step,
            'balance': self.current_balance,
            'reward': reward,
            'positions': self.positions.copy(),
            'prices': self.prices.copy()
        }
        self.episode_history.append(step_info)
        
        return observation, reward, done, {}

    def _execute_actions(self, action: np.ndarray) -> float:
        """Execute trading actions and calculate reward"""
        try:
            total_reward = 0.0
            
            for i, symbol in enumerate(self.symbols):
                action_type = action[i]  # 0: hold, 1: buy, 2: sell
                
                if action_type == 1:  # Buy
                    reward = self._buy_symbol(symbol)
                elif action_type == 2:  # Sell
                    reward = self._sell_symbol(symbol)
                else:  # Hold
                    reward = self._hold_symbol(symbol)
                
                total_reward += reward
            
            return total_reward
            
        except Exception as e:
            logging.error(f"Error executing actions: {e}")
            return 0.0

    def _buy_symbol(self, symbol: str) -> float:
        """Buy symbol and calculate reward"""
        try:
            # Simple buy logic - buy 1 unit if we have enough balance
            if self.current_balance >= self.prices[symbol]:
                self.current_balance -= self.prices[symbol]
                self.positions[symbol] += 1.0
                return 0.0  # No immediate reward for buying
            return -0.1  # Penalty for insufficient balance
            
        except Exception as e:
            logging.error(f"Error buying {symbol}: {e}")
            return -0.1

    def _sell_symbol(self, symbol: str) -> float:
        """Sell symbol and calculate reward"""
        try:
            # Simple sell logic - sell 1 unit if we have position
            if self.positions[symbol] > 0:
                self.current_balance += self.prices[symbol]
                self.positions[symbol] -= 1.0
                return 0.1  # Small reward for selling
            return -0.1  # Penalty for selling without position
            
        except Exception as e:
            logging.error(f"Error selling {symbol}: {e}")
            return -0.1

    def _hold_symbol(self, symbol: str) -> float:
        """Hold symbol and calculate reward"""
        try:
            # Small reward for holding based on price movement
            # This is a simplified implementation
            return 0.0
            
        except Exception as e:
            logging.error(f"Error holding {symbol}: {e}")
            return 0.0

    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        try:
            observation = []
            
            # Add balance
            observation.append(self.current_balance / self.initial_balance)
            
            # Add symbol data
            for symbol in self.symbols:
                # Price (normalized)
                observation.append(self.prices[symbol])
                
                # Position
                observation.append(self.positions[symbol])
                
                # Technical indicators (simplified)
                for _ in range(8):  # Placeholder for technical indicators
                    observation.append(0.0)
            
            return np.array(observation, dtype=np.float32)
            
        except Exception as e:
            logging.error(f"Error getting observation: {e}")
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

    def update_prices(self, new_prices: Dict[str, float]):
        """Update symbol prices"""
        self.prices.update(new_prices)

    def get_portfolio_value(self) -> float:
        """Get total portfolio value"""
        try:
            total_value = self.current_balance
            for symbol, position in self.positions.items():
                total_value += position * self.prices[symbol]
            return total_value
        except:
            return self.current_balance


class RLAgent:
    """
    Reinforcement Learning Agent for Trading
    Handles RL model training and inference
    """

    def __init__(self, symbol: str = None, model_type: str = "PPO"):
        """Initialize the RL Agent"""
        self.logger = BOT_LOGGERS['MLModels']
        self.symbol = symbol
        self.model_type = model_type
        self.logger.info(f"🤖 [RLAgent] Initializing RL Agent for {symbol}...")
        
        # Environment
        self.env = None
        self.model = None
        self.is_trained = False
        
        # Training parameters
        self.training_config = {
            'total_timesteps': 100000,
            'learning_rate': 3e-4,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5
        }
        
        # Performance tracking
        self.training_history = []
        self.performance_metrics = {}
        self.last_training_date = None
        
        self.logger.info("✅ [RLAgent] RL Agent initialized successfully")

    def initialize_environment(self, symbols: List[str] = None):
        """Initialize the trading environment"""
        try:
            if symbols is None:
                symbols = SYMBOLS
            
            self.env = PortfolioEnvironment(symbols)
            self.logger.info(f"🌍 [RLAgent] Environment initialized with {len(symbols)} symbols")
            
        except Exception as e:
            self.logger.error(f"❌ [RLAgent] Error initializing environment: {e}")
            raise

    def initialize_model(self):
        """Initialize the RL model"""
        try:
            if self.env is None:
                self.initialize_environment()
            
            # For now, we'll use a simple policy network
            # In a real implementation, you would use stable-baselines3 or similar
            self.model = self._create_simple_model()
            
            self.logger.info("🤖 [RLAgent] Model initialized")
            
        except Exception as e:
            self.logger.error(f"❌ [RLAgent] Error initializing model: {e}")
            raise

    def _create_simple_model(self):
        """Create a simple neural network model"""
        try:
            # This is a placeholder for a real RL model
            # In practice, you would use stable-baselines3 or implement your own
            class SimpleModel:
                def __init__(self, env):
                    self.env = env
                    self.weights = np.random.randn(env.observation_space.shape[0], env.action_space.nvec[0])
                
                def predict(self, observation):
                    # Simple linear policy
                    action_probs = np.dot(observation, self.weights)
                    action_probs = np.exp(action_probs) / np.sum(np.exp(action_probs))
                    action = np.random.choice(len(action_probs), p=action_probs)
                    return np.array([action] * len(self.env.symbols))
                
                def learn(self, total_timesteps):
                    # Placeholder for learning
                    pass
            
            return SimpleModel(self.env)
            
        except Exception as e:
            self.logger.error(f"❌ [RLAgent] Error creating simple model: {e}")
            return None

    def train(self, total_timesteps: int = None) -> Dict[str, Any]:
        """
        Train the RL agent
        
        Args:
            total_timesteps: Number of training timesteps
            
        Returns:
            Training results
        """
        try:
            if total_timesteps is None:
                total_timesteps = self.training_config['total_timesteps']
            
            if self.model is None:
                self.initialize_model()
            
            self.logger.info(f"🎓 [RLAgent] Starting training for {total_timesteps} timesteps...")
            
            # Training loop (simplified)
            episode_rewards = []
            episode_lengths = []
            
            for episode in range(100):  # Simplified training loop
                obs = self.env.reset()
                episode_reward = 0
                episode_length = 0
                
                while True:
                    action = self.model.predict(obs)
                    obs, reward, done, info = self.env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    
                    if done:
                        break
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                if episode % 10 == 0:
                    avg_reward = np.mean(episode_rewards[-10:])
                    self.logger.info(f"🎓 [RLAgent] Episode {episode}, Avg Reward: {avg_reward:.4f}")
            
            # Store training results
            training_results = {
                'total_timesteps': total_timesteps,
                'episodes': len(episode_rewards),
                'avg_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'avg_episode_length': np.mean(episode_lengths),
                'training_date': datetime.utcnow().isoformat()
            }
            
            self.training_history.append(training_results)
            self.is_trained = True
            self.last_training_date = datetime.utcnow()
            
            self.logger.info(f"✅ [RLAgent] Training completed - Avg Reward: {training_results['avg_reward']:.4f}")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"❌ [RLAgent] Error during training: {e}")
            raise

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """
        Make prediction using trained model
        
        Args:
            observation: Current state observation
            
        Returns:
            Predicted action
        """
        try:
            if not self.is_trained or self.model is None:
                # Return random action if not trained
                return np.random.randint(0, 3, size=len(SYMBOLS))
            
            action = self.model.predict(observation)
            return action
            
        except Exception as e:
            self.logger.error(f"❌ [RLAgent] Error making prediction: {e}")
            return np.random.randint(0, 3, size=len(SYMBOLS))

    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluate the trained agent
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Evaluation results
        """
        try:
            if not self.is_trained or self.model is None:
                raise ValueError("Agent must be trained before evaluation")
            
            self.logger.info(f"📊 [RLAgent] Evaluating agent over {num_episodes} episodes...")
            
            episode_rewards = []
            episode_lengths = []
            final_portfolio_values = []
            
            for episode in range(num_episodes):
                obs = self.env.reset()
                episode_reward = 0
                episode_length = 0
                
                while True:
                    action = self.model.predict(obs)
                    obs, reward, done, info = self.env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    
                    if done:
                        break
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                final_portfolio_values.append(self.env.get_portfolio_value())
            
            # Calculate evaluation metrics
            evaluation_results = {
                'num_episodes': num_episodes,
                'avg_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'min_reward': np.min(episode_rewards),
                'max_reward': np.max(episode_rewards),
                'avg_episode_length': np.mean(episode_lengths),
                'avg_final_portfolio_value': np.mean(final_portfolio_values),
                'std_final_portfolio_value': np.std(final_portfolio_values),
                'evaluation_date': datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"✅ [RLAgent] Evaluation completed - Avg Reward: {evaluation_results['avg_reward']:.4f}")
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"❌ [RLAgent] Error during evaluation: {e}")
            raise

    def should_retrain(self, performance_threshold: float = None) -> bool:
        """
        Check if agent should be retrained
        
        Args:
            performance_threshold: Minimum performance threshold
            
        Returns:
            True if agent should be retrained
        """
        try:
            if performance_threshold is None:
                performance_threshold = ML_CONFIG.get('MIN_F1_SCORE', 0.35)
            
            # Check if agent is trained
            if not self.is_trained:
                return True
            
            # Check training history
            if self.training_history:
                latest_avg_reward = self.training_history[-1].get('avg_reward', 0)
                if latest_avg_reward < performance_threshold:
                    self.logger.info(f"🔄 [RLAgent] Agent performance ({latest_avg_reward:.4f}) below threshold ({performance_threshold})")
                    return True
            
            # Check if agent is stale
            if self.last_training_date:
                days_since_training = (datetime.utcnow() - self.last_training_date).days
                if days_since_training >= 7:  # Retrain weekly
                    self.logger.info(f"🔄 [RLAgent] Agent is stale ({days_since_training} days since last training)")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ [RLAgent] Error checking retrain condition: {e}")
            return True

    def save_model(self, filepath: str = None):
        """Save trained model to file"""
        try:
            if not self.is_trained or self.model is None:
                raise ValueError("Model must be trained before saving")
            
            if filepath is None:
                filepath = f"models/rl_agent_{self.symbol}.joblib"
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model data
            model_data = {
                'model': self.model,
                'training_history': self.training_history,
                'performance_metrics': self.performance_metrics,
                'last_training_date': self.last_training_date,
                'symbol': self.symbol,
                'model_type': self.model_type,
                'training_config': self.training_config
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"💾 [RLAgent] Model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ [RLAgent] Error saving model: {e}")
            raise

    def load_model(self, filepath: str = None):
        """Load trained model from file"""
        try:
            if filepath is None:
                filepath = f"models/rl_agent_{self.symbol}.joblib"
            
            if not os.path.exists(filepath):
                self.logger.warning(f"⚠️ [RLAgent] Model file not found: {filepath}")
                return False
            
            # Load model data
            model_data = joblib.load(filepath)
            
            # Restore model state
            self.model = model_data.get('model')
            self.training_history = model_data.get('training_history', [])
            self.performance_metrics = model_data.get('performance_metrics', {})
            self.last_training_date = model_data.get('last_training_date')
            self.model_type = model_data.get('model_type', self.model_type)
            self.training_config = model_data.get('training_config', self.training_config)
            
            self.is_trained = self.model is not None
            
            self.logger.info(f"📂 [RLAgent] Model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ [RLAgent] Error loading model: {e}")
            return False

    def get_agent_summary(self) -> Dict[str, Any]:
        """Get agent summary information"""
        try:
            summary = {
                'symbol': self.symbol,
                'model_type': self.model_type,
                'is_trained': self.is_trained,
                'last_training_date': self.last_training_date,
                'training_history_length': len(self.training_history)
            }
            
            if self.training_history:
                latest_training = self.training_history[-1]
                summary['latest_avg_reward'] = latest_training.get('avg_reward', 0)
                summary['latest_episodes'] = latest_training.get('episodes', 0)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ [RLAgent] Error getting agent summary: {e}")
            return {}