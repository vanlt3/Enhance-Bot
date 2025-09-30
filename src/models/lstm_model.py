"""
LSTM Model - Long Short-Term Memory neural network implementation
Refactored from the original Bot-Trading_Swing.py
"""

import numpy as np
import pandas as pd
import logging
import joblib
import os
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import configuration and constants
from Bot_Trading_Swing_Refactored import (
    ML_CONFIG, BOT_LOGGERS, FeatureConstants
)

# Try to import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. LSTM functionality will be limited.")


class LSTMModel:
    """
    LSTM Model for time series prediction
    Handles LSTM neural network training and inference
    """

    def __init__(self, symbol: str = None, sequence_length: int = 60):
        """Initialize the LSTM Model"""
        self.logger = BOT_LOGGERS['MLModels']
        self.symbol = symbol
        self.sequence_length = sequence_length
        self.logger.info(f"🧠 [LSTMModel] Initializing LSTM Model for {symbol}...")
        
        if not TENSORFLOW_AVAILABLE:
            self.logger.warning("⚠️ [LSTMModel] TensorFlow not available. Using simplified implementation.")
        
        # Model configuration
        self.model_config = {
            'lstm_units': [50, 50],
            'dropout_rate': 0.2,
            'recurrent_dropout': 0.2,
            'dense_units': [25, 1],
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'validation_split': 0.2,
            'early_stopping_patience': 10,
            'reduce_lr_patience': 5
        }
        
        # Model components
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        self.feature_names = []
        
        # Performance tracking
        self.training_history = []
        self.performance_metrics = {}
        self.last_training_date = None
        
        # Data storage
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        self.logger.info("✅ [LSTMModel] LSTM Model initialized successfully")

    def prepare_data(self, data: pd.DataFrame, target_column: str = 'close', 
                    feature_columns: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training
        
        Args:
            data: Input DataFrame
            target_column: Name of target column
            feature_columns: List of feature column names
            
        Returns:
            Tuple of (X, y) arrays
        """
        try:
            self.logger.info("📊 [LSTMModel] Preparing data for LSTM...")
            
            if data.empty:
                raise ValueError("Input data is empty")
            
            # Select features
            if feature_columns is None:
                # Use all numeric columns except target
                numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                if target_column in numeric_columns:
                    numeric_columns.remove(target_column)
                feature_columns = numeric_columns
            
            self.feature_names = feature_columns
            
            # Prepare feature matrix
            features = data[feature_columns].values
            target = data[target_column].values
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Create sequences
            X, y = self._create_sequences(features_scaled, target)
            
            self.logger.info(f"✅ [LSTMModel] Data prepared - X shape: {X.shape}, y shape: {y.shape}")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"❌ [LSTMModel] Error preparing data: {e}")
            raise

    def _create_sequences(self, features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input"""
        try:
            X, y = [], []
            
            for i in range(self.sequence_length, len(features)):
                X.append(features[i-self.sequence_length:i])
                y.append(target[i])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            self.logger.error(f"❌ [LSTMModel] Error creating sequences: {e}")
            raise

    def build_model(self, input_shape: Tuple[int, int]) -> Any:
        """
        Build LSTM model architecture
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            
        Returns:
            Compiled Keras model
        """
        try:
            if not TENSORFLOW_AVAILABLE:
                self.logger.warning("⚠️ [LSTMModel] TensorFlow not available. Using placeholder model.")
                return self._create_placeholder_model()
            
            self.logger.info("🏗️ [LSTMModel] Building LSTM model architecture...")
            
            # Create model
            model = Sequential()
            
            # First LSTM layer
            model.add(LSTM(
                units=self.model_config['lstm_units'][0],
                return_sequences=True,
                input_shape=input_shape,
                dropout=self.model_config['dropout_rate'],
                recurrent_dropout=self.model_config['recurrent_dropout']
            ))
            model.add(BatchNormalization())
            
            # Second LSTM layer
            if len(self.model_config['lstm_units']) > 1:
                model.add(LSTM(
                    units=self.model_config['lstm_units'][1],
                    return_sequences=False,
                    dropout=self.model_config['dropout_rate'],
                    recurrent_dropout=self.model_config['recurrent_dropout']
                ))
                model.add(BatchNormalization())
            
            # Dense layers
            for units in self.model_config['dense_units'][:-1]:
                model.add(Dense(units, activation='relu'))
                model.add(Dropout(self.model_config['dropout_rate']))
                model.add(BatchNormalization())
            
            # Output layer
            model.add(Dense(self.model_config['dense_units'][-1], activation='linear'))
            
            # Compile model
            optimizer = Adam(learning_rate=self.model_config['learning_rate'])
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            self.logger.info("✅ [LSTMModel] LSTM model built successfully")
            
            return model
            
        except Exception as e:
            self.logger.error(f"❌ [LSTMModel] Error building model: {e}")
            raise

    def _create_placeholder_model(self):
        """Create placeholder model when TensorFlow is not available"""
        class PlaceholderModel:
            def __init__(self):
                self.is_trained = False
            
            def fit(self, X, y, **kwargs):
                self.is_trained = True
                return self
            
            def predict(self, X):
                # Return random predictions
                return np.random.randn(len(X))
            
            def evaluate(self, X, y):
                return [0.0, 0.0]  # [loss, mae]
        
        return PlaceholderModel()

    def train(self, X: np.ndarray, y: np.ndarray, validation_data: Tuple[np.ndarray, np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the LSTM model
        
        Args:
            X: Training features
            y: Training targets
            validation_data: Validation data tuple (X_val, y_val)
            
        Returns:
            Training results
        """
        try:
            self.logger.info(f"🎓 [LSTMModel] Starting LSTM training for {self.symbol}...")
            
            # Store training data
            self.X_train = X
            self.y_train = y
            
            # Build model
            input_shape = (X.shape[1], X.shape[2])
            self.model = self.build_model(input_shape)
            
            if not TENSORFLOW_AVAILABLE:
                # Use placeholder training
                self.model.fit(X, y)
                self.is_trained = True
                self.last_training_date = datetime.utcnow()
                
                training_results = {
                    'epochs': 0,
                    'final_loss': 0.0,
                    'final_mae': 0.0,
                    'training_date': self.last_training_date.isoformat()
                }
                
                self.training_history.append(training_results)
                self.logger.info("✅ [LSTMModel] Placeholder training completed")
                return training_results
            
            # Prepare callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.model_config['early_stopping_patience'],
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=self.model_config['reduce_lr_patience'],
                    min_lr=1e-7
                )
            ]
            
            # Train model
            history = self.model.fit(
                X, y,
                batch_size=self.model_config['batch_size'],
                epochs=self.model_config['epochs'],
                validation_split=self.model_config['validation_split'],
                callbacks=callbacks,
                verbose=0
            )
            
            # Store training results
            training_results = {
                'epochs': len(history.history['loss']),
                'final_loss': history.history['loss'][-1],
                'final_mae': history.history['mae'][-1],
                'val_loss': history.history.get('val_loss', [0])[-1],
                'val_mae': history.history.get('val_mae', [0])[-1],
                'training_date': datetime.utcnow().isoformat(),
                'history': history.history
            }
            
            self.training_history.append(training_results)
            self.is_trained = True
            self.last_training_date = datetime.utcnow()
            
            self.logger.info(f"✅ [LSTMModel] Training completed - Final Loss: {training_results['final_loss']:.6f}")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"❌ [LSTMModel] Error during training: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using trained model
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        try:
            if not self.is_trained or self.model is None:
                raise ValueError("Model must be trained before making predictions")
            
            predictions = self.model.predict(X)
            return predictions.flatten()
            
        except Exception as e:
            self.logger.error(f"❌ [LSTMModel] Error making predictions: {e}")
            raise

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X: Test features
            y: Test targets
            
        Returns:
            Evaluation metrics
        """
        try:
            if not self.is_trained or self.model is None:
                raise ValueError("Model must be trained before evaluation")
            
            self.logger.info(f"📊 [LSTMModel] Evaluating LSTM model for {self.symbol}...")
            
            # Make predictions
            y_pred = self.predict(X)
            
            # Calculate metrics
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, y_pred)
            
            # Calculate additional metrics
            mape = np.mean(np.abs((y - y_pred) / y)) * 100
            directional_accuracy = np.mean(np.sign(y[1:] - y[:-1]) == np.sign(y_pred[1:] - y_pred[:-1])) * 100
            
            evaluation_results = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape,
                'directional_accuracy': directional_accuracy,
                'evaluation_date': datetime.utcnow().isoformat()
            }
            
            self.performance_metrics = evaluation_results
            
            self.logger.info(f"✅ [LSTMModel] Evaluation completed - RMSE: {rmse:.6f}, R²: {r2:.4f}")
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"❌ [LSTMModel] Error during evaluation: {e}")
            raise

    def forecast(self, data: pd.DataFrame, steps: int = 1) -> np.ndarray:
        """
        Forecast future values
        
        Args:
            data: Historical data
            steps: Number of steps to forecast
            
        Returns:
            Forecasted values
        """
        try:
            if not self.is_trained or self.model is None:
                raise ValueError("Model must be trained before forecasting")
            
            self.logger.info(f"🔮 [LSTMModel] Forecasting {steps} steps ahead for {self.symbol}...")
            
            # Prepare data
            features = data[self.feature_names].values
            features_scaled = self.scaler.transform(features)
            
            # Get last sequence
            last_sequence = features_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            
            # Forecast
            forecasts = []
            current_sequence = last_sequence.copy()
            
            for _ in range(steps):
                # Predict next value
                next_pred = self.model.predict(current_sequence, verbose=0)
                forecasts.append(next_pred[0, 0])
                
                # Update sequence (shift and add prediction)
                # Note: This is a simplified approach. In practice, you might want to use the actual next features
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = next_pred[0, 0]  # Assuming first feature is the target
            
            return np.array(forecasts)
            
        except Exception as e:
            self.logger.error(f"❌ [LSTMModel] Error during forecasting: {e}")
            raise

    def should_retrain(self, performance_threshold: float = None) -> bool:
        """
        Check if model should be retrained
        
        Args:
            performance_threshold: Minimum performance threshold
            
        Returns:
            True if model should be retrained
        """
        try:
            if performance_threshold is None:
                performance_threshold = ML_CONFIG.get('MIN_F1_SCORE', 0.35)
            
            # Check if model is trained
            if not self.is_trained:
                return True
            
            # Check performance metrics
            if self.performance_metrics:
                r2 = self.performance_metrics.get('r2', 0)
                if r2 < performance_threshold:
                    self.logger.info(f"🔄 [LSTMModel] Model R² ({r2:.4f}) below threshold ({performance_threshold})")
                    return True
            
            # Check if model is stale
            if self.last_training_date:
                days_since_training = (datetime.utcnow() - self.last_training_date).days
                if days_since_training >= 7:  # Retrain weekly
                    self.logger.info(f"🔄 [LSTMModel] Model is stale ({days_since_training} days since last training)")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ [LSTMModel] Error checking retrain condition: {e}")
            return True

    def save_model(self, filepath: str = None):
        """Save trained model to file"""
        try:
            if not self.is_trained or self.model is None:
                raise ValueError("Model must be trained before saving")
            
            if filepath is None:
                filepath = f"models/lstm_model_{self.symbol}.joblib"
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model data
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'sequence_length': self.sequence_length,
                'model_config': self.model_config,
                'training_history': self.training_history,
                'performance_metrics': self.performance_metrics,
                'last_training_date': self.last_training_date,
                'symbol': self.symbol
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"💾 [LSTMModel] Model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ [LSTMModel] Error saving model: {e}")
            raise

    def load_model(self, filepath: str = None):
        """Load trained model from file"""
        try:
            if filepath is None:
                filepath = f"models/lstm_model_{self.symbol}.joblib"
            
            if not os.path.exists(filepath):
                self.logger.warning(f"⚠️ [LSTMModel] Model file not found: {filepath}")
                return False
            
            # Load model data
            model_data = joblib.load(filepath)
            
            # Restore model state
            self.model = model_data.get('model')
            self.scaler = model_data.get('scaler')
            self.feature_names = model_data.get('feature_names', [])
            self.sequence_length = model_data.get('sequence_length', self.sequence_length)
            self.model_config = model_data.get('model_config', self.model_config)
            self.training_history = model_data.get('training_history', [])
            self.performance_metrics = model_data.get('performance_metrics', {})
            self.last_training_date = model_data.get('last_training_date')
            
            self.is_trained = self.model is not None
            
            self.logger.info(f"📂 [LSTMModel] Model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ [LSTMModel] Error loading model: {e}")
            return False

    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information"""
        try:
            summary = {
                'symbol': self.symbol,
                'is_trained': self.is_trained,
                'sequence_length': self.sequence_length,
                'num_features': len(self.feature_names),
                'feature_names': self.feature_names,
                'last_training_date': self.last_training_date,
                'training_history_length': len(self.training_history)
            }
            
            if self.performance_metrics:
                summary.update({
                    'r2_score': self.performance_metrics.get('r2', 0),
                    'rmse': self.performance_metrics.get('rmse', 0),
                    'mae': self.performance_metrics.get('mae', 0),
                    'directional_accuracy': self.performance_metrics.get('directional_accuracy', 0)
                })
            
            if self.training_history:
                latest_training = self.training_history[-1]
                summary.update({
                    'latest_epochs': latest_training.get('epochs', 0),
                    'latest_loss': latest_training.get('final_loss', 0)
                })
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ [LSTMModel] Error getting model summary: {e}")
            return {}