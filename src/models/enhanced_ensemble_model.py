"""
Enhanced Ensemble Model - Machine learning ensemble implementation
Refactored from the original Bot-Trading_Swing.py
"""

import pandas as pd
import numpy as np
import logging
import joblib
import os
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
import lightgbm as lgb

# Import configuration and constants
from Bot_Trading_Swing_Refactored import (
    ML_CONFIG, BOT_LOGGERS, FeatureConstants
)


class EnhancedEnsembleModel:
    """
    Enhanced Ensemble Model - Handles ensemble machine learning models
    Refactored to be more modular and maintainable
    """

    def __init__(self, symbol: str = None):
        """Initialize the Enhanced Ensemble Model"""
        self.logger = BOT_LOGGERS['MLModels']
        self.symbol = symbol
        self.logger.info(f"🤖 [EnsembleModel] Initializing Enhanced Ensemble Model for {symbol}...")
        
        # Model configuration
        self.model_config = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42
            },
            'lightgbm': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42
            },
            'logistic_regression': {
                'random_state': 42,
                'max_iter': 1000
            },
            'svm': {
                'kernel': 'rbf',
                'random_state': 42,
                'probability': True
            }
        }
        
        # Initialize models
        self.models = {}
        self.model_scores = {}
        self.ensemble_weights = {}
        self.is_trained = False
        self.feature_importance = {}
        
        # Performance tracking
        self.performance_history = []
        self.last_retrain_date = None
        
        self.logger.info("✅ [EnsembleModel] Enhanced Ensemble Model initialized successfully")

    def initialize_models(self):
        """Initialize all ensemble models"""
        try:
            self.logger.info("🤖 [EnsembleModel] Initializing ensemble models...")
            
            # Tree-based models
            self.models['random_forest'] = RandomForestClassifier(**self.model_config['random_forest'])
            self.models['gradient_boosting'] = GradientBoostingClassifier(**self.model_config['gradient_boosting'])
            self.models['xgboost'] = xgb.XGBClassifier(**self.model_config['xgboost'])
            self.models['lightgbm'] = lgb.LGBMClassifier(**self.model_config['lightgbm'])
            
            # Linear models
            self.models['logistic_regression'] = LogisticRegression(**self.model_config['logistic_regression'])
            
            # Non-linear models
            self.models['svm'] = SVC(**self.model_config['svm'])
            
            # Initialize weights (equal weights initially)
            num_models = len(self.models)
            self.ensemble_weights = {name: 1.0 / num_models for name in self.models.keys()}
            
            self.logger.info(f"✅ [EnsembleModel] Initialized {num_models} models")
            
        except Exception as e:
            self.logger.error(f"❌ [EnsembleModel] Error initializing models: {e}")
            raise

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, float]:
        """
        Train all ensemble models
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary of model scores
        """
        try:
            self.logger.info(f"🤖 [EnsembleModel] Training ensemble models for {self.symbol}...")
            
            if not self.models:
                self.initialize_models()
            
            # Handle missing values
            X_train = X_train.fillna(X_train.median())
            if X_val is not None:
                X_val = X_val.fillna(X_train.median())
            
            # Train each model
            for name, model in self.models.items():
                try:
                    self.logger.info(f"🤖 [EnsembleModel] Training {name}...")
                    
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Evaluate model
                    if X_val is not None and y_val is not None:
                        y_pred = model.predict(X_val)
                        score = f1_score(y_val, y_pred, average='weighted')
                    else:
                        # Use cross-validation if no validation set
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
                        score = cv_scores.mean()
                    
                    self.model_scores[name] = score
                    self.logger.info(f"✅ [EnsembleModel] {name} trained with score: {score:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"❌ [EnsembleModel] Error training {name}: {e}")
                    self.model_scores[name] = 0.0
            
            # Update ensemble weights based on performance
            self._update_ensemble_weights()
            
            # Calculate feature importance
            self._calculate_feature_importance(X_train.columns)
            
            self.is_trained = True
            self.last_retrain_date = datetime.utcnow()
            
            self.logger.info(f"✅ [EnsembleModel] All models trained successfully")
            return self.model_scores
            
        except Exception as e:
            self.logger.error(f"❌ [EnsembleModel] Error training models: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using ensemble
        
        Args:
            X: Features for prediction
            
        Returns:
            Predicted labels
        """
        try:
            if not self.is_trained:
                raise ValueError("Models must be trained before making predictions")
            
            # Handle missing values
            X = X.fillna(X.median())
            
            # Get predictions from each model
            predictions = {}
            probabilities = {}
            
            for name, model in self.models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X)
                        predictions[name] = model.predict(X)
                        probabilities[name] = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
                    else:
                        predictions[name] = model.predict(X)
                        probabilities[name] = predictions[name]
                except Exception as e:
                    self.logger.warning(f"⚠️ [EnsembleModel] Error getting predictions from {name}: {e}")
                    continue
            
            # Combine predictions using weighted average
            if probabilities:
                ensemble_proba = np.zeros(len(X))
                total_weight = 0
                
                for name, proba in probabilities.items():
                    weight = self.ensemble_weights.get(name, 0)
                    ensemble_proba += proba * weight
                    total_weight += weight
                
                if total_weight > 0:
                    ensemble_proba /= total_weight
                    ensemble_predictions = (ensemble_proba > 0.5).astype(int)
                else:
                    # Fallback to majority vote
                    ensemble_predictions = self._majority_vote(predictions)
            else:
                # Fallback to majority vote
                ensemble_predictions = self._majority_vote(predictions)
            
            return ensemble_predictions
            
        except Exception as e:
            self.logger.error(f"❌ [EnsembleModel] Error making predictions: {e}")
            raise

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities using ensemble
        
        Args:
            X: Features for prediction
            
        Returns:
            Prediction probabilities
        """
        try:
            if not self.is_trained:
                raise ValueError("Models must be trained before making predictions")
            
            # Handle missing values
            X = X.fillna(X.median())
            
            # Get probabilities from each model
            probabilities = {}
            
            for name, model in self.models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X)
                        probabilities[name] = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
                    else:
                        # Convert predictions to probabilities
                        pred = model.predict(X)
                        probabilities[name] = pred.astype(float)
                except Exception as e:
                    self.logger.warning(f"⚠️ [EnsembleModel] Error getting probabilities from {name}: {e}")
                    continue
            
            # Combine probabilities using weighted average
            if probabilities:
                ensemble_proba = np.zeros(len(X))
                total_weight = 0
                
                for name, proba in probabilities.items():
                    weight = self.ensemble_weights.get(name, 0)
                    ensemble_proba += proba * weight
                    total_weight += weight
                
                if total_weight > 0:
                    ensemble_proba /= total_weight
                else:
                    ensemble_proba = np.full(len(X), 0.5)
            else:
                ensemble_proba = np.full(len(X), 0.5)
            
            return ensemble_proba
            
        except Exception as e:
            self.logger.error(f"❌ [EnsembleModel] Error getting prediction probabilities: {e}")
            raise

    def _update_ensemble_weights(self):
        """Update ensemble weights based on model performance"""
        try:
            if not self.model_scores:
                return
            
            # Normalize scores to get weights
            total_score = sum(self.model_scores.values())
            if total_score > 0:
                self.ensemble_weights = {
                    name: score / total_score 
                    for name, score in self.model_scores.items()
                }
            else:
                # Equal weights if all scores are 0
                num_models = len(self.model_scores)
                self.ensemble_weights = {
                    name: 1.0 / num_models 
                    for name in self.model_scores.keys()
                }
            
            self.logger.info(f"🔧 [EnsembleModel] Updated ensemble weights: {self.ensemble_weights}")
            
        except Exception as e:
            self.logger.error(f"❌ [EnsembleModel] Error updating ensemble weights: {e}")

    def _majority_vote(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Perform majority vote on predictions"""
        try:
            if not predictions:
                return np.array([])
            
            # Convert to numpy array
            pred_array = np.array(list(predictions.values()))
            
            # Majority vote
            ensemble_predictions = np.round(pred_array.mean(axis=0)).astype(int)
            
            return ensemble_predictions
            
        except Exception as e:
            self.logger.error(f"❌ [EnsembleModel] Error in majority vote: {e}")
            return np.array([])

    def _calculate_feature_importance(self, feature_names: List[str]):
        """Calculate feature importance across all models"""
        try:
            self.feature_importance = {}
            
            for name, model in self.models.items():
                try:
                    if hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                        self.feature_importance[name] = dict(zip(feature_names, importance))
                    elif hasattr(model, 'coef_'):
                        # For linear models, use absolute coefficients
                        coef = np.abs(model.coef_[0])
                        self.feature_importance[name] = dict(zip(feature_names, coef))
                except Exception as e:
                    self.logger.warning(f"⚠️ [EnsembleModel] Error calculating feature importance for {name}: {e}")
                    continue
            
            # Calculate ensemble feature importance
            if self.feature_importance:
                ensemble_importance = {}
                for feature in feature_names:
                    importance_sum = 0
                    total_weight = 0
                    
                    for model_name, importance_dict in self.feature_importance.items():
                        if feature in importance_dict:
                            weight = self.ensemble_weights.get(model_name, 0)
                            importance_sum += importance_dict[feature] * weight
                            total_weight += weight
                    
                    if total_weight > 0:
                        ensemble_importance[feature] = importance_sum / total_weight
                    else:
                        ensemble_importance[feature] = 0
                
                self.feature_importance['ensemble'] = ensemble_importance
            
            self.logger.info("✅ [EnsembleModel] Feature importance calculated")
            
        except Exception as e:
            self.logger.error(f"❌ [EnsembleModel] Error calculating feature importance: {e}")

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate ensemble model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        try:
            if not self.is_trained:
                raise ValueError("Models must be trained before evaluation")
            
            self.logger.info(f"📊 [EnsembleModel] Evaluating ensemble model for {self.symbol}...")
            
            # Handle missing values
            X_test = X_test.fillna(X_test.median())
            
            # Get predictions
            y_pred = self.predict(X_test)
            y_proba = self.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Model-specific evaluations
            individual_scores = {}
            for name, model in self.models.items():
                try:
                    if hasattr(model, 'predict'):
                        individual_pred = model.predict(X_test)
                        individual_f1 = f1_score(y_test, individual_pred, average='weighted')
                        individual_scores[name] = individual_f1
                except Exception as e:
                    self.logger.warning(f"⚠️ [EnsembleModel] Error evaluating {name}: {e}")
                    continue
            
            # Compile results
            evaluation_results = {
                'accuracy': accuracy,
                'f1_score': f1,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist(),
                'individual_scores': individual_scores,
                'ensemble_weights': self.ensemble_weights,
                'feature_importance': self.feature_importance.get('ensemble', {}),
                'evaluation_date': datetime.utcnow().isoformat()
            }
            
            # Store performance history
            self.performance_history.append(evaluation_results)
            
            self.logger.info(f"✅ [EnsembleModel] Evaluation completed - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"❌ [EnsembleModel] Error evaluating model: {e}")
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
            
            # Check performance history
            if self.performance_history:
                latest_f1 = self.performance_history[-1].get('f1_score', 0)
                if latest_f1 < performance_threshold:
                    self.logger.info(f"🔄 [EnsembleModel] Model performance ({latest_f1:.4f}) below threshold ({performance_threshold})")
                    return True
            
            # Check if model is stale
            if self.last_retrain_date:
                days_since_retrain = (datetime.utcnow() - self.last_retrain_date).days
                if days_since_retrain >= 7:  # Retrain weekly
                    self.logger.info(f"🔄 [EnsembleModel] Model is stale ({days_since_retrain} days since last retrain)")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ [EnsembleModel] Error checking retrain condition: {e}")
            return True

    def save_model(self, filepath: str = None):
        """Save trained model to file"""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before saving")
            
            if filepath is None:
                filepath = f"models/ensemble_model_{self.symbol}.joblib"
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model data
            model_data = {
                'models': self.models,
                'model_scores': self.model_scores,
                'ensemble_weights': self.ensemble_weights,
                'feature_importance': self.feature_importance,
                'performance_history': self.performance_history,
                'last_retrain_date': self.last_retrain_date,
                'symbol': self.symbol,
                'model_config': self.model_config
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"💾 [EnsembleModel] Model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ [EnsembleModel] Error saving model: {e}")
            raise

    def load_model(self, filepath: str = None):
        """Load trained model from file"""
        try:
            if filepath is None:
                filepath = f"models/ensemble_model_{self.symbol}.joblib"
            
            if not os.path.exists(filepath):
                self.logger.warning(f"⚠️ [EnsembleModel] Model file not found: {filepath}")
                return False
            
            # Load model data
            model_data = joblib.load(filepath)
            
            # Restore model state
            self.models = model_data.get('models', {})
            self.model_scores = model_data.get('model_scores', {})
            self.ensemble_weights = model_data.get('ensemble_weights', {})
            self.feature_importance = model_data.get('feature_importance', {})
            self.performance_history = model_data.get('performance_history', [])
            self.last_retrain_date = model_data.get('last_retrain_date')
            self.model_config = model_data.get('model_config', self.model_config)
            
            self.is_trained = len(self.models) > 0
            
            self.logger.info(f"📂 [EnsembleModel] Model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ [EnsembleModel] Error loading model: {e}")
            return False

    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information"""
        try:
            summary = {
                'symbol': self.symbol,
                'is_trained': self.is_trained,
                'num_models': len(self.models),
                'model_names': list(self.models.keys()),
                'ensemble_weights': self.ensemble_weights,
                'last_retrain_date': self.last_retrain_date,
                'performance_history_length': len(self.performance_history)
            }
            
            if self.performance_history:
                latest_performance = self.performance_history[-1]
                summary['latest_accuracy'] = latest_performance.get('accuracy', 0)
                summary['latest_f1_score'] = latest_performance.get('f1_score', 0)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ [EnsembleModel] Error getting model summary: {e}")
            return {}