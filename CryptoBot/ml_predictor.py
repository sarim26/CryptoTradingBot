"""
ML Predictor Module for Crypto Trading Bot
Uses machine learning to analyze price trends and predict optimal buy percentages.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
from datetime import datetime, timedelta
import config
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class MLPredictor:
    """
    Machine Learning predictor for determining optimal buy percentages
    based on historical price trends and market conditions.
    """
    
    def __init__(self):
        """Initialize the ML predictor with default parameters."""
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.last_prediction_time = None
        self.cached_prediction = None
        
        print(f"\n{'='*60}")
        print(f"🤖 ML PREDICTOR INITIALIZED")
        print(f"{'='*60}")
        print(f"Lookback Period: {config.ML_LOOKBACK_HOURS} hours")
        print(f"Timeframe: {config.ML_TIMEFRAME}")
        print(f"Buy Range: {config.ML_MIN_BUY_PERCENTAGE}% - {config.ML_MAX_BUY_PERCENTAGE}%")
        print(f"Confidence Threshold: {config.ML_CONFIDENCE_THRESHOLD}")
        print(f"{'='*60}\n")
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for feature engineering.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional technical indicators
        """
        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Moving averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        
        # Price relative to moving averages
        df['price_vs_sma5'] = df['close'] / df['sma_5']
        df['price_vs_sma10'] = df['close'] / df['sma_10']
        df['price_vs_sma20'] = df['close'] / df['sma_20']
        
        # Volatility indicators
        df['volatility'] = df['price_change'].rolling(window=10).std()
        df['atr'] = self._calculate_atr(df, 14)
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
        bb_std_val = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std_val * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std_val * bb_std)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Trend strength
        df['trend_strength'] = (df['sma_5'] - df['sma_20']) / df['sma_20']
        
        # Support and resistance levels
        df['support_level'] = df['low'].rolling(window=20).min()
        df['resistance_level'] = df['high'].rolling(window=20).max()
        df['support_distance'] = (df['close'] - df['support_level']) / df['close']
        df['resistance_distance'] = (df['resistance_level'] - df['close']) / df['close']
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range (ATR)."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index (RSI)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features for ML model training/prediction.
        
        Args:
            df: DataFrame with OHLCV and technical indicators
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        # Select relevant features
        feature_columns = [
            'price_change', 'high_low_ratio', 'close_open_ratio',
            'price_vs_sma5', 'price_vs_sma10', 'price_vs_sma20',
            'volatility', 'atr', 'volume_ratio', 'rsi',
            'bb_position', 'trend_strength', 'support_distance', 'resistance_distance'
        ]
        
        # Filter out columns that might not exist
        available_features = [col for col in feature_columns if col in df.columns]
        
        # Create feature matrix
        features = df[available_features].fillna(method='ffill').fillna(0)
        
        # Add lagged features (previous periods)
        for lag in [1, 2, 3, 5]:
            for col in ['price_change', 'volatility', 'rsi']:
                if col in features.columns:
                    features[f'{col}_lag_{lag}'] = features[col].shift(lag)
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(0)
        
        return features.values, features.columns.tolist()
    
    def _create_target_variable(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create target variable for training based on current market conditions.
        Uses reactive approach instead of looking far ahead.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Array of target buy percentages
        """
        targets = []
        
        for i in range(len(df)):
            if i < 20:  # Need some history
                targets.append(config.BUY_DROP_PERCENTAGE)  # Default
                continue
            
            current_price = df.iloc[i]['close']
            
            # Calculate current market conditions
            # 1. RSI (most important for oversold/overbought)
            rsi = self._calculate_rsi(df.iloc[:i+1]['close'], 14).iloc[-1] if i >= 14 else 50
            
            # 2. Recent volatility (last 10 periods)
            recent_volatility = df.iloc[i-10:i]['close'].pct_change().std() * 100 if i >= 10 else 0
            
            # 3. Distance to support (last 20 periods)
            support = df.iloc[i-20:i]['low'].min() if i >= 20 else current_price
            support_distance = ((current_price - support) / current_price) * 100
            
            # 4. Short-term price momentum (last 3 periods)
            price_momentum = ((current_price - df.iloc[i-3]['close']) / df.iloc[i-3]['close']) * 100 if i >= 3 else 0
            
            # 5. Quick look-ahead (only 3 periods = 15 minutes)
            future_prices = df.iloc[i+1:i+4]['close'].values if i < len(df) - 3 else []
            if len(future_prices) > 0:
                min_future_price = np.min(future_prices)
                future_drop = ((current_price - min_future_price) / current_price) * 100
            else:
                future_drop = 0
            
            # Determine optimal buy threshold based on conditions
            base_threshold = config.BUY_DROP_PERCENTAGE
            
            # RSI-based adjustments (most important)
            if rsi < 25:  # Very oversold
                rsi_adjustment = -0.8  # Much lower threshold
            elif rsi < 30:  # Oversold
                rsi_adjustment = -0.5  # Lower threshold
            elif rsi > 70:  # Overbought
                rsi_adjustment = 0.8   # Higher threshold
            elif rsi > 60:  # Getting overbought
                rsi_adjustment = 0.3   # Slightly higher threshold
            else:
                rsi_adjustment = 0
            
            # Support distance adjustments
            if support_distance < 0.5:  # Very close to support
                support_adjustment = -0.4
            elif support_distance < 1.0:  # Close to support
                support_adjustment = -0.2
            elif support_distance > 3.0:  # Far from support
                support_adjustment = 0.3
            else:
                support_adjustment = 0
            
            # Volatility adjustments
            if recent_volatility > 2.0:  # High volatility
                volatility_adjustment = 0.5
            elif recent_volatility < 0.5:  # Low volatility
                volatility_adjustment = -0.2
            else:
                volatility_adjustment = 0
            
            # Momentum adjustments
            if price_momentum < -1.0:  # Strong downward momentum
                momentum_adjustment = -0.3
            elif price_momentum > 1.0:  # Strong upward momentum
                momentum_adjustment = 0.3
            else:
                momentum_adjustment = 0
            
            # Future drop consideration (limited to 15 minutes)
            if future_drop > 0:
                future_adjustment = min(0.3, future_drop * 0.1)  # Cap at 0.3%
            else:
                future_adjustment = 0
            
            # Combine all adjustments
            optimal_threshold = (base_threshold + rsi_adjustment + support_adjustment + 
                               volatility_adjustment + momentum_adjustment + future_adjustment)
            
            # Clamp to reasonable range
            optimal_threshold = max(config.ML_MIN_BUY_PERCENTAGE, 
                                  min(config.ML_MAX_BUY_PERCENTAGE, optimal_threshold))
            
            targets.append(optimal_threshold)
        
        return np.array(targets)
    
    def train_model(self, exchange, symbol: str) -> bool:
        """
        Train the ML model on historical data.
        
        Args:
            exchange: Exchange connector instance
            symbol: Trading pair symbol
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            print(f"🔄 Training ML model for {symbol}...")
            
            # Calculate number of candles needed
            timeframe_minutes = {
                '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30, '1h': 60
            }.get(config.ML_TIMEFRAME, 5)
            
            total_minutes = config.ML_LOOKBACK_HOURS * 60
            limit = max(500, total_minutes // timeframe_minutes)
            
            # Fetch historical data
            candles = exchange.get_ohlcv(symbol, timeframe=config.ML_TIMEFRAME, limit=limit)
            
            if not candles or len(candles) < 100:
                print(f"❌ Insufficient historical data for {symbol}")
                return False
            
            # Convert to DataFrame
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Calculate technical indicators
            df = self._calculate_technical_indicators(df)
            
            # Prepare features and targets
            features, feature_names = self._prepare_features(df)
            targets = self._create_target_variable(df)
            
            # Remove rows with NaN targets
            valid_indices = ~np.isnan(targets)
            features = features[valid_indices]
            targets = targets[valid_indices]
            
            if len(features) < 50:
                print(f"❌ Insufficient valid data for training")
                return False
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train model
            self.model.fit(features_scaled, targets)
            self.feature_names = feature_names
            self.is_trained = True
            
            # Calculate training accuracy
            predictions = self.model.predict(features_scaled)
            mse = mean_squared_error(targets, predictions)
            rmse = np.sqrt(mse)
            
            print(f"✅ ML model trained successfully!")
            print(f"   Training samples: {len(features)}")
            print(f"   Features: {len(feature_names)}")
            print(f"   RMSE: {rmse:.3f}%")
            
            return True
            
        except Exception as e:
            print(f"❌ Error training ML model: {e}")
            return False
    
    def predict_buy_percentage(self, exchange, symbol: str, current_price: float) -> Tuple[float, float]:
        """
        Predict optimal buy percentage using trained ML model.
        
        Args:
            exchange: Exchange connector instance
            symbol: Trading pair symbol
            current_price: Current market price
            
        Returns:
            Tuple of (predicted_buy_percentage, confidence_score)
        """
        # Check if we can use cached prediction (configurable duration)
        now = datetime.now()
        cache_duration = getattr(config, 'ML_CACHE_DURATION_SECONDS', 30)
        if (self.cached_prediction and 
            self.last_prediction_time and 
            (now - self.last_prediction_time).seconds < cache_duration):
            return self.cached_prediction
        
        if not self.is_trained:
            print("⚠️  ML model not trained, using fallback")
            return config.BUY_DROP_PERCENTAGE, 0.0
        
        try:
            # Get recent data for prediction
            timeframe_minutes = {
                '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30, '1h': 60
            }.get(config.ML_TIMEFRAME, 5)
            
            # Get enough data for technical indicators
            limit = max(100, (config.ML_LOOKBACK_HOURS * 60) // timeframe_minutes)
            candles = exchange.get_ohlcv(symbol, timeframe=config.ML_TIMEFRAME, limit=limit)
            
            if not candles or len(candles) < 50:
                print("⚠️  Insufficient data for ML prediction")
                return config.BUY_DROP_PERCENTAGE, 0.0
            
            # Convert to DataFrame
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Calculate technical indicators
            df = self._calculate_technical_indicators(df)
            
            # Prepare features for latest data point
            features, _ = self._prepare_features(df)
            
            if len(features) == 0:
                print("⚠️  No features available for prediction")
                return config.BUY_DROP_PERCENTAGE, 0.0
            
            # Use the most recent data point
            latest_features = features[-1].reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(latest_features)
            
            # Make prediction
            predicted_percentage = self.model.predict(features_scaled)[0]
            
            # Apply real-time market condition adjustments
            current_price = df['close'].iloc[-1]
            
            # Get current market conditions
            current_rsi = self._calculate_rsi(df['close'], 14).iloc[-1] if len(df) >= 14 else 50
            support = df['low'].rolling(window=20).min().iloc[-1]
            support_distance = ((current_price - support) / current_price) * 100
            recent_volatility = df['close'].pct_change().rolling(window=10).std().iloc[-1] * 100
            
            # Real-time adjustments based on current conditions
            adjustment_factor = 1.0
            adjustment_reason = ""
            
            # 1. RSI-based adjustments (most important)
            if current_rsi < 25:  # Very oversold
                adjustment_factor *= 0.4  # Reduce threshold by 60%
                adjustment_reason = f"Very oversold RSI ({current_rsi:.1f})"
            elif current_rsi < 30:  # Oversold
                adjustment_factor *= 0.6  # Reduce threshold by 40%
                adjustment_reason = f"Oversold RSI ({current_rsi:.1f})"
            elif current_rsi > 70:  # Overbought
                adjustment_factor *= 1.5  # Increase threshold by 50%
                adjustment_reason = f"Overbought RSI ({current_rsi:.1f})"
            
            # 2. Support distance adjustments
            if support_distance < 0.5:  # Very close to support
                adjustment_factor *= 0.7  # Reduce threshold by 30%
                if adjustment_reason:
                    adjustment_reason += f" + very close to support ({support_distance:.2f}%)"
                else:
                    adjustment_reason = f"Very close to support ({support_distance:.2f}%)"
            elif support_distance < 1.0:  # Close to support
                adjustment_factor *= 0.8  # Reduce threshold by 20%
                if adjustment_reason:
                    adjustment_reason += f" + close to support ({support_distance:.2f}%)"
                else:
                    adjustment_reason = f"Close to support ({support_distance:.2f}%)"
            
            # 3. Volatility adjustments
            if recent_volatility > 2.0:  # High volatility
                adjustment_factor *= 1.3  # Increase threshold by 30%
                if adjustment_reason:
                    adjustment_reason += f" + high volatility ({recent_volatility:.2f}%)"
                else:
                    adjustment_reason = f"High volatility ({recent_volatility:.2f}%)"
            elif recent_volatility < 0.5:  # Low volatility
                adjustment_factor *= 0.9  # Slightly reduce threshold
                if adjustment_reason:
                    adjustment_reason += f" + low volatility ({recent_volatility:.2f}%)"
                else:
                    adjustment_reason = f"Low volatility ({recent_volatility:.2f}%)"
            
            # Apply adjustment
            if adjustment_factor != 1.0:
                predicted_percentage = max(config.ML_MIN_BUY_PERCENTAGE, 
                                         predicted_percentage * adjustment_factor)
                print(f"🎯 Real-time adjustment: {adjustment_reason}")
                print(f"   Threshold: {predicted_percentage:.2f}% (factor: {adjustment_factor:.2f})")
            
            # Calculate confidence based on multiple factors
            feature_importance = self.model.feature_importances_
            
            # Factor 1: Feature importance consistency (how evenly distributed importance is)
            importance_std = np.std(feature_importance)
            importance_consistency = 1.0 - min(1.0, importance_std * 10)  # Lower std = higher consistency
            
            # Factor 2: Model training quality (based on RMSE from training)
            # We'll use a reasonable default since we don't store training RMSE
            training_quality = 0.7  # Assume decent training quality
            
            # Factor 3: Data quality (based on recent volatility and data completeness)
            recent_volatility = df['close'].pct_change().rolling(window=10).std().iloc[-1]
            volatility_factor = max(0.3, 1.0 - min(1.0, recent_volatility * 100))  # Lower volatility = higher confidence
            
            # Factor 4: Feature completeness (how many features are non-zero)
            non_zero_features = np.count_nonzero(latest_features[0])
            feature_completeness = non_zero_features / len(latest_features[0])
            
            # Combine all factors
            confidence = (importance_consistency * 0.3 + 
                         training_quality * 0.3 + 
                         volatility_factor * 0.2 + 
                         feature_completeness * 0.2)
            
            confidence = min(1.0, max(0.0, confidence))
            
            # Clamp prediction to reasonable range
            predicted_percentage = max(config.ML_MIN_BUY_PERCENTAGE,
                                    min(config.ML_MAX_BUY_PERCENTAGE, predicted_percentage))
            
            # Cache the prediction
            self.cached_prediction = (predicted_percentage, confidence)
            self.last_prediction_time = now
            
            return predicted_percentage, confidence
            
        except Exception as e:
            print(f"⚠️  Error in ML prediction: {e}")
            return config.BUY_DROP_PERCENTAGE, 0.0
    
    def get_trend_analysis(self, exchange, symbol: str) -> Dict[str, any]:
        """
        Analyze current market trends and provide insights.
        
        Args:
            exchange: Exchange connector instance
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with trend analysis results
        """
        try:
            # Get recent data
            candles = exchange.get_ohlcv(symbol, timeframe=config.ML_TIMEFRAME, limit=100)
            
            if not candles or len(candles) < 20:
                return {"error": "Insufficient data"}
            
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Calculate basic trend indicators
            current_price = df['close'].iloc[-1]
            sma_5 = df['close'].rolling(window=5).mean().iloc[-1]
            sma_20 = df['close'].rolling(window=20).mean().iloc[-1]
            
            # Trend direction
            if sma_5 > sma_20:
                trend = "Bullish"
            elif sma_5 < sma_20:
                trend = "Bearish"
            else:
                trend = "Sideways"
            
            # Volatility
            volatility = df['close'].pct_change().rolling(window=10).std().iloc[-1] * 100
            
            # Support and resistance
            support = df['low'].rolling(window=20).min().iloc[-1]
            resistance = df['high'].rolling(window=20).max().iloc[-1]
            
            # Distance to support/resistance
            support_distance = ((current_price - support) / current_price) * 100
            resistance_distance = ((resistance - current_price) / current_price) * 100
            
            return {
                "trend": trend,
                "volatility": volatility,
                "support_level": support,
                "resistance_level": resistance,
                "support_distance": support_distance,
                "resistance_distance": resistance_distance,
                "sma_5": sma_5,
                "sma_20": sma_20,
                "current_price": current_price
            }
            
        except Exception as e:
            return {"error": f"Analysis failed: {e}"}
    
    def should_use_ml_prediction(self, confidence: float) -> bool:
        """
        Determine if ML prediction should be used based on confidence.
        
        Args:
            confidence: Confidence score from ML prediction
            
        Returns:
            True if ML prediction should be used, False otherwise
        """
        return confidence >= config.ML_CONFIDENCE_THRESHOLD
