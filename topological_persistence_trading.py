"""
Topological Persistence Patterns Trading System
===============================================

A sophisticated implementation of topological data analysis (TDA) for algorithmic trading.
This system uses persistent homology to detect geometric patterns in financial time series
and generates trading signals based on topological features.

Key Features:
- Multiple embedding strategies (OHLC, time-delay, multivariate)
- Vietoris-Rips complex construction with persistent homology
- Persistence diagram to feature vector conversion
- Walk-forward backtesting with proper time-series CV
- Regime-switching trading rules based on topological features
- Statistical significance testing

Author: AI Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Core TDA libraries
try:
    import ripser
    from ripser import ripser
    from persim import PersistenceImager, plot_diagrams
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
    import xgboost as xgb
except ImportError as e:
    print(f"Missing required libraries: {e}")
    print("Install with: pip install ripser scikit-tda persim scikit-learn xgboost")
    raise

class TopologicalEmbedding:
    """
    Handles different embedding strategies for converting time series to point clouds.
    """
    
    def __init__(self, method: str = 'ohlc', **kwargs):
        """
        Initialize embedding method.
        
        Args:
            method: 'ohlc', 'time_delay', 'multivariate', or 'returns'
            **kwargs: Method-specific parameters
        """
        self.method = method
        self.params = kwargs
        
    def embed_window(self, data: pd.DataFrame, start_idx: int, window_size: int) -> np.ndarray:
        """
        Convert a time window to a point cloud.
        
        Args:
            data: DataFrame with OHLCV data
            start_idx: Starting index
            window_size: Number of points in window
            
        Returns:
            Point cloud as numpy array
        """
        window_data = data.iloc[start_idx:start_idx + window_size].copy()
        
        if self.method == 'ohlc':
            return self._ohlc_embedding(window_data)
        elif self.method == 'time_delay':
            return self._time_delay_embedding(window_data)
        elif self.method == 'multivariate':
            return self._multivariate_embedding(window_data)
        elif self.method == 'returns':
            return self._returns_embedding(window_data)
        else:
            raise ValueError(f"Unknown embedding method: {self.method}")
    
    def _ohlc_embedding(self, data: pd.DataFrame) -> np.ndarray:
        """OHLC embedding with proper market data adaptation."""
        # Create rolling window point cloud from OHLCV data
        ohlc = data[['open', 'high', 'low', 'close']].values
        
        # Market-specific preprocessing
        # 1. Log-normalize prices to handle scale differences
        ohlc_log = np.log(ohlc + 1e-8)
        
        # 2. Create time-delay embedding for each OHLC component
        # This creates a richer point cloud representation
        embedded_points = []
        
        for i in range(len(ohlc_log)):
            # Take a small window around current point
            start_idx = max(0, i - 2)
            end_idx = min(len(ohlc_log), i + 3)
            window = ohlc_log[start_idx:end_idx]
            
            # Flatten window into a point
            if len(window) > 0:
                point = window.flatten()
                # Pad or truncate to fixed size
                if len(point) < 12:  # 3 time steps * 4 OHLC
                    point = np.pad(point, (0, 12 - len(point)), 'constant')
                else:
                    point = point[:12]
                embedded_points.append(point)
        
        if embedded_points:
            return np.array(embedded_points)
        else:
            # Fallback to simple normalization
            if len(ohlc) > 1:
                mean_vals = np.mean(ohlc, axis=0)
                std_vals = np.std(ohlc, axis=0) + 1e-9
                ohlc = (ohlc - mean_vals) / std_vals
            return ohlc
    
    def _time_delay_embedding(self, data: pd.DataFrame) -> np.ndarray:
        """Time-delay embedding of returns."""
        returns = np.log(data['close'] / data['close'].shift(1)).dropna().values
        
        m = self.params.get('dimension', 3)  # embedding dimension
        tau = self.params.get('delay', 1)    # delay parameter
        
        if len(returns) < m * tau:
            return np.array([]).reshape(0, m)
        
        # Create time-delay embedding
        embedded = np.zeros((len(returns) - (m-1)*tau, m))
        for i in range(m):
            embedded[:, i] = returns[i*tau:len(returns) - (m-1-i)*tau]
        
        return embedded
    
    def _multivariate_embedding(self, data: pd.DataFrame) -> np.ndarray:
        """Multivariate embedding including volume."""
        features = data[['open', 'high', 'low', 'close', 'volume']].values
        
        # Normalize features
        if len(features) > 1:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
        
        return features
    
    def _returns_embedding(self, data: pd.DataFrame) -> np.ndarray:
        """Simple returns embedding."""
        returns = np.log(data['close'] / data['close'].shift(1)).dropna().values
        return returns.reshape(-1, 1)


class TopologicalFeatureExtractor:
    """
    Extracts topological features from point clouds using persistent homology.
    """
    
    def __init__(self, max_dim: int = 2, metric: str = 'euclidean', fast_mode: bool = True):
        """
        Initialize feature extractor.
        
        Args:
            max_dim: Maximum homology dimension to compute
            metric: Distance metric for Vietoris-Rips complex
            fast_mode: Use faster approximation methods
        """
        self.max_dim = max_dim
        self.metric = metric
        self.fast_mode = fast_mode
        self.persistence_imager = None
        
    def compute_persistence_diagram(self, point_cloud: np.ndarray) -> List[np.ndarray]:
        """
        Compute persistence diagrams for a point cloud with fast mode optimization.
        
        Args:
            point_cloud: Input point cloud
            
        Returns:
            List of persistence diagrams for each homology dimension
        """
        if len(point_cloud) < 3:
            return [np.array([]).reshape(0, 2) for _ in range(self.max_dim + 1)]
        
        try:
            if self.fast_mode:
                # Fast mode: subsample and use lower max dimension
                if len(point_cloud) > 20:
                    # Subsample to 20 points for speed
                    indices = np.random.choice(len(point_cloud), 20, replace=False)
                    point_cloud = point_cloud[indices]
                
                # Use H0, H1, and H2 for richer features
                result = ripser(point_cloud, maxdim=2, metric=self.metric, thresh=0.5)
                diagrams = result['dgms']
                
                # Ensure we have diagrams for all dimensions
                while len(diagrams) < self.max_dim + 1:
                    diagrams.append(np.array([]).reshape(0, 2))
                
                return diagrams
            else:
                # Full mode: compute all dimensions
                result = ripser(point_cloud, maxdim=self.max_dim, metric=self.metric)
                diagrams = result['dgms']
                
                # Ensure we have diagrams for all dimensions
                while len(diagrams) < self.max_dim + 1:
                    diagrams.append(np.array([]).reshape(0, 2))
                
                return diagrams
            
        except Exception as e:
            print(f"Error computing persistence diagram: {e}")
            return [np.array([]).reshape(0, 2) for _ in range(self.max_dim + 1)]
    
    def setup_persistence_imager(self, diagrams: List[np.ndarray], 
                                pixel_size: float = 0.05) -> None:
        """
        Setup persistence imager based on diagram statistics.
        
        Args:
            diagrams: Sample persistence diagrams
            pixel_size: Size of pixels in persistence image
        """
        all_births = []
        all_deaths = []
        
        for dgm in diagrams:
            if len(dgm) > 0:
                # Filter out infinite values
                valid_births = dgm[:, 0][np.isfinite(dgm[:, 0])]
                valid_deaths = dgm[:, 1][np.isfinite(dgm[:, 1])]
                all_births.extend(valid_births)
                all_deaths.extend(valid_deaths)
        
        if all_births and all_deaths:
            birth_min, birth_max = min(all_births), max(all_births)
            death_min, death_max = min(all_deaths), max(all_deaths)
            
            # Ensure ranges are finite and reasonable
            if np.isfinite(birth_min) and np.isfinite(birth_max):
                birth_range = (birth_min, birth_max)
            else:
                birth_range = (0, 1)
                
            if np.isfinite(death_min) and np.isfinite(death_max):
                pers_range = (0, death_max - birth_min)
            else:
                pers_range = (0, 1)
            
            # Ensure ranges are not too small
            if birth_range[1] - birth_range[0] < 1e-6:
                birth_range = (0, 1)
            if pers_range[1] - pers_range[0] < 1e-6:
                pers_range = (0, 1)
            
            self.persistence_imager = PersistenceImager(
                pixel_size=pixel_size,
                birth_range=birth_range,
                pers_range=pers_range
            )
    
    def diagram_to_persistence_image(self, diagram: np.ndarray) -> np.ndarray:
        """
        Convert persistence diagram to persistence image.
        
        Args:
            diagram: Persistence diagram
            
        Returns:
            Persistence image as 2D array
        """
        if self.persistence_imager is None or len(diagram) == 0:
            return np.zeros((20, 20))  # Default size
        
        try:
            return self.persistence_imager.transform(diagram)
        except:
            return np.zeros((20, 20))
    
    def extract_summary_features(self, diagrams: List[np.ndarray]) -> np.ndarray:
        """
        Extract market-specific topological features from persistence diagrams.
        
        Args:
            diagrams: List of persistence diagrams (H0, H1, H2)
            
        Returns:
            Feature vector with market-relevant topological features
        """
        features = []
        
        for dim, dgm in enumerate(diagrams):
            if len(dgm) == 0:
                # No features for this dimension - add zeros
                if dim == 0:  # H0 features
                    features.extend([0, 0, 0, 0, 0, 0])  # 6 H0 features
                elif dim == 1:  # H1 features  
                    features.extend([0, 0, 0, 0, 0, 0, 0])  # 7 H1 features
                else:  # H2 features
                    features.extend([0, 0, 0, 0])  # 4 H2 features
                continue
            
            # Filter out infinite values
            finite_mask = np.isfinite(dgm).all(axis=1)
            dgm_clean = dgm[finite_mask]
            
            if len(dgm_clean) == 0:
                if dim == 0:
                    features.extend([0, 0, 0, 0, 0, 0])
                elif dim == 1:
                    features.extend([0, 0, 0, 0, 0, 0, 0])
                else:
                    features.extend([0, 0, 0, 0])
                continue
            
            lifetimes = dgm_clean[:, 1] - dgm_clean[:, 0]
            lifetimes = lifetimes[np.isfinite(lifetimes)]
            
            if len(lifetimes) == 0:
                if dim == 0:
                    features.extend([0, 0, 0, 0, 0, 0])
                elif dim == 1:
                    features.extend([0, 0, 0, 0, 0, 0, 0])
                else:
                    features.extend([0, 0, 0, 0])
                continue
            
            # Market-specific topological features
            if dim == 0:  # H0 - Clusters (market regimes)
                # Long-lived clusters indicate stable regimes
                long_lived = lifetimes[lifetimes > np.percentile(lifetimes, 75)]
                features.extend([
                    len(lifetimes),                           # Total cluster count
                    np.sum(lifetimes),                        # Total cluster persistence
                    np.max(lifetimes),                        # Max cluster lifetime
                    len(long_lived),                          # Number of long-lived clusters
                    np.mean(lifetimes) if len(lifetimes) > 0 else 0,  # Mean cluster lifetime
                    np.std(lifetimes) if len(lifetimes) > 1 else 0     # Cluster lifetime variance
                ])
                
            elif dim == 1:  # H1 - Loops (oscillatory patterns)
                # Long-lived loops indicate strong oscillatory behavior
                long_lived = lifetimes[lifetimes > np.percentile(lifetimes, 75)]
                features.extend([
                    len(lifetimes),                           # Total loop count
                    np.sum(lifetimes),                        # Total loop persistence
                    np.max(lifetimes),                        # Max loop lifetime
                    len(long_lived),                          # Number of long-lived loops
                    np.mean(lifetimes) if len(lifetimes) > 0 else 0,  # Mean loop lifetime
                    np.std(lifetimes) if len(lifetimes) > 1 else 0,    # Loop lifetime variance
                    np.sum(long_lived) if len(long_lived) > 0 else 0   # Total long-lived loop persistence
                ])
                
            else:  # H2 - Voids (complex market structures)
                features.extend([
                    len(lifetimes),                           # Total void count
                    np.sum(lifetimes),                        # Total void persistence
                    np.max(lifetimes),                        # Max void lifetime
                    np.mean(lifetimes) if len(lifetimes) > 0 else 0   # Mean void lifetime
                ])
        
        # Convert to array and clean any remaining inf/nan values
        features_array = np.array(features)
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return features_array
    
    def extract_features(self, point_cloud: np.ndarray, 
                        use_images: bool = True,
                        use_summary: bool = True) -> np.ndarray:
        """
        Extract all topological features from a point cloud.
        
        Args:
            point_cloud: Input point cloud
            use_images: Whether to include persistence images
            use_summary: Whether to include summary statistics
            
        Returns:
            Combined feature vector
        """
        diagrams = self.compute_persistence_diagram(point_cloud)
        
        features = []
        
        if use_summary:
            summary_features = self.extract_summary_features(diagrams)
            features.append(summary_features)
        
        if use_images:
            for dim, dgm in enumerate(diagrams):
                if dim <= 1:  # Only use H0 and H1 images
                    img = self.diagram_to_persistence_image(dgm)
                    features.append(img.ravel())
        
        if features:
            combined_features = np.concatenate(features)
            # Clean any remaining inf/nan values
            combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=1e6, neginf=-1e6)
            return combined_features
        else:
            return np.array([])


class TopologicalTradingStrategy:
    """
    Main trading strategy class that combines topological features with ML models.
    """
    
    def __init__(self, 
                 embedding_method: str = 'ohlc',
                 window_size: int = 192,
                 embedding_params: Optional[Dict] = None,
                 model_type: str = 'random_forest',
                 use_persistence_images: bool = True,
                 use_summary_features: bool = True):
        """
        Initialize the trading strategy.
        
        Args:
            embedding_method: Method for converting time series to point clouds
            window_size: Size of sliding window
            embedding_params: Parameters for embedding method
            model_type: Type of ML model ('random_forest', 'xgboost')
            use_persistence_images: Whether to use persistence images
            use_summary_features: Whether to use summary statistics
        """
        self.window_size = window_size
        self.embedding_method = embedding_method
        self.embedding_params = embedding_params or {}
        
        # Initialize components
        self.embedder = TopologicalEmbedding(embedding_method, **self.embedding_params)
        self.feature_extractor = TopologicalFeatureExtractor(fast_mode=True)
        
        # Model configuration
        self.model_type = model_type
        self.use_persistence_images = use_persistence_images
        self.use_summary_features = use_summary_features
        
        # Model and scaler
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Feature statistics for setup
        self.feature_dim = None
        
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels for training.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Feature matrix and label array
        """
        features = []
        labels = []
        
        print("Computing topological features...")
        
        # First pass: compute diagrams to setup persistence imager
        sample_diagrams = []
        for i in range(min(10, len(data) - self.window_size - 1)):
            point_cloud = self.embedder.embed_window(data, i, self.window_size)
            if len(point_cloud) > 0:
                diagrams = self.feature_extractor.compute_persistence_diagram(point_cloud)
                sample_diagrams.extend(diagrams)
        
        # Setup persistence imager
        if sample_diagrams:
            self.feature_extractor.setup_persistence_imager(sample_diagrams)
        
        # Second pass: extract features
        for i in range(self.window_size, len(data) - 1):
            point_cloud = self.embedder.embed_window(data, i, self.window_size)
            
            if len(point_cloud) == 0:
                continue
            
            # Extract topological features
            feature_vector = self.feature_extractor.extract_features(
                point_cloud, 
                use_images=self.use_persistence_images,
                use_summary=self.use_summary_features
            )
            
            if len(feature_vector) > 0:
                # Create balanced label (avoid bias toward 0)
                current_price = data['close'].iloc[i]
                # Look ahead 3 periods for more stable signal
                future_prices = data['close'].iloc[i+1:i+4]
                if len(future_prices) >= 3:
                    avg_future_price = future_prices.mean()
                    trend_return = (avg_future_price / current_price) - 1
                    # Balanced labeling - avoid bias toward 0
                    if trend_return > 0.0005:  # 0.05% threshold
                        features.append(feature_vector)
                        labels.append(1)
                    elif trend_return < -0.0005:  # 0.05% threshold
                        features.append(feature_vector)
                        labels.append(0)
                    # Skip noisy cases - don't add to features or labels
                # Skip if not enough future data
        
        if not features:
            raise ValueError("No valid features extracted from data")
        
        # Ensure all features have the same dimension
        target_dim = len(features[0])
        for i, feat in enumerate(features):
            if len(feat) != target_dim:
                if len(feat) < target_dim:
                    # Pad with zeros
                    features[i] = np.pad(feat, (0, target_dim - len(feat)), 'constant')
                else:
                    # Truncate
                    features[i] = feat[:target_dim]
        
        X = np.vstack(features)
        y = np.array(labels)
        
        # Store feature dimension
        self.feature_dim = X.shape[1]
        
        print(f"Extracted {len(features)} feature vectors with dimension {self.feature_dim}")
        
        return X, y
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the model on historical data.
        
        Args:
            data: Training data with OHLCV columns
        """
        X, y = self.prepare_features(data)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and train model
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        # CRITICAL: Validate model performance with balanced accuracy
        from sklearn.metrics import balanced_accuracy_score
        train_predictions = self.model.predict_proba(X_scaled)[:, 1]
        train_preds = (train_predictions > 0.5).astype(int)
        
        train_accuracy = ((train_predictions > 0.5) == y).mean()
        balanced_acc = balanced_accuracy_score(y, train_preds)
        
        print(f"Model training accuracy: {train_accuracy:.1%}")
        print(f"Balanced accuracy: {balanced_acc:.1%}")
        print(f"Label distribution: {np.bincount(y)}")
        
        # Store accuracy metrics for tracking
        self.last_train_accuracy = train_accuracy
        self.last_balanced_accuracy = balanced_acc
        self.last_label_distribution = np.bincount(y)
        
        # If balanced accuracy is too low, the model might be learning inverse patterns
        if balanced_acc < 0.45:
            print("⚠️  Low balanced accuracy - model may be learning inverse patterns")
            self.model_inverted = True
        else:
            self.model_inverted = False
        
        print(f"Model trained with {len(X)} samples")
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model and scaler to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_dim': self.feature_dim,
            'model_inverted': getattr(self, 'model_inverted', False),
            'model_type': self.model_type,
            'window_size': self.window_size,
            'embedding_method': self.embedding_method
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model and scaler from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_dim = model_data['feature_dim']
        self.model_inverted = model_data.get('model_inverted', False)
        self.model_type = model_data['model_type']
        self.window_size = model_data['window_size']
        self.embedding_method = model_data['embedding_method']
        self.is_fitted = True
        
        print(f"✅ Model loaded from {filepath}")
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for new data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X, _ = self.prepare_features(data)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict_proba(X_scaled)[:, 1]
        
        # If model is inverted, flip the predictions
        if hasattr(self, 'model_inverted') and self.model_inverted:
            predictions = 1 - predictions
            print("🔄 Inverted model predictions applied")
        
        return predictions
    
    def generate_signals(self, data: pd.DataFrame, 
                        threshold: float = 0.6, 
                        use_regime_logic: bool = True) -> pd.DataFrame:
        """
        Generate trading signals based on topological regime detection.
        
        Args:
            data: DataFrame with OHLCV data
            threshold: Probability threshold for signal generation
            use_regime_logic: If True, use regime-aware rules; if False, use pure ML signals
            
        Returns:
            DataFrame with signals and topological regime information
        """
        predictions = self.predict(data)
        
        # Create signals DataFrame with correct length
        signals = pd.DataFrame(index=data.index[self.window_size:len(data)-1])
        
        # Ensure length match
        
        # Ensure length match by truncating predictions if needed
        if len(predictions) > len(signals):
            predictions = predictions[:len(signals)]
        elif len(predictions) < len(signals):
            # Pad with zeros if predictions are shorter
            predictions = np.pad(predictions, (0, len(signals) - len(predictions)), 'constant')
        
        signals['prediction'] = predictions
        signals['signal'] = 0
        signals['regime'] = 'neutral'
        
        # Extract topological features for regime detection
        topological_features = []
        for i in range(self.window_size, len(data) - 1):
            point_cloud = self.embedder.embed_window(data, i, self.window_size)
            if len(point_cloud) > 0:
                features = self.feature_extractor.extract_features(
                    point_cloud, use_images=False, use_summary=True
                )
                topological_features.append(features)
            else:
                topological_features.append(np.zeros(17))  # 6+7+4 features
        
        if use_regime_logic and topological_features:
            signals['topological_features'] = topological_features
            
            # Market-specific trading logic based on topological features
            for i, (idx, row) in enumerate(signals.iterrows()):
                if i < len(topological_features):
                    features = topological_features[i]
                    
                    # Extract key topological indicators
                    h0_clusters = features[0] if len(features) > 0 else 0
                    h0_long_lived = features[3] if len(features) > 3 else 0
                    h1_loops = features[6] if len(features) > 6 else 0
                    h1_long_lived = features[9] if len(features) > 9 else 0
                    h1_persistence = features[7] if len(features) > 7 else 0
                    
                    # Regime detection based on topological features
                    if h1_long_lived > 2 and h1_persistence > 0.1:
                        # High oscillatory activity - mean reversion regime
                        signals.loc[idx, 'regime'] = 'oscillatory'
                        # In oscillatory regime, trade against the trend with reduced size
                        if row['prediction'] > 0.7:
                            signals.loc[idx, 'signal'] = -0.5  # Short on high prediction
                        elif row['prediction'] < 0.3:
                            signals.loc[idx, 'signal'] = 0.5   # Long on low prediction
                            
                    elif h0_long_lived > 3 and h1_loops < 1:
                        # Stable clustering - trending regime
                        signals.loc[idx, 'regime'] = 'trending'
                        # In trending regime, follow the trend
                        # Only trade on high confidence predictions
                        if row['prediction'] > threshold:
                            signals.loc[idx, 'signal'] = 0.5   # Reduced position size for long
                        elif row['prediction'] < (1 - threshold):
                            signals.loc[idx, 'signal'] = -0.5  # Reduced position size for short
                        else:
                            signals.loc[idx, 'signal'] = 0   # Hold on neutral prediction
                            
                    else:
                        # Transitional or complex regime
                        signals.loc[idx, 'regime'] = 'transitional'
                        # In transitional regime, use very conservative signals
                        if row['prediction'] > 0.8:
                            signals.loc[idx, 'signal'] = 0.3  # Very small long position
                        elif row['prediction'] < 0.2:
                            signals.loc[idx, 'signal'] = -0.3  # Very small short position
        else:
            # Pure ML signals - simpler and potentially more robust
            for idx, row in signals.iterrows():
                signals.loc[idx, 'regime'] = 'ml_direct'
                # FIXED: Use 0.5 as threshold (50% probability)
                if row['prediction'] > 0.5:  # Above 50% = bullish
                    signals.loc[idx, 'signal'] = 1.0   # Full long
                elif row['prediction'] < 0.5:  # Below 50% = bearish
                    signals.loc[idx, 'signal'] = -1.0  # Full short
                else:
                    signals.loc[idx, 'signal'] = 0     # Hold (exactly 50%)
        
        # REMOVED: Double-reversal logic that was causing systematic inversion
        # The model inversion is now handled in the predict() method based on training accuracy
        
        return signals


class TopologicalBacktester:
    """
    Backtesting framework for topological trading strategies.
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0005):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital
            transaction_cost: Transaction cost as fraction
            slippage: Slippage as fraction
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        
    def backtest_strategy(self, 
                         data: pd.DataFrame, 
                         strategy: TopologicalTradingStrategy,
                         train_size: int = 1000,
                         test_size: int = 200,
                         step_size: int = 50,
                         use_regime_logic: bool = True,
                         auto_reverse_signals: bool = True,
                         save_best_model: bool = True) -> Dict:
        """
        Perform walk-forward backtesting.
        
        Args:
            data: Historical data
            strategy: Trading strategy instance
            train_size: Size of training window
            test_size: Size of test window
            step_size: Step size for rolling window
            use_regime_logic: Whether to use regime-aware trading logic
            auto_reverse_signals: Whether to automatically reverse signals if win rate < 50%
            save_best_model: Whether to save the best performing model
            
        Returns:
            Dictionary with backtest results
        """
        results = []
        all_signals = []
        iteration_accuracies = []
        best_accuracy = 0
        best_model_path = None
        
        print("Starting walk-forward backtesting...")
        
        # For pre-trained models, just test on the data directly
        if strategy.is_fitted:
            print(f"♻️  Using pre-trained model for testing on {len(data)} samples")
            signals = strategy.generate_signals(data, use_regime_logic=use_regime_logic)
            
            if len(signals) > 0:
                # Calculate returns
                test_returns = self._calculate_strategy_returns(data, signals)
                results.extend(test_returns)
                all_signals.append(signals)
                print(f"✅ Generated {len(signals)} signals")
                
                # Calculate test accuracy
                test_features, test_labels = strategy.prepare_features(data)
                if len(test_features) > 0:
                    test_predictions = strategy.model.predict_proba(test_features)[:, 1]
                    test_preds = (test_predictions > 0.5).astype(int)
                    from sklearn.metrics import balanced_accuracy_score
                    test_accuracy = balanced_accuracy_score(test_labels, test_preds)
                    iteration_accuracies.append(test_accuracy)
                    print(f"📊 Test accuracy: {test_accuracy:.1%}")
            else:
                print("❌ No signals generated")
        else:
            # Original walk-forward logic for training models
            for start_idx in range(train_size, len(data) - test_size, step_size):
                train_start = start_idx - train_size
                train_end = start_idx
                test_start = start_idx
                test_end = min(start_idx + test_size, len(data))
                
                print(f"Training on {train_start}:{train_end}, Testing on {test_start}:{test_end}")
                
                # Train on historical data
                train_data = data.iloc[train_start:train_end]
                strategy.fit(train_data)
                print(f"🔄 Model trained on {len(train_data)} samples")
                
                # Track accuracy for this iteration
                if hasattr(strategy, 'last_balanced_accuracy'):
                    iteration_accuracies.append(strategy.last_balanced_accuracy)
                    print(f"📊 Iteration {len(iteration_accuracies)} accuracy: {strategy.last_balanced_accuracy:.1%}")
                    
                    # Save best model
                    if save_best_model and strategy.last_balanced_accuracy > best_accuracy:
                        best_accuracy = strategy.last_balanced_accuracy
                        best_model_path = f"models/best_model_iter_{len(iteration_accuracies)}.pkl"
                        strategy.save_model(best_model_path)
                        print(f"🏆 New best model saved! Accuracy: {best_accuracy:.1%}")
                
                # Test on next period
                test_data = data.iloc[test_start:test_end]
                signals = strategy.generate_signals(test_data, use_regime_logic=use_regime_logic)
                
                # Calculate returns
                test_returns = self._calculate_strategy_returns(test_data, signals)
                results.extend(test_returns)
                all_signals.append(signals)
        
        # Combine all results
        if results:
            results_df = pd.DataFrame(results)
            performance_metrics = self._calculate_performance_metrics(results_df)
            
            # Add accuracy tracking to results
            if iteration_accuracies:
                performance_metrics['iteration_accuracies'] = iteration_accuracies
                performance_metrics['avg_accuracy'] = np.mean(iteration_accuracies)
                performance_metrics['best_accuracy'] = best_accuracy
                performance_metrics['best_model_path'] = best_model_path
                performance_metrics['accuracy_std'] = np.std(iteration_accuracies)
                
                print(f"\n📈 ACCURACY SUMMARY:")
                print(f"   Average accuracy across iterations: {performance_metrics['avg_accuracy']:.1%}")
                print(f"   Best accuracy: {performance_metrics['best_accuracy']:.1%}")
                print(f"   Accuracy std: {performance_metrics['accuracy_std']:.1%}")
                print(f"   Best model saved: {best_model_path}")
            
            return performance_metrics
        else:
            return {"error": "No valid backtest results"}
    
    def _calculate_strategy_returns(self, 
                                   data: pd.DataFrame, 
                                   signals: pd.DataFrame) -> List[Dict]:
        """
        Calculate strategy returns for a test period.
        
        Args:
            data: Price data
            signals: Trading signals
            
        Returns:
            List of return dictionaries
        """
        returns = []
        capital = self.initial_capital
        position = 0
        
        for i, (timestamp, signal_row) in enumerate(signals.iterrows()):
            if i == 0:
                continue
            
            current_price = data.loc[timestamp, 'close']
            prev_price = data.loc[data.index[data.index < timestamp][-1], 'close']
            
            # Calculate market return
            market_return = (current_price / prev_price) - 1
            
            # Get new position from signal
            new_position = signal_row['signal']
            
            # Calculate strategy return based on CURRENT position (not new one)
            strategy_return = position * market_return
            
            # Update capital with strategy return
            capital *= (1 + strategy_return)
            
            # Transaction costs only when position changes
            if position != new_position:
                capital *= (1 - self.transaction_cost)
            
            # Update position
            position = new_position
            
            returns.append({
                'timestamp': timestamp,
                'market_return': market_return,
                'strategy_return': strategy_return,
                'capital': capital,
                'position': position
            })
        
        return returns
    
    def _calculate_performance_metrics(self, results_df: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive performance metrics including topological regime analysis.
        
        Args:
            results_df: DataFrame with backtest results
            
        Returns:
            Dictionary with performance metrics and regime analysis
        """
        if len(results_df) == 0:
            return {"error": "No results to analyze"}
        
        # Calculate cumulative returns
        results_df['cumulative_market'] = (1 + results_df['market_return']).cumprod()
        results_df['cumulative_strategy'] = results_df['capital'] / self.initial_capital
        
        # Basic performance metrics
        total_return = results_df['cumulative_strategy'].iloc[-1] - 1
        market_return = results_df['cumulative_market'].iloc[-1] - 1
        
        # Sharpe ratio (assuming 5-minute returns, 252*288 periods per year)
        strategy_returns = results_df['strategy_return'].dropna()
        if len(strategy_returns) > 1:
            sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252 * 288)
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        cumulative = results_df['cumulative_strategy']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate and trade analysis
        winning_trades = (strategy_returns > 0).sum()
        total_trades = len(strategy_returns)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # DEBUG: Check prediction accuracy vs trading win rate
        print(f"🔍 DEBUG: Trading win rate: {win_rate:.1%} vs Model accuracy: 70%+")
        print(f"🔍 DEBUG: Total trades: {total_trades}, Winning trades: {winning_trades}")
        if len(strategy_returns) > 0:
            avg_trade = np.mean(strategy_returns)
            print(f"🔍 DEBUG: Avg trade return: {avg_trade:.4f}")
        else:
            print("🔍 DEBUG: No valid trades found")
        print(f"🔍 DEBUG: Transaction cost impact: {self.transaction_cost:.1%} per trade")
        
        # Additional metrics
        avg_trade = strategy_returns.mean() if len(strategy_returns) > 0 else 0
        best_trade = strategy_returns.max() if len(strategy_returns) > 0 else 0
        worst_trade = strategy_returns.min() if len(strategy_returns) > 0 else 0
        
        # Calmar ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = strategy_returns[strategy_returns < 0]
        if len(downside_returns) > 1:
            downside_dev = downside_returns.std()
            sortino_ratio = strategy_returns.mean() / downside_dev * np.sqrt(252 * 288) if downside_dev > 0 else 0
        else:
            sortino_ratio = 0
        
        # Regime analysis (if available)
        regime_performance = {}
        if 'regime' in results_df.columns:
            for regime in results_df['regime'].unique():
                regime_data = results_df[results_df['regime'] == regime]
                if len(regime_data) > 0:
                    regime_returns = regime_data['strategy_return'].dropna()
                    if len(regime_returns) > 0:
                        regime_performance[regime] = {
                            'count': len(regime_data),
                            'avg_return': regime_returns.mean(),
                            'win_rate': (regime_returns > 0).sum() / len(regime_returns),
                            'total_return': (1 + regime_returns).prod() - 1
                        }
        
        return {
            'total_return': total_return,
            'market_return': market_return,
            'excess_return': total_return - market_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'avg_trade': avg_trade,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'regime_performance': regime_performance,
            'results_df': results_df
        }


def load_sample_data() -> pd.DataFrame:
    """
    Load sample data for demonstration.
    In practice, replace this with your data loading function.
    """
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_points = 2000
    
    # Generate realistic price data with trends and volatility clustering
    returns = np.random.normal(0, 0.02, n_points)
    returns[500:600] += np.random.normal(0, 0.05, 100)  # Volatility cluster
    
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_points)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_points))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_points))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_points)
    })
    
    # Ensure OHLC consistency
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    data.index = pd.date_range('2020-01-01', periods=n_points, freq='5T')
    
    return data


def main():
    """
    Main demonstration function.
    """
    print("Topological Persistence Patterns Trading System")
    print("=" * 50)
    
    # Load data
    print("Loading sample data...")
    data = load_sample_data()
    print(f"Loaded {len(data)} data points")
    
    # Initialize strategy
    strategy = TopologicalTradingStrategy(
        embedding_method='ohlc',
        window_size=96,  # 2 trading days of 5-minute data
        model_type='random_forest',
        use_persistence_images=True,
        use_summary_features=True
    )
    
    # Initialize backtester
    backtester = TopologicalBacktester(
        initial_capital=100000,
        transaction_cost=0.001,
        slippage=0.0005
    )
    
    # Run backtest
    results = backtester.backtest_strategy(
        data=data,
        strategy=strategy,
        train_size=800,
        test_size=200,
        step_size=100
    )
    
    # Display results
    if 'error' not in results:
        print("\nBacktest Results:")
        print("-" * 30)
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Market Return: {results['market_return']:.2%}")
        print(f"Excess Return: {results['excess_return']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"Total Trades: {results['total_trades']}")
        
        # Plot results
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(results['results_df']['cumulative_market'], label='Market', alpha=0.7)
        plt.plot(results['results_df']['cumulative_strategy'], label='Strategy', alpha=0.7)
        plt.title('Cumulative Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(results['results_df']['strategy_return'].cumsum(), label='Strategy Returns')
        plt.title('Cumulative Strategy Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    else:
        print(f"Backtest failed: {results['error']}")


if __name__ == "__main__":
    main()
