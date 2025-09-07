import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class EnhancedIsolationForestDetector:
    """
    Unsupervised Isolation Forest detector for fraud/outlier detection.
    Automatically determines anomalies without a fixed contamination rate.
    """

    def __init__(self, max_samples=30000):
        self.max_samples = max_samples
        self.iso_forest = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.feature_stats = {}

    def preprocess_features(self, df, fit_encoders=True):
        """Preprocess dataset: numeric transformations, datetime features, categorical encoding"""
        df_processed = df.copy()

        # Datetime features (your dataset already has hour and day_of_week)
        if 'hour' in df_processed.columns:
            df_processed['hour_sin'] = np.sin(2 * np.pi * df_processed['hour'] / 24)
            df_processed['hour_cos'] = np.cos(2 * np.pi * df_processed['hour'] / 24)
        if 'day_of_week' in df_processed.columns:
            df_processed['day_sin'] = np.sin(2 * np.pi * df_processed['day_of_week'] / 7)
            df_processed['day_cos'] = np.cos(2 * np.pi * df_processed['day_of_week'] / 7)

        # Numeric features
        df_processed['amount_log'] = np.log1p(df_processed['amount'])
        df_processed['is_same_state'] = (df_processed['user_home_state'] == df_processed['transaction_state']).astype(int)
        df_processed['is_home_ip'] = (df_processed['ip_address'] == df_processed['user_home_ip']).astype(int)
        df_processed['amount_percentile'] = df_processed['amount'].rank(pct=True)
        df_processed['is_night_transaction'] = ((df_processed['hour'] >= 22) | (df_processed['hour'] <= 5)).astype(int)
        df_processed['is_weekend'] = (df_processed['day_of_week'] >= 5).astype(int)

        if 'distance_from_home_km' in df_processed.columns:
            df_processed['very_far_from_home'] = (df_processed['distance_from_home_km'] > 500).astype(int)
            df_processed['distance_log'] = np.log1p(df_processed['distance_from_home_km'])

        # Categorical encoding
        categorical_cols = ['channel', 'user_home_state', 'transaction_state', 'agent_type', 'time_of_day']
        for col in categorical_cols:
            if col in df_processed.columns:
                if fit_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_processed[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df_processed[col].astype(str))
                else:
                    if col in self.label_encoders:
                        known_categories = set(self.label_encoders[col].classes_)
                        df_processed[col] = df_processed[col].astype(str).apply(lambda x: x if x in known_categories else 'unknown')
                        if 'unknown' not in known_categories:
                            self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, 'unknown')
                        df_processed[f'{col}_encoded'] = self.label_encoders[col].transform(df_processed[col])

        # Combine all feature columns
        feature_cols = [
            'amount','amount_log','amount_percentile',
            'lat','lon','distance_from_home_km',
            'hour','day_of_week','hour_sin','hour_cos','day_sin','day_cos',
            'is_same_state','is_home_ip','is_night_transaction','is_weekend'
        ]
        if 'very_far_from_home' in df_processed.columns:
            feature_cols.extend(['very_far_from_home', 'distance_log'])
        for col in categorical_cols:
            if f'{col}_encoded' in df_processed.columns:
                feature_cols.append(f'{col}_encoded')

        available_features = [col for col in feature_cols if col in df_processed.columns]
        self.feature_names = available_features

        return df_processed[available_features]

    def train_isolation_forest(self, X_train):
        """Train Isolation Forest without fixed contamination"""
        print(f"Training Isolation Forest with {X_train.shape[0]} samples and {X_train.shape[1]} features...")
        self.iso_forest = IsolationForest(
            contamination='auto',  # automatically determines anomaly threshold
            random_state=42,
            n_estimators=100,
            max_samples=min(self.max_samples, len(X_train))
        )
        self.iso_forest.fit(X_train)

        feature_df = pd.DataFrame(X_train, columns=self.feature_names)
        self.feature_stats = {
            'means': feature_df.mean().to_dict(),
            'stds': feature_df.std().to_dict(),
            'mins': feature_df.min().to_dict(),
            'maxs': feature_df.max().to_dict()
        }

        print("Training complete!")
        return self.iso_forest

    def predict_fraud(self, X_scaled, return_scores=False):
        """Predict anomalies (fraud)"""
        if self.iso_forest is None:
            raise ValueError("Model not trained yet. Call train_isolation_forest first.")

        if_pred = self.iso_forest.predict(X_scaled)
        if_scores = self.iso_forest.decision_function(X_scaled)
        if_anomalies = (if_pred == -1).astype(int)

        results = pd.DataFrame({
            'is_fraud': if_anomalies,
            'isolation_forest_flagged': if_anomalies,
            'confidence_score': self.normalize_scores(if_scores)
        })
        if return_scores:
            results['raw_anomaly_score'] = if_scores

        return results

    def normalize_scores(self, scores):
        min_score = scores.min()
        max_score = scores.max()
        if min_score == max_score:
            return np.zeros_like(scores)
        normalized = (max_score - scores) / (max_score - min_score)
        return np.clip(normalized, 0, 1)

    def save_model(self, filepath_prefix='isolation_forest_model'):
        """Save model and preprocessing objects"""
        model_data = {
            'iso_forest': self.iso_forest,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'feature_stats': self.feature_stats,
            'max_samples': self.max_samples
        }
        joblib.dump(model_data, f'{filepath_prefix}.pkl')
        print(f"✅ Model saved as {filepath_prefix}.pkl")

    def load_model(self, filepath):
        """Load previously trained model"""
        model_data = joblib.load(filepath)
        self.iso_forest = model_data['iso_forest']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.feature_stats = model_data['feature_stats']
        self.max_samples = model_data['max_samples']
        print(f"✅ Model loaded from {filepath}")

