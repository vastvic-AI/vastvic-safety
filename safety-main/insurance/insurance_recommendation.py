import pandas as pd
import numpy as np
import tensorflow as tf
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
import uuid
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('insurance_recommendation.log')
    ]
)
logger = logging.getLogger(__name__)

# Simulated insurance dataset (mimicking thesis structure)
def generate_synthetic_data(num_users: int = 75308, num_products: int = 22) -> pd.DataFrame:
    logger.info("Generating synthetic dataset")
    np.random.seed(42)
    products = ['Travel_Insurance', 'Health_Insurance', 'SPPI', 'Sickness_Benefit', 'Disability', 'Retirement_Annuity', 'Life_Assurance',
                'Wellness_Lifestyle', 'Accidental_Death', 'Comprehensive_Medical', 'Professional_Health',
                'General_Health', 'Professional_Disability'] + [f'Product_{i}' for i in range(10)]
    data = {
        'user_id': range(num_users),
        'age': np.random.randint(18, 70, num_users),
        'gender': np.random.choice(['Male', 'Female'], num_users),
        'occupation': np.random.choice(['Medical', 'Legal', 'Engineer', 'Academic', 'Financial', 'Other'], num_users),
        'smoking_habit': np.random.choice(['Smoker', 'Non-smoker'], num_users),
        'policy_time': np.random.randint(1, 25, num_users),
        'premium': np.random.uniform(300, 2500, num_users),
        'product_id': np.random.choice(range(num_products), num_users),
        'user_behavior_score': np.random.uniform(0, 1, num_users),
        'exit_strategy': np.random.choice(['Early Exit', 'Long-Term', 'Churn Risk'], num_users),
        'walking_pattern_score': np.random.uniform(0, 1, num_users),
        # Simulation features for integration:
        'exit_choice': np.random.choice(['North', 'South', 'East', 'West'], num_users),
        'panic_level': np.random.uniform(0, 1, num_users),
        'exit_time': np.random.uniform(10, 300, num_users)
    }
    df = pd.DataFrame(data)
    df['product_name'] = [products[i] for i in df['product_id']]
    return df

# XGBoost-based recommendation for travel and health insurance
class TravelHealthInsuranceRecommender:
    def __init__(self):
        self.product_le = LabelEncoder()
        self.gender_le = LabelEncoder()
        self.occupation_le = LabelEncoder()
        self.smoking_le = LabelEncoder()
        self.exit_choice_le = LabelEncoder()
        self.model = None
        self.price_model = None
        self.products = ['Travel_Insurance', 'Health_Insurance']
        self.is_trained = False

    def fit(self, df: pd.DataFrame):
        # Focus only on travel and health insurance
        df = df[df['product_name'].isin(self.products)].copy()
        # Add movement/behavior features, fill with random/sensible values if missing
        if 'path_length' not in df:
            df['path_length'] = np.random.uniform(10, 200, len(df))
        if 'waited' not in df:
            df['waited'] = np.random.randint(0, 50, len(df))
        if 'collisions' not in df:
            df['collisions'] = np.random.randint(0, 5, len(df))
        X = pd.DataFrame({
            'age': df['age'],
            'gender': self.gender_le.fit_transform(df['gender']),
            'occupation': self.occupation_le.fit_transform(df['occupation']),
            'smoking_habit': self.smoking_le.fit_transform(df['smoking_habit']),
            'user_behavior_score': df['user_behavior_score'],
            'walking_pattern_score': df['walking_pattern_score'],
            'exit_choice': self.exit_choice_le.fit_transform(df['exit_choice']),
            'panic_level': df['panic_level'],
            'exit_time': df['exit_time'],
            'path_length': df['path_length'],
            'waited': df['waited'],
            'collisions': df['collisions']
        })
        y = self.product_le.fit_transform(df['product_name'])
        price = df['premium']
        self.model = xgb.XGBClassifier(n_estimators=80, max_depth=4, learning_rate=0.11, use_label_encoder=False, eval_metric='mlogloss')
        self.model.fit(X, y)
        self.price_model = xgb.XGBRegressor(n_estimators=80, max_depth=4, learning_rate=0.11)
        self.price_model.fit(X, price)
        self.is_trained = True

    def recommend(self, user_features: dict):
        if not self.is_trained:
            raise Exception("Model not trained. Call fit() first.")
        # Prepare features
        X = pd.DataFrame([{
            'age': user_features['age'],
            'gender': self.gender_le.transform([user_features['gender']])[0],
            'occupation': self.occupation_le.transform([user_features['occupation']])[0],
            'smoking_habit': self.smoking_le.transform([user_features['smoking_habit']])[0],
            'user_behavior_score': user_features.get('user_behavior_score', 0.5),
            'walking_pattern_score': user_features.get('walking_pattern_score', 0.5),
            'exit_choice': self.exit_choice_le.transform([user_features.get('exit_choice', 'North')])[0],
            'panic_level': user_features.get('panic_level', 0.5),
            'exit_time': user_features.get('exit_time', 100),
            'path_length': user_features.get('path_length', 50.0),
            'waited': user_features.get('waited', 10),
            'collisions': user_features.get('collisions', 0)
        }])
        product_idx = self.model.predict(X)[0]
        product = self.product_le.inverse_transform([product_idx])[0]
        price = float(self.price_model.predict(X)[0])
        return {'product': product, 'premium': max(300, min(price, 2500))}

# Singleton pattern for integration
_recommender = None
def get_travel_health_recommender():
    global _recommender
    if _recommender is None or not getattr(_recommender, 'is_trained', False):
        df = generate_synthetic_data(num_users=8000, num_products=14)
        _recommender = TravelHealthInsuranceRecommender()
        _recommender.fit(df)
    return _recommender

def recommend_for_agent(agent_features: dict) -> dict:
    """
    Recommend best travel/health insurance and price for a user/agent.
    agent_features: dict with keys: age, gender, occupation, smoking_habit, user_behavior_score, walking_pattern_score, exit_choice, panic_level, exit_time
    Returns: dict with 'product' and 'premium'
    """
    recommender = get_travel_health_recommender()
    return recommender.recommend(agent_features)

# Preprocessing class
class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.le_gender = LabelEncoder()
        self.le_occupation = LabelEncoder()
        self.le_smoking = LabelEncoder()
        self.le_product = LabelEncoder()
        self.le_exit_strategy = LabelEncoder()
        
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocessing data")
        df = df.copy()
        df['gender'] = self.le_gender.fit_transform(df['gender'])
        df['occupation'] = self.le_occupation.fit_transform(df['occupation'])
        df['smoking_habit'] = self.le_smoking.fit_transform(df['smoking_habit'])
        df['exit_strategy'] = self.le_exit_strategy.fit_transform(df['exit_strategy'])
        df['product_id'] = self.le_product.fit_transform(df['product_name'])
        numerical_cols = ['age', 'policy_time', 'premium', 'user_behavior_score', 'walking_pattern_score']
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        return df
    
    def transform_input(self, customer_data: Dict) -> np.ndarray:
        input_data = np.array([[
            customer_data['age'],
            self.le_gender.transform([customer_data['gender']])[0],
            self.le_occupation.transform([customer_data['occupation']])[0],
            self.le_smoking.transform([customer_data['smoking_habit']])[0],
            customer_data['policy_time'],
            customer_data['premium'],
            customer_data['user_behavior_score'],
            self.le_exit_strategy.transform([customer_data['exit_strategy']])[0],
            customer_data['walking_pattern_score']
        ]])
        # Scale numerical columns: age, policy_time, premium, user_behavior_score, walking_pattern_score
        input_data[:, [0, 4, 5, 6, 8]] = self.scaler.transform(input_data[:, [0, 4, 5, 6, 8]])
        return input_data

# WALS-based Collaborative Filtering
class WALSCollaborativeFilter:
    def __init__(self, num_factors: int = 5, reg: float = 0.1, num_iters: int = 20):
        self.num_factors = num_factors
        self.reg = reg
        self.num_iters = num_iters
        self.user_factors = None
        self.item_factors = None
    
    def create_sparse_tensor(self, matrix: pd.DataFrame) -> tf.SparseTensor:
        logger.info("Creating sparse tensor for user-item matrix")
        coo = matrix.stack().reset_index()
        coo = coo[coo[0] != 0]  # Remove zero entries
        indices = coo[['user_id', 'product_id']].values
        values = coo[0].values
        dense_shape = matrix.shape
        return tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    
    def fit(self, user_item_matrix: pd.DataFrame) -> None:
        try:
            sparse_tensor = self.create_sparse_tensor(user_item_matrix)
            num_users, num_items = sparse_tensor.dense_shape
            self.user_factors = tf.Variable(tf.random.normal([num_users, self.num_factors], stddev=0.1))
            self.item_factors = tf.Variable(tf.random.normal([num_items, self.num_factors], stddev=0.1))
            optimizer = tf.keras.optimizers.Adam(0.01)
            
            logger.info("Training WALS model")
            for iteration in range(self.num_iters):
                # Update user factors
                for i in range(num_users):
                    user_ratings = tf.sparse.slice(sparse_tensor, [i, 0], [1, num_items])
                    item_indices = user_ratings.indices[:, 1]
                    ratings = user_ratings.values
                    items = tf.gather(self.item_factors, item_indices)
                    user_factor = tf.linalg.lstsq(items, ratings[:, None], l2_regularizer=self.reg)
                    self.user_factors[i].assign(user_factor)
                
                # Update item factors
                for j in range(num_items):
                    item_ratings = tf.sparse.slice(sparse_tensor, [0, j], [num_users, 1])
                    user_indices = item_ratings.indices[:, 0]
                    ratings = item_ratings.values
                    users = tf.gather(self.user_factors, user_indices)
                    item_factor = tf.linalg.lstsq(users, ratings[:, None], l2_regularizer=self.reg)
                    self.item_factors[j].assign(item_factor)
                
                logger.debug(f"WALS iteration {iteration + 1}/{self.num_iters} completed")
        
        except Exception as e:
            logger.error(f"Error in WALS training: {str(e)}")
            raise
    
    def recommend(self, user_id: int, top_n: int = 3) -> List[str]:
        try:
            user_vector = self.user_factors[user_id]
            scores = tf.matmul(user_vector[None, :], self.item_factors, transpose_b=True)
            top_indices = tf.argsort(scores, direction='DESCENDING')[0][:top_n]
            return top_indices.numpy()
        except Exception as e:
            logger.error(f"Error in WALS recommendation: {str(e)}")
            return []

# Neural Network for Content-Based Recommendation
class NeuralNetworkRecommender:
    def __init__(self, input_dim: int, num_products: int, hyperparams: Dict):
        self.model = self._build_model(input_dim, num_products, hyperparams)
    
    def _build_model(self, input_dim: int, num_products: int, hyperparams: Dict) -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(hyperparams['hidden_layers'][0], activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dropout(hyperparams['dropout']),
            tf.keras.layers.Dense(hyperparams['hidden_layers'][1], activation='relu'),
            tf.keras.layers.Dropout(hyperparams['dropout']),
            tf.keras.layers.Dense(hyperparams['hidden_layers'][2], activation='relu'),
            tf.keras.layers.Dense(num_products, activation='softmax')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparams['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2, epochs: int = 75) -> None:
        logger.info("Training neural network")
        try:
            self.model.fit(X, y, validation_split=validation_split, epochs=epochs, batch_size=32, verbose=0)
            logger.info("Neural network training completed")
        except Exception as e:
            logger.error(f"Error in neural network training: {str(e)}")
            raise
    
    def recommend(self, input_data: np.ndarray, top_n: int = 3) -> List[int]:
        try:
            probabilities = self.model.predict(input_data, verbose=0)
            top_indices = np.argsort(probabilities[0])[::-1][:top_n]
            return top_indices
        except Exception as e:
            logger.error(f"Error in neural network recommendation: {str(e)}")
            return []

# Premium Prediction Model
class PremiumPredictor:
    def __init__(self):
        self.base_premium = 500.0
    
    def predict(self, customer_data: Dict) -> float:
        try:
            risk_factor = 1.0
            if customer_data['smoking_habit'] == 'Smoker':
                risk_factor += 0.2
            if customer_data['occupation'] in ['Medical', 'Legal']:
                risk_factor += 0.3
            if customer_data['age'] > 50:
                risk_factor += 0.15
            if customer_data['policy_time'] < 5:
                risk_factor += 0.1
            premium = self.base_premium * risk_factor
            logger.info(f"Predicted premium: ${premium:.2f} for customer")
            return premium
        except Exception as e:
            logger.error(f"Error in premium prediction: {str(e)}")
            return self.base_premium

# Hybrid Recommendation Engine
class InsuranceRecommendationEngine:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.cf_model = WALSCollaborativeFilter(num_factors=5, reg=0.1, num_iters=20)
        self.nn_model = NeuralNetworkRecommender(
            input_dim=9,
            num_products=22,
            hyperparams={
                'hidden_layers': [100, 80, 110],
                'dropout': 0.0053,
                'learning_rate': 0.017
            }
        )
        self.premium_predictor = PremiumPredictor()
        self.product_names = None
    
    def train(self, df: pd.DataFrame) -> None:
        logger.info("Initializing training of recommendation engine")
        try:
            # Preprocess data
            df_processed = self.preprocessor.preprocess(df)
            self.product_names = self.preprocessor.le_product.classes_
            
            # Create user-item matrix for CF
            user_item_matrix = pd.pivot_table(
                df_processed,
                values='policy_time',
                index='user_id',
                columns='product_id',
                fill_value=0
            )
            self.cf_model.fit(user_item_matrix)
            
            # Train neural network
            features = ['age', 'gender', 'occupation', 'smoking_habit', 'policy_time', 'premium', 'user_behavior_score', 'exit_strategy', 'walking_pattern_score']
            X = df_processed[features].values
            y = df_processed['product_id'].values
            self.nn_model.train(X, y)
            
            logger.info("Training completed successfully")
        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            raise
    
    def recommend(self, customer_data: Dict, user_id: Optional[int] = None) -> Dict:
        request_id = str(uuid.uuid4())
        logger.info(f"Processing recommendation request {request_id}")
        
        try:
            # Prepare response
            response = {
                'request_id': request_id,
                'timestamp': datetime.utcnow().isoformat(),
                'recommendations': [],
                'estimated_premium': 0.0,
                'source': ''
            }
            
            # Get recommendations
            if user_id is not None and user_id < 75308:  # Assuming max users from thesis
                top_indices = self.cf_model.recommend(user_id)
                response['source'] = 'Collaborative Filtering'
            else:
                input_data = self.preprocessor.transform_input(customer_data)
                top_indices = self.nn_model.recommend(input_data)
                response['source'] = 'Neural Network'
            
            # Map indices to product names
            response['recommendations'] = [self.product_names[i] for i in top_indices]
            
            # Predict premium
            response['estimated_premium'] = self.premium_predictor.predict(customer_data)
            
            logger.info(f"Recommendation generated: {response['source']} with products {response['recommendations']}")
            return response
        
        except Exception as e:
            logger.error(f"Error in recommendation for request {request_id}: {str(e)}")
            response['error'] = str(e)
            return response

# Main execution
if __name__ == "__main__":
    try:
        # Generate and preprocess data
        df = generate_synthetic_data()
        engine = InsuranceRecommendationEngine()
        engine.train(df)
        
        # Example customer
        customer = {
            'age': 35,
            'gender': 'Male',
            'occupation': 'Medical',
            'smoking_habit': 'Non-smoker',
            'policy_time': 5,
            'premium': 1000,
            'user_behavior_score': 0.7,  # Example: moderately engaged user
            'exit_strategy': 'Long-Term',  # Example: prefers long-term policies
            'walking_pattern_score': 0.8  # Example: high physical activity
        }
        
        # Test for existing user
        result_existing = engine.recommend(customer, user_id=42)
        print(json.dumps({
            'Existing User': {
                'Source': result_existing['source'],
                'Products': result_existing['recommendations'],
                'Estimated Premium': f"${result_existing['estimated_premium']:.2f}",
                'Request ID': result_existing['request_id'],
                'Timestamp': result_existing['timestamp']
            }
        }, indent=2))
        
        # Test for new user
        result_new = engine.recommend(customer)
        print(json.dumps({
            'New User': {
                'Source': result_new['source'],
                'Products': result_new['recommendations'],
                'Estimated Premium': f"${result_new['estimated_premium']:.2f}",
                'Request ID': result_new['request_id'],
                'Timestamp': result_new['timestamp']
            }
        }, indent=2))
        
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        print(json.dumps({'error': str(e)}))