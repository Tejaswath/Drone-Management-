# model.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import pickle

# Load the dataset
data_set = pd.read_csv('drone_dataset.csv')

# Define numeric columns and categorical columns
numeric_columns = ['flight_radius', 'flight_height', 'cost', 'battery_life', 'wind_resistance', 
                   'payload_capacity', 'noise_level', 'camera_quality', 'user_rating']
categorical_columns = ['regulatory_compliance', 'obstacle_avoidance']

# Create the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numeric_columns),
        ('cat', OneHotEncoder(), categorical_columns)
    ])

# Fit and transform the data
features = data_set.drop(columns=['drone_name'])
target = data_set['drone_name']
preprocessed_features = preprocessor.fit_transform(features)

# Apply PCA
pca = PCA(n_components=5)
pca_features = pca.fit_transform(preprocessed_features)

# Train a Nearest Neighbors model
knn = NearestNeighbors(n_neighbors=3)
knn.fit(pca_features)

# Save the model and preprocessor
with open('drone_model.pkl', 'wb') as f:
    pickle.dump((preprocessor, pca, knn, target), f)
