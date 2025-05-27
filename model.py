import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

class PropertyModel:
    def __init__(self, json_path):
        self.df = pd.read_json(json_path)
        self.features = ['price', 'area', 'propertyType', 'inventoryType', 'bhk',
                         'furnishing', 'reraApproved', 'possession', 'facing']
        self.target = 'score'
        self.model = None
        self.preprocessor = None

    def preprocess_data(self):
        df = self.df.dropna(subset=[self.target])
        X = df[self.features]
        y = df[self.target]

        categorical_cols = ['area', 'propertyType', 'inventoryType', 'furnishing', 'possession', 'facing']
        numeric_cols = ['price', 'bhk', 'reraApproved']

        cat_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        num_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='mean'))
        ])

        self.preprocessor = ColumnTransformer([
            ('cat', cat_transformer, categorical_cols),
            ('num', num_transformer, numeric_cols)
        ])

        X_processed = self.preprocessor.fit_transform(X)
        return X_processed, np.array(y)

    def build_model(self, input_dim):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)  # Output score
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def train(self):
        X, y = self.preprocess_data()
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = self.build_model(X.shape[1])
        self.model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_val, y_val), verbose=0)

    def predict(self, input_data):
        input_df = pd.DataFrame([input_data])
        input_processed = self.preprocessor.transform(input_df)
        return float(self.model.predict(input_processed, verbose=0)[0][0])
