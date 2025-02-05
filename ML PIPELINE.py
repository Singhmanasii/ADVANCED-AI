import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from tpot import TPOTClassifier

# Step 1: Load and Prepare Data
# Here we use the Iris dataset as an example
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Preprocessing and Feature Engineering
# Define which columns are numeric
numeric_features = X.columns.tolist()

# We can apply different transformations to numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
    ('scaler', StandardScaler())  # Normalize the data
])

# Combine everything into a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ]
)

# Step 4: Define a Model Selection Pipeline using TPOT
# Use TPOT to automatically select the best model and parameters
tpot = TPOTClassifier(generations=5, population_size=20, random_state=42, verbosity=2)

# Step 5: Train the model using the training data (apply preprocessor and train TPOT)
# We apply preprocessing manually for TPOT
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

tpot.fit(X_train_transformed, y_train)

# Step 6: Evaluate the model on the test set
y_pred = tpot.predict(X_test_transformed)

print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')

# Optional: Export the best model pipeline found by TPOT
tpot.export('best_model_pipeline.py')
