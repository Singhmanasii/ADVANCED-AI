from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the TPOTClassifier (using evolutionary algorithms for hyperparameter optimization)
tpot = TPOTClassifier(generations=5, population_size=20, random_state=42, verbosity=2)

# Train the model and optimize hyperparameters
tpot.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = tpot.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')

# Export the best model pipeline found by TPOT
tpot.export('best_model_pipeline.py')
