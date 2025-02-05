import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from sklearn.model_selection import train_test_split

# Sample data (user, item, rating)
data = np.array([
    [1, 101, 5], [1, 102, 3], [2, 101, 4], [2, 103, 2],
    [3, 102, 4], [3, 104, 1]
])

users = data[:, 0]
items = data[:, 1]
ratings = data[:, 2]

# Normalize ratings to be between 0 and 1
ratings = (ratings - ratings.min()) / (ratings.max() - ratings.min())

# Create mappings for zero-based indices
unique_users = np.unique(users)
unique_items = np.unique(items)

user_id_map = {old: new for new, old in enumerate(unique_users)}
item_id_map = {old: new for new, old in enumerate(unique_items)}

# Convert user and item IDs to zero-based indices
users = np.array([user_id_map[u] for u in users])
items = np.array([item_id_map[i] for i in items])

# Train-test split
train_users, test_users, train_items, test_items, train_ratings, test_ratings = train_test_split(
    users, items, ratings, test_size=0.2, random_state=42
)

# Get number of unique users and items
num_users = len(unique_users)
num_items = len(unique_items)

# Model architecture
user_input = Input(shape=(1,))
item_input = Input(shape=(1,))

user_embedding = Embedding(num_users, 50)(user_input)  # No need for +1
item_embedding = Embedding(num_items, 50)(item_input)

user_vec = Flatten()(user_embedding)
item_vec = Flatten()(item_embedding)

concat = Concatenate()([user_vec, item_vec])
dense = Dense(128, activation='relu')(concat)
output = Dense(1, activation='sigmoid')(dense)

model = Model(inputs=[user_input, item_input], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit([train_users, train_items], train_ratings, epochs=10, batch_size=16, verbose=1)

# Test the model
predicted_ratings = model.predict([test_users, test_items])
print(f"Predicted ratings: {predicted_ratings.flatten()}")
