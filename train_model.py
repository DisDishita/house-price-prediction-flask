import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Simple dataset
data = {
    'sqft': [1000, 1500, 2000, 2500],
    'price': [200000, 300000, 400000, 500000]
}

df = pd.DataFrame(data)

# Train model
model = LinearRegression()
model.fit(df[['sqft']], df['price'])

# Save model
pickle.dump(model, open('model.pkl', 'wb'))

print("Model trained and saved as model.pkl")