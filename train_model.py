import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("dataset.csv")

X = data[['Area', 'Bedrooms', 'Bathrooms', 'Location']]
y = data['Price']

# Train model on FULL DATA (important)
model = LinearRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open("model/house_model.pkl", "wb"))

print("Model trained successfully")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
