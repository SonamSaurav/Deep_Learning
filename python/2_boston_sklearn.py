from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load California housing data
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
# Save the model
import joblib
joblib.dump(model, 'california_housing_model.pkl')
# Load the model
loaded_model = joblib.load('california_housing_model.pkl')
# Predict using the loaded model
loaded_predictions = loaded_model.predict(X_test)
# Evaluate the loaded model
mse_loaded = mean_squared_error(y_test, loaded_predictions)
print("Mean Squared Error (original model):", mse)
print("Mean Squared Error (loaded model):", mse_loaded)
# Print the mean squared error
print("Mean Squared Error:", mse)
# Print the coefficients
print("Coefficients:", model.coef_)
# Print the intercept
print("Intercept:", model.intercept_)
# Print the first 5 predictions
print("First 5 predictions:", predictions[:5])
# Print the first 5 actual values
print("First 5 actual values:", y_test[:5])
# Print the first 5 features
print("First 5 features:", X_test[:5])
# Print the shape of the data
print("Shape of the data:", X.shape)
# Print the shape of the training data
print("Shape of the training data:", X_train.shape)
# Print the shape of the test data
print("Shape of the test data:", X_test.shape)
# Print the number of features
print("Number of features:", X.shape[1])
# Print the number of samples
print("Number of samples:", X.shape[0])
# Print the number of training samples
print("Number of training samples:", X_train.shape[0])
# Print the number of test samples
print("Number of test samples:", X_test.shape[0])
# Print the number of training samples
print("Number of training samples:", X_train.shape[0])
# Print the number of test samples
print("Number of test samples:", X_test.shape[0])
# Print the number of features
print("Number of features:", X.shape[1])