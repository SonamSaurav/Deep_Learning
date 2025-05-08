from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = fetch_20newsgroups(subset='all')
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

# Pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
# Save the model
import joblib
joblib.dump(model, 'news_model.pkl')
# Load the model
loaded_model = joblib.load('news_model.pkl')
# Predict using the loaded model
loaded_predictions = loaded_model.predict(X_test)
# Evaluate the loaded model
mse_loaded = accuracy_score(y_test, loaded_predictions)
print("Mean Squared Error (loaded model):", mse_loaded)
# Print the accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
# Print the first 5 predictions
print("First 5 predictions:", y_pred[:5])
# Print the first 5 actual values
print("First 5 actual values:", y_test[:5])
# Print the first 5 features
print("First 5 features:", X_test[:5])
