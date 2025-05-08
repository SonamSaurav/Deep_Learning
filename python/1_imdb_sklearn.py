from sklearn.datasets import load_files # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
import nltk # type: ignore
import os

# Download required NLTK resources
nltk.download('movie_reviews')
from nltk.corpus import movie_reviews # type: ignore

# Load dataset
fileids = movie_reviews.fileids()
docs = [" ".join(movie_reviews.words(fileid)) for fileid in fileids]
labels = [1 if fileid.startswith("pos") else 0 for fileid in fileids]

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(docs)
y = labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Logistic Regression classifier
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))