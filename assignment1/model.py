from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def train_model():
  """
  Trains a logistic regression model on the Iris dataset.

  Returns:
      A trained LogisticRegression model.
  """

  # Load the Iris dataset
  iris = datasets.load_iris()
  X = iris.data  # Features
  y = iris.target  # Target variable (iris flower type)

  # Split data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

  # Create and train the logistic regression model
  model = LogisticRegression()
  model.fit(X_train, y_train)

  # Make predictions on the test set
  predictions = model.predict(X_test)

  # Calculate accuracy
  accuracy = accuracy_score(y_test, predictions)

  # Print the accuracy
  print("Training Completed ")

  joblib.dump(model,'iris_modelv01.joblib')



  return model

if __name__ == "__main__":
  trained_model = train_model()
  # You can now use the trained_model for further tasks

 