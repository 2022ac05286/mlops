from joblib import load

def predict(model_path, data_point):
  # Load the model
  model = load(model_path)

  # Make prediction using your model's predict method
  prediction = model.predict([data_point])[0]  # Adjust based on your model
  return prediction

if __name__ == '__main__':
  model_path = "C:/Users/divya/OneDrive/Desktop/iris_modelv01.joblib"  # Replace with your model path

  # Example data point
  data_point = [1, 2, 3,3]  # Replace with your actual data point format

  prediction = predict(model_path, data_point)
  print('Prediction:', prediction)
