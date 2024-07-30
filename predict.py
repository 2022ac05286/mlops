import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from datetime import datetime

# Load the best model
model_path = "2024-07-30_17-47-24_0.84.joblib"  # Update this with the actual path of your saved best model
best_model = joblib.load(model_path)

# Load the test data
test_df = pd.read_csv("test.csv", index_col="id")

# Separate features from the original test data
X_test = test_df.copy()

# Process the test data
preprocessed_test_data = best_model.named_steps['preprocessor'].transform(X_test)

# Make predictions
predictions = best_model.named_steps['train'].predict(preprocessed_test_data)

# Add predictions to the original DataFrame
X_test['willExit'] = predictions

# Filter to keep only records where prediction value is 1
filtered_df = X_test[X_test['willExit'] == 1].copy()

# Convert the prediction value 1 to "Yes"
filtered_df['willExit'] = filtered_df['willExit'].map({1: 'Yes'})

# Save the updated DataFrame to a CSV file with a timestamp
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
output_filename = f"prediction_{timestamp}.csv"
filtered_df.to_csv(output_filename)

print(f"Predictions saved to {output_filename}.")
