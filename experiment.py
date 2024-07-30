import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime
import skops.io as sio
import joblib

# Loading the data
bank_df = pd.read_csv("train.csv", index_col="id", nrows=1000)
bank_df = bank_df.drop(["CustomerId", "Surname"], axis=1)
bank_df = bank_df.sample(frac=1)

# Splitting data into training and testing sets
X = bank_df.drop(["Exited"], axis=1)
y = bank_df.Exited

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=125)

# Identify numerical and categorical columns
cat_col = [1, 2]
num_col = [0, 3, 4, 5, 6, 7, 8, 9]

# Transformers for numerical data
numerical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", MinMaxScaler())]
)

# Transformers for categorical data
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder()),
    ]
)

# Combine pipelines using ColumnTransformer
preproc_pipe = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, num_col),
        ("cat", categorical_transformer, cat_col),
    ],
    remainder="passthrough",
)

# Fit the preprocessing pipeline
preproc_pipe.fit(X_train)



# Selecting the best features
KBest = SelectKBest(chi2, k="all")

# Define hyperparameters for different iterations
hyperparameters = [
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 150, "max_depth": 15}
]

# Initialize variables to track the best model
best_accuracy = 0
best_model = None
best_timestamp = ""

# MLflow Tracking
for iteration, params in enumerate(hyperparameters):
    with mlflow.start_run() as run:
        # Update model with hyperparameters
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=125
        )
        
        # KBest and model pipeline
        train_pipe = Pipeline(
            steps=[
                ("KBest", KBest),
                ("RFmodel", model),
            ]
        )

        # Combining the preprocessing and training pipelines
        complete_pipe = Pipeline(
            steps=[
                ("preprocessor", preproc_pipe),
                ("train", train_pipe),
            ]
        )
        
        # Train the model
        complete_pipe.fit(X_train, y_train)
        
        # Evaluate the model
        predictions = complete_pipe.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="macro")
        # Get the current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # Log parameters
        mlflow.log_param("iteration", iteration + 1)
        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("max_depth", params["max_depth"])
        mlflow.log_param("random_state", model.random_state)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        
        # Log model
        mlflow.sklearn.log_model(complete_pipe, "model")
        
        # Save metrics to file
        with open("experiments_metrics.txt", "a") as outfile:
            outfile.write(f"DateTime {timestamp}-Iteration {iteration + 1} - Hyperparameters: {params} - Accuracy: {round(accuracy, 2)}, F1 Score: {round(f1, 2)}\n")
        
        # Update best model if current model is better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = complete_pipe
            best_timestamp = timestamp

# Save the best model with the timestamp and accuracy in the filename
if best_model is not None:
    joblib.dump(best_model, f"{best_timestamp}_{round(best_accuracy, 2)}.joblib")

print("Experiments completed and Best Model Saved.")
