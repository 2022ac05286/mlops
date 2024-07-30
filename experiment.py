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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import skops.io as sio

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

# Selecting the best features
KBest = SelectKBest(chi2, k="all")

# Define hyperparameters for different iterations
hyperparameters = [
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 200, "max_depth": 15},
    {"n_estimators": 150, "max_depth": 20}
]

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
        with open("metrics.txt", "a") as outfile:
            outfile.write(f"Iteration {iteration + 1} - Hyperparameters: {params} - Accuracy: {round(accuracy, 2)}, F1 Score: {round(f1, 2)}\n")

        # Confusion Matrix Plot
        cm = confusion_matrix(y_test, predictions, labels=complete_pipe.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=complete_pipe.classes_)
        disp.plot()
        plt.savefig(f"model_results_iteration_{iteration + 1}.png", dpi=120)
        plt.close()
        
        # Optionally save the pipeline
        sio.dump(complete_pipe, f"bank_pipeline_iteration_{iteration + 1}.skops")

print("Experiments completed.")
