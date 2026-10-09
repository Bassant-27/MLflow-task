from sklearn.datasets import fetch_california_housing

import pandas as pd
import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
data = fetch_california_housing()
X = data.data
y = data.target

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=42)

mlflow.set_experiment("California Housing Experiments")

configs = [
    {"max_depth": 3, "learning_rate": 0.1},
    {"max_depth": 5, "learning_rate": 0.05},
    {"max_depth": 7, "learning_rate": 0.01}
]

results = []

for i, config in enumerate(configs, start=1):

    with mlflow.start_run(run_name=f"Run {i}"):

        model = GradientBoostingRegressor(
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        rmse = mean_squared_error(y_val, predictions) ** 0.5

        mae = mean_absolute_error(y_val, predictions)

        r2 = r2_score(y_val, predictions)

        mlflow.log_params(config)

        mlflow.log_metrics({
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })

        signature = infer_signature(X_train, model.predict(X_train))


        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=X_train[:5],
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        results.append({
            "Run": f"Run {i}",
            "Max Depth": config["max_depth"],
            "Learning Rate": config["learning_rate"],
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })

        print(f"\nRun {i}")
        print(f"Max Depth: {config['max_depth']}")
        print(f"Learning Rate: {config['learning_rate']}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2: {r2:.4f}")

best_result = min(results, key=lambda result: result["RMSE"])

for result in results:
    print(result)

print("\nBest Model:")
print(best_result)

print("\nStart the MLflow UI using:")
print("mlflow ui")