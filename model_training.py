import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_preprocessing import preprocess_data
from mlflow.tracking import MlflowClient

# Load and prepare data
df = preprocess_data(r"D:\GUVI_PROJECT_3\PROJECT_3\medical_insurance.csv")
df.drop(columns=['bmi_class'], inplace=True)

X = df.drop('charges', axis=1)
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluation function
def evaluate_model(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }

# Train and log model to MLflow
def train_and_log_model(name, model):
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate_model(y_test, preds)

        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name)

        print(f"\n{name} Model:")
        print(metrics)

# Get the best run ID automatically (based on R2)
def get_best_run_id(experiment_name="Default", metric_name="R2"):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=[f"metrics.{metric_name} DESC"],
        max_results=1
    )

    if runs:
        best_run_id = runs[0].info.run_id
        print(f"\n✅ Best Run ID (by {metric_name}): {best_run_id}")
        return best_run_id
    else:
        raise Exception("No runs found.")

# Main model training pipeline
if __name__ == "__main__":
    train_and_log_model("LinearRegression", LinearRegression())
    train_and_log_model("RandomForest", RandomForestRegressor(n_estimators=100, random_state=42))
    train_and_log_model("XGBoost", XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    train_and_log_model("DecisionTree", DecisionTreeRegressor(max_depth=5))
    train_and_log_model("RidgeRegression", Ridge(alpha=1.0))

    # Get and print the best Run ID
    get_best_run_id()
