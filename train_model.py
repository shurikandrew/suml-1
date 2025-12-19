from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import label_binarize
import joblib
import os
import json
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("iris-model-zoo")

iris = load_iris()
flowers_data, flowers_labels = iris.data, iris.target

flowers_data_train, flowers_data_test, flowers_labels_train, flowers_labels_test = train_test_split(
    flowers_data,
    flowers_labels,
    test_size=0.3,
    random_state=10
)

models = {
    "KNN": KNeighborsClassifier(n_neighbors=4),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True),
    "RandomForest": RandomForestClassifier(random_state=10)
}

os.makedirs("artifacts", exist_ok=True)
os.makedirs("app", exist_ok=True)

best_f1 = 0.0
best_model_name = None
best_model = None
best_metrics = None
best_run_id = None

for model_name, model in models.items():

    with mlflow.start_run(run_name=model_name) as run:

        model.fit(flowers_data_train, flowers_labels_train)
        labels_predictions = model.predict(flowers_data_test)

        accuracy = accuracy_score(flowers_labels_test, labels_predictions)
        f1 = f1_score(flowers_labels_test, labels_predictions, average="macro")
        precision = precision_score(flowers_labels_test, labels_predictions, average="macro")
        recall = recall_score(flowers_labels_test, labels_predictions, average="macro")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        if hasattr(model, "predict_proba"):
            y_test_bin = label_binarize(flowers_labels_test, classes=[0, 1, 2])
            y_proba = model.predict_proba(flowers_data_test)
            roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr")
            mlflow.log_metric("roc_auc", roc_auc)

        mlflow.log_params(model.get_params())
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("version", "v1.0.0")

        cm = confusion_matrix(flowers_labels_test, labels_predictions)
        ConfusionMatrixDisplay(cm).plot()
        cm_path = f"artifacts/confusion_matrix_{model_name}.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

        report_path = f"artifacts/classification_report_{model_name}.txt"
        with open(report_path, "w") as f:
            f.write(classification_report(flowers_labels_test, labels_predictions))
        mlflow.log_artifact(report_path)

        mlflow.sklearn.log_model(model, name="model")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = model_name
            best_model = model
            best_metrics = {
                "accuracy": round(accuracy, 3),
                "f1_macro": round(f1, 3)
            }
            best_run_id = run.info.run_id

        print(f"{model_name} accuracy: {accuracy * 100:.2f}%")

final_model_path = "app/model.joblib"
joblib.dump(best_model, final_model_path)

model_meta = {
    "best_model": best_model_name,
    "metrics": best_metrics,
    "mlflow_run_id": best_run_id,
    "version": "v1.0.0"
}

with open("app/model_meta.json", "w") as f:
    json.dump(model_meta, f, indent=2)

print(f"\nBest model: {best_model_name}")
print("Saved app/model.joblib and app/model_meta.json")

client = MlflowClient()

registered_model_name = "IrisModel"
existing_models = [m.name for m in client.search_registered_models()]
if registered_model_name not in existing_models:
    client.create_registered_model(name=registered_model_name)

model_uri = f"runs:/{best_run_id}/model"
client.create_model_version(
    name=registered_model_name,
    source=model_uri,
    run_id=best_run_id
)

print(f"Registered model '{registered_model_name}' from run {best_run_id}")
