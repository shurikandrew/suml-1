from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

iris = load_iris()
flowers_data, flowers_labels = iris.data, iris.target

flowers_data_train, flowers_data_test, flowers_labels_train, flowers_labels_test = train_test_split(flowers_data, flowers_labels, test_size=0.3, random_state=10)

model = KNeighborsClassifier(n_neighbors=4)
model.fit(flowers_data_train, flowers_labels_train)

labels_predictions = model.predict(flowers_data_test)
accuracy = accuracy_score(flowers_labels_test, labels_predictions)*100
print(f"Accuracy: {accuracy}%")

model_path = os.path.join("app", "model.joblib")
joblib.dump(model, model_path)