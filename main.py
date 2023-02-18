# Importing Essentials
import joblib
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict


# Fetch MNIST Dataset
mnist = fetch_openml("mnist_784", version=1, parser="auto")

# Divide Input and Label
X, y = mnist["data"], mnist["target"]
X = X.to_numpy()
y = y.to_numpy()
y = y.astype(np.uint8)

# Split Train and Test Set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

## -- Train KNeighborsClassifier() on our Training Dataset and Fine Tune it Using GridSearchCV -- ##
# Define the hyperparameters to be searched
param_grid = {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]}

# Create the KNeighborsClassifier model
knn = KNeighborsClassifier()

# Create a GridSearchCV object to search the hyperparameters
grid_search = GridSearchCV(knn, param_grid, cv=3)

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_n_neighbors = grid_search.best_params_["n_neighbors"]
best_weights = grid_search.best_params_["weights"]

# Train the KNeighborsClassifier model with the best hyperparameters
knn = KNeighborsClassifier(n_neighbors=best_n_neighbors, weights=best_weights)
knn.fit(X_train, y_train)


## ------------- Calcualte Its accuracy and f1_score ------------- ##
print(
    "--------------------- Performance of KNeighboursClassifier on Training Dataset ---------------------"
)

# Accuracy from cross_val_score()
accuracy = cross_val_score(knn, X_train, y_train, cv=3, scoring="accuracy").mean()*100
print(f"Accuracy = {accuracy}%")

# Get Predictions
y_train_pred = cross_val_predict(knn, X_train, y_train, cv=3)

# Precision Score Count
precision_value = precision_score(y_train, y_train_pred, average="macro")*100
print(f"Precision = {precision_value}%")

# Recall Score Count
recall_value = recall_score(y_train, y_train_pred, average="macro")*100
print(f"Recall = {recall_value}")

# f1_score Value
f1_score_value = f1_score(y_train, y_train_pred, average="macro")*100
print(f"f1_score = {f1_score_value}%")

print(
    "--------------------- Performance of KNeighboursClassifier on Tesiting Dataset ---------------------"
)

# Accuracy Using cross_val_score()
accuracy = cross_val_score(knn, X_test, y_test, cv=3, scoring="accuracy").mean()*100
print(f"Accuracy = {accuracy}%")

# Get Predictions
y_test_pred = cross_val_predict(knn, X_test, y_test, cv=3)

# Precision Score Count
precision_value = precision_score(y_test, y_test_pred, average="macro")*100
print(f"Precision = {precision_value}%")

# Recall Score Count
recall_value = recall_score(y_test, y_test_pred, average="macro")*100
print(f"Recall = {recall_value}")

# f1_score Value
f1_score_value = f1_score(y_test, y_test_pred, average="macro")*100
print(f"f1_score = {f1_score_value}%")


# Save Our Model Using Joblib
joblib.dump(knn, "mnist_classifier_96.pkl")