from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd

# Read in data
X_train = np.genfromtxt("data/train_features.csv")
y_train = np.genfromtxt("data/train_labels.csv")
X_test = np.genfromtxt("data/test_features.csv")
y_test = np.genfromtxt("data/test_labels.csv")

# Fit a model

clf =  RandomForestClassifier()
clf.fit(X_train, y_train)

# Get overall accuracy
acc = clf.score(X_test, y_test)

# Get precision and recall
y_score = clf.predict(X_test)
prec = precision_score(y_test, y_score)
rec = recall_score(y_test, y_score)
# Get the loss

pd.DataFrame({'predicted': y_score, 'actual': y_test}).to_csv('classes.csv', index=False)

with open("metrics.json", 'w') as outfile:
    json.dump({"accuracy": acc, "precision": prec, "recall": rec}, outfile)

# Plot it
disp = ConfusionMatrixDisplay.from_estimator(
    clf, X_test, y_test, normalize="true", cmap=plt.cm.Blues
)
plt.savefig("plot.png")
