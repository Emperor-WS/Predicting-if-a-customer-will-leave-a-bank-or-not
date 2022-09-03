#Importing libraries

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing DS
dataset = pd.read_csv("Churn_Modelling.csv")
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#Encoding independant variable
labelencoder_X1 = LabelEncoder()
x[:, 1] = labelencoder_X1.fit_transform(x[:, 1])
labelencoder_X2 = LabelEncoder()
x[:, 2] = labelencoder_X2.fit_transform(x[:, 2])
ct = ColumnTransformer([("Geography", OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
x = x[:, 1:]

#Spilitting the dataset into the training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#Fitting XGBoost to the trainning set
classifier = XGBClassifier()
classifier.fit(x_train, y_train)

#Predicitng The Test Set Results
y_pred = classifier.predict(x_test)

#Making The Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

#Applying k-Fold cross validation
accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10)
accuracies.mean()
accuracies.std()
