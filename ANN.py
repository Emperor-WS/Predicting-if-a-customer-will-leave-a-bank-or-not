#Importing libraries

from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import model_from_json


#Importing DS
dataset = pd.read_csv("Churn_Modelling.csv")
x = dataset.iloc[:, 3: 13].values
y = dataset.iloc[:, 13].values

#Encoding independant variable
le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])

# One Hot Encoding the "Geography" column
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
x = x[:, 1:]

#Spilitting the dataset into the training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#Feature Scaling
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#Making the ANN
#Importing the Keras libraries and packages

#Initialising the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(units=6, kernel_initializer="uniform", activation='relu'))

#Adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer="uniform", activation='relu'))

#Adding the output layer
classifier.add(Dense(units=1, kernel_initializer="uniform", activation='sigmoid'))

#Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Fitting the ANN to the trainning set
classifier.fit(x_train, y_train, batch_size=10, epochs=200)

#Predicitng The Test Set Results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > .5)

#Making The Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)

# serialize model to JSON
model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("model.h5")
print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

#Predicitng The Test Set Results
y_pred = loaded_model.predict(x_test)
y_pred = (y_pred > .5)

#Making The Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)

print(loaded_model.predict(sc_x.transform(
    [[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
