import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier

import pickle


from utils.utils import display, log_all_predictions, save_model

def train_linear_model(model_type: str, x, y):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=0.1
    )

    if model_type == "linear":
        model = linear_model.LinearRegression()
    else:
        model = KNeighborsClassifier(n_neighbors=7)

    model.fit(x_train, y_train)
    print(model.score(x_test, y_test))
    
    save_model(model, "studentsModel.pickle")


TARGET = "G3"
data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "health", "studytime", "freetime"]]

x = np.array(data.drop([TARGET], 1))
y = np.array(data[TARGET])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x, y, test_size=0.1
)

pickle_in = open("studentsModel.pickle", "rb")
linear = pickle.load(pickle_in)

predictions = linear.predict(x_test)
log_all_predictions(predictions, x_test, y_test)

display(parameter="G2", data=data, target=TARGET)
