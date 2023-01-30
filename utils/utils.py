import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle


def display(parameter: str, data, target: str):
    style.use("ggplot")
    pyplot.scatter(data[parameter], data[target])
    pyplot.xlabel("P")
    pyplot.ylabel("Target")
    pyplot.show()


def log_all_predictions(predictions, x_test, y_test):
    for x in range(len(predictions)):
        print("Prediction:", predictions[x], "Data:", x_test[x], "Target:",y_test[x])


def encode_label(encoder, data, column):
    return encoder.fit_transform(list(data[column]))

def save_model(model, name: str):
    with open(name, "wb") as f:
        pickle.dump(model, f)